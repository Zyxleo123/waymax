from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from model.diffusion.modules.mlp import MLP



class SelfAttention(nnx.Module):
    """Multi-head self-attention over a sequence of feature tokens."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        rngs: nnx.Rngs = None,
    ):
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)

        # Kept consistent with your CrossAttention implementation.
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

        self.norm = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(
        self,
        x_btc: jnp.ndarray,
        mask_bt: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply self-attention.

        Args:
            x_btc:
                [B, T, C]

            mask_bt:
                [B, T] boolean mask.
                True = valid token.
                False = padding token.

        Returns:
            [B, T, C]
        """
        B, T, C = x_btc.shape

        # Pre-normalization
        x_btc = self.norm(x_btc)

        Q = self.q_proj(x_btc)  # [B, T, C]
        K = self.k_proj(x_btc)
        V = self.v_proj(x_btc)

        # [B, T, C]
        # ->
        # [B, H, T, D]
        Q = Q.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        K = K.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        V = V.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Flatten batch and heads:
        # [B * H, T, D]
        Q = Q.reshape(B * self.num_heads, T, self.head_dim)
        K = K.reshape(B * self.num_heads, T, self.head_dim)
        V = V.reshape(B * self.num_heads, T, self.head_dim)

        # [B * H, T, T]
        attn_scores = (
            Q @ K.transpose(0, 2, 1)
        ) * self.scale

        # ---------------------------------------------------------
        # Key mask
        #
        # Each query is prevented from attending to invalid keys.
        # [B, T]
        # -> [B, H, 1, T]
        # -> [B * H, 1, T]
        # ---------------------------------------------------------
        key_mask = jnp.broadcast_to(
            mask_bt[:, None, None, :],
            (B, self.num_heads, 1, T),
        ).reshape(
            B * self.num_heads, 1, T
        )

        attn_scores = jnp.where(
            key_mask,
            attn_scores,
            jnp.array(-1e9, dtype=attn_scores.dtype),
        )

        attn_weights = jax.nn.softmax(
            attn_scores,
            axis=-1,
        )

        # [B * H, T, D]
        out = attn_weights @ V

        # [B * H, T, D]
        # -> [B, H, T, D]
        # -> [B, T, H, D]
        # -> [B, T, C]
        out = (
            out.reshape(
                B,
                self.num_heads,
                T,
                self.head_dim,
            )
            .transpose(0, 2, 1, 3)
            .reshape(B, T, C)
        )

        out = self.out_proj(out)

        # Optional but usually useful:
        # force padded query positions to remain zero.
        out = out * mask_bt[..., None].astype(out.dtype)

        return out

class CrossAttention(nnx.Module):
    """Multi-head cross-attention: query from features, key/value from condition tokens.

    This implementation assumes `context_mask_bn` (bool array of shape [B, n_tokens])
    is always provided by the caller. There is no Python branching on mask presence
    to avoid jax.jit retrace overhead.
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        rngs: nnx.Rngs = None,
    ):
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nnx.Linear(query_dim, query_dim, rngs=rngs)
        self.k_proj = nnx.Linear(context_dim, query_dim, rngs=rngs)
        self.v_proj = nnx.Linear(context_dim, query_dim, rngs=rngs)
        self.out_proj = nnx.Linear(query_dim, query_dim, rngs=rngs)

        self.query_norm = nnx.LayerNorm(query_dim, rngs=rngs)
        self.context_norm = nnx.LayerNorm(context_dim, rngs=rngs)

    def __call__(self, query_btc: jnp.ndarray, context_bnc: jnp.ndarray, context_mask_bn: jnp.ndarray) -> jnp.ndarray:
        """Apply cross-attention.

        Args:
            query_btc: [B_q, T, C] query (may be B_q==1 and will be broadcast over batch)
            context_bnc: [B, n_tokens, C_context] condition tokens
            context_mask_bn: [B, n_tokens] boolean mask for condition tokens (True=keep)

        Returns:
            [B, T, C] attended features (batch size equals `context_bnc.shape[0]`)
        """
        # Allow query to be a single set of tokens that is broadcast across the batch.
        Bq, Tq, Cq = query_btc.shape
        Bc = context_bnc.shape[0]
        if Bq != Bc:
            if Bq == 1:
                query_btc = jnp.broadcast_to(query_btc, (Bc, Tq, Cq))
            else:
                raise ValueError("Batch dimension of query and context must match, or query batch==1 for broadcasting")

        B = context_bnc.shape[0]
        T = query_btc.shape[1]
        n_tokens = context_bnc.shape[1]

        # Normalize query and context
        query_btc = self.query_norm(query_btc)
        context_bnc = self.context_norm(context_bnc)

        # Project query, key, value
        Q = self.q_proj(query_btc)  # [B, T, C]
        K = self.k_proj(context_bnc)  # [B, n_tokens, C]
        V = self.v_proj(context_bnc)  # [B, n_tokens, C]

        # Reshape for multi-head attention: [B*num_heads, T, head_dim]
        Q = Q.reshape((B, T, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B * self.num_heads, T, self.head_dim))
        K = K.reshape((B, n_tokens, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B * self.num_heads, n_tokens, self.head_dim))
        V = V.reshape((B, n_tokens, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B * self.num_heads, n_tokens, self.head_dim))

        # Attention scores: [B*num_heads, T, n_tokens]
        attn_scores = (Q @ K.transpose((0, 2, 1))) * self.scale

        # Broadcast boolean mask -> [B*num_heads, 1, n_tokens]
        mask_bhn = jnp.tile(context_mask_bn[:, None, :], (1, self.num_heads, 1))  # [B, H, n]
        mask_flat = mask_bhn.reshape((B * self.num_heads, 1, n_tokens))
        attn_scores = jnp.where(mask_flat, attn_scores, jnp.array(-1e9, dtype=attn_scores.dtype))

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)  # [B*num_heads, T, n_tokens]

        # Apply attention to values: [B*num_heads, T, head_dim]
        out = attn_weights @ V

        # Reshape back: [B, T, C]
        out = out.reshape((B, self.num_heads, T, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B, T, Cq))
        out = self.out_proj(out)
        return out


class AttentionBlock(nnx.Module):
    """Transformer-style block:

        x = x + SelfAttention(x)
        x = x + CrossAttention(x, context)
        x = x + FFN(x)

    Self-attention and cross-attention use pre-norm internally.
    FFN uses its own pre-norm.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        attn_hidden_dim: int,
        ffn_hidden_dim: int,
        num_heads: int = 8,
        n_layers: int = 2,
        rngs: nnx.Rngs = None,
    ):
        self.self_attn_layers = nnx.List([
            SelfAttention(
                dim=dim,
                hidden_dim=attn_hidden_dim,
                num_heads=num_heads,
                rngs=rngs,
            )
            for _ in range(n_layers)
        ])
        self.cross_attn_layers = nnx.List([
            CrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                hidden_dim=attn_hidden_dim,
                num_heads=num_heads,
                rngs=rngs,
            )
            for _ in range(n_layers)
        ])
        self.ffn_norms = nnx.List([
            nnx.LayerNorm(dim, rngs=rngs)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nnx.List([
            MLP(
                [dim, ffn_hidden_dim, dim],
                rngs=rngs,
            )
            for _ in range(n_layers)
        ])

    def __call__(
        self,
        x_btc: jnp.ndarray,
        context_bnc: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply attention block.

        Args:
            x_btc:
                [B, T, C]

            context_bnc:
                [B, N, C_context]

            x_mask_bt:
                [B, T]
                True for valid x tokens.

            context_mask_bn:
                [B, N]
                True for valid context tokens.

        Returns:
            [B, T, C]
        """
        x_mask_bt = jnp.ones((x_btc.shape[0], x_btc.shape[1]), dtype=bool)
        context_mask_bn = jnp.ones((context_bnc.shape[0], context_bnc.shape[1]), dtype=bool)
        for self_attn, cross_attn, ffn_norm, ffn in zip(
            self.self_attn_layers,
            self.cross_attn_layers,
            self.ffn_norms,
            self.ffn_layers,
        ):
            x_btc = x_btc + self_attn(
                x_btc,
                x_mask_bt,
            )

            cross_out = cross_attn(
                x_btc,
                context_bnc,
                context_mask_bn,
            )

            cross_out = (
                cross_out
                * x_mask_bt[..., None].astype(cross_out.dtype)
            )

            x_btc = x_btc + cross_out

            ffn_out = ffn(
                ffn_norm(x_btc)
            )

            ffn_out = (
                ffn_out
                * x_mask_bt[..., None].astype(ffn_out.dtype)
            )

            x_btc = x_btc + ffn_out
        return x_btc

class CrossAttentionLayers(nnx.Module):
    """Stack of cross-attention layers with residual connections."""
    def __init__(self, query_dim: int, context_dim: int, hidden_dim: int = 512, num_heads: int = 8, num_layers: int = 4, rngs: nnx.Rngs = None):
        self.attn_layers = nnx.List([CrossAttention(query_dim, context_dim, hidden_dim, num_heads, rngs=rngs) for _ in range(num_layers)])

    def __call__(self, query_btc: jnp.ndarray, context_bnc: jnp.ndarray, context_mask_bn: jnp.ndarray) -> jnp.ndarray:
        h = query_btc
        for layer in self.attn_layers:
            attn_out = layer(h, context_bnc, context_mask_bn)
            h = h + attn_out  # Residual connection
        return h


__all__ = ["CrossAttention", "CrossAttentionLayers"]
