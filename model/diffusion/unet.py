from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from model.diffusion.modules.attention_2 import AttentionBlock


def sinusoidal_timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10_000) -> jnp.ndarray:
    t = jnp.asarray(t)
    if not jnp.issubdtype(t.dtype, jnp.floating):
        t = t.astype(jnp.float32)
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
    return emb


class Conv1dBlock(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, rngs: nnx.Rngs, kernel: int = 3, groups: int = 8):
        self.conv = nnx.Conv(
            in_features=in_ch,
            out_features=out_ch,
            kernel_size=(kernel,),
            padding="SAME",
            rngs=rngs,
        )
        self.gn = nnx.GroupNorm(num_features=out_ch, num_groups=min(groups, out_ch), epsilon=1e-5, rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        h_btc = self.conv(x_btc)
        h_btc = self.gn(h_btc)
        return nnx.silu(h_btc)


class Identity1d(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class ResBlock1d(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, rngs: nnx.Rngs, groups: int = 8):
        self.time = nnx.Conv(in_features=time_ch, out_features=in_ch, kernel_size=(1,), padding="SAME", rngs=rngs)
        self.block1 = Conv1dBlock(in_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.skip = Identity1d() if in_ch == out_ch else nnx.Conv(in_features=in_ch, out_features=out_ch, kernel_size=(1,), padding="SAME", rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray, time_feat_b1c: jnp.ndarray) -> jnp.ndarray:
        """Apply residual block.

        Scene conditioning is applied by the surrounding attention blocks.

        Args:
            x_btc: [B, T, C_in] input features
            time_feat_b1c: [B, 1, C_time] time embedding

        Returns:
            [B, T, C_out] output features
        """
        time_b1c = self.time(time_feat_b1c)
        h = x_btc + time_b1c
        h = self.block1(h)
        h = self.block2(h)

        if isinstance(self.skip, Identity1d):
            return h + x_btc
        skip = self.skip(x_btc)
        return h + skip


class Downsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(4,), strides=(2,), padding=((1, 1),), rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x_btc)


class Upsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        x_btc = jax.image.resize(x_btc, shape=(x_btc.shape[0], x_btc.shape[1] * 2, x_btc.shape[2]), method="nearest")
        return self.conv(x_btc)


class Projector(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, in_feat: jnp.ndarray) -> jnp.ndarray:
        h = self.fc1(in_feat)
        h = nnx.silu(h)
        return self.fc2(h)
    

class UNet1DConditioned(nnx.Module):
    def __init__(
        self,
        in_ch: int,
        rngs: nnx.Rngs,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        time_emb_dim: int = 128,
        hidden_dim: int = 512,
        groups: int = 8,
        horizon: int = 25,
    ):
        self.in_ch = in_ch
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.n_down = len(ch_mult)
        self.horizon = horizon

        # Condition projection: project tokens to base_ch dimension for cross-attention
        self.cond_proj = nnx.Linear(cond_dim, base_ch, rngs=rngs)
        self.time_proj = Projector(time_emb_dim, base_ch, hidden_dim=hidden_dim, rngs=rngs)
        self.stem = nnx.Conv(in_features=in_ch, out_features=base_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

        ch = base_ch
        down_blocks = []
        downsamples = []
        skip_channels = []
        down_attn_blocks = []

        for i, m in enumerate(ch_mult):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                down_blocks.append(ResBlock1d(ch, out_ch, base_ch, rngs=rngs, groups=groups))
                down_attn_blocks.append(AttentionBlock(out_ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs))
                ch = out_ch
            skip_channels.append(ch)
            downsamples.append(Downsample1d(ch, rngs=rngs) if i != len(ch_mult) - 1 else Identity1d())

        self.mid_block1 = ResBlock1d(ch, ch, base_ch, rngs=rngs, groups=groups)
        self.mid_attn_block1 = AttentionBlock(ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs)
        self.mid_block2 = ResBlock1d(ch, ch, base_ch, rngs=rngs, groups=groups)
        self.mid_attn_block2 = AttentionBlock(ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs)

        up_blocks = []
        upsamples = []
        up_attn_blocks = []
        for i, m in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                up_blocks.append(ResBlock1d(ch + skip_channels[i], out_ch, base_ch, rngs=rngs, groups=groups))
                up_attn_blocks.append(AttentionBlock(out_ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs))
                ch = out_ch
            upsamples.append(Upsample1d(ch, rngs=rngs) if i != 0 else Identity1d())

        # Register containers as NNX modules in a single assignment.
        self.down_blocks = nnx.List(down_blocks)
        self.down_attn_blocks = nnx.List(down_attn_blocks)
        self.downsamples = nnx.List(downsamples)
        self.skip_channels = tuple(skip_channels)
        self.up_blocks = nnx.List(up_blocks)
        self.up_attn_blocks = nnx.List(up_attn_blocks)
        self.upsamples = nnx.List(upsamples)
        self.out_norm = nnx.GroupNorm(num_features=ch, num_groups=min(groups, ch), epsilon=1e-5, rngs=rngs)
        self.out_conv = nnx.Conv(in_features=ch, out_features=in_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """Forward pass conditioned on scene tokens.

        Args:
            x: [B, C, T] input trajectory
            t: [B] timestep values
            cond: [B, n_cond_tokens, cond_dim] scene tokens

        Returns:
            [B, C, T] network prediction
        """
        divisor = 2 ** (self.n_down - 1)
        if self.horizon % divisor != 0:
            target_t = ((self.horizon // divisor) + 1) * divisor
            x = jnp.pad(x, ((0, 0), (0, 0), (0, target_t - self.horizon)), mode="constant")

        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        # Project condition tokens: [B, n_tokens, cond_dim] -> [B, n_tokens, base_ch]
        cond = self.cond_proj(cond)  # [B, n_tokens, base_ch]
        time_feat = self.time_proj(t_emb)[:, None, :]

        h = self.stem(jnp.swapaxes(x, 1, 2))

        skips = []
        rb = 0
        for i in range(self.n_down):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[rb](h, time_feat)
                h = self.down_attn_blocks[rb](h, cond)
                rb += 1
            skips.append(h)
            h = self.downsamples[i](h)

        h = self.mid_block1(h, time_feat)
        h = self.mid_attn_block1(h, cond)
        h = self.mid_block2(h, time_feat)
        h = self.mid_attn_block2(h, cond)

        rb_up = 0
        for i in range(self.n_down):
            skip = skips.pop()
            if h.shape[1] != skip.shape[1]:
                min_len = min(h.shape[1], skip.shape[1])
                h = h[:, :min_len, :]
                skip = skip[:, :min_len, :]

            for _ in range(self.num_res_blocks):
                h = jnp.concatenate([h, skip], axis=-1)
                h = self.up_blocks[rb_up](h, time_feat)
                h = self.up_attn_blocks[rb_up](h, cond)
                rb_up += 1

            h = self.upsamples[i](h)

        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)
        return jnp.swapaxes(h[:, :self.horizon, :], 1, 2)

