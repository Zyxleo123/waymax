from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .mlp import MLP


class CrossAttention(nn.Module):
    """Multi-head cross-attention with explicit token masking."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.out_proj = MLP([query_dim, hidden_dim, query_dim])

        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        query_btc: torch.Tensor,
        context_bnc: torch.Tensor,
        context_mask_bn: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bq, tq, cq = query_btc.shape
        bc = context_bnc.shape[0]
        if bq != bc:
            if bq == 1:
                query_btc = query_btc.expand(bc, tq, cq)
            else:
                raise ValueError("Batch dimension of query and context must match, or query batch==1 for broadcasting")

        b, t, _ = query_btc.shape
        n_tokens = context_bnc.shape[1]

        if context_mask_bn is None:
            context_mask_bn = torch.ones((b, n_tokens), device=context_bnc.device, dtype=torch.bool)

        query_btc = self.query_norm(query_btc)
        context_bnc = self.context_norm(context_bnc)

        q = self.q_proj(query_btc)
        k = self.k_proj(context_bnc)
        v = self.v_proj(context_bnc)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2).reshape(b * self.num_heads, t, self.head_dim)
        k = k.view(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2).reshape(b * self.num_heads, n_tokens, self.head_dim)
        v = v.view(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2).reshape(b * self.num_heads, n_tokens, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        mask = context_mask_bn[:, None, :].expand(b, self.num_heads, n_tokens).reshape(b * self.num_heads, 1, n_tokens)
        attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.view(b, self.num_heads, t, self.head_dim).transpose(1, 2).reshape(b, t, cq)
        return self.out_proj(out)


class CrossAttentionLayers(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, hidden_dim: int = 512, num_heads: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [CrossAttention(query_dim, context_dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, query_btc: torch.Tensor, context_bnc: torch.Tensor, context_mask_bn: torch.Tensor) -> torch.Tensor:
        h = query_btc
        for layer in self.attn_layers:
            h = h + layer(h, context_bnc, context_mask_bn)
        return h