from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .mlp import MLP


class PointNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layer: int = 4,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        end_layer_activation: bool = True,
        dropout_p: Optional[float] = None,
        pool_mode: str = "max",
    ) -> None:
        super().__init__()
        if pool_mode != "max":
            raise ValueError(f"pool_mode {pool_mode} not supported")

        self.pool_mode = pool_mode
        self.hidden_dim = hidden_dim
        self.input_mlp = MLP(
            [input_dim, hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
        )

        self.mlp_layers = nn.ModuleList(
            [
                MLP(
                    [hidden_dim, hidden_dim // 2],
                    dropout_p=dropout_p,
                    use_layernorm=use_layernorm,
                    use_batchnorm=use_batchnorm,
                )
                for _ in range(n_layer - 1)
            ]
        )

        self.mlp_out = MLP(
            [hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
            end_layer_activation=end_layer_activation,
        )

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None, *, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        if x.ndim != 3:
            raise ValueError(f"Input x should be 3D tensor, got {tuple(x.shape)}")

        b, n, _ = x.shape
        if valid is None:
            valid = torch.ones((b, n), dtype=torch.bool, device=x.device)

        has_any = torch.any(valid, dim=1)
        neg_inf = torch.tensor(-1e9, dtype=x.dtype, device=x.device)

        h = self.input_mlp(x)

        for mlp in self.mlp_layers:
            feature_encoded = mlp(h)
            feature_pooled = torch.where(valid[..., None], feature_encoded, neg_inf).amax(dim=1, keepdim=True)
            feature_pooled = torch.where(has_any[:, None, None], feature_pooled, torch.zeros_like(feature_pooled))
            h = torch.cat([feature_encoded, feature_pooled.expand(b, n, feature_pooled.shape[-1])], dim=-1)

        h = torch.where(valid[..., None], h, neg_inf).amax(dim=1, keepdim=False)
        h = torch.where(has_any[:, None], h, torch.zeros_like(h))
        return self.mlp_out(h)