from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.vla.modules.attention import SelfAttentionLayers
from model.vla.modules.mlp import MLP


class TrajectoryEncoder(nn.Module):
	def __init__(
		self,
		trajectory_dim: int = 5,
		hidden_dim: int = 512,
		cond_dim: int = 256,
		num_heads: int = 4,
		num_attn_layers: int = 4,
		max_steps: int = 64,
	) -> None:
		super().__init__()
		self.trajectory_dim = int(trajectory_dim)
		self.hidden_dim = int(hidden_dim)
		self.cond_dim = int(cond_dim)
		self.max_steps = int(max_steps)

		self.input_proj = MLP([trajectory_dim, hidden_dim, hidden_dim, cond_dim])
		self.time_embedding = nn.Parameter(torch.zeros(1, self.max_steps, cond_dim))
		self.self_attention = SelfAttentionLayers(
			dim=cond_dim,
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			num_layers=num_attn_layers,
		)
		self.output_proj = MLP([cond_dim, hidden_dim, cond_dim], end_layer_activation=False)

	def forward(
		self,
		trajectory: torch.Tensor,
		valid: Optional[torch.Tensor] = None,
		*,
		deterministic: bool = True,
	) -> tuple[torch.Tensor, torch.Tensor]:
		if trajectory.ndim != 3:
			raise ValueError(f"trajectory must have shape [B, T, D], got {tuple(trajectory.shape)}")
		if trajectory.shape[-1] != self.trajectory_dim:
			raise ValueError(
				f"trajectory last dim must be {self.trajectory_dim}, got {trajectory.shape[-1]}"
			)

		batch_size, num_steps, _ = trajectory.shape
		if num_steps > self.max_steps:
			raise ValueError(
				f"trajectory has {num_steps} steps, exceeding max_steps={self.max_steps}"
			)

		if valid is None:
			valid = torch.ones((batch_size, num_steps), dtype=torch.bool, device=trajectory.device)
		else:
			valid = valid.to(device=trajectory.device)
			if valid.dtype != torch.bool:
				valid = valid > 0
			if valid.shape != (batch_size, num_steps):
				raise ValueError(
					f"valid mask must have shape [{batch_size}, {num_steps}], got {tuple(valid.shape)}"
				)

		tokens = self.input_proj(trajectory.reshape(-1, self.trajectory_dim), deterministic=deterministic)
		tokens = tokens.reshape(batch_size, num_steps, self.cond_dim)
		tokens = tokens + self.time_embedding[:, :num_steps, :]
		tokens = self.self_attention(tokens, mask=valid)
		tokens = self.output_proj(tokens, deterministic=deterministic)
		tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
		return tokens, valid


__all__ = ["TrajectoryEncoder"]