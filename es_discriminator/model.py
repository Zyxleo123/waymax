"""Scene and trajectory encoders shared by `real_es.wm_model.WMTrajectoryDiscriminator`; requires PyTorch.

`TrajectoryDiscriminator` holds the agent / map / traffic-light / goal encoders and the
candidate GRU + cross-attention; the WM discriminator extends it with world-model and
road-edge branches. Scores are uncalibrated logits (higher = better).
"""

from __future__ import annotations

import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, hidden: int = 96, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.agent_embed = _MLP(8, hidden)
        self.map_embed = _MLP(5, hidden)
        self.light_embed = _MLP(4, hidden)
        self.goal_embed = _MLP(4, hidden)  # [goal_xy, current_ego_vxy]
        self.traj_embed = _MLP(6, hidden)
        self.traj_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _encode_scene(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = batch["agents"].shape[0]
        agent_hist_valid = batch["agent_mask"].bool()
        encoded_hist = self.agent_embed(batch["agents"])
        denom = agent_hist_valid.sum(dim=2).clamp(min=1).unsqueeze(-1)
        agent_tokens = (encoded_hist * agent_hist_valid.unsqueeze(-1)).sum(dim=2) / denom
        agent_valid = agent_hist_valid.any(dim=2)

        map_tokens = self.map_embed(batch["map"])
        map_valid = batch["map_mask"].bool()
        light_tokens = self.light_embed(batch["lights"])
        light_valid = batch["lights_mask"].bool()
        goal_token = self.goal_embed(torch.cat([batch["goal"], batch["ego_vel"]], dim=-1)).unsqueeze(1)
        tokens = torch.cat([agent_tokens, map_tokens, light_tokens, goal_token], dim=1)
        valid = torch.cat([
            agent_valid, map_valid, light_valid,
            torch.ones((bsz, 1), dtype=torch.bool, device=tokens.device),
        ], dim=1)
        return tokens, valid, goal_token


def path_yaw_torch(xy: torch.Tensor, fallback_yaw: torch.Tensor, min_step_m: float = 0.05) -> torch.Tensor:
    """Heading of travel from the origin through xy [...,T,2]; holds the last defined heading
    (or `fallback_yaw` [...,T] before any motion) while the step is below `min_step_m`."""
    diff = torch.diff(torch.cat([torch.zeros_like(xy[..., :1, :]), xy], dim=-2), dim=-2)
    heading = torch.atan2(diff[..., 1], diff[..., 0])
    steps = torch.arange(xy.shape[-2], device=xy.device).expand(xy.shape[:-1])
    moving_idx = torch.where(diff.norm(dim=-1) > min_step_m, steps, torch.full_like(steps, -1))
    last_idx = torch.cummax(moving_idx, dim=-1).values
    held = torch.gather(heading, -1, last_idx.clamp(min=0))
    return torch.where(last_idx >= 0, held, fallback_yaw)


def make_candidate_features_torch(xyyaw: torch.Tensor, dt: float = 0.2, *, pos_scale: float = 100.0,
                                  vel_scale: float = 25.0) -> torch.Tensor:
    """Convert [...,T,3] ego-frame meters/radians into [...,T,6], yaw from the path."""
    if xyyaw.shape[-1] != 3 or dt <= 0:
        raise ValueError("Expected [...,T,3] local trajectory and positive dt")
    xy = xyyaw[..., :2]
    velocity = torch.diff(
        torch.cat([torch.zeros_like(xy[..., :1, :]), xy], dim=-2), dim=-2
    ) / dt
    heading = path_yaw_torch(xy, xyyaw[..., 2]).unsqueeze(-1)
    return torch.cat([xy / pos_scale, velocity / vel_scale, heading.sin(), heading.cos()], dim=-1)
