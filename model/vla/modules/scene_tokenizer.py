from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from model.vla.modules.attention import CrossAttentionLayers
from model.vla.modules.mlp import MLP
from model.vla.modules.point_net import PointNet
from model.vla.modules.trajectory_encoder import TrajectoryEncoder


class SceneTokenizer(nn.Module):
    def __init__(
        self,
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 15,
        map_attr_dim: int = 25,
        tl_attr_dim: int = 11,
        use_ego_trajectory: bool = False,
        trajectory_dim: int = 5,
        hidden_dim: int = 1024,
        cond_dim: int = 256,
        num_tokens: int = 32,
        num_heads: int = 4,
        num_attn_layers: int = 4,
        trajectory_hidden_dim: int | None = None,
        trajectory_max_steps: int = 64,
    ) -> None:
        super().__init__()
        self.token_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_other_agents = 128
        self.use_ego_trajectory = bool(use_ego_trajectory)
        self.other_pos_embedding = nn.Parameter(torch.zeros(1, self.max_other_agents, self.token_dim))
        self.ego_tokenizer = MLP([ego_dim, hidden_dim, hidden_dim, cond_dim])
        self.goal_tokenizer = MLP([goal_dim, hidden_dim, hidden_dim, cond_dim])
        self.other_tokenizer = MLP([other_dim, hidden_dim, hidden_dim, cond_dim])
        self.tl_tokenizer = MLP([tl_attr_dim, hidden_dim, hidden_dim, cond_dim])
        self.map_tokenizer = PointNet(map_attr_dim, cond_dim)
        self.cond_tokens = nn.Parameter(torch.zeros(1, num_tokens, cond_dim))
        self.attention = CrossAttentionLayers(cond_dim, cond_dim, num_heads=num_heads, num_layers=num_attn_layers)
        self.ego_trajectory_encoder = None
        if self.use_ego_trajectory:
            trajectory_hidden_dim = hidden_dim if trajectory_hidden_dim is None else int(trajectory_hidden_dim)
            self.ego_trajectory_encoder = TrajectoryEncoder(
                trajectory_dim=trajectory_dim,
                hidden_dim=trajectory_hidden_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                num_attn_layers=num_attn_layers,
                max_steps=trajectory_max_steps,
            )

    def _tokenize_scene_features(
        self,
        input_features: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        ego_state = input_features["ego_state"]
        goal_xy = input_features["goal_xy"]
        remaining_timesteps = input_features["remaining_timesteps"]
        other_states = input_features["other_states"]
        other_valid = input_features["other_valid"]
        map_features = input_features["map_features"]
        map_valid = input_features["map_valid"]
        traffic_light_features = input_features["traffic_light_features"]
        traffic_light_valid = input_features["traffic_light_valid"]
        ego_trajectory = input_features.get("ego_trajectory")
        ego_trajectory_valid = input_features.get("ego_trajectory_valid")

        if remaining_timesteps.ndim == 1:
            remaining_timesteps = remaining_timesteps[:, None]

        goal_input = torch.cat([goal_xy, remaining_timesteps], dim=-1)

        batch_size, num_others, other_dim = other_states.shape
        _, num_map_segments, num_points_per_segment, map_dim = map_features.shape
        _, num_tl, tl_dim = traffic_light_features.shape

        ego_token = self.ego_tokenizer(ego_state, deterministic=deterministic).reshape(batch_size, 1, self.token_dim)
        goal_token = self.goal_tokenizer(goal_input, deterministic=deterministic).reshape(batch_size, 1, self.token_dim)

        other_token = self.other_tokenizer(
            other_states.reshape(-1, other_dim),
            deterministic=deterministic,
        ).reshape(batch_size, num_others, self.token_dim)
        other_token = other_token + self.other_pos_embedding[:, :num_others, :]

        map_token = self.map_tokenizer(
            map_features.reshape(-1, num_points_per_segment, map_dim),
            map_valid.reshape(-1, num_points_per_segment),
            deterministic=deterministic,
        ).reshape(batch_size, num_map_segments, self.token_dim)

        tl_token = self.tl_tokenizer(
            traffic_light_features.reshape(-1, tl_dim),
            deterministic=deterministic,
        ).reshape(batch_size, num_tl, self.token_dim)

        traj_token = None
        traj_valid = None
        if self.ego_trajectory_encoder is not None and ego_trajectory is not None:
            if ego_trajectory.ndim != 3:
                raise ValueError(f"ego_trajectory must have shape [B, T, D], got {tuple(ego_trajectory.shape)}")
            traj_token, traj_valid = self.ego_trajectory_encoder(
                ego_trajectory,
                valid=ego_trajectory_valid,
                deterministic=deterministic,
            )

        token_parts = [ego_token, goal_token, other_token, map_token, tl_token]
        valid_parts = [
            torch.ones((batch_size, 2), dtype=torch.bool, device=ego_token.device),
            other_valid,
            map_valid.any(dim=2),
            traffic_light_valid,
        ]
        if traj_token is not None and traj_valid is not None:
            token_parts.append(traj_token)
            valid_parts.append(traj_valid)

        scene_device = ego_token.device
        scene_tokens = torch.cat(token_parts, dim=1)
        scene_valid = torch.cat(valid_parts, dim=1)

        return self.attention(
            query_btc=self.cond_tokens,
            context_bnc=scene_tokens,
            context_mask_bn=scene_valid,
        )

    def forward(self, input_features: Mapping[str, torch.Tensor], *, deterministic: bool = True) -> torch.Tensor:
        return self._tokenize_scene_features(input_features, deterministic=deterministic)


__all__ = ["SceneTokenizer"]
