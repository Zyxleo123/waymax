from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WorldModelConfig


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentMultiAgentWorldModel(nn.Module):
    """Encoder -> latent dynamics -> decoder world model.

    Inputs
    ------
    agent_history: [B, N, H, D]
        Ego-relative historical state for all tracked objects (including ego).
    agent_history_valid: [B, N, H]
    agent_valid: [B, N]
        Current validity. Invalid agent tokens are masked in scene attention.
    map_features: [B, M, P, map_dim]
    map_valid: [B, M, P]
    traffic_light_features: [B, L, tl_dim]
    traffic_light_valid: [B, L]

    Outputs
    -------
    pred_delta_xy_norm: [B, N, T, 2]
        Future displacement from each agent's current position, expressed in the
        current ego frame and normalized by config.max_range.
    pred_yaw_vec: [B, N, T, 2]
        sin/cos of future yaw delta relative to the agent's current yaw.
    latent_scene: [B, N, C]
        Interaction-aware per-agent latent at the anchor time.
    latent_future: [B, N, T, C]
        Predicted future latent sequence before the physical-state decoder.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        c = int(config.hidden_dim)

        self.agent_step_encoder = _MLP(
            config.agent_feature_dim,
            c,
            c,
            dropout=config.dropout,
        )
        # A GRU provides a stable history encoder without requiring every padded
        # agent to have a non-empty attention sequence.
        self.agent_history_gru = nn.GRU(c, c, batch_first=True)

        self.map_point_encoder = _MLP(
            config.map_attr_dim,
            c,
            c,
            dropout=config.dropout,
        )
        self.tl_encoder = _MLP(
            config.tl_attr_dim,
            c,
            c,
            dropout=config.dropout,
        )

        self.agent_type_token = nn.Parameter(torch.zeros(1, 1, c))
        self.map_type_token = nn.Parameter(torch.zeros(1, 1, c))
        self.tl_type_token = nn.Parameter(torch.zeros(1, 1, c))

        scene_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=config.num_heads,
            dim_feedforward=4 * c,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scene_transformer = nn.TransformerEncoder(
            scene_layer,
            num_layers=config.num_scene_layers,
            enable_nested_tensor=False,
        )
        self.scene_norm = nn.LayerNorm(c)

        self.future_time_embedding = nn.Parameter(
            torch.zeros(1, 1, config.future_steps, c)
        )
        nn.init.normal_(self.future_time_embedding, std=0.02)

        future_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=config.num_heads,
            dim_feedforward=4 * c,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.future_dynamics = nn.TransformerEncoder(
            future_layer,
            num_layers=config.num_future_layers,
            enable_nested_tensor=False,
        )
        self.future_norm = nn.LayerNorm(c)
        self.state_decoder = _MLP(c, c, 4, dropout=config.dropout)

    @staticmethod
    def _masked_map_pool(
        point_tokens: torch.Tensor,
        point_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [B,M,P,C] -> [B,M,C]. A large finite negative number avoids NaNs in
        # mixed precision while still making invalid points lose the max.
        mask = point_valid.to(dtype=torch.bool)
        masked = torch.where(
            mask[..., None],
            point_tokens,
            torch.full_like(point_tokens, -1.0e4),
        )
        pooled = masked.max(dim=2).values
        seg_valid = mask.any(dim=2)
        pooled = torch.where(seg_valid[..., None], pooled, torch.zeros_like(pooled))
        return pooled, seg_valid

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        history = features["agent_history"].float()
        history_valid = features["agent_history_valid"].bool()
        agent_valid = features["agent_valid"].bool()
        map_features = features["map_features"].float()
        map_valid = features["map_valid"].bool()
        tl_features = features["traffic_light_features"].float()
        tl_valid = features["traffic_light_valid"].bool()

        b, n, h, _ = history.shape
        c = int(self.config.hidden_dim)

        # Temporal per-agent encoder.
        step_tokens = self.agent_step_encoder(history)
        step_tokens = step_tokens * history_valid[..., None].to(step_tokens.dtype)
        flat = step_tokens.reshape(b * n, h, c)
        _, h_n = self.agent_history_gru(flat)
        agent_tokens = h_n[-1].reshape(b, n, c)
        agent_tokens = agent_tokens * agent_valid[..., None].to(agent_tokens.dtype)
        agent_tokens = agent_tokens + self.agent_type_token

        # Map polyline encoder.
        map_point_tokens = self.map_point_encoder(map_features)
        map_tokens, map_seg_valid = self._masked_map_pool(map_point_tokens, map_valid)
        map_tokens = map_tokens + self.map_type_token

        # Traffic-light encoder.
        tl_tokens = self.tl_encoder(tl_features) + self.tl_type_token
        tl_tokens = tl_tokens * tl_valid[..., None].to(tl_tokens.dtype)

        scene_tokens = torch.cat([agent_tokens, map_tokens, tl_tokens], dim=1)
        scene_valid = torch.cat([agent_valid, map_seg_valid, tl_valid], dim=1)

        # PyTorch expects True for padding positions.
        scene_tokens = self.scene_transformer(
            scene_tokens,
            src_key_padding_mask=~scene_valid,
        )# [B, N + M + L, C] = [B, 208, 256]: 208 agents, 256 features per agent.
        # N = max_agents = 128 
        # M = max_map_segments = 64
        # L = max_tl_points = 16
        scene_tokens = self.scene_norm(scene_tokens)
        latent_scene = scene_tokens[:, :n]
        latent_scene = latent_scene * agent_valid[..., None].to(latent_scene.dtype)

        # Latent future rollout. This predicts a latent sequence first; the
        # decoder then maps it to explicit physical future states.
        future = latent_scene[:, :, None, :] + self.future_time_embedding
        future = future.reshape(b * n, self.config.future_steps, c)
        future = self.future_dynamics(future)
        future = self.future_norm(future)
        latent_future = future.reshape(b, n, self.config.future_steps, c)

        decoded = self.state_decoder(latent_future)
        pred_delta_xy_norm = decoded[..., :2]
        pred_yaw_vec = F.normalize(decoded[..., 2:4], dim=-1, eps=1e-6)

        valid_scale = agent_valid[:, :, None, None].to(decoded.dtype)
        pred_delta_xy_norm = pred_delta_xy_norm * valid_scale
        pred_yaw_vec = pred_yaw_vec * valid_scale

        return {
            "pred_delta_xy_norm": pred_delta_xy_norm,
            "pred_yaw_vec": pred_yaw_vec,
            "latent_scene": latent_scene,
            "latent_future": latent_future,
        }
