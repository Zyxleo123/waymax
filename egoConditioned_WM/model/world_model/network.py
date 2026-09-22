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
    """Shared scene encoder + optional ego-candidate-conditioned decoder.

    Expensive modules (history GRU, map/TL, scene transformer, future
    transformer) run once per scene. Each ES candidate only pays for:
      ego GRU over T steps + a C-dim fuse MLP + the state decoder.

    Typical ES cost (B=1, K=64, N=128, T=25, C=256): ~2e5 decoder tokens,
    vs rerunning a 208-token scene transformer K times.

    Outputs keep the unimodal keys when K=1:
      pred_delta_xy_norm: [B, N, T, 2]
    and always also:
      pred_delta_xy_norm_k: [B, K, N, T, 2]
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        c = int(config.hidden_dim)

        self.agent_step_encoder = _MLP(
            config.agent_feature_dim, c, c, dropout=config.dropout
        )
        self.agent_history_gru = nn.GRU(c, c, batch_first=True)
        self.map_point_encoder = _MLP(config.map_attr_dim, c, c, dropout=config.dropout)
        self.tl_encoder = _MLP(config.tl_attr_dim, c, c, dropout=config.dropout)

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

        self.future_time_embedding = nn.Parameter(torch.zeros(1, 1, config.future_steps, c))
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

        self.ego_step_encoder = _MLP(config.ego_future_dim, c, c, dropout=config.dropout)
        self.ego_history_gru = nn.GRU(c, c, batch_first=True)
        # Residual fuse: zero last layer so conditioned == unconditioned at init.
        self.ego_fuse = _MLP(2 * c, c, c, dropout=config.dropout)
        nn.init.zeros_(self.ego_fuse.net[-1].weight)
        nn.init.zeros_(self.ego_fuse.net[-1].bias)

        self.state_decoder = _MLP(c, c, 6, dropout=config.dropout)
        last = self.state_decoder.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        last.bias.data[2:4] = float(config.sigma_raw_bias)

    @staticmethod
    def _masked_map_pool(
        point_tokens: torch.Tensor,
        point_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def encode_scene(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run history/map/TL + scene transformer once. Safe to cache across ES candidates."""
        history = features["agent_history"].float()
        history_valid = features["agent_history_valid"].bool()
        agent_valid = features["agent_valid"].bool()
        map_features = features["map_features"].float()
        map_valid = features["map_valid"].bool()
        tl_features = features["traffic_light_features"].float()
        tl_valid = features["traffic_light_valid"].bool()

        b, n, h, _ = history.shape
        c = int(self.config.hidden_dim)

        step_tokens = self.agent_step_encoder(history)
        step_tokens = step_tokens * history_valid[..., None].to(step_tokens.dtype)
        flat = step_tokens.reshape(b * n, h, c)
        _, h_n = self.agent_history_gru(flat)
        agent_tokens = h_n[-1].reshape(b, n, c)
        agent_tokens = agent_tokens * agent_valid[..., None].to(agent_tokens.dtype)
        agent_tokens = agent_tokens + self.agent_type_token

        map_point_tokens = self.map_point_encoder(map_features)
        map_tokens, map_seg_valid = self._masked_map_pool(map_point_tokens, map_valid)
        map_tokens = map_tokens + self.map_type_token

        tl_tokens = self.tl_encoder(tl_features) + self.tl_type_token
        tl_tokens = tl_tokens * tl_valid[..., None].to(tl_tokens.dtype)

        scene_tokens = torch.cat([agent_tokens, map_tokens, tl_tokens], dim=1)
        scene_valid = torch.cat([agent_valid, map_seg_valid, tl_valid], dim=1)
        scene_tokens = self.scene_transformer(scene_tokens, src_key_padding_mask=~scene_valid)
        scene_tokens = self.scene_norm(scene_tokens)
        latent_scene = scene_tokens[:, :n] * agent_valid[..., None].to(scene_tokens.dtype)
        return {"latent_scene": latent_scene, "agent_valid": agent_valid}

    def encode_ego_future(
        self,
        ego_future: torch.Tensor,
        ego_future_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ego_future: [B, K, T, F] or [B, T, F] -> [B, K, C]."""
        if ego_future.ndim == 3:
            ego_future = ego_future[:, None]
        b, k, t, _ = ego_future.shape
        c = int(self.config.hidden_dim)
        tokens = self.ego_step_encoder(ego_future.float())
        if ego_future_valid is not None:
            if ego_future_valid.ndim == 2:
                ego_future_valid = ego_future_valid[:, None]
            tokens = tokens * ego_future_valid[..., None].to(tokens.dtype)
        flat = tokens.reshape(b * k, t, c)
        _, h_n = self.ego_history_gru(flat)
        return h_n[-1].reshape(b, k, c)

    def decode_from_scene(
        self,
        scene: Mapping[str, torch.Tensor],
        *,
        ego_future: torch.Tensor | None = None,
        ego_future_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent_scene = scene["latent_scene"]
        agent_valid = scene["agent_valid"].bool()
        b, n, c = latent_scene.shape
        t = int(self.config.future_steps)

        future = latent_scene[:, :, None, :] + self.future_time_embedding
        future = future.reshape(b * n, t, c)
        future = self.future_dynamics(future)
        future = self.future_norm(future)
        latent_future = future.reshape(b, n, t, c)

        use_ego = bool(self.config.ego_conditioning) and ego_future is not None
        if use_ego:
            ego_lat = self.encode_ego_future(ego_future, ego_future_valid)
            k = ego_lat.shape[1]
            fut_k = latent_future[:, None, :, :, :].expand(b, k, n, t, c)
            ego_k = ego_lat[:, :, None, None, :].expand(b, k, n, t, c)
            fused = fut_k + self.ego_fuse(torch.cat([fut_k, ego_k], dim=-1))
            decoded = self.state_decoder(fused)
        else:
            k = 1
            decoded = self.state_decoder(latent_future)[:, None]

        pred_delta_k = decoded[..., :2]
        raw_sigma_k = decoded[..., 2:4]
        sigma_k = F.softplus(raw_sigma_k) + float(self.config.sigma_eps)
        yaw_k = F.normalize(decoded[..., 4:6], dim=-1, eps=1e-6)

        valid_scale = agent_valid[:, None, :, None, None].to(decoded.dtype)
        pred_delta_k = pred_delta_k * valid_scale
        sigma_k = sigma_k * valid_scale
        yaw_k = yaw_k * valid_scale
        raw_sigma_k = raw_sigma_k * valid_scale

        pred_delta = pred_delta_k[:, 0]
        pred_yaw = yaw_k[:, 0]
        pred_sigma = sigma_k[:, 0]
        return {
            "pred_delta_xy_norm": pred_delta,
            "pred_yaw_vec": pred_yaw,
            "pred_sigma_xy_norm": pred_sigma,
            "pred_sigma_xy_m": pred_sigma * float(self.config.max_range),
            "pred_raw_sigma_xy": raw_sigma_k[:, 0],
            "pred_delta_xy_norm_k": pred_delta_k,
            "pred_yaw_vec_k": yaw_k,
            "pred_sigma_xy_norm_k": sigma_k,
            "pred_sigma_xy_m_k": sigma_k * float(self.config.max_range),
            "latent_scene": latent_scene,
            "latent_future": latent_future,
            "num_candidates": torch.tensor(k, device=latent_scene.device),
        }

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scene = self.encode_scene(features)
        ego = features.get("ego_future") if self.config.ego_conditioning else None
        ego_valid = features.get("ego_future_valid") if ego is not None else None
        return self.decode_from_scene(scene, ego_future=ego, ego_future_valid=ego_valid)
