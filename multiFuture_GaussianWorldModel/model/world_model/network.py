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


def gather_mode(x: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
    """Index mode dimension 2. x: [B, N, K, ...], mode_idx: [B, N] -> [B, N, ...]."""
    rest = x.shape[3:]
    idx = mode_idx.view(*mode_idx.shape, 1, *([1] * len(rest))).expand(-1, -1, 1, *rest)
    return x.gather(2, idx).squeeze(2)


def rotate_xy(xy: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate xy by angle (sin, cos). Pass -sin to rotate by the negative angle."""
    x = xy[..., 0]
    y = xy[..., 1]
    return torch.stack([x * cos - y * sin, x * sin + y * cos], dim=-1)


class LatentMultiAgentWorldModel(nn.Module):
    """Shared encoder + K-mode Gaussian decoder.

    decoder_type="per_step" (legacy): shared future transformer over T, K mode
    embeddings added per step, per-step MLP decoder [B, N, K, T, C] -> 6;
    classifier on latent_scene.

    decoder_type="direct": mode queries [B, N, K, C] = mode_fuse(latent + mode_k),
    traj_head MLP -> whole [T, 6] trajectory per query; classifier scores each
    mode from concat[query, embed(predicted waypoints)]. With mode_map_context,
    the queries first pass through a transformer decoder (self-attn over K,
    cross-attn to the agent's nearest map/TL tokens in its own frame).

    Outputs
    -------
    pred_delta_xy_norm_k: [B, N, K, T, 2]  ego frame
    pred_delta_xy_agent_norm_k: [B, N, K, T, 2]  agent frame (== ego frame if
                          agent_frame_decoding is off)
    agent_heading_sc:     [B, N, 2]  (sin, cos) of agent yaw relative to ego;
                          (0, 1) if agent_frame_decoding is off
    pred_yaw_vec_k:       [B, N, K, T, 2]
    pred_sigma_xy_norm_k: [B, N, K, T, 2]  in the frame of pred_delta_xy_agent_norm_k
    pred_sigma_xy_m_k:    [B, N, K, T, 2]
    mode_logits:          [B, N, K]
    mode_probs:           [B, N, K]
    top1_mode:            [B, N]
    pred_delta_xy_norm:   [B, N, T, 2]  top-1 mean (planner-compatible)
    pred_yaw_vec:         [B, N, T, 2]
    pred_sigma_xy_norm:   [B, N, T, 2]
    pred_sigma_xy_m:      [B, N, T, 2]
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        c = int(config.hidden_dim)
        k = int(config.num_modes)

        self.agent_step_encoder = _MLP(
            config.agent_feature_dim,
            c,
            c,
            dropout=config.dropout,
        )
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

        # K learned mode queries. Orthogonal rows keep modes distinct at init.
        # Rows are added to a LayerNorm'd latent of norm ~sqrt(c), so scale them
        # to match; unit-norm rows barely perturb the latent and modes collapse.
        self.mode_embeddings = nn.Parameter(torch.empty(k, c))
        embed_scale = (
            float(c) ** 0.5 if config.mode_embed_scale is None else float(config.mode_embed_scale)
        )
        if k <= c:
            nn.init.orthogonal_(self.mode_embeddings)
            with torch.no_grad():
                self.mode_embeddings.mul_(embed_scale)
        else:
            nn.init.normal_(self.mode_embeddings, std=embed_scale / float(c) ** 0.5)

        self.mode_fuse = _MLP(c, c, c, dropout=config.dropout)

        if config.decoder_type == "per_step":
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

            self.mode_classifier = _MLP(c, c, k, dropout=config.dropout)
            nn.init.zeros_(self.mode_classifier.net[-1].weight)
            nn.init.zeros_(self.mode_classifier.net[-1].bias)

            # mu_x, mu_y, raw_sigma_x, raw_sigma_y, sin d yaw, cos d yaw.
            self.state_decoder = _MLP(c, c, 6, dropout=config.dropout)
            last = self.state_decoder.net[-1]
            # Small nonzero weights so mode queries can separate trajectories at init.
            # Zero last-layer weights would make every mode identical (μ=0).
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)
            last.bias.data[2:4] = float(config.sigma_raw_bias)
        else:
            # One query per (agent, mode) decodes the whole [T, 6] trajectory.
            t = int(config.future_steps)
            self.traj_head = _MLP(c, c, t * 6, dropout=config.dropout)
            last = self.traj_head.net[-1]
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)
            last.bias.data.view(t, 6)[:, 2:4] = float(config.sigma_raw_bias)

            # Classifier scores each mode from its query + its predicted trajectory.
            waypoint_idx = [(i + 1) * t // 5 - 1 for i in range(5)]
            self.register_buffer(
                "waypoint_idx",
                torch.tensor(waypoint_idx, dtype=torch.long),
                persistent=False,
            )
            self.traj_embed = _MLP(2 * len(waypoint_idx), c, c, dropout=config.dropout)
            self.mode_cls = _MLP(2 * c, c, 1, dropout=config.dropout)
            nn.init.zeros_(self.mode_cls.net[-1].weight)
            nn.init.zeros_(self.mode_cls.net[-1].bias)

            if config.mode_map_context:
                # Per-agent local context: map/TL tokens keyed by (xy, dir) relative
                # to the agent, plus an always-valid null token for empty neighbourhoods.
                self.ctx_pos_encoder = _MLP(4, c, c, dropout=config.dropout)
                self.ctx_null_token = nn.Parameter(torch.zeros(1, 1, c))
                ctx_layer = nn.TransformerDecoderLayer(
                    d_model=c,
                    nhead=config.num_heads,
                    dim_feedforward=4 * c,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.mode_context = nn.TransformerDecoder(
                    ctx_layer,
                    num_layers=config.num_mode_context_layers,
                    norm=nn.LayerNorm(c),
                )

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

    def _attend_map_context(
        self,
        mode_queries: torch.Tensor,
        ctx_tokens: torch.Tensor,
        agent_xy: torch.Tensor,
        heading_sc: torch.Tensor,
        map_features: torch.Tensor,
        map_valid: torch.Tensor,
        tl_features: torch.Tensor,
        tl_valid: torch.Tensor,
    ) -> torch.Tensor:
        """mode_queries [B, N, K, C] attend to each agent's nearest map/TL tokens.

        ctx_tokens [B, M + L, C] are the scene-transformer outputs for map segments
        then TL points. Each segment is located by its point closest to the agent.
        """
        b, n, k, c = mode_queries.shape
        max_range = float(self.config.max_range)

        # Map: nearest valid point of each segment -> [B, N, M].
        map_xy = map_features[..., 0:2]
        map_dir = map_features[..., 2:4]
        diff = map_xy[:, None] - agent_xy[:, :, None, None, :]  # [B, N, M, P, 2]
        d2 = diff.square().sum(-1).masked_fill(~map_valid[:, None], float("inf"))
        seg_d2, near = d2.min(dim=-1)
        idx = near[..., None, None].expand(-1, -1, -1, 1, 2)
        map_rel = diff.gather(3, idx).squeeze(3)
        map_rel_dir = map_dir[:, None].expand(-1, n, -1, -1, -1).gather(3, idx).squeeze(3)

        # TL points have no direction.
        tl_rel = tl_features[:, None, :, 0:2] - agent_xy[:, :, None, :]  # [B, N, L, 2]
        tl_d2 = tl_rel.square().sum(-1).masked_fill(~tl_valid[:, None], float("inf"))

        rel_xy = torch.cat([map_rel, tl_rel], dim=2)
        rel_dir = torch.cat([map_rel_dir, torch.zeros_like(tl_rel)], dim=2)
        key_d2 = torch.cat([seg_d2, tl_d2], dim=2)
        radius = float(self.config.mode_context_radius_m) / max_range
        key_d2 = key_d2.masked_fill(key_d2 > radius * radius, float("inf"))

        s = min(int(self.config.mode_context_keys), key_d2.shape[-1])
        near_d2, sel = key_d2.topk(s, dim=-1, largest=False)
        sel_valid = torch.isfinite(near_d2)

        sin = -heading_sc[..., 0, None]
        cos = heading_sc[..., 1, None]
        # Agent frame, in units of 10 m (keys lie within ~radius).
        rel_xy = rotate_xy(rel_xy, sin, cos) * (max_range / 10.0)
        rel_dir = rotate_xy(rel_dir, sin, cos)
        pos = torch.cat([rel_xy, rel_dir], dim=-1)
        pos = pos.gather(2, sel[..., None].expand(-1, -1, -1, 4))
        pos = torch.where(sel_valid[..., None], pos, torch.zeros_like(pos))

        tok = ctx_tokens[:, None].expand(-1, n, -1, -1).gather(2, sel[..., None].expand(-1, -1, -1, c))
        memory = tok + self.ctx_pos_encoder(pos)
        memory = torch.cat(
            [self.ctx_null_token.expand(b * n, 1, c), memory.reshape(b * n, s, c)], dim=1
        )
        pad = torch.cat(
            [sel_valid.new_zeros(b * n, 1), ~sel_valid.reshape(b * n, s)], dim=1
        )
        out = self.mode_context(
            mode_queries.reshape(b * n, k, c),
            memory,
            memory_key_padding_mask=pad,
        )
        return out.view(b, n, k, c)

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
        k = int(self.config.num_modes)
        t = int(self.config.future_steps)

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
        scene_tokens = self.scene_transformer(
            scene_tokens,
            src_key_padding_mask=~scene_valid,
        )
        scene_tokens = self.scene_norm(scene_tokens)
        latent_scene = scene_tokens[:, :n]
        latent_scene = latent_scene * agent_valid[..., None].to(latent_scene.dtype)

        if self.config.agent_frame_decoding:
            # Last history step is the current step: features 4:6 = sin/cos of yaw rel. ego.
            heading_sc = history[:, :, -1, 4:6]
        else:
            heading_sc = torch.zeros_like(history[:, :, -1, 4:6])
            heading_sc[..., 1] = 1.0

        latent_future = None
        if self.config.decoder_type == "per_step":
            # Shared future rollout: still [B * N, T, C], independent of K.
            future = latent_scene[:, :, None, :] + self.future_time_embedding
            future = future.reshape(b * n, t, c)
            future = self.future_dynamics(future)
            future = self.future_norm(future)
            latent_future = future.reshape(b, n, t, c)

            # Broadcast K mode queries onto the shared future, then one decoder.
            mode_latents = latent_future[:, :, None, :, :] + self.mode_embeddings.view(1, 1, k, 1, c)
            mode_latents = self.mode_fuse(mode_latents)
            decoded = self.state_decoder(mode_latents)
            mode_logits = self.mode_classifier(latent_scene)
        else:
            # Fuse modes before decoding: one query per (agent, mode).
            mode_queries = self.mode_fuse(
                latent_scene[:, :, None, :] + self.mode_embeddings.view(1, 1, k, c)
            )
            if self.config.mode_map_context:
                mode_queries = self._attend_map_context(
                    mode_queries,
                    scene_tokens[:, n:],
                    history[:, :, -1, 0:2],
                    heading_sc,
                    map_features,
                    map_valid,
                    tl_features,
                    tl_valid,
                )
            decoded = self.traj_head(mode_queries).view(b, n, k, t, 6)
            waypoints = decoded[..., self.waypoint_idx, :2].flatten(-2)
            traj_feat = self.traj_embed(waypoints)
            mode_logits = self.mode_cls(torch.cat([mode_queries, traj_feat], dim=-1)).squeeze(-1)

        pred_delta_xy_agent_norm_k = decoded[..., :2]
        sin = heading_sc[..., 0, None, None]
        cos = heading_sc[..., 1, None, None]
        pred_delta_xy_norm_k = rotate_xy(pred_delta_xy_agent_norm_k, sin, cos)
        raw_sigma_k = decoded[..., 2:4]
        sigma_xy_norm_k = F.softplus(raw_sigma_k) + float(self.config.sigma_eps)
        pred_yaw_vec_k = F.normalize(decoded[..., 4:6], dim=-1, eps=1e-6)

        mode_probs = torch.softmax(mode_logits, dim=-1)

        valid_scale_k = agent_valid[:, :, None, None, None].to(decoded.dtype)
        pred_delta_xy_norm_k = pred_delta_xy_norm_k * valid_scale_k
        pred_delta_xy_agent_norm_k = pred_delta_xy_agent_norm_k * valid_scale_k
        sigma_xy_norm_k = sigma_xy_norm_k * valid_scale_k
        pred_yaw_vec_k = pred_yaw_vec_k * valid_scale_k
        raw_sigma_k = raw_sigma_k * valid_scale_k
        valid_agent = agent_valid[..., None].to(mode_logits.dtype)
        mode_logits = mode_logits * valid_agent
        mode_probs = mode_probs * valid_agent

        top1_mode = mode_probs.argmax(dim=-1)
        pred_delta_xy_norm = gather_mode(pred_delta_xy_norm_k, top1_mode)
        pred_yaw_vec = gather_mode(pred_yaw_vec_k, top1_mode)
        pred_sigma_xy_norm = gather_mode(sigma_xy_norm_k, top1_mode)

        out = {
            "pred_delta_xy_norm_k": pred_delta_xy_norm_k,
            "pred_delta_xy_agent_norm_k": pred_delta_xy_agent_norm_k,
            "agent_heading_sc": heading_sc,
            "pred_yaw_vec_k": pred_yaw_vec_k,
            "pred_sigma_xy_norm_k": sigma_xy_norm_k,
            "pred_sigma_xy_m_k": sigma_xy_norm_k * float(self.config.max_range),
            "pred_raw_sigma_xy_k": raw_sigma_k,
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "top1_mode": top1_mode,
            "pred_delta_xy_norm": pred_delta_xy_norm,
            "pred_yaw_vec": pred_yaw_vec,
            "pred_sigma_xy_norm": pred_sigma_xy_norm,
            "pred_sigma_xy_m": pred_sigma_xy_norm * float(self.config.max_range),
            "latent_scene": latent_scene,
        }
        if latent_future is not None:
            out["latent_future"] = latent_future
        return out
