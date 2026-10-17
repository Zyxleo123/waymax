"""World-model-conditioned trajectory discriminator (Gaussian Future Encoder + road-edge tokens).

Extends `TrajectoryDiscriminator` (scene, map, goal and trajectory encoders are reused
unchanged) with two candidate-relative, time-aligned cross-attention branches:

  WM branch    tokens per (agent, mode, step) from the cached world-model mixture (6 modes,
               probabilities and std): an MLP over the mode-mean pose, step velocity, agent
               size and log std, a depthwise temporal encoder over the 25 steps and a
               mode-probability embedding. The candidate's step-tau query attends to every
               (agent, mode) token at the same tau; keys/values and the logit bias also carry
               geometry of that mode relative to the candidate pose at tau (offset and heading
               in the candidate frame, distance, box clearance, Mahalanobis distance and the
               std rotated into the candidate frame). log p is added to the logits, so the
               attention follows the mixture weights.
  Edge branch  static road-edge tokens (point + direction); geometry adds the signed offset of
               the candidate from each edge's tangent line, so "which side of the road edge"
               is directly available for offroad reasoning.

Each candidate gets one score; the per-step branch outputs are pooled (mean and max over
time) and fused with the base pooled vector.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from es_discriminator.model import TrajectoryDiscriminator, _MLP

VARIANT = "full_gaussian"
STEP_DT = 0.2


def rotate_cov_torch(cov: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """(sxx, syy, sxy) expressed in a frame rotated by -angle (i.e. R(-a) S R(-a)^T)."""
    c, s = torch.cos(angle), torch.sin(angle)
    sxx, syy, sxy = cov[..., 0], cov[..., 1], cov[..., 2]
    return torch.stack([c * c * sxx + 2 * c * s * sxy + s * s * syy,
                        s * s * sxx - 2 * c * s * sxy + c * c * syy,
                        -c * s * sxx + c * s * syy + (c * c - s * s) * sxy], -1)


class _Temporal(nn.Module):
    """Two residual depthwise Conv1d blocks over the step axis of [..., T, H]."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = x.reshape(-1, shape[-2], shape[-1])
        for conv, norm in zip(self.convs, self.norms):
            y = norm(y + torch.nn.functional.gelu(conv(y.transpose(1, 2)).transpose(1, 2)))
        return y.reshape(shape)


class _GeoAttention(nn.Module):
    """Candidate-step queries over tokens with a geometry MLP in logits and values.

    q [B,K,T,H]; tokens [B,Tt,N,H] (Tt = T for time-aligned tokens, 1 for static);
    geo [B,K,T,N,F]; bias [B,1|K,T|1,N] additive logit bias (or None); mask [B,1,Tt,N].
    """

    def __init__(self, hidden: int, heads: int, geo_dim: int, geo_hidden: int = 32) -> None:
        super().__init__()
        self.heads, self.dh = heads, hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.geo = nn.Sequential(nn.Linear(geo_dim, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, geo_hidden))
        self.geo_logit = nn.Linear(geo_hidden, heads)
        self.geo_value = nn.Linear(geo_hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, q, tokens, geo, bias, mask):
        b, k, t, h = q.shape
        nh, dh = self.heads, self.dh
        static = tokens.shape[1] == 1
        qh = self.q(q).view(b, k, t, nh, dh)
        kh = self.k(tokens).view(b, tokens.shape[1], tokens.shape[2], nh, dh)
        vh = self.v(tokens).view(b, tokens.shape[1], tokens.shape[2], nh, dh)
        g = self.geo(geo)                                                           # [B,K,T,N,G]
        if static:
            logits = torch.einsum("bktnd,bmnd->bktmn", qh, kh[:, 0])
        else:
            logits = torch.einsum("bktnd,btmnd->bktmn", qh, kh)                     # same step only
        logits = logits / math.sqrt(dh) + self.geo_logit(g)
        if bias is not None:
            logits = logits + bias.unsqueeze(-1)
        keep = mask.unsqueeze(-1).to(logits.dtype)
        logits = logits.masked_fill(keep == 0, -1e9)
        attn = torch.softmax(logits, dim=3) * keep                                   # rows with no token -> 0
        if static:
            v_tok = torch.einsum("bktmn,bmnd->bktnd", attn, vh[:, 0])
        else:
            v_tok = torch.einsum("bktmn,btmnd->bktnd", attn, vh)
        g_pooled = torch.einsum("bktm,bktmg->bktg", attn.mean(-1), g)
        return self.out(v_tok.reshape(b, k, t, h) + self.geo_value(g_pooled))


class WMTrajectoryDiscriminator(TrajectoryDiscriminator):
    def __init__(self, hidden: int = 128, heads: int = 4, dropout: float = 0.1, *,
                 pos_scale: float = 20.0, vel_scale: float = 5.0) -> None:
        super().__init__(hidden, heads, dropout)
        self.variant = VARIANT
        self.pos_scale, self.vel_scale = float(pos_scale), float(vel_scale)
        self.mode_embed = _MLP(10, hidden)
        self.mode_temporal = _Temporal(hidden)
        self.mode_index = nn.Embedding(6, hidden)
        self.prob_embed = nn.Linear(1, hidden)
        self.wm_attn = _GeoAttention(hidden, heads, 12)
        self.edge_embed = _MLP(5, hidden)
        self.edge_attn = _GeoAttention(hidden, heads, 7)
        self.head = nn.Sequential(
            nn.LayerNorm(7 * hidden),
            nn.Linear(7 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    # ---- per-branch geometry ----------------------------------------------------------

    def _wm_tokens(self, wm: dict) -> torch.Tensor:
        mean = wm["mean"]                                                            # [B,A,M,T,3]
        b, a, m, t, _ = mean.shape
        prev = torch.cat([wm["current"][:, :, None, None, :2].expand(b, a, m, 1, 2), mean[..., :-1, :2]], dim=3)
        vel = (mean[..., :2] - prev) / STEP_DT
        size = wm["size"][:, :, None, None].expand(b, a, m, t, 2)
        feats = [mean[..., :2] / self.pos_scale, torch.sin(mean[..., 2:3]), torch.cos(mean[..., 2:3]),
                 vel / self.vel_scale, size / 10.0, 0.5 * torch.log(wm["cov"][..., :2].clamp(min=1e-4))]
        tok = self.mode_temporal(self.mode_embed(torch.cat(feats, -1)))             # [B,A,M,T,H]
        tok = tok + self.mode_index.weight[:m].view(1, 1, m, 1, -1)
        tok = tok + self.prob_embed(torch.log(wm["probs"].clamp(min=1e-4))[..., None])[:, :, :, None]
        return tok.permute(0, 3, 1, 2, 4).reshape(b, t, a * m, -1)                  # [B,T,A*M,H]

    def _wm_geometry(self, cand_xy, cand_yaw, ego_size, wm: dict) -> torch.Tensor:
        mean = wm["mean"]
        b, a, m, t, _ = mean.shape
        agent_xy = mean[..., :2].permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 2)  # [B,1,T,N,2]
        agent_yaw = mean[..., 2].permute(0, 3, 1, 2).reshape(b, 1, t, a * m)
        d = agent_xy - cand_xy.unsqueeze(3)                                         # [B,K,T,N,2]
        c, s = torch.cos(cand_yaw).unsqueeze(-1), torch.sin(cand_yaw).unsqueeze(-1)
        dx, dy = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        rel = agent_yaw - cand_yaw.unsqueeze(-1)
        size = wm["size"][:, :, None].expand(b, a, m, 2).reshape(b, 1, 1, a * m, 2)
        reach = 0.5 * torch.linalg.vector_norm(size, dim=-1) + 0.5 * torch.linalg.vector_norm(ego_size, dim=-1).view(b, 1, 1, 1)
        geo = [dx / 10.0, dy / 10.0, torch.log1p(dist), torch.sin(rel), torch.cos(rel), (dist - reach) / 5.0,
               torch.exp(-dist / 2.0), torch.exp(-(dist - reach).clamp(min=0) / 1.0)]
        cov = wm["cov"].permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 3) + torch.tensor(
            [0.05, 0.05, 0.0], device=mean.device)
        det = (cov[..., 0] * cov[..., 1] - cov[..., 2] ** 2).clamp(min=1e-6)
        maha = (cov[..., 1] * d[..., 0] ** 2 - 2 * cov[..., 2] * d[..., 0] * d[..., 1]
                + cov[..., 0] * d[..., 1] ** 2) / det
        cand_cov = rotate_cov_torch(cov.expand(b, cand_xy.shape[1], t, a * m, 3), cand_yaw.unsqueeze(-1))
        geo += [torch.log1p(maha), 0.5 * torch.log(cand_cov[..., 0].clamp(min=1e-4)),
                0.5 * torch.log(cand_cov[..., 1].clamp(min=1e-4)), 0.25 * torch.log(det).expand_as(maha)]
        return torch.stack(geo, -1)

    def _edge_geometry(self, cand_xy, cand_yaw, ego_size, edges: dict) -> torch.Tensor:
        exy = edges["xy"][:, None, None]                                            # [B,1,1,E,2]
        edir = edges["dir"][:, None, None]
        d = cand_xy.unsqueeze(3) - exy                                              # [B,K,T,E,2]
        lateral = edir[..., 0] * d[..., 1] - edir[..., 1] * d[..., 0]               # signed, left of edge > 0
        along = edir[..., 0] * d[..., 0] + edir[..., 1] * d[..., 1]
        dist = torch.sqrt((d * d).sum(-1) + 1e-6)
        rel = torch.atan2(edir[..., 1], edir[..., 0]) - cand_yaw.unsqueeze(-1)
        half_w = 0.5 * ego_size[:, 1].view(-1, 1, 1, 1)
        return torch.stack([lateral / 5.0, along / 10.0, torch.log1p(dist), torch.sin(rel), torch.cos(rel),
                            (lateral.abs() - half_w) / 2.0, torch.exp(-dist / 2.0)], -1)

    # ---- scoring ------------------------------------------------------------------------

    def score_groups(self, batch: dict, candidates: torch.Tensor, cand_xyyaw: torch.Tensor | None = None,
                     wm: dict | None = None, edges: dict | None = None, ego_size: torch.Tensor | None = None,
                     chunk: int = 0) -> torch.Tensor:
        """context [B,...], features [B,K,T,6], raw candidates [B,K,T,3] (m, rad), WM / edge dicts -> [B,K]."""
        return self.score_encoded(self.encode_context(batch, wm, edges, ego_size), candidates, cand_xyyaw, chunk)

    def encode_context(self, batch: dict, wm: dict, edges: dict, ego_size: torch.Tensor) -> dict:
        """Candidate-independent part (scene, WM and edge tokens); reusable across ES iterations."""
        tokens, valid, goal_token = self._encode_scene(batch)
        edge_tok = self.edge_embed(torch.cat([edges["xy"] / self.pos_scale, edges["dir"],
                                              edges["mask"][..., None].float()], -1))  # [B,E,H]
        wm_tok = self._wm_tokens(wm)
        return dict(tokens=tokens, valid=valid, goal_token=goal_token, wm=wm, wm_tok=wm_tok, edges=edges,
                    edge_tok=edge_tok, ego_size=ego_size)

    def score_encoded(self, enc: dict, candidates: torch.Tensor, cand_xyyaw: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        count = candidates.shape[1]
        if enc["wm"]["mean"].shape[3] != candidates.shape[2]:
            raise ValueError(f"WM steps {enc['wm']['mean'].shape[3]} != candidate steps {candidates.shape[2]}")
        parts = []
        step = chunk if chunk and chunk < count else count
        for s0 in range(0, count, step):
            sl = slice(s0, s0 + step)
            parts.append(self._score_chunk(enc["tokens"], enc["valid"], enc["goal_token"], candidates[:, sl],
                                           cand_xyyaw[:, sl], enc["wm"], enc["wm_tok"], enc["edges"],
                                           enc["edge_tok"], enc["ego_size"]))
        return torch.cat(parts, dim=1)

    def _score_chunk(self, tokens, valid, goal_token, candidates, cand_xyyaw, wm, wm_tok, edges, edge_tok, ego_size):
        bsz, count, steps, _ = candidates.shape
        traj = self.traj_embed(candidates.reshape(bsz * count, steps, -1))
        encoded_traj, _ = self.traj_gru(traj)
        cross, _ = self.cross_attn(encoded_traj.reshape(bsz, count * steps, -1), tokens, tokens,
                                   key_padding_mask=~valid, need_weights=False)
        cross = cross.reshape(bsz * count, steps, -1)
        q = encoded_traj.reshape(bsz, count, steps, -1)
        cand_xy = cand_xyyaw[..., :2]
        cand_yaw = torch.atan2(candidates[..., 4], candidates[..., 5])               # path yaw, as in features
        pooled = [encoded_traj[:, -1], cross.mean(dim=1), goal_token[:, 0].repeat_interleave(count, dim=0)]

        e_geo = self._edge_geometry(cand_xy, cand_yaw, ego_size, edges)
        e_out = self.edge_attn(q, edge_tok[:, None], e_geo, None, edges["mask"][:, None, None, :])
        pooled += [e_out.mean(2).reshape(bsz * count, -1), e_out.amax(2).reshape(bsz * count, -1)]

        b, a, m, t, _ = wm["mean"].shape
        w_geo = self._wm_geometry(cand_xy, cand_yaw, ego_size, wm)
        mask = (wm["valid"][:, :, None] & (wm["probs"] > 0)).reshape(b, 1, 1, a * m).expand(b, 1, t, a * m)
        bias = torch.log(wm["probs"].clamp(min=1e-4)).reshape(b, 1, 1, a * m)
        w_out = self.wm_attn(q, wm_tok, w_geo, bias, mask)
        pooled += [w_out.mean(2).reshape(bsz * count, -1), w_out.amax(2).reshape(bsz * count, -1)]
        return self.head(torch.cat(pooled, -1)).squeeze(-1).reshape(bsz, count)
