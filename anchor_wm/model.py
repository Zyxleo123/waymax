"""AnchorWM network: MF-Gaussian scene encoder + anchor-query decoder + ego-plan scorer.

Encoder      modules and names copied from multiFuture_GaussianWorldModel's
             LatentMultiAgentWorldModel, so `warm_start_encoder` loads its weights as is.
Decoder      one query per (agent, anchor): encoder token + embed(anchor trajectory) (+ goal token
             for the SDC) -> MLP -> per-step residual around the anchor, sigma, yaw, and a logit
             scored from the query and its predicted waypoints (MF's `direct` head, with the learned
             mode embeddings replaced by fixed k-means anchors). Decoding is in each agent's frame.
Scorer       es_discriminator.real_es.wm_model.WMTrajectoryDiscriminator ported onto the shared
             encoder: candidate-step queries cross-attend to the scene tokens, to time-aligned
             (agent, mode) tokens of the decoded futures (with candidate-relative geometry) and to
             road-edge tokens. The WM tokens include the decoder's query embeddings, so with
             `grad_scale > 0` the scoring loss trains the decoder and encoder (the joint arm A2);
             `grad_scale = 0` detaches every WM input of the scorer (the separate arm A1).
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor_wm.config import DYNAMIC_TYPES, SUB_SCORES, AnchorWMConfig
from anchor_wm.geometry import box_corners, candidate_features, edge_signed_distance, path_yaw, rotate_xy, sat_gap

ENCODER_PREFIXES = (
    "agent_step_encoder.", "agent_history_gru.", "map_point_encoder.", "tl_encoder.",
    "agent_type_token", "map_type_token", "tl_type_token", "scene_transformer.", "scene_norm.",
)
SEL_MIN_PROB = 0.05
SEL_SPEED_RATIOS = (0.0, 0.5, 1.0, 1.5)


class _MLP(nn.Module):
    """MF-Gaussian's MLP (attribute name `net` matters for warm starting)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TokenMLP(nn.Module):
    """es_discriminator's token MLP (Linear-GELU-Linear-LayerNorm)."""

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                    nn.LayerNorm(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward; gradient multiplied by `scale` (0 detaches)."""
    if scale == 0.0:
        return x.detach()
    if scale == 1.0:
        return x
    return _GradScale.apply(x, float(scale))


def anchor_group(object_types: torch.Tensor, is_sdc: torch.Tensor) -> torch.Tensor:
    """0 vehicle (and SDC / other types), 1 pedestrian, 2 cyclist."""
    g = torch.zeros_like(object_types, dtype=torch.long)
    g = torch.where(object_types == 2, torch.ones_like(g), g)
    g = torch.where(object_types == 3, torch.full_like(g, 2), g)
    return torch.where(is_sdc.bool(), torch.zeros_like(g), g)


def gather_k(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B,N,K,...], idx [B,N,M] -> [B,N,M,...]."""
    rest = x.shape[3:]
    return x.gather(2, idx.view(*idx.shape, *([1] * len(rest))).expand(*idx.shape, *rest))


# ---- scorer blocks, copied from es_discriminator/real_es/wm_model.py ---------------------

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
            y = norm(y + F.gelu(conv(y.transpose(1, 2)).transpose(1, 2)))
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
        g = self.geo(geo)
        if static:
            logits = torch.einsum("bktnd,bmnd->bktmn", qh, kh[:, 0])
        else:
            logits = torch.einsum("bktnd,btmnd->bktmn", qh, kh)
        logits = logits / math.sqrt(dh) + self.geo_logit(g)
        if bias is not None:
            logits = logits + bias.unsqueeze(-1)
        keep = mask.unsqueeze(-1).to(logits.dtype)
        logits = logits.masked_fill(keep == 0, -1e4)
        attn = torch.softmax(logits, dim=3) * keep
        if static:
            v_tok = torch.einsum("bktmn,bmnd->bktnd", attn, vh[:, 0])
        else:
            v_tok = torch.einsum("bktmn,btmnd->bktnd", attn, vh)
        g_pooled = torch.einsum("bktm,bktmg->bktg", attn.mean(-1), g)
        return self.out(v_tok.reshape(b, k, t, h) + self.geo_value(g_pooled))


def _zero_mlp(inputs: int, hidden: int, out: int) -> nn.Sequential:
    """MLP whose output is exactly zero at init (last layer zero), for warm-start-safe additions."""
    mlp = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, out))
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


def goal_step_features(cand_xy: torch.Tensor, cand_yaw: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
    """[B,K,T,2], [B,K,T], goal [B,2] -> [B,K,T,5]: goal offset in the candidate frame, log distance,
    sin / cos of the bearing to the goal relative to the heading."""
    d = goal_xy[:, None, None, :] - cand_xy
    c, s = torch.cos(cand_yaw), torch.sin(cand_yaw)
    lx, ly = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
    dist = torch.sqrt(lx * lx + ly * ly + 1e-6)
    return torch.stack([lx / 50.0, ly / 50.0, torch.log1p(dist) / 5.0, ly / dist, lx / dist], -1)


def goal_end_features(cand_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
    """Endpoint features [B,K,5]: final distance to the goal (linear / log), offset, and goal gain."""
    d = goal_xy[:, None, :] - cand_xy[:, :, -1]
    dist = d.norm(dim=-1)
    gain = goal_xy.norm(dim=-1)[:, None] - dist
    return torch.stack([dist / 20.0, torch.log1p(dist) / 5.0, d[..., 0] / 50.0, d[..., 1] / 50.0, gain / 20.0], -1)


def route_step_features(cand_xy: torch.Tensor, cand_yaw: torch.Tensor, lanes: dict, chunk: int = 4096) -> torch.Tensor:
    """[B,K,T,3]: distance to the nearest lane centerline point and sin / cos of the heading error
    to that lane's direction (zeros, with distance 2.0, when the scene has no lane points)."""
    b, k, t, _ = cand_xy.shape
    q = cand_xy.reshape(b, k * t, 2)
    lxy, ldir, lval = lanes["xy"], lanes["dir"], lanes["valid"].bool()
    dists, dirs = [], []
    for s0 in range(0, q.shape[1], chunk):
        qq = q[:, s0:s0 + chunk]
        d2 = (qq.unsqueeze(2) - lxy.unsqueeze(1)).square().sum(-1).masked_fill(~lval.unsqueeze(1), float("inf"))
        m, i = d2.min(-1)
        dists.append(m)
        dirs.append(torch.gather(ldir, 1, i.unsqueeze(-1).expand(-1, -1, 2)))
    dist = torch.sqrt(torch.cat(dists, 1).clamp(max=1e4)).reshape(b, k, t)
    ldir_q = torch.cat(dirs, 1).reshape(b, k, t, 2)
    c, s = torch.cos(cand_yaw), torch.sin(cand_yaw)
    err_sin = c * ldir_q[..., 1] - s * ldir_q[..., 0]
    err_cos = c * ldir_q[..., 0] + s * ldir_q[..., 1]
    has = lval.any(1).view(b, 1, 1)
    feat = torch.stack([(dist / 5.0).clamp(max=2.0), err_sin, err_cos], -1)
    return torch.where(has[..., None], feat, torch.tensor([2.0, 0.0, 0.0], device=cand_xy.device, dtype=feat.dtype))


RULE_STEP_DIM, RULE_END_DIM = 8, 8


@torch.no_grad()
def rule_features(cand_xy: torch.Tensor, cand_yaw: torch.Tensor, ego_lw: torch.Tensor, wm: dict,
                  edges: dict, edge_budget: float = 8e6, pair_budget: float = 4e5) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact rule-reward geometry of each candidate step: per step [B,K,T,RULE_STEP_DIM], per candidate
    [B,K,RULE_END_DIM]. Fixed inputs (no gradient), computed in fp32 whatever the autocast.

    Forecasts: SAT box gaps of the ego box against every selected (agent, mode) box -> smallest gap
    (any mode / modes with p >= SEL_MIN_PROB) and independent-agent collision probabilities
    1 - prod_a (1 - sum_m p_am * c(gap)) with c = sigmoid(-gap / 0.25), 1[gap < 0] and a 1 m near-miss
    sigmoid. Road edges: Waymax-rule signed distance of the ego box corners (max over corners, > 0
    offroad) to the label edges, clipped, as an offroad sigmoid and as a 0.5 m near-curb sigmoid.
    Summary: the per-step maxima / minima, the fraction of the horizon before the first hard
    violation (1 = none) and the mean soft collision probability.

    Memory stays small (it runs next to JAX in closed loop): box gaps are computed for about
    `pair_budget` (candidate step, agent mode) pairs at a time and the edge search uses `edge_budget`."""
    with torch.autocast(cand_xy.device.type, enabled=False):
        cand_xy, cand_yaw, ego_lw = cand_xy.float(), cand_yaw.float(), ego_lw.float()
        b, k, t, _ = cand_xy.shape
        xy, yaw = wm["xy"].float(), wm["yaw"].float()
        a, m = xy.shape[1], xy.shape[2]
        probs = wm["probs"].float()                                                           # [B,A,M]
        live = wm["valid"][:, :, None] & (probs > 0)
        ec = box_corners(cand_xy, cand_yaw, ego_lw[:, None, None].expand(b, k, t, 2))          # [B,K,T,4,2]
        size = wm["size"].float()[:, :, None, None].expand(b, a, m, t, 2)
        oc = box_corners(xy, yaw, size).permute(0, 3, 1, 2, 4, 5).reshape(b, 1, t, a * m, 4, 2)
        o_yaw = yaw.permute(0, 3, 1, 2).reshape(b, 1, t, a * m)
        n_step = max(1, int(pair_budget // max(1, b * k * t)))
        gap = torch.cat([sat_gap(ec.unsqueeze(3), cand_yaw.unsqueeze(-1), oc[:, :, :, n0:n0 + n_step],
                                 o_yaw[:, :, :, n0:n0 + n_step]) for n0 in range(0, a * m, n_step)], -1).view(b, k, t, a, m)
        big = torch.full_like(gap, 50.0)
        g_min = torch.where(live[:, None, None], gap, big).amin((-1, -2))
        likely = live & (probs >= SEL_MIN_PROB)
        g_likely = torch.where(likely[:, None, None], gap, big).amin((-1, -2))
        p = probs[:, None, None]                                                              # [B,1,1,A,M]
        def ind(c):
            # 1 - prod_a (1 - q_a), multiplied out over agents: torch.prod on CUDA is JIT-compiled
            # (nvrtc), which the closed-loop environment cannot load.
            q = (p * c).sum(-1).clamp(0, 1)
            keep = torch.ones_like(q[..., 0])
            for i in range(q.shape[-1]):
                keep = keep * (1 - q[..., i])
            return 1 - keep

        p_soft, p_hard = ind(torch.sigmoid(-gap / 0.25)), ind((gap < 0).float())
        p_near = ind(torch.sigmoid(-(gap - 1.0) / 0.25))

        q = ec.reshape(b, k * t * 4, 2)
        sd = edge_signed_distance(q, edges, chunk=1 << 20, budget=edge_budget).view(b, k, t, 4).amax(-1).clamp(-5.0, 5.0)
        off = torch.sigmoid(sd / 0.1)
        step = torch.stack([g_min.clamp(-2, 10) / 5, g_likely.clamp(-2, 10) / 5, p_soft, p_hard, p_near,
                            sd / 2, off, torch.sigmoid((sd + 0.5) / 0.2)], -1)

        hard = (p_hard > 0.5) | (sd > 0)
        idx = torch.arange(1, t + 1, device=cand_xy.device, dtype=cand_xy.dtype)
        first = torch.where(hard, idx, torch.full_like(idx, t + 1.0)).amin(-1)
        end = torch.stack([p_soft.amax(-1), p_hard.amax(-1), p_near.amax(-1), off.amax(-1),
                           g_min.amin(-1).clamp(-2, 10) / 5, sd.amax(-1) / 2, ((first - 1) / t).clamp(max=1.0),
                           p_soft.mean(-1)], -1)
    return step, end


def _rotate_cov(cov: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """(sxx, syy, sxy) expressed in a frame rotated by -angle (i.e. R(-a) S R(-a)^T)."""
    c, s = torch.cos(angle), torch.sin(angle)
    sxx, syy, sxy = cov[..., 0], cov[..., 1], cov[..., 2]
    return torch.stack([c * c * sxx + 2 * c * s * sxy + s * s * syy,
                        s * s * sxx - 2 * c * s * sxy + c * c * syy,
                        -c * s * sxx + c * s * syy + (c * c - s * s) * sxy], -1)


class PlanScorer(nn.Module):
    """Scores ego candidates [B,C,T] against the decoded futures. Outputs a total score (higher is
    better, trained by ranking) and sub-score logits for SUB_SCORES."""

    def __init__(self, cfg: AnchorWMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, c, heads = cfg.scorer_hidden, cfg.hidden_dim, cfg.scorer_heads
        self.pos_scale, self.vel_scale = float(cfg.scorer_pos_scale), float(cfg.scorer_vel_scale)
        self.scene_proj = nn.Sequential(nn.Linear(c, h), nn.LayerNorm(h))
        self.goal_embed = _TokenMLP(4, h)
        self.traj_embed = _TokenMLP(6, h)
        self.traj_gru = nn.GRU(h, h, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(h, heads, dropout=cfg.dropout, batch_first=True)
        self.query_proj = nn.Sequential(nn.Linear(c, h), nn.LayerNorm(h))
        self.mode_embed = _TokenMLP(10, h)
        self.mode_temporal = _Temporal(h)
        self.prob_embed = nn.Linear(1, h)
        self.wm_attn = _GeoAttention(h, heads, 12)
        self.edge_embed = _TokenMLP(5, h)
        self.edge_attn = _GeoAttention(h, heads, 7)
        self.trunk = nn.Sequential(nn.LayerNorm(7 * h), nn.Linear(7 * h, h), nn.GELU(), nn.Dropout(cfg.dropout))
        self.sub_head = nn.Linear(h, len(SUB_SCORES))
        self.total_head = nn.Linear(h, 1)
        if cfg.scorer_goal_features:
            self.goal_step = _zero_mlp(5, h, h)      # added to the candidate step embedding
            self.goal_end = _zero_mlp(5, h, h)       # added to the pooled representation
        if cfg.scorer_route_features:
            self.route_step = _zero_mlp(3, h, h)
        if cfg.scorer_rule_features:
            self.rule_step = _zero_mlp(RULE_STEP_DIM, h, h)   # added to the candidate step embedding
            self.rule_end = _zero_mlp(RULE_END_DIM, h, h)     # added to the pooled representation
        if cfg.scorer_aux_heads:
            self.ttv_head = nn.Linear(h, 1)          # time to first violation / horizon (1 = none)
            self.goaldist_head = nn.Linear(h, 1)     # normalized final distance to the goal
        self.last_aux: dict[str, torch.Tensor] = {}

    # ---- candidate-independent context (reusable across ES iterations) -------------------

    @torch.no_grad()
    def select_agents(self, f: Mapping[str, torch.Tensor], mean: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Agents [B,A] nearest to the ego now or along constant-velocity ego rollouts under any mode
        with p >= 0.05 (es_discriminator.real_es.wm_cache.select_agents, batched)."""
        cfg = self.cfg
        b, n, m, t, _ = mean.shape
        valid = f["agent_valid"].bool() & ~f["is_sdc"].bool() & torch.isin(
            f["object_types"], torch.tensor(DYNAMIC_TYPES, device=mean.device))
        speed = f["ego_vel"].float().norm(dim=-1)                                             # [B]
        steps = torch.arange(1, t + 1, device=mean.device, dtype=mean.dtype) * cfg.future_dt
        ratios = torch.tensor(SEL_SPEED_RATIOS, device=mean.device, dtype=mean.dtype)
        ego_x = speed[:, None, None] * ratios[None, :, None] * steps[None, None, :]           # [B,R,T]
        ego = torch.stack([ego_x, torch.zeros_like(ego_x)], -1)                               # [B,R,T,2]
        d_mode = (mean[:, :, :, None] - ego[:, None, None]).norm(dim=-1).amin(dim=(3, 4))     # [B,N,M]
        d_fut = torch.where(probs >= SEL_MIN_PROB, d_mode, torch.full_like(d_mode, float("inf"))).amin(-1)
        d = torch.minimum(f["current_local_xy"].float().norm(dim=-1), d_fut)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        a = min(int(cfg.scorer_agents), n)
        dist, idx = torch.topk(d, a, dim=1, largest=False)
        return idx, torch.isfinite(dist)

    def context(self, enc: dict, dec: dict, f: Mapping[str, torch.Tensor], scale: float) -> dict:
        cfg = self.cfg
        rng = float(cfg.max_range)
        tokens = self.scene_proj(grad_scale(enc["scene_tokens"], scale))
        goal_token = self.goal_embed(torch.cat([f["goal_xy"].float() / 100.0, f["ego_vel"].float() / 25.0], -1))
        tokens = torch.cat([tokens, goal_token[:, None]], 1)
        valid = torch.cat([enc["scene_valid"], torch.ones_like(enc["scene_valid"][:, :1])], 1)

        # Top modes per agent, then the agents that matter for the ego.
        ms = int(cfg.scorer_modes)
        top_p, top_k = torch.topk(dec["probs"].detach(), ms, dim=-1)                          # [B,N,M]
        cur = f["current_local_xy"].float()
        mean_all = cur[:, :, None, None] + gather_k(dec["pred_xy_k"].detach(), top_k) * rng
        aidx, avalid = self.select_agents(f, mean_all, top_p)                                 # [B,A]
        a = aidx.shape[1]
        g2 = lambda x: x.gather(1, aidx.view(*aidx.shape, *([1] * (x.dim() - 2))).expand(  # noqa: E731
            *aidx.shape, *x.shape[2:]))
        k_sel = g2(top_k)                                                                     # [B,A,M]
        sel = lambda x: gather_k(g2(x), k_sel)  # noqa: E731

        xy = g2(cur)[:, :, None, None] + grad_scale(sel(dec["pred_xy_k"]), scale) * rng       # [B,A,M,T,2]
        yaw_vec = grad_scale(sel(dec["yaw_vec_k"]), scale)
        cur_yaw = g2(f["current_yaw_local"].float())
        yaw = cur_yaw[:, :, None, None] + torch.atan2(yaw_vec[..., 0], yaw_vec[..., 1])
        sigma = grad_scale(sel(dec["sigma_k"]), scale) * rng                                  # agent frame, m
        var = sigma.square()
        diag = torch.stack([var[..., 0], var[..., 1], torch.zeros_like(var[..., 0])], -1)
        cov = _rotate_cov(diag, -cur_yaw[:, :, None, None])                                   # ego frame
        probs = grad_scale(sel(dec["probs"]), scale)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-6)
        probs = probs * avalid[..., None]
        size = g2(f["agent_history"][:, :, -1, 6:8].float() * cfg.max_width)                  # length, width
        query = self.query_proj(grad_scale(sel(dec["queries"]), scale))                       # [B,A,M,H]

        wm = {"xy": xy, "yaw": yaw, "cov": cov, "sigma": sigma, "probs": probs, "size": size,
              "valid": avalid, "current": g2(cur)}
        wm_tok = self._wm_tokens(wm, query)
        e_mask = f["tok_edge_valid"].bool()
        edges = {"xy": f["tok_edge_xy"].float(), "dir": f["tok_edge_dir"].float(), "mask": e_mask}
        edge_tok = self.edge_embed(torch.cat([edges["xy"] / self.pos_scale, edges["dir"],
                                              e_mask[..., None].float()], -1))
        ctx = dict(tokens=tokens, valid=valid, goal_token=goal_token, wm=wm, wm_tok=wm_tok, edges=edges,
                   edge_tok=edge_tok, ego_lw=f["ego_lw"].float(), agent_idx=aidx, mode_idx=k_sel,
                   goal_xy=f["goal_xy"].float())
        if cfg.scorer_route_features:
            ctx["lanes"] = {"xy": f["lane_xy"].float(), "dir": f["lane_dir"].float(), "valid": f["lane_valid"].bool()}
        if cfg.scorer_rule_features:
            ctx["label_edges"] = {"edge_xy": f["edge_xy"].float(), "edge_dir": f["edge_dir"].float(),
                                  "edge_prev_dir": f["edge_prev_dir"].float(), "edge_same_prev": f["edge_same_prev"],
                                  "edge_valid": f["edge_valid"]}
        return ctx

    def _wm_tokens(self, wm: dict, query: torch.Tensor) -> torch.Tensor:
        xy = wm["xy"]
        b, a, m, t, _ = xy.shape
        prev = torch.cat([wm["current"][:, :, None, None].expand(b, a, m, 1, 2), xy[..., :-1, :]], dim=3)
        vel = (xy - prev) / self.cfg.future_dt
        size = wm["size"][:, :, None, None].expand(b, a, m, t, 2)
        feats = [xy / self.pos_scale, torch.sin(wm["yaw"])[..., None], torch.cos(wm["yaw"])[..., None],
                 vel / self.vel_scale, size / 10.0, torch.log(wm["sigma"].clamp(min=1e-2))]
        tok = self.mode_temporal(self.mode_embed(torch.cat(feats, -1)))                       # [B,A,M,T,H]
        tok = tok + query[:, :, :, None] + self.prob_embed(torch.log(wm["probs"].clamp(min=1e-4))[..., None])[:, :, :, None]
        return tok.permute(0, 3, 1, 2, 4).reshape(b, t, a * m, -1)                           # [B,T,A*M,H]

    # ---- candidate geometry (ported) -----------------------------------------------------

    def _wm_geometry(self, cand_xy, cand_yaw, ego_lw, wm: dict) -> torch.Tensor:
        xy = wm["xy"]
        b, a, m, t, _ = xy.shape
        agent_xy = xy.permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 2)
        agent_yaw = wm["yaw"].permute(0, 3, 1, 2).reshape(b, 1, t, a * m)
        d = agent_xy - cand_xy.unsqueeze(3)                                                   # [B,K,T,N,2]
        c, s = torch.cos(cand_yaw).unsqueeze(-1), torch.sin(cand_yaw).unsqueeze(-1)
        dx, dy = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        rel = agent_yaw - cand_yaw.unsqueeze(-1)
        size = wm["size"][:, :, None].expand(b, a, m, 2).reshape(b, 1, 1, a * m, 2)
        reach = 0.5 * size.norm(dim=-1) + 0.5 * ego_lw.norm(dim=-1).view(b, 1, 1, 1)
        geo = [dx / 10.0, dy / 10.0, torch.log1p(dist), torch.sin(rel), torch.cos(rel), (dist - reach) / 5.0,
               torch.exp(-dist / 2.0), torch.exp(-(dist - reach).clamp(min=0) / 1.0)]
        cov = wm["cov"].permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 3) + torch.tensor(
            [0.05, 0.05, 0.0], device=xy.device, dtype=xy.dtype)
        det = (cov[..., 0] * cov[..., 1] - cov[..., 2] ** 2).clamp(min=1e-6)
        maha = (cov[..., 1] * d[..., 0] ** 2 - 2 * cov[..., 2] * d[..., 0] * d[..., 1]
                + cov[..., 0] * d[..., 1] ** 2) / det
        cand_cov = _rotate_cov(cov.expand(b, cand_xy.shape[1], t, a * m, 3), cand_yaw.unsqueeze(-1))
        geo += [torch.log1p(maha), 0.5 * torch.log(cand_cov[..., 0].clamp(min=1e-4)),
                0.5 * torch.log(cand_cov[..., 1].clamp(min=1e-4)), 0.25 * torch.log(det).expand_as(maha)]
        return torch.stack(geo, -1)

    @staticmethod
    def _edge_geometry(cand_xy, cand_yaw, ego_lw, edges: dict) -> torch.Tensor:
        exy = edges["xy"][:, None, None]
        edir = edges["dir"][:, None, None]
        d = cand_xy.unsqueeze(3) - exy
        lateral = edir[..., 0] * d[..., 1] - edir[..., 1] * d[..., 0]
        along = edir[..., 0] * d[..., 0] + edir[..., 1] * d[..., 1]
        dist = torch.sqrt((d * d).sum(-1) + 1e-6)
        rel = torch.atan2(edir[..., 1], edir[..., 0]) - cand_yaw.unsqueeze(-1)
        half_w = 0.5 * ego_lw[:, 1].view(-1, 1, 1, 1)
        return torch.stack([lateral / 5.0, along / 10.0, torch.log1p(dist), torch.sin(rel), torch.cos(rel),
                            (lateral.abs() - half_w) / 2.0, torch.exp(-dist / 2.0)], -1)

    # ---- scoring ----------------------------------------------------------------------------

    def forward(self, ctx: dict, cand_xy: torch.Tensor, cand_yaw_fallback: torch.Tensor,
                chunk: int = 0, heading: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """cand_xy [B,C,T,2] (m, ego frame), heading fallback [B,C,T] -> total [B,C], sub-logits [B,C,S].

        `heading` [B,C,T]: the candidates' own headings, used as is when cfg.scorer_input_heading (else
        the heading is the travel direction from xy, falling back to `cand_yaw_fallback` while still)."""
        count = cand_xy.shape[1]
        step = chunk if chunk and chunk < count else count
        use_heading = heading is not None and self.cfg.scorer_input_heading
        totals, subs, auxs = [], [], []
        for s0 in range(0, count, step):
            sl = slice(s0, s0 + step)
            tot, sub, aux = self._score_chunk(ctx, cand_xy[:, sl], cand_yaw_fallback[:, sl],
                                              heading[:, sl] if use_heading else None)
            totals.append(tot)
            subs.append(sub)
            auxs.append(aux)
        # Auxiliary heads (fine-tuning only) are exposed on the side, so callers keep (total, sub).
        self.last_aux = {key: torch.cat([a[key] for a in auxs], 1) for key in auxs[0]} if auxs[0] else {}
        return torch.cat(totals, 1), torch.cat(subs, 1)

    def _score_chunk(self, ctx: dict, cand_xy: torch.Tensor, yaw_fallback: torch.Tensor,
                     heading: torch.Tensor | None = None):
        bsz, count, steps, _ = cand_xy.shape
        cfg = self.cfg
        cand_yaw = heading if heading is not None else path_yaw(cand_xy, yaw_fallback)
        feats = candidate_features(cand_xy, cand_yaw, cfg.future_dt, self.pos_scale, self.vel_scale)
        traj = self.traj_embed(feats.reshape(bsz * count, steps, -1))
        if cfg.scorer_goal_features:
            traj = traj + self.goal_step(goal_step_features(cand_xy, cand_yaw, ctx["goal_xy"])).reshape(bsz * count, steps, -1)
        if cfg.scorer_route_features:
            traj = traj + self.route_step(route_step_features(cand_xy, cand_yaw, ctx["lanes"])).reshape(bsz * count, steps, -1)
        if cfg.scorer_rule_features:
            r_step, r_end = rule_features(cand_xy, cand_yaw, ctx["ego_lw"], ctx["wm"], ctx["label_edges"])
            traj = traj + self.rule_step(r_step).reshape(bsz * count, steps, -1)
        encoded, _ = self.traj_gru(traj)
        cross, _ = self.cross_attn(encoded.reshape(bsz, count * steps, -1), ctx["tokens"], ctx["tokens"],
                                   key_padding_mask=~ctx["valid"], need_weights=False)
        cross = cross.reshape(bsz * count, steps, -1)
        q = encoded.reshape(bsz, count, steps, -1)
        pooled = [encoded[:, -1], cross.mean(1), ctx["goal_token"].repeat_interleave(count, dim=0)]

        e_geo = self._edge_geometry(cand_xy, cand_yaw, ctx["ego_lw"], ctx["edges"])
        e_out = self.edge_attn(q, ctx["edge_tok"][:, None], e_geo, None, ctx["edges"]["mask"][:, None, None, :])
        pooled += [e_out.mean(2).reshape(bsz * count, -1), e_out.amax(2).reshape(bsz * count, -1)]

        wm = ctx["wm"]
        b, a, m, t, _ = wm["xy"].shape
        w_geo = self._wm_geometry(cand_xy, cand_yaw, ctx["ego_lw"], wm)
        mask = (wm["valid"][:, :, None] & (wm["probs"] > 0)).reshape(b, 1, 1, a * m).expand(b, 1, t, a * m)
        bias = torch.log(wm["probs"].clamp(min=1e-4)).reshape(b, 1, 1, a * m)
        w_out = self.wm_attn(q, ctx["wm_tok"], w_geo, bias, mask)
        pooled += [w_out.mean(2).reshape(bsz * count, -1), w_out.amax(2).reshape(bsz * count, -1)]

        z = self.trunk(torch.cat(pooled, -1))
        if cfg.scorer_goal_features:
            z = z + self.goal_end(goal_end_features(cand_xy, ctx["goal_xy"])).reshape(bsz * count, -1)
        if cfg.scorer_rule_features:
            z = z + self.rule_end(r_end).reshape(bsz * count, -1)
        aux = {}
        if cfg.scorer_aux_heads:
            aux = {"ttv": self.ttv_head(z).squeeze(-1).reshape(bsz, count),
                   "goaldist": self.goaldist_head(z).squeeze(-1).reshape(bsz, count)}
        return self.total_head(z).squeeze(-1).reshape(bsz, count), self.sub_head(z).reshape(bsz, count, -1), aux


class AnchorWorldModel(nn.Module):
    def __init__(self, cfg: AnchorWMConfig, anchors_m: torch.Tensor):
        super().__init__()
        self.config = cfg
        c, t = int(cfg.hidden_dim), int(cfg.future_steps)
        if tuple(anchors_m.shape[1:]) != (cfg.num_anchors, t, 2):
            raise ValueError(f"anchors {tuple(anchors_m.shape)} != [G,{cfg.num_anchors},{t},2]")

        # ---- encoder (MF-Gaussian names) ----
        self.agent_step_encoder = _MLP(cfg.agent_feature_dim, c, c, dropout=cfg.dropout)
        self.agent_history_gru = nn.GRU(c, c, batch_first=True)
        self.map_point_encoder = _MLP(cfg.map_attr_dim, c, c, dropout=cfg.dropout)
        self.tl_encoder = _MLP(cfg.tl_attr_dim, c, c, dropout=cfg.dropout)
        self.agent_type_token = nn.Parameter(torch.zeros(1, 1, c))
        self.map_type_token = nn.Parameter(torch.zeros(1, 1, c))
        self.tl_type_token = nn.Parameter(torch.zeros(1, 1, c))
        layer = nn.TransformerEncoderLayer(d_model=c, nhead=cfg.num_heads, dim_feedforward=4 * c,
                                           dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.scene_transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_scene_layers,
                                                       enable_nested_tensor=False)
        self.scene_norm = nn.LayerNorm(c)

        # ---- anchor decoder ----
        self.register_buffer("anchors", (anchors_m.float() / float(cfg.max_range)).contiguous())
        self.anchor_embed = _MLP(t * 2, c, c, dropout=cfg.dropout)
        self.goal_embed = _MLP(4, c, c, dropout=cfg.dropout)
        self.query_fuse = _MLP(c, c, c, dropout=cfg.dropout)
        self.traj_head = _MLP(c, c, t * 6, dropout=cfg.dropout)
        last = self.traj_head.net[-1]
        nn.init.normal_(last.weight, std=0.01)     # residual ~0 at init: predictions start at the anchors
        nn.init.zeros_(last.bias)
        bias = last.bias.data.view(t, 6)
        bias[:, 2:4] = float(cfg.sigma_raw_bias)
        bias[:, 5] = 1.0                           # yaw delta starts at 0 (sin 0, cos 1)
        waypoint_idx = [(i + 1) * t // 5 - 1 for i in range(5)]
        self.register_buffer("waypoint_idx", torch.tensor(waypoint_idx, dtype=torch.long), persistent=False)
        self.traj_embed = _MLP(2 * len(waypoint_idx), c, c, dropout=cfg.dropout)
        self.mode_cls = _MLP(2 * c, c, 1, dropout=cfg.dropout)
        nn.init.zeros_(self.mode_cls.net[-1].weight)
        nn.init.zeros_(self.mode_cls.net[-1].bias)

        self.scorer = PlanScorer(cfg)

    # ---- encoder ----------------------------------------------------------------------------

    @staticmethod
    def _masked_map_pool(point_tokens, point_valid):
        mask = point_valid.bool()
        pooled = torch.where(mask[..., None], point_tokens, torch.full_like(point_tokens, -1.0e4)).max(dim=2).values
        seg_valid = mask.any(dim=2)
        return torch.where(seg_valid[..., None], pooled, torch.zeros_like(pooled)), seg_valid

    def encode(self, f: Mapping[str, torch.Tensor]) -> dict:
        history = f["agent_history"].float()
        history_valid = f["agent_history_valid"].bool()
        agent_valid = f["agent_valid"].bool()
        b, n, h, _ = history.shape
        c = int(self.config.hidden_dim)

        step_tokens = self.agent_step_encoder(history) * history_valid[..., None].to(history.dtype)
        _, h_n = self.agent_history_gru(step_tokens.reshape(b * n, h, c))
        agent_tokens = h_n[-1].reshape(b, n, c) * agent_valid[..., None].to(history.dtype) + self.agent_type_token
        map_tokens, map_seg_valid = self._masked_map_pool(self.map_point_encoder(f["map_features"].float()),
                                                          f["map_valid"].bool())
        map_tokens = map_tokens + self.map_type_token
        tl_valid = f["traffic_light_valid"].bool()
        tl_tokens = (self.tl_encoder(f["traffic_light_features"].float()) + self.tl_type_token) * tl_valid[..., None].to(history.dtype)

        scene_tokens = torch.cat([agent_tokens, map_tokens, tl_tokens], dim=1)
        scene_valid = torch.cat([agent_valid, map_seg_valid, tl_valid], dim=1)
        scene_tokens = self.scene_norm(self.scene_transformer(scene_tokens, src_key_padding_mask=~scene_valid))
        latent = scene_tokens[:, :n] * agent_valid[..., None].to(scene_tokens.dtype)
        return {"scene_tokens": scene_tokens, "scene_valid": scene_valid, "latent_agent": latent}

    # ---- decoder ----------------------------------------------------------------------------

    def decode(self, f: Mapping[str, torch.Tensor], enc: dict) -> dict:
        cfg = self.config
        latent = enc["latent_agent"]
        b, n, c = latent.shape
        k, t = int(cfg.num_anchors), int(cfg.future_steps)
        agent_valid = f["agent_valid"].bool()
        is_sdc = f["is_sdc"].bool()

        group = anchor_group(f["object_types"], is_sdc)                                       # [B,N]
        anchors = self.anchors[group]                                                         # [B,N,K,T,2]
        anchor_feat = self.anchor_embed(self.anchors.flatten(-2))[group]                      # [B,N,K,C]
        goal = self.goal_embed(torch.cat([f["goal_xy"].float() / 100.0, f["ego_vel"].float() / 25.0], -1))
        latent = latent + is_sdc[..., None].to(latent.dtype) * goal[:, None]
        queries = self.query_fuse(latent[:, :, None, :] + anchor_feat)                        # [B,N,K,C]
        raw = self.traj_head(queries).view(b, n, k, t, 6)

        xy_agent = anchors + raw[..., :2]
        heading_sc = f["agent_history"][:, :, -1, 4:6].float()                                # sin, cos rel. ego
        heading_sc = torch.where(agent_valid[..., None], heading_sc, torch.tensor([0.0, 1.0], device=latent.device))
        xy_ego = rotate_xy(xy_agent, heading_sc[..., 0, None, None], heading_sc[..., 1, None, None])
        sigma = F.softplus(raw[..., 2:4]) + float(cfg.sigma_eps)
        yaw_vec = F.normalize(raw[..., 4:6], dim=-1, eps=1e-6)

        waypoints = xy_agent[..., self.waypoint_idx, :].flatten(-2)
        logits = self.mode_cls(torch.cat([queries, self.traj_embed(waypoints)], -1)).squeeze(-1)
        logits = logits.masked_fill(~agent_valid[..., None], 0.0)
        probs = torch.softmax(logits.float(), dim=-1) * agent_valid[..., None]
        return {
            "pred_xy_agent_k": xy_agent,      # [B,N,K,T,2] normalized, agent frame
            "pred_xy_k": xy_ego,              # [B,N,K,T,2] normalized displacement, ego frame
            "sigma_k": sigma,                 # [B,N,K,T,2] normalized, agent frame
            "yaw_vec_k": yaw_vec,             # [B,N,K,T,2] (sin, cos) of yaw change
            "logits": logits, "probs": probs, # [B,N,K]
            "queries": queries,               # [B,N,K,C]
            "anchors_agent": anchors,         # [B,N,K,T,2] normalized, agent frame
            "heading_sc": heading_sc,
        }

    def forward(self, f: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
        enc = self.encode(f)
        return enc, self.decode(f, enc)


@torch.no_grad()
def select_modes(dec: dict, m: int, nms_dist_m: float, max_range: float) -> torch.Tensor:
    """Planner modes [B,N,m]: greedy by probability with endpoint NMS; when every remaining anchor is
    suppressed the next most probable unselected one is taken."""
    probs = dec["probs"]
    end = dec["pred_xy_k"][..., -1, :] * max_range                                            # [B,N,K,2]
    suppressed = torch.zeros_like(probs, dtype=torch.bool)
    selected = torch.zeros_like(probs, dtype=torch.bool)
    picks = []
    for _ in range(m):
        score = torch.where(suppressed, probs - 10.0, probs).masked_fill(selected, float("-inf"))
        pick = score.argmax(-1)                                                               # [B,N]
        picks.append(pick)
        selected.scatter_(-1, pick[..., None], True)
        pe = end.gather(2, pick[..., None, None].expand(-1, -1, 1, 2))
        suppressed |= (end - pe).norm(dim=-1) < nms_dist_m
    return torch.stack(picks, -1)


def warm_start_encoder(model: AnchorWorldModel, ckpt_path: str) -> dict:
    """Load the encoder of an MF-Gaussian checkpoint (`model_state_dict`); every encoder tensor of
    `model` must be found with the same shape."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt["model_state_dict"]
    wm_cfg = dict(ckpt.get("config", {}))
    for key in ("hidden_dim", "num_heads", "num_scene_layers", "max_agents", "max_map_segments",
                "max_points_per_segment", "num_object_types", "num_map_type_classes", "history_steps"):
        if key in wm_cfg and wm_cfg[key] != getattr(model.config, key):
            raise ValueError(f"warm-start config mismatch on {key}: {wm_cfg[key]} != {getattr(model.config, key)}")
    own = model.state_dict()
    wanted = [k for k in own if k.startswith(ENCODER_PREFIXES)]
    missing = [k for k in wanted if k not in src]
    bad = [k for k in wanted if k in src and src[k].shape != own[k].shape]
    if missing or bad:
        raise ValueError(f"warm start: missing {missing[:5]}, shape mismatch {bad[:5]}")
    model.load_state_dict({k: src[k] for k in wanted}, strict=False)
    return {"checkpoint": ckpt_path, "tensors": len(wanted), "epoch": ckpt.get("epoch")}
