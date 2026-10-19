"""Self-generated ego candidates and their online labels (log replay), all on the GPU.

Candidates per scene (C = K + 1 + 16):
  * the K decoded ego anchors (what the model would propose itself),
  * the logged ego future,
  * 16 smooth perturbations of it along its own path: lateral offset ramps (+-0.75, 1.5, 3 m),
    speed scaling x0.5 / 0.8 / 1.2, two constant-deceleration stops, a 2.5 m nudge toward the
    nearest agent ahead, lateral + speed combinations and two random combinations.

Labels (Waymax semantics, other agents follow the log; 0.2 s steps over the valid horizon):
  overlap      any step where the ego box overlaps a valid logged box (separating-axis test)
  clearance    min separating-axis gap over steps and objects (m; < 0 = overlap)
  offroad      any step where a box corner has positive signed road-edge distance
  offroad_sd   max corner signed distance (m)
  progress     arc length of the end point projected on the logged ego path (m)
  goal_gain    |goal| - |goal - end| (m)
  comfort      exceedance of lateral 4.5 / longitudinal 7 m/s^2 (m/s^2, < 0 = within limits)
"""

from __future__ import annotations

import torch

from anchor_wm.geometry import (box_corners, comfort_excess, edge_signed_distance, interp_path, path_yaw,
                                polyline_with_extension, project_progress, sat_gap, wrap_angle)

KIND_ANCHOR, KIND_GT, KIND_PERTURB = 0, 1, 2
_LATERAL = (-3.0, -1.5, -0.75, 0.75, 1.5, 3.0)
_SPEED = (0.5, 0.8, 1.2)
_STOP_DECEL = (2.5, 5.0)
NUM_PERTURB = len(_LATERAL) + len(_SPEED) + len(_STOP_DECEL) + 1 + 2 + 2


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def perturb_gt(gt_xy: torch.Tensor, ego_speed: torch.Tensor, nudge_sign: torch.Tensor, dt: float,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """gt_xy [B,T,2] -> [B,P,T,2] perturbations that follow the logged path's geometry."""
    b, t, _ = gt_xy.shape
    dev, dtype = gt_xy.device, gt_xy.dtype
    path = torch.cat([torch.zeros_like(gt_xy[:, :1]), gt_xy], 1)                         # origin + T
    ext, cum = polyline_with_extension(path)
    s_gt = cum[:, 1:t + 1]                                                                 # [B,T]
    time = torch.arange(1, t + 1, device=dev, dtype=dtype) * dt                            # [T]

    lat, scale, ramp, stop = [], [], [], []
    def add(d=0.0, f=1.0, r=2.0, a=0.0):
        lat.append(d if torch.is_tensor(d) else torch.full((b,), d, device=dev, dtype=dtype))
        scale.append(f if torch.is_tensor(f) else torch.full((b,), f, device=dev, dtype=dtype))
        ramp.append(r if torch.is_tensor(r) else torch.full((b,), r, device=dev, dtype=dtype))
        stop.append(a)
    for d in _LATERAL:
        add(d=d)
    for f in _SPEED:
        add(f=f)
    for a in _STOP_DECEL:
        add(a=a)
    add(d=2.5 * nudge_sign.to(dtype), r=1.5)
    add(d=1.5, f=1.2)
    add(d=-1.5, f=1.2)
    for _ in range(2):
        u = torch.rand((3, b), device=dev, dtype=dtype, generator=generator)
        add(d=-3.0 + 6.0 * u[0], f=0.6 + 0.7 * u[1], r=1.0 + 2.0 * u[2])

    d = torch.stack(lat, 1)[..., None]                                                     # [B,P,1]
    f = torch.stack(scale, 1)[..., None]
    r = torch.stack(ramp, 1)[..., None]
    s = f * s_gt[:, None]                                                                  # [B,P,T]
    decel = torch.tensor(stop, device=dev, dtype=dtype)[None, :, None]                     # [1,P,1]
    v0 = ego_speed.to(dtype)[:, None, None]
    a = decel.clamp_min(1e-3)
    t_stop = v0 / a
    s_stop = torch.where(time < t_stop, v0 * time - 0.5 * a * time ** 2, v0 ** 2 / (2.0 * a))
    s = torch.where(decel > 0, s_stop, s)
    pos, tangent = interp_path(ext, cum, s)                                                # [B,P,T,2]
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], -1)
    return pos + (d * _smoothstep(time / r))[..., None] * normal


def nearest_agent_side(f: dict) -> torch.Tensor:
    """+1 / -1: side (left / right) of the nearest valid agent within 30 m ahead (+1 if none)."""
    xy = f["current_local_xy"].float()
    ok = f["agent_valid"].bool() & ~f["is_sdc"].bool() & (xy[..., 0] > 0) & (xy.norm(dim=-1) < 30.0)
    d = torch.where(ok, xy.norm(dim=-1), torch.full_like(xy[..., 0], float("inf")))
    idx = d.argmin(1)
    y = xy[torch.arange(xy.shape[0], device=xy.device), idx, 1]
    return torch.where(torch.isfinite(d.amin(1)) & (y < 0), -torch.ones_like(y), torch.ones_like(y))


def build_candidates(f: dict, dec: dict, max_range: float, dt: float,
                     generator: torch.Generator | None = None) -> dict:
    """Ego candidates [B,C,T,2] (m, ego frame), heading fallbacks, kinds [C] and validity [B,C]."""
    b = f["ego_idx"].shape[0]
    ar = torch.arange(b, device=f["ego_idx"].device)
    ego_xy = dec["pred_xy_k"][ar, f["ego_idx"]].detach().float() * max_range               # [B,K,T,2]
    ego_yaw = torch.atan2(dec["yaw_vec_k"][ar, f["ego_idx"], ..., 0],
                          dec["yaw_vec_k"][ar, f["ego_idx"], ..., 1]).detach().float()     # [B,K,T]
    gt = f["ego_future_xy"].float()
    gt_ok = f["ego_future_valid"].bool().all(-1)                                           # [B]
    gt_safe = torch.where(gt_ok[:, None, None], gt, torch.zeros_like(gt))
    pert = perturb_gt(gt_safe, f["ego_vel"].float().norm(dim=-1), nearest_agent_side(f), dt, generator)
    k, p = ego_xy.shape[1], pert.shape[1]
    xy = torch.cat([ego_xy, gt_safe[:, None], pert], 1)
    yaw_fb = torch.cat([ego_yaw, f["ego_future_yaw"].float()[:, None],
                        torch.zeros_like(pert[..., 0])], 1)
    kinds = torch.tensor([KIND_ANCHOR] * k + [KIND_GT] + [KIND_PERTURB] * p, device=xy.device)
    valid = torch.cat([torch.ones((b, k), dtype=torch.bool, device=xy.device),
                       gt_ok[:, None].expand(b, 1 + p)], 1)
    return {"xy": xy, "yaw_fallback": yaw_fb, "kinds": kinds, "valid": valid}


def _with_midpoints(x: torch.Tensor, angle: bool = False) -> torch.Tensor:
    """[...,T,(D)] on the step axis (-2 for vectors, -1 for scalars) -> [...,2T-1,(D)]: each step
    followed by the midpoint to the next one (angles interpolated on the circle)."""
    vec = not angle and x.dim() >= 2 and x.shape[-1] == 2
    ax = -2 if vec else -1
    a = x.narrow(ax, 0, x.shape[ax] - 1)
    b = x.narrow(ax, 1, x.shape[ax] - 1)
    mid = a + 0.5 * wrap_angle(b - a) if angle else 0.5 * (a + b)
    inter = torch.stack([a, mid], dim=ax).flatten(ax - 1, ax) if vec else torch.stack([a, mid], dim=-1).flatten(-2)
    return torch.cat([inter, x.narrow(ax, x.shape[ax] - 1, 1)], dim=ax)


@torch.no_grad()
def label_candidates(f: dict, cand_xy: torch.Tensor, cand_yaw: torch.Tensor, dt: float,
                     horizon: torch.Tensor | None = None, obj_chunk: int = 32, midpoints: bool = True) -> dict:
    """Online labels for candidates [B,C,T,2] / headings [B,C,T] against the batch's logged futures.

    `horizon` [B] limits the label horizon in steps (default: all T). With `midpoints`, overlap and
    offroad are also checked halfway between steps (linear interpolation of both boxes), which
    recovers short overlaps that a 0.2 s grid skips over."""
    b, c, t, _ = cand_xy.shape
    dev = cand_xy.device
    steps_ok = torch.ones((b, t), dtype=torch.bool, device=dev) if horizon is None else (
        torch.arange(1, t + 1, device=dev)[None] <= horizon[:, None])
    o_xy, o_yaw, o_lw = f["fut_xy"].float(), f["fut_yaw"].float(), f["fut_lw"].float()   # [B,N,T,...]
    o_valid = f["fut_valid"].bool()
    g_xy, g_yaw, g_ok = cand_xy, cand_yaw, steps_ok
    if midpoints and t > 1:
        g_xy, g_yaw = _with_midpoints(cand_xy), _with_midpoints(cand_yaw, angle=True)
        g_ok = _with_midpoints(steps_ok.float()) > 0.75                                    # both ends in horizon
        o_xy, o_yaw, o_lw = _with_midpoints(o_xy), _with_midpoints(o_yaw, angle=True), _with_midpoints(o_lw)
        o_valid = _with_midpoints(o_valid.float()) > 0.75
    tg = g_xy.shape[2]
    ego_lw = f["ego_lw"].float()[:, None, None].expand(b, c, tg, 2)
    ego_c = box_corners(g_xy, g_yaw, ego_lw)                                               # [B,C,Tg,4,2]

    # Overlap / clearance against every valid logged object, chunked over objects.
    o_valid = o_valid & g_ok[:, None]
    gap_min = torch.full((b, c, tg), float("inf"), device=dev)
    for n0 in range(0, o_xy.shape[1], obj_chunk):
        sl = slice(n0, n0 + obj_chunk)
        oc = box_corners(o_xy[:, sl], o_yaw[:, sl], o_lw[:, sl])                           # [B,n,Tg,4,2]
        gap = sat_gap(ego_c[:, :, None], g_yaw[:, :, None], oc[:, None], o_yaw[:, None, sl])  # [B,C,n,Tg]
        gap = gap.masked_fill(~o_valid[:, None, sl], float("inf"))
        gap_min = torch.minimum(gap_min, gap.amin(2))
    clearance = gap_min.masked_fill(~g_ok[:, None], float("inf")).amin(-1)                # [B,C]
    clearance = torch.where(torch.isfinite(clearance), clearance, torch.full_like(clearance, 50.0))

    # Offroad: Waymax signed distance of every box corner.
    edges = {k: f[k] for k in ("edge_xy", "edge_dir", "edge_prev_dir", "edge_same_prev", "edge_valid")}
    edges = {k: (v.float() if v.is_floating_point() else v) for k, v in edges.items()}
    sd = edge_signed_distance(ego_c.reshape(b, -1, 2), edges).reshape(b, c, tg, 4).amax(-1)
    sd = sd.masked_fill(~g_ok[:, None], -100.0)
    offroad_sd = sd.amax(-1)

    # Progress along the logged ego path, goal gain, comfort.
    gt = f["ego_future_xy"].float()
    gt_valid = f["ego_future_valid"].bool()
    last_valid = torch.where(gt_valid.any(1), t - 1 - gt_valid.flip(1).float().argmax(1), torch.zeros_like(gt_valid[:, 0], dtype=torch.long))
    gt_fill = torch.where(gt_valid[..., None], gt, gt[torch.arange(b, device=dev), last_valid][:, None])
    gt_fill = torch.where(gt_valid.any(1)[:, None, None], gt_fill, torch.zeros_like(gt_fill))
    ext, cum = polyline_with_extension(torch.cat([torch.zeros_like(gt[:, :1]), gt_fill], 1))
    h_idx = (steps_ok.sum(1) - 1).clamp_min(0)
    end = cand_xy[torch.arange(b, device=dev)[:, None], torch.arange(c, device=dev)[None], h_idx[:, None]]
    progress = project_progress(end, ext, cum)
    goal = f["goal_xy"].float()
    goal_gain = goal.norm(dim=-1, keepdim=True) - (goal[:, None] - end).norm(dim=-1)
    gt_progress = project_progress(gt_fill[torch.arange(b, device=dev), h_idx][:, None], ext, cum)[:, 0]
    comfort = comfort_excess(cand_xy, f["ego_vel"].float()[:, None].expand(b, c, 2), dt)

    return {"overlap": clearance <= 0.0, "clearance": clearance, "offroad": offroad_sd > 0.0,
            "offroad_sd": offroad_sd, "progress": progress, "goal_gain": goal_gain,
            "gt_progress": gt_progress, "comfort": comfort}


def candidate_headings(cand_xy: torch.Tensor, yaw_fallback: torch.Tensor) -> torch.Tensor:
    return path_yaw(cand_xy, yaw_fallback)
