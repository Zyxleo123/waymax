"""Frozen multi-future Gaussian world-model predictions and road-edge tokens per recorded replan.

For every `<batch>.npz` of the ES candidate bank this reruns the frozen WM once per
replan step on the state the planner saw (`replaced_state`: original log with the
executed ego history) and writes, row-aligned with the batch's groups:

  <wm_dir>/<batch>.wm.npz
    wm_agent_idx [G,A] int16 (object index, -1 pad)   wm_type [G,A] int8
    wm_size      [G,A,2]  length, width at t
    wm_current   [G,A,3]  x, y, yaw at t (group ego frame)
    wm_mean      [G,A,M,T,3] mode-mean x, y, yaw (group ego frame), step i = t + 2(i+1), 0.2 s
    wm_sigma     [G,A,M,T,2] WM's diagonal std (m) in the WM anchor frame
    wm_cov       [G,A,M,T,3] the same Gaussian as (sxx, syy, sxy) in the group ego frame
    wm_probs     [G,A,M]  mode probabilities         wm_valid [G,A]
    wm_sel_dist  [G,A]    selection distance (m)
  <edge_dir>/<batch>.edges.npz
    edge_xy [G,E,2], edge_dir [G,E,2] (group ego frame), edge_type [G,E] int8, edge_mask [G,E]

The WM predicts deltas and a diagonal Gaussian in its anchor frame (SDC pose at t, the
frame of its training targets); the group frame is the same pose, and `wm_cov` carries
the covariance through `rotate_cov` so any residual rotation is handled exactly.

Agents (A = 32) are chosen independently of any candidate: the smallest of the current
distance to the ego and the distance between any mode mean (p >= 0.05) and constant-
velocity ego rollouts (speed x {0, 0.5, 1, 1.5}) at the same step. Road-edge tokens
(E = 256) are boundary/median points in a forward-biased box (100 m ahead, 25 m behind,
40 m to each side), farthest-point sampled from the point nearest the ego.

Validation (fails loudly): rerun means / probabilities equal the recorded live-planner
`wm_traj` / `wm_mode_probs` (float16 storage), and the WM anchor equals the group frame.
On `--validate_files` files it also reports a logged-state control, collision-agent
coverage (new selection vs the recorded nearest-32) and road-edge coverage (new tokens vs
the 256-point context map).

  python -m es_discriminator.real_es.wm_cache --data_dir .../es_candidates --tfrecord_dir ... \
      --wm_dir .../wm_predictions/<run> --edge_dir .../road_edges/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from waymax import datatypes

from es_discriminator.analyze_es_decisions import batch_files, load_batch_state
from es_discriminator.real_es.check_alignment import replaced_state

REPO = Path(__file__).resolve().parents[2]
WM_PKG = REPO / "multiFuture_GaussianWorldModel"
DEFAULT_CKPT = ("/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/"
                "multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt")
DEFAULT_TFRECORDS = "/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords/training"
NUM_AGENTS = 32
NUM_EDGES = 256
EDGE_BOX = (-25.0, 100.0, 40.0)        # behind, ahead, half-width (m)
EDGE_MIN_SPACING = 1.0
SEL_MIN_PROB = 0.05
SEL_SPEED_RATIOS = (0.0, 0.5, 1.0, 1.5)
DYNAMIC_TYPES = (1, 2, 3)              # vehicle, pedestrian, cyclist
WM_SUFFIX = ".wm.npz"
EDGE_SUFFIX = ".edges.npz"
MEAN_TOL_M = 0.1                       # logged-state control threshold
EXCESS_TOL_M = 0.05                    # error beyond the float16 half-step of the recorded wm_traj value
PROB_TOL = 2e-3


def rot(xy: np.ndarray, angle) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], -1)


def rotate_cov(cov: np.ndarray, angle) -> np.ndarray:
    """(sxx, syy, sxy) of a 2-D Gaussian expressed in a frame rotated by +angle -> R S R^T."""
    c, s = np.cos(angle), np.sin(angle)
    sxx, syy, sxy = cov[..., 0], cov[..., 1], cov[..., 2]
    return np.stack([c * c * sxx - 2 * c * s * sxy + s * s * syy,
                     s * s * sxx + 2 * c * s * sxy + c * c * syy,
                     c * s * (sxx - syy) + (c * c - s * s) * sxy], -1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_forecaster(ckpt: str, device: str):
    from simulation.run_simulation import _load_isolated_wm_forecaster
    return _load_isolated_wm_forecaster(WM_PKG, checkpoint_path=ckpt, device=device, planning_mode=None)


def jax_state(state):
    return jax.tree_util.tree_map(jnp.asarray, state)


@torch.no_grad()
def predict_step(forecaster, sim_state, t: int) -> dict:
    """One WM forward for every world of `sim_state` at step t, on the model's native grid.

    Returns world-frame means [B,N,M,T,2], yaws [B,N,M,T], anchor-frame sigma [B,N,M,T,2],
    probs [B,N,M], validity [B,N] and the anchor pose, using the same decoding as
    `WorldModelForecaster._decode_boxes` (current pose + delta rotated by the anchor yaw).
    """
    features, meta = forecaster._prepare_features(sim_state, int(t))
    pred = forecaster.model(features)
    cfg = forecaster.config
    n = int(meta["modeled_agents"])
    delta = pred["pred_delta_xy_norm_k"].float().cpu().numpy() * float(cfg.max_range)      # [B,n,M,T,2]
    yaw_vec = pred["pred_yaw_vec_k"].float().cpu().numpy()
    sigma = pred["pred_sigma_xy_m_k"].float().cpu().numpy()
    probs = pred["mode_probs"].float().cpu().numpy()
    ego_yaw = meta["ego_yaw"].astype(np.float64)
    tt = int(meta["timestep"])
    cur_xy = np.stack([meta["x"][:, :n, tt], meta["y"][:, :n, tt]], -1).astype(np.float64)
    cur_yaw = meta["yaw"][:, :n, tt].astype(np.float64)
    mean_xy = cur_xy[:, :, None, None] + rot(delta.astype(np.float64), ego_yaw[:, None, None, None])
    mean_yaw = wrap(cur_yaw[:, :, None, None] + np.arctan2(yaw_vec[..., 0], yaw_vec[..., 1]))
    valid = (meta["valid"][:, :n, tt].astype(bool) & ~meta["is_sdc"][:, :n].astype(bool)
             & np.isin(meta["object_types"][:, :n], DYNAMIC_TYPES))
    return dict(mean_xy=mean_xy, mean_yaw=mean_yaw, sigma=sigma, probs=probs, valid=valid,
                cur_xy=cur_xy, cur_yaw=cur_yaw, length=meta["length"][:, :n, tt], width=meta["width"][:, :n, tt],
                types=meta["object_types"][:, :n], origin=meta["origin_xy"].astype(np.float64), ego_yaw=ego_yaw)


def select_agents(cur_local: np.ndarray, mean_local: np.ndarray, probs: np.ndarray, valid: np.ndarray,
                  ego_speed: float, dt: float, k: int = NUM_AGENTS) -> tuple[np.ndarray, np.ndarray]:
    """Candidate-independent agent choice from current and WM-predicted occupancy.

    cur_local [N,2], mean_local [N,M,T,2], probs [N,M], valid [N] -> (indices [<=k], distance [<=k]).
    """
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return idx, np.zeros(0)
    steps = (np.arange(mean_local.shape[2]) + 1) * dt
    ego = np.stack([np.outer(SEL_SPEED_RATIOS, steps) * ego_speed,
                    np.zeros((len(SEL_SPEED_RATIOS), len(steps)))], -1)                      # [R,T,2]
    d_now = np.linalg.norm(cur_local[idx], axis=-1)
    diff = mean_local[idx][:, :, None] - ego[None, None]                                       # [n,M,R,T,2]
    d_mode = np.linalg.norm(diff, axis=-1).min(axis=(2, 3))                                    # [n,M]
    d_fut = np.where(probs[idx] >= SEL_MIN_PROB, d_mode, np.inf).min(axis=1)
    d = np.minimum(d_now, d_fut)
    order = np.argsort(d, kind="stable")[:k]
    return idx[order], d[order]


def farthest_point_sample(xy: np.ndarray, k: int, min_spacing: float) -> np.ndarray:
    """Indices of up to k points, greedy farthest-point from the point nearest the origin."""
    if len(xy) == 0:
        return np.zeros(0, np.int64)
    chosen = [int(np.argmin(np.linalg.norm(xy, axis=-1)))]
    dist = np.linalg.norm(xy - xy[chosen[0]], axis=-1)
    while len(chosen) < k:
        j = int(np.argmax(dist))
        if dist[j] < min_spacing:
            break
        chosen.append(j)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[j], axis=-1))
    return np.asarray(chosen, np.int64)


def edge_tokens(rg_np: dict, w: int, origin: np.ndarray, heading: float, k: int = NUM_EDGES) -> dict:
    keep = rg_np["valid"][w].astype(bool) & np.isin(rg_np["types"][w], (15, 16))
    xy = rot(np.stack([rg_np["x"][w][keep], rg_np["y"][w][keep]], -1).astype(np.float64) - origin, -heading)
    dirs = rot(np.stack([rg_np["dir_x"][w][keep], rg_np["dir_y"][w][keep]], -1).astype(np.float64), -heading)
    types = rg_np["types"][w][keep]
    back, ahead, half = EDGE_BOX
    box = (xy[:, 0] >= back) & (xy[:, 0] <= ahead) & (np.abs(xy[:, 1]) <= half)
    xy, dirs, types = xy[box], dirs[box], types[box]
    pick = farthest_point_sample(xy, k, EDGE_MIN_SPACING)
    out = dict(edge_xy=np.zeros((k, 2), np.float32), edge_dir=np.zeros((k, 2), np.float32),
               edge_type=np.zeros((k,), np.int8), edge_mask=np.zeros((k,), bool))
    n = len(pick)
    norm = np.maximum(np.linalg.norm(dirs[pick], axis=-1, keepdims=True), 1e-6)
    out["edge_xy"][:n] = xy[pick]
    out["edge_dir"][:n] = dirs[pick] / norm
    out["edge_type"][:n] = types[pick]
    out["edge_mask"][:n] = True
    return out


def group_inputs(p: dict, w: int, origin: np.ndarray, heading: float, ego_speed: float, dt: float) -> dict:
    """Cached arrays (`wm_*` keys) of one replan group from a `predict_step` output, world w."""
    M, T = p["mean_xy"].shape[2:4]
    A = NUM_AGENTS
    cur = rot(p["cur_xy"][w] - origin, -heading)                                       # [n,2]
    mean = rot(p["mean_xy"][w] - origin, -heading)                                     # [n,M,T,2]
    sel, dist = select_agents(cur, mean, p["probs"][w], p["valid"][w], ego_speed, dt)
    k = len(sel)
    out = dict(
        wm_agent_idx=np.full((A,), -1, np.int16), wm_type=np.zeros((A,), np.int8),
        wm_size=np.zeros((A, 2), np.float32), wm_current=np.zeros((A, 3), np.float32),
        wm_mean=np.zeros((A, M, T, 3), np.float32), wm_sigma=np.zeros((A, M, T, 2), np.float32),
        wm_cov=np.zeros((A, M, T, 3), np.float32), wm_probs=np.zeros((A, M), np.float32),
        wm_valid=np.zeros((A,), bool), wm_sel_dist=np.zeros((A,), np.float32),
    )
    sig = p["sigma"][w][sel]
    cov = np.stack([sig[..., 0] ** 2, sig[..., 1] ** 2, np.zeros_like(sig[..., 0])], -1)
    out["wm_agent_idx"][:k] = sel
    out["wm_type"][:k] = p["types"][w][sel]
    out["wm_size"][:k] = np.stack([p["length"][w][sel], p["width"][w][sel]], -1)
    out["wm_current"][:k] = np.concatenate([cur[sel], wrap(p["cur_yaw"][w][sel] - heading)[:, None]], -1)
    out["wm_mean"][:k] = np.concatenate([mean[sel], wrap(p["mean_yaw"][w][sel] - heading)[..., None]], -1)
    out["wm_sigma"][:k] = sig
    out["wm_cov"][:k] = rotate_cov(cov, p["ego_yaw"][w] - heading)
    out["wm_probs"][:k] = p["probs"][w][sel]
    out["wm_valid"][:k] = True
    out["wm_sel_dist"][:k] = dist
    return out


def _roadgraph_np(st) -> dict:
    rg = st.roadgraph_points
    return {k: np.asarray(getattr(rg, k)) for k in ("x", "y", "dir_x", "dir_y", "types", "valid")}


def cache_file(path: str, forecaster, tfrecord_dir: str, wm_dir: str, edge_dir: str, *,
               validate: bool = True, timings: list | None = None) -> dict:
    """Write the WM and edge caches for one batch file; return validation statistics."""
    name = os.path.basename(path)[:-4]
    d = dict(np.load(path))
    st = load_batch_state(d, tfrecord_dir)
    rs = jax_state(replaced_state(jax.device_get(st), d))
    rg_np = _roadgraph_np(st)
    G = len(d["group_t"])
    T = int(forecaster.config.future_steps)
    dt = 0.1 * int(forecaster.config.future_stride)
    rows = []
    edges = {k: [] for k in ("edge_xy", "edge_dir", "edge_type", "edge_mask")}
    stats = dict(groups=0, mean_err_m=0.0, mean_excess_m=0.0, prob_err=0.0, origin_err_m=0.0, heading_err_rad=0.0,
                 recorded_agents_not_modeled=0, agents_valid=0)
    preds = {}
    for t in np.unique(d["group_t"]):
        t0 = time.time()
        preds[int(t)] = predict_step(forecaster, rs, int(t))
        if timings is not None:
            timings.append(time.time() - t0)
    for g in range(G):
        t, w = int(d["group_t"][g]), int(d["group_scen"][g])
        p = preds[t]
        origin, heading = d["group_origin"][g].astype(np.float64), float(d["group_heading"][g])
        stats["origin_err_m"] = max(stats["origin_err_m"], float(np.abs(p["origin"][w] - origin).max()))
        stats["heading_err_rad"] = max(stats["heading_err_rad"], float(abs(wrap(p["ego_yaw"][w] - heading))))
        speed = float(np.hypot(d["group_ego_state"][g, 3], d["group_ego_state"][g, 4]))
        rows.append(group_inputs(p, w, origin, heading, speed, dt))
        stats["agents_valid"] += int(rows[-1]["wm_valid"].sum())
        mean = rot(p["mean_xy"][w] - origin, -heading)
        for key, v in edge_tokens(rg_np, w, origin, heading).items():
            edges[key].append(v)

        if validate and d["group_valid"][g]:
            rec_idx = d["track_agent_idx"][g]
            n_mod = p["mean_xy"].shape[1]
            for slot, a in enumerate(rec_idx):
                if a < 0:
                    continue
                if a >= n_mod or not p["valid"][w][a]:
                    stats["recorded_agents_not_modeled"] += int(d["wm_valid"][g, slot].any())
                    continue
                if not d["wm_valid"][g, slot].any():
                    continue
                rec16 = d["wm_traj"][g, slot][:, 2:2 * T + 1:2, :2]                         # [M,T,2]
                err = np.abs(mean[a] - rec16.astype(np.float64))
                half_step = np.spacing(np.abs(rec16)).astype(np.float64) / 2
                stats["mean_err_m"] = max(stats["mean_err_m"], float(err.max()))
                stats["mean_excess_m"] = max(stats["mean_excess_m"], float((err - half_step).max()))
                stats["prob_err"] = max(stats["prob_err"], float(np.abs(
                    p["probs"][w][a] - d["wm_mode_probs"][g, slot].astype(np.float64)).max()))
            stats["groups"] += 1
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)
    out = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    for target, arrays in ((os.path.join(wm_dir, name + WM_SUFFIX), out),
                           (os.path.join(edge_dir, name + EDGE_SUFFIX), {k: np.stack(v) for k, v in edges.items()})):
        tmp = target + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, target)
    return stats


def check_stats(stats: dict) -> list[str]:
    bad = []
    if stats["mean_excess_m"] > EXCESS_TOL_M:
        bad.append(f"mode means differ from the recorded live-planner forecast by {stats['mean_excess_m']:.3f} m "
                   f"beyond float16 storage precision (max raw error {stats['mean_err_m']:.3f} m)")
    if stats["prob_err"] > PROB_TOL:
        bad.append(f"mode probabilities differ by {stats['prob_err']:.4f}")
    if stats["origin_err_m"] > 1e-3 or stats["heading_err_rad"] > 1e-4:
        bad.append(f"WM anchor != group frame ({stats['origin_err_m']:.2e} m, {stats['heading_err_rad']:.2e} rad)")
    return bad


# ---- extra validation on a few files ---------------------------------------------------

def logged_state_control(path: str, forecaster, tfrecord_dir: str) -> dict:
    """Rerun on the original log (ego never replaced): must NOT reproduce the recorded forecast."""
    d = dict(np.load(path))
    st = load_batch_state(d, tfrecord_dir)
    T = int(forecaster.config.future_steps)
    errs = []
    for t in np.unique(d["group_t"]):
        p = predict_step(forecaster, st, int(t))
        for g in np.flatnonzero(d["group_t"] == t):
            w = int(d["group_scen"][g])
            origin, heading = d["group_origin"][g].astype(np.float64), float(d["group_heading"][g])
            for slot, a in enumerate(d["track_agent_idx"][g]):
                if a < 0 or a >= p["mean_xy"].shape[1] or not d["wm_valid"][g, slot].any():
                    continue
                mean = rot(p["mean_xy"][w][a] - origin, -heading)
                rec = d["wm_traj"][g, slot][:, 2:2 * T + 1:2, :2].astype(np.float64)
                errs.append(float(np.abs(mean - rec).max()))
    errs = np.asarray(errs)
    return {"agents": int(len(errs)), "median_err_m": float(np.median(errs)) if len(errs) else None,
            "frac_err_above_tol": float((errs > MEAN_TOL_M).mean()) if len(errs) else None}


def collision_agent_coverage(path: str, wm_path: str, tfrecord_dir: str, max_groups: int = 40) -> dict:
    """Share of Waymax-overlapping candidates whose colliding logged agent(s) are in the selected set."""
    from es_discriminator.label_bank import _to_world
    from es_discriminator.real_es.waymax_labels import _boxes
    from waymax.utils import geometry

    @jax.jit
    def per_object(ego_kt5, obj_ot5, valid_ot):
        k, t = ego_kt5.shape[:2]
        o = obj_ot5.shape[0]
        a = jnp.broadcast_to(ego_kt5[:, None], (k, o, t, 5))
        b = jnp.broadcast_to(obj_ot5[None], (k, o, t, 5))
        return jnp.any(geometry.has_overlap(a, b) & valid_ot[None], axis=2)              # [K,O]

    d = dict(np.load(path))
    lab = dict(np.load(path[:-4] + ".labels_waymax.npz"))
    wm = dict(np.load(wm_path))
    st = load_batch_state(d, tfrecord_dir)
    log = st.log_trajectory
    lx, ly, lyaw = np.asarray(log.x), np.asarray(log.y), np.asarray(log.yaw)
    llen, lwid, lvalid = np.asarray(log.length), np.asarray(log.width), np.asarray(log.valid).astype(bool)
    is_sdc = np.asarray(st.object_metadata.is_sdc).astype(bool)
    T = d["cand_traj"].shape[1]
    res = {"overlapping_candidates": 0, "covered_new": 0, "covered_recorded_nearest32": 0,
           "with_agent_absent_at_t": 0, "observable": 0, "observable_covered_new": 0}
    for g in range(min(len(d["group_t"]), max_groups)):
        rows = np.flatnonzero((d["cand_group"] == g) & (lab["gt_overlap_step"] > 0))
        if len(rows) == 0:
            continue
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        h = int(lab["label_horizon"][g])
        e = int(np.argmax(is_sdc[w]))
        steps_c = np.minimum(np.arange(t + 1, t + 1 + T), lx.shape[2] - 1)
        in_h = np.arange(1, T + 1) <= h
        obj = _boxes(np.stack([lx[w][:, steps_c], ly[w][:, steps_c], lyaw[w][:, steps_c]], -1),
                     llen[w][:, steps_c], lwid[w][:, steps_c])
        others = np.ones(lx.shape[1], bool)
        others[e] = False
        valid = lvalid[w][:, steps_c] & in_h[None] & others[:, None]
        world = _to_world(np.nan_to_num(d["cand_traj"][rows].astype(np.float64)), d["group_origin"][g],
                          float(d["group_heading"][g]))
        L, W = d["group_ego_state"][g, 5], d["group_ego_state"][g, 6]
        hit = np.asarray(per_object(jnp.asarray(_boxes(world, L, W)), jnp.asarray(obj), jnp.asarray(valid)))
        new = set(int(a) for a in wm["wm_agent_idx"][g] if a >= 0)
        old = set(int(a) for a in d["track_agent_idx"][g] if a >= 0)
        for r in range(len(rows)):
            agents = set(np.flatnonzero(hit[r]).tolist())
            if not agents:
                continue
            res["overlapping_candidates"] += 1
            res["covered_new"] += int(agents <= new)
            res["covered_recorded_nearest32"] += int(agents <= old)
            if all(lvalid[w, a, t] for a in agents):
                res["observable"] += 1
                res["observable_covered_new"] += int(agents <= new)
            else:
                res["with_agent_absent_at_t"] += 1
    return res


def edge_coverage(path: str, edge_path: str, tfrecord_dir: str, cands_per_group: int = 32,
                  near_m: float = 5.0, tol_m: float = 0.5) -> dict:
    """How well tokens describe the nearest road edge at candidate positions within `near_m` of one.

    Reference: Waymax signed distance to the nearest road-edge point of the full roadgraph
    (candidate centre at the logged ego height). A token set "covers" a query when the
    perpendicular distance to the tangent line of its nearest token is within `tol_m` of
    the reference. Compared for the new edge tokens and the edge-type points of the
    256-point context map.
    """
    from es_discriminator.analyze_es_decisions import road_edges
    from waymax.metrics.roadgraph import compute_signed_distance_to_nearest_road_edge_point

    d = dict(np.load(path))
    ed = dict(np.load(edge_path))
    st = load_batch_state(d, tfrecord_dir)
    z = np.asarray(st.log_trajectory.z)
    is_sdc = np.asarray(st.object_metadata.is_sdc).astype(bool)
    rng = np.random.default_rng(0)
    sdf = jax.jit(compute_signed_distance_to_nearest_road_edge_point)
    cache = {}
    tot = {"queries": 0, "new_tokens": 0, "context_map": 0}
    for g in np.flatnonzero(d["group_valid"].astype(bool)):
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        if w not in cache:
            cache[w] = road_edges(st, w)
        rows = np.flatnonzero(d["cand_group"] == g)
        rows = rng.choice(rows, min(cands_per_group, len(rows)), replace=False)
        local = np.nan_to_num(d["cand_traj"][rows][:, 1:50:2, :2].astype(np.float64)).reshape(-1, 2)
        origin, heading = d["group_origin"][g].astype(np.float64), float(d["group_heading"][g])
        world = rot(local, heading) + origin
        e = int(np.argmax(is_sdc[w]))
        q = np.concatenate([world, np.full((len(world), 1), z[w, e, t])], -1).astype(np.float32)
        pad = int(np.ceil(len(q) / 1024) * 1024)
        qp = np.zeros((pad, 3), np.float32)
        qp[:len(q)] = q
        ref = np.abs(np.asarray(sdf(jnp.asarray(qp), cache[w]))[:len(q)])
        near = ref <= near_m
        if not near.any():
            continue
        lq, rq = local[near], ref[near]
        tot["queries"] += int(near.sum())

        def approx(xy, dirs):
            if len(xy) == 0:
                return np.full(len(lq), np.inf)
            dd = np.linalg.norm(lq[:, None] - xy[None], axis=-1)
            j = dd.argmin(1)
            v = lq - xy[j]
            return np.abs(v[:, 0] * dirs[j, 1] - v[:, 1] * dirs[j, 0])

        m = ed["edge_mask"][g]
        tot["new_tokens"] += int((np.abs(approx(ed["edge_xy"][g][m], ed["edge_dir"][g][m]) - rq) <= tol_m).sum())
        cm = d["map"][g][d["map_mask"][g]]
        is_edge = np.isin(np.rint(cm[:, 4] * 32).astype(int), (15, 16))
        dirs = cm[is_edge, 2:4] / np.maximum(np.linalg.norm(cm[is_edge, 2:4], axis=-1, keepdims=True), 1e-6)
        tot["context_map"] += int((np.abs(approx(cm[is_edge, :2] * 100.0, dirs) - rq) <= tol_m).sum())
    n = max(tot["queries"], 1)
    return {"queries_within_%gm_of_edge" % near_m: tot["queries"],
            "covered_by_new_edge_tokens": tot["new_tokens"] / n,
            "covered_by_context_map_256": tot["context_map"] / n, "tolerance_m": tol_m}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tfrecord_dir", default=DEFAULT_TFRECORDS)
    p.add_argument("--wm_dir", required=True)
    p.add_argument("--edge_dir", required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--validate_files", type=int, default=6,
                   help="files that also get the logged-state control and agent/edge coverage")
    p.add_argument("--summary_out", default="")
    p.add_argument("--overwrite", action="store_true", help="recompute batches that already have both caches")
    a = p.parse_args()

    forecaster = load_forecaster(a.checkpoint, a.device)
    cfg = forecaster.config
    os.makedirs(a.wm_dir, exist_ok=True)
    manifest = {
        "checkpoint": a.checkpoint, "checkpoint_sha256": sha256(a.checkpoint),
        "num_modes": int(cfg.num_modes), "future_steps": int(cfg.future_steps),
        "step_dt_s": 0.1 * int(cfg.future_stride), "modeled_agents": int(cfg.max_agents),
        "frame": "group ego frame = WM anchor frame (SDC pose at t of replaced_state); wm_sigma is the WM's "
                 "diagonal std in the anchor frame, wm_cov the same Gaussian in the group frame",
        "time_grid": "wm_mean[..., i, :] is world step t + 2(i+1) = candidate step i of cand_traj[:, 1:50:2]",
        "agents": {"count": NUM_AGENTS, "rule": "min(current distance, distance of p>=%.2f mode means to ego CV "
                   "rollouts at speed x %s)" % (SEL_MIN_PROB, list(SEL_SPEED_RATIOS))},
        "edges": {"count": NUM_EDGES, "box_behind_ahead_halfwidth_m": EDGE_BOX, "types": [15, 16],
                  "sampling": "farthest point from the point nearest the ego, min spacing %.1f m" % EDGE_MIN_SPACING},
    }
    with open(os.path.join(a.wm_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    files = batch_files(a.data_dir)
    if a.max_files:
        files = files[:a.max_files]
    files = files[a.shard::a.num_shards]
    agg = dict(files=0, groups=0, mean_err_m=0.0, mean_excess_m=0.0, prob_err=0.0, origin_err_m=0.0, heading_err_rad=0.0,
               recorded_agents_not_modeled=0, agents_valid=0)
    timings: list[float] = []
    failures = []
    t0 = time.time()
    skipped = 0
    for i, f in enumerate(files):
        base = os.path.basename(f)[:-4]
        if (not a.overwrite and os.path.exists(os.path.join(a.wm_dir, base + WM_SUFFIX))
                and os.path.exists(os.path.join(a.edge_dir, base + EDGE_SUFFIX))):
            skipped += 1
            continue
        s = cache_file(f, forecaster, a.tfrecord_dir, a.wm_dir, a.edge_dir, timings=timings)
        bad = check_stats(s)
        if bad:
            failures.append({"file": os.path.basename(f), "problems": bad})
        agg["files"] += 1
        for k in ("groups", "recorded_agents_not_modeled", "agents_valid"):
            agg[k] += s[k]
        for k in ("mean_err_m", "mean_excess_m", "prob_err", "origin_err_m", "heading_err_rad"):
            agg[k] = max(agg[k], s[k])
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"wm cache {i + 1}/{len(files)}  {(time.time() - t0) / (i + 1):.1f}s/batch  "
                  f"max mean err {agg['mean_err_m']:.3f} m  failures {len(failures)}", flush=True)

    extra = {}
    vf = files[:a.validate_files]
    if vf:
        extra["logged_state_control"] = logged_state_control(vf[0], forecaster, a.tfrecord_dir)
        cov = [collision_agent_coverage(f, os.path.join(a.wm_dir, os.path.basename(f)[:-4] + WM_SUFFIX),
                                        a.tfrecord_dir) for f in vf]
        n = max(sum(c["overlapping_candidates"] for c in cov), 1)
        n_obs = max(sum(c["observable"] for c in cov), 1)
        extra["collision_agent_coverage"] = {
            "files": len(vf), "overlapping_candidates": n,
            "new_selection": sum(c["covered_new"] for c in cov) / n,
            "recorded_nearest32": sum(c["covered_recorded_nearest32"] for c in cov) / n,
            "frac_hitting_agent_absent_at_t": sum(c["with_agent_absent_at_t"] for c in cov) / n,
            "new_selection_on_observable": sum(c["observable_covered_new"] for c in cov) / n_obs}
        ec = [edge_coverage(f, os.path.join(a.edge_dir, os.path.basename(f)[:-4] + EDGE_SUFFIX), a.tfrecord_dir)
              for f in vf]
        q = max(sum(e["queries_within_5m_of_edge"] for e in ec), 1)
        extra["edge_coverage"] = {
            "files": len(vf), "queries_within_5m_of_edge": q, "tolerance_m": 0.5,
            "new_edge_tokens": sum(e["covered_by_new_edge_tokens"] * e["queries_within_5m_of_edge"] for e in ec) / q,
            "context_map_256": sum(e["covered_by_context_map_256"] * e["queries_within_5m_of_edge"] for e in ec) / q}

    summary = {
        **agg, "skipped_existing": skipped, "tolerances": {"mean_beyond_float16_m": EXCESS_TOL_M, "prob": PROB_TOL,
                                                             "logged_state_control_m": MEAN_TOL_M},
        "wm_forward_s_per_replan_step": {"median": float(np.median(timings)) if timings else None,
                                         "p90": float(np.percentile(timings, 90)) if timings else None,
                                         "worlds_per_forward": 5, "device": a.device},
        "failures": failures, **extra,
    }
    print(json.dumps(summary, indent=1))
    if a.summary_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.summary_out)), exist_ok=True)
        with open(a.summary_out, "w") as f:
            json.dump(summary, f, indent=1)
    if failures:
        raise SystemExit(f"{len(failures)} file(s) failed WM cache validation")
    ctl = extra.get("logged_state_control", {})
    if ctl and ctl.get("frac_err_above_tol") is not None and ctl["frac_err_above_tol"] < 0.05:
        raise SystemExit("logged-state control reproduced the recorded forecast; validation is not discriminative")


if __name__ == "__main__":
    main()
