"""WOMD / Waymax-style evaluation metrics for non-goal-conditioned ego policies.

Open-loop (Waymo Open Motion Dataset motion-prediction style, K samples per scenario):
  - minADE / minFDE / meanADE / meanFDE at fixed horizons (default 3s, 5s)
  - Miss rate with the WOMD speed-scaled lateral/longitudinal thresholds
  - Overlap rate / offroad rate of the K samples against the logged scene

Closed-loop (Waymax style, ego re-plans periodically, other agents replay the log):
  - Collision (bbox overlap) and offroad, plus the first step at which they occur
  - Log divergence (ADE/FDE against the logged ego trajectory) at fixed horizons
  - Route progress: arc length travelled along the logged ego path, and its ratio
    to the logged progress

Conventions: all trajectories are world-frame numpy arrays sampled at the world dt
(0.1s for WOMD). Index 0 of a "future" array is the anchor (current) step.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from waymax.metrics.roadgraph import compute_signed_distance_to_nearest_road_edge_point
from waymax.utils import geometry

# WOMD miss-rate thresholds per horizon (seconds): (lateral_m, longitudinal_m).
WOMD_MISS_THRESHOLDS: dict[float, tuple[float, float]] = {
    3.0: (1.0, 2.0),
    5.0: (1.8, 3.6),
    8.0: (3.0, 6.0),
}


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def womd_speed_scale(speed_b: np.ndarray) -> np.ndarray:
    """WOMD threshold scaling: 0.5 below 1.4 m/s, 1.0 above 11 m/s, linear between."""
    return 0.5 + 0.5 * np.clip((speed_b - 1.4) / (11.0 - 1.4), 0.0, 1.0)


def _nanmean(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        count = np.sum(~np.isnan(x), axis=axis)
        total = np.nansum(x, axis=axis)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------


def extract_log_arrays(sim_state) -> dict[str, np.ndarray]:
    """Copies the logged trajectories needed by the metrics to host numpy arrays.

    Must be called on the *original* sim_state, before the ego log is overwritten
    by the closed-loop rollout.
    """
    traj = sim_state.log_trajectory
    is_sdc_bn = np.asarray(sim_state.object_metadata.is_sdc).astype(bool)
    ego_idx_b = np.argmax(is_sdc_bn.astype(np.int32), axis=1)
    b_idx = np.arange(is_sdc_bn.shape[0])

    def _ego(field) -> np.ndarray:
        return np.asarray(field)[b_idx, ego_idx_b]

    return {
        "ego_idx": ego_idx_b,
        "is_sdc": is_sdc_bn,
        "ego_xy": np.stack([_ego(traj.x), _ego(traj.y)], axis=-1),  # [B, T, 2]
        "ego_yaw": _ego(traj.yaw),  # [B, T]
        "ego_vel": np.stack([_ego(traj.vel_x), _ego(traj.vel_y)], axis=-1),  # [B, T, 2]
        "ego_valid": _ego(traj.valid).astype(bool),  # [B, T]
        "ego_length": _ego(traj.length),  # [B, T]
        "ego_width": _ego(traj.width),  # [B, T]
        "ego_z": _ego(traj.z),  # [B, T]
        # Logged boxes of every object, [B, N, T, 5] as (x, y, length, width, yaw).
        "obj_box": np.stack(
            [
                np.asarray(traj.x),
                np.asarray(traj.y),
                np.asarray(traj.length),
                np.asarray(traj.width),
                np.asarray(traj.yaw),
            ],
            axis=-1,
        ),
        "obj_valid": np.asarray(traj.valid).astype(bool),  # [B, N, T]
    }


# ---------------------------------------------------------------------------
# Displacement metrics (minADE / minFDE / miss rate)
# ---------------------------------------------------------------------------


def displacement_metrics(
    pred_xy_bkt2: np.ndarray,
    gt_xy_bt2: np.ndarray,
    gt_yaw_bt: np.ndarray,
    gt_valid_bt: np.ndarray,
    init_speed_b: np.ndarray,
    horizon_steps: int,
    miss_thresholds: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
    """WOMD-style displacement metrics over future steps 1..horizon_steps.

    Args:
        pred_xy_bkt2: K predicted futures, index 0 = anchor step. [B, K, T>=H+1, 2]
        gt_xy_bt2 / gt_yaw_bt / gt_valid_bt: logged ego future aligned with the
            prediction (index 0 = anchor step). [B, T>=H+1, ...]
        init_speed_b: ego speed at the anchor step, used for miss-rate scaling.
        horizon_steps: number of world steps in the horizon (e.g. 30 for 3s @ 10Hz).
        miss_thresholds: (lateral_m, longitudinal_m) or None to skip miss rate.

    Returns per-scenario arrays [B] (NaN where the ground truth is unavailable).
    """
    sl = slice(1, horizon_steps + 1)
    pred = pred_xy_bkt2[:, :, sl]
    gt = gt_xy_bt2[:, sl]
    valid = gt_valid_bt[:, sl]  # [B, H]

    err_bkh = np.linalg.norm(pred - gt[:, None], axis=-1)
    num_valid_b = valid.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ade_bk = np.where(
            num_valid_b[:, None] > 0,
            (err_bkh * valid[:, None]).sum(axis=-1) / np.maximum(num_valid_b[:, None], 1),
            np.nan,
        )
    final_valid_b = valid[:, -1]
    fde_bk = np.where(final_valid_b[:, None], err_bkh[:, :, -1], np.nan)

    out = {
        "minADE": np.min(ade_bk, axis=1),
        "minFDE": np.min(fde_bk, axis=1),
        "meanADE": np.mean(ade_bk, axis=1),
        "meanFDE": np.mean(fde_bk, axis=1),
    }

    if miss_thresholds is not None:
        lat_thr, lon_thr = miss_thresholds
        scale_b = womd_speed_scale(init_speed_b)
        # Rotate the final-step error into the ground-truth heading frame.
        diff_bk2 = pred[:, :, -1] - gt[:, None, -1]
        yaw_b = gt_yaw_bt[:, horizon_steps]
        c, s = np.cos(yaw_b)[:, None], np.sin(yaw_b)[:, None]
        lon_err = np.abs(c * diff_bk2[..., 0] + s * diff_bk2[..., 1])
        lat_err = np.abs(-s * diff_bk2[..., 0] + c * diff_bk2[..., 1])
        hit_bk = (lon_err <= lon_thr * scale_b[:, None]) & (lat_err <= lat_thr * scale_b[:, None])
        out["miss"] = np.where(final_valid_b, (~np.any(hit_bk, axis=1)).astype(np.float64), np.nan)

    return out


# ---------------------------------------------------------------------------
# Scene interaction metrics (overlap / offroad)
# ---------------------------------------------------------------------------


@jax.jit
def _ego_overlap_btm(ego_box_btm5, other_box_btn5, other_valid_btn):
    b, t, m, _ = ego_box_btm5.shape
    n = other_box_btn5.shape[2]
    ego = jnp.broadcast_to(ego_box_btm5[:, :, :, None, :], (b, t, m, n, 5))
    other = jnp.broadcast_to(other_box_btn5[:, :, None, :, :], (b, t, m, n, 5))
    overlap = geometry.has_overlap(ego, other) & other_valid_btn[:, :, None, :]
    return jnp.any(overlap, axis=-1)


@jax.jit
def _ego_offroad_btm(ego_box_btm5, ego_z_b, roadgraph_points):
    def _per_world(boxes_tm5, z, rg):
        def _per_step(boxes_m5):
            corners_xy = geometry.corners_from_bboxes(boxes_m5).reshape(-1, 2)  # [M*4, 2]
            query = jnp.concatenate([corners_xy, jnp.full((corners_xy.shape[0], 1), z)], axis=-1)
            dist = compute_signed_distance_to_nearest_road_edge_point(query, rg)
            return jnp.any(dist.reshape(boxes_m5.shape[0], 4) > 0.0, axis=-1)

        # Map over time to keep the [queries, roadgraph_points] buffer small.
        return jax.lax.map(_per_step, boxes_tm5)

    return jax.vmap(_per_world)(ego_box_btm5, ego_z_b, roadgraph_points)


def ego_overlap_and_offroad(
    ego_xyyaw_btm3: np.ndarray,
    t_abs: np.ndarray,
    log: dict[str, np.ndarray],
    roadgraph_points,
    anchor_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlap (with logged non-ego objects) and offroad flags for ego candidates.

    Args:
        ego_xyyaw_btm3: M candidate ego poses per step, [B, T', M, 3] (x, y, yaw).
        t_abs: absolute scenario timestep for each of the T' steps, [T'].
        log: output of `extract_log_arrays` on the original state.
        roadgraph_points: batched waymax RoadgraphPoints [B, P].
        anchor_step: step used for the ego box size and z.

    Returns:
        (overlap_btm, offroad_btm) boolean arrays.
    """
    b = ego_xyyaw_btm3.shape[0]
    b_idx = np.arange(b)
    length_b = log["ego_length"][:, anchor_step]
    width_b = log["ego_width"][:, anchor_step]
    z_b = log["ego_z"][:, anchor_step]
    size = np.broadcast_to(
        np.stack([length_b, width_b], axis=-1)[:, None, None, :],
        ego_xyyaw_btm3.shape[:3] + (2,),
    )
    ego_box = np.concatenate([ego_xyyaw_btm3[..., :2], size, ego_xyyaw_btm3[..., 2:3]], axis=-1)

    other_box = np.transpose(log["obj_box"][:, :, t_abs], (0, 2, 1, 3))  # [B, T', N, 5]
    other_valid = np.transpose(log["obj_valid"][:, :, t_abs], (0, 2, 1)).copy()  # [B, T', N]
    other_valid[b_idx, :, log["ego_idx"]] = False

    overlap = _ego_overlap_btm(
        jnp.asarray(ego_box, jnp.float32),
        jnp.asarray(other_box, jnp.float32),
        jnp.asarray(other_valid),
    )
    offroad = _ego_offroad_btm(
        jnp.asarray(ego_box, jnp.float32),
        jnp.asarray(z_b, jnp.float32),
        roadgraph_points,
    )
    return np.asarray(overlap), np.asarray(offroad)


def _first_true_step(flags_bt: np.ndarray, t_abs: np.ndarray) -> np.ndarray:
    any_b = np.any(flags_bt, axis=1)
    first = t_abs[np.argmax(flags_bt, axis=1)]
    return np.where(any_b, first, -1).astype(np.int32)


# ---------------------------------------------------------------------------
# Route progress
# ---------------------------------------------------------------------------


def _arclength_on_path(points_p2: np.ndarray, path_n2: np.ndarray) -> np.ndarray:
    """Arc length of the projection of each point onto a polyline.

    The last segment is extended so a point beyond the end of the path keeps
    accumulating progress (ratio can exceed 1).
    """
    seg = path_n2[1:] - path_n2[:-1]  # [S, 2]
    seg_len = np.linalg.norm(seg, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    good = seg_len > 1e-6
    if not np.any(good):
        return np.full((points_p2.shape[0],), np.nan)

    d = points_p2[:, None, :] - path_n2[None, :-1, :]  # [P, S, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.sum(d * seg[None], axis=-1) / np.maximum(seg_len**2, 1e-12)[None]
    upper = np.ones_like(seg_len)
    upper[np.flatnonzero(good)[-1]] = np.inf
    u = np.clip(u, 0.0, upper[None])
    proj = path_n2[None, :-1, :] + u[..., None] * seg[None]
    dist = np.linalg.norm(points_p2[:, None, :] - proj, axis=-1)
    dist = np.where(good[None], dist, np.inf)
    j = np.argmin(dist, axis=1)
    p = np.arange(points_p2.shape[0])
    return cum[j] + u[p, j] * seg_len[j]


def route_progress(
    sim_xy_bt2: np.ndarray,
    log_xy_bt2: np.ndarray,
    log_valid_bt: np.ndarray,
    anchor_step: int,
    min_log_progress_m: float = 1.0,
) -> dict[str, np.ndarray]:
    """Progress along the logged ego path, from the anchor step to the episode end.

    progress_ratio is NaN when the logged ego barely moves (< min_log_progress_m),
    since the ratio is undefined for a stationary expert.
    """
    b = sim_xy_bt2.shape[0]
    sim_prog = np.full((b,), np.nan)
    log_prog = np.full((b,), np.nan)
    for i in range(b):
        valid_t = np.flatnonzero(log_valid_bt[i, anchor_step:]) + anchor_step
        if valid_t.size < 2:
            continue
        path = log_xy_bt2[i, valid_t]
        log_prog[i] = float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=-1)))
        s = _arclength_on_path(sim_xy_bt2[i, [anchor_step, -1]], path)
        sim_prog[i] = float(s[1] - s[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(log_prog >= min_log_progress_m, sim_prog / log_prog, np.nan)
    return {"progress_m": sim_prog, "log_progress_m": log_prog, "progress_ratio": ratio}


# ---------------------------------------------------------------------------
# Top-level evaluators
# ---------------------------------------------------------------------------


def evaluate_open_loop(
    pred_world_bkt5: np.ndarray,
    pred_valid_bt: np.ndarray,
    log: dict[str, np.ndarray],
    roadgraph_points,
    anchor_step: int,
    world_dt: float,
    horizons_s: list[float],
) -> dict[str, np.ndarray]:
    """WOMD-style open-loop metrics of K samples predicted at `anchor_step`.

    Args:
        pred_world_bkt5: [B, K, T, 5] world trajectories (x, y, yaw, vx, vy) at the
            world dt, index 0 = anchor step (planner postprocess output).
        pred_valid_bt: [B, T] validity of the predicted steps.
    """
    num_steps_total = int(log["ego_xy"].shape[1])
    pred_steps = int(np.min(np.sum(pred_valid_bt, axis=1)))  # includes the anchor
    max_steps = min(pred_steps, num_steps_total - anchor_step)
    t_abs = np.arange(anchor_step, anchor_step + max_steps)

    gt_xy = log["ego_xy"][:, t_abs]
    gt_yaw = log["ego_yaw"][:, t_abs]
    gt_valid = log["ego_valid"][:, t_abs]
    init_speed = np.linalg.norm(log["ego_vel"][:, anchor_step], axis=-1)
    pred = pred_world_bkt5[:, :, :max_steps]

    out: dict[str, np.ndarray] = {}
    for h in horizons_s:
        h_steps = int(round(h / world_dt))
        if h_steps >= max_steps:
            continue
        tag = f"{h:g}s"
        disp = displacement_metrics(
            pred[..., :2],
            gt_xy,
            gt_yaw,
            gt_valid,
            init_speed,
            h_steps,
            WOMD_MISS_THRESHOLDS.get(float(h)),
        )
        for name, value in disp.items():
            out[f"ol_{name}@{tag}"] = value

    # Overlap / offroad over the future steps of every sample.
    ego_xyyaw = np.transpose(pred[:, :, 1:, :3], (0, 2, 1, 3))  # [B, T', K, 3]
    overlap_btk, offroad_btk = ego_overlap_and_offroad(
        ego_xyyaw, t_abs[1:], log, roadgraph_points, anchor_step
    )
    overlap_bk = np.any(overlap_btk, axis=1)
    offroad_bk = np.any(offroad_btk, axis=1)
    # WOMD overlap rate uses the top-1 trajectory; samples have no confidence, so
    # we report the mean over the K samples (expected rate of a random sample).
    out["ol_overlap_rate"] = overlap_bk.mean(axis=1)
    out["ol_offroad_rate"] = offroad_bk.mean(axis=1)
    return out


def evaluate_closed_loop(
    sim_world_bt5: np.ndarray,
    log: dict[str, np.ndarray],
    roadgraph_points,
    anchor_step: int,
    world_dt: float,
    horizons_s: list[float],
) -> dict[str, np.ndarray]:
    """Waymax-style closed-loop metrics of the simulated ego trajectory.

    Args:
        sim_world_bt5: [B, T, 5] ego trajectory (x, y, yaw, vx, vy) over the full
            scenario; steps < anchor_step equal the log.
    """
    num_steps_total = int(sim_world_bt5.shape[1])
    t_abs = np.arange(anchor_step, num_steps_total)
    sim_xy = sim_world_bt5[..., :2]

    # Future steps only (the anchor pose is the logged pose).
    ego_xyyaw = sim_world_bt5[:, t_abs[1:], None, :3]  # [B, T', 1, 3]
    overlap_bt1, offroad_bt1 = ego_overlap_and_offroad(
        ego_xyyaw, t_abs[1:], log, roadgraph_points, anchor_step
    )
    overlap_bt = overlap_bt1[..., 0]
    offroad_bt = offroad_bt1[..., 0]

    out: dict[str, np.ndarray] = {
        "cl_collision": np.any(overlap_bt, axis=1).astype(np.float64),
        "cl_offroad": np.any(offroad_bt, axis=1).astype(np.float64),
        "cl_collision_step": _first_true_step(overlap_bt, t_abs[1:]),
        "cl_offroad_step": _first_true_step(offroad_bt, t_abs[1:]),
    }

    # Log divergence (closed-loop ADE / FDE vs the logged ego).
    gt_xy = log["ego_xy"][:, t_abs]
    gt_yaw = log["ego_yaw"][:, t_abs]
    gt_valid = log["ego_valid"][:, t_abs]
    init_speed = np.linalg.norm(log["ego_vel"][:, anchor_step], axis=-1)
    sim_future = sim_xy[:, None, t_abs]  # [B, 1, T', 2]
    full_steps = len(t_abs) - 1
    for h in list(horizons_s) + [None]:
        h_steps = full_steps if h is None else int(round(h / world_dt))
        if h_steps > full_steps or h_steps <= 0:
            continue
        disp = displacement_metrics(sim_future, gt_xy, gt_yaw, gt_valid, init_speed, h_steps, None)
        tag = "full" if h is None else f"{h:g}s"
        out[f"cl_ADE@{tag}"] = disp["minADE"]
        out[f"cl_FDE@{tag}"] = disp["minFDE"]

    out.update(
        {
            f"cl_{k}": v
            for k, v in route_progress(sim_xy, log["ego_xy"], log["ego_valid"], anchor_step).items()
        }
    )
    out["cl_success"] = ((out["cl_collision"] == 0) & (out["cl_offroad"] == 0)).astype(np.float64)
    return out


def summarize(per_scenario: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    """NaN-aware mean of each metric over scenarios (plus valid counts)."""
    summary: dict[str, float] = {}
    for key in keys:
        if key.endswith("_step"):
            continue
        values = np.array(
            [np.nan if r.get(key) is None else float(r[key]) for r in per_scenario],
            dtype=np.float64,
        )
        summary[key] = float(_nanmean(values))
        summary[f"{key}/count"] = int(np.sum(~np.isnan(values)))
    return summary
