"""Rulebook-style ranking of ES plans (Censi et al. 2019) with slack on collision probability (Wray et al. 2015).

Each rule is a violation >= 0 that is 0 when the rule holds, so plans that satisfy a rule tie on it and
only magnitudes of real violations are compared. Rules are checked over the executed window plus a
lookahead (``near_s``) first, then over the whole horizon. Keys, in order (lower is better):
  feasibility    1 - kinematically feasible
  collision      P(overlap) within near_s, P(overlap) over the horizon     (slack ``collision_slack``)
                 -expected first-contact time                             (later contact first)
  off-road       depth past the road edge within near_s, over the horizon, -first off-road time
  goal           1 - goal_progress (0 = on pace to reach the goal by the deadline)
  tiebreak       -min(edge clearance, OFFROAD_MARGIN_M), lane-keeping cost, jerk

``rank_violations`` filters the pool key by key (survivors within the key's slack, 1e-6 if none, of the
best survivor) and scores (d + 0.99 r) / n_keys, d = keys passed, r = rank within the group by the key
that failed. Top-k elites are the deepest survivors, then the nearest misses. Scores are relative to
the ranked set, so a pool that merges elites with new samples must be re-ranked as a whole.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scores import helpers as scorer_helpers

VIOLATION_KEYS = (
    "v_infeasible",
    "v_coll_near",
    "v_coll_all",
    "v_coll_time",
    "v_off_near",
    "v_off_all",
    "v_off_time",
    "v_goal",
    "v_clearance",
    "v_lane",
    "v_jerk",
)
SLACK_KEYS = ("v_coll_near", "v_coll_all")


def ego_kinematics(sim_state, timestep: int, world_idx: int = 0, lookback: int = 5) -> tuple[float, float, float]:
    """(speed, acceleration, curvature) of the ego at ``timestep``, from the last ``lookback`` steps (0.1 s each)."""
    ego = scorer_helpers.get_ego_idx(sim_state, world_idx)
    tr = sim_state.sim_trajectory
    t1 = int(timestep)
    t0 = max(t1 - lookback, 0)
    speed = np.hypot(np.asarray(tr.vel_x[world_idx, ego, t0 : t1 + 1]), np.asarray(tr.vel_y[world_idx, ego, t0 : t1 + 1]))
    if t1 == t0:
        return float(speed[-1]), 0.0, 0.0
    accel = float(speed[-1] - speed[0]) / (0.1 * (t1 - t0))
    dyaw = float(np.angle(np.exp(1j * (float(tr.yaw[world_idx, ego, t1]) - float(tr.yaw[world_idx, ego, t0])))))
    dist = float(speed.mean()) * 0.1 * (t1 - t0)
    return float(speed[-1]), accel, dyaw / dist if dist > 0.5 else 0.0


def kinematic_feasibility(
    traj_pt5: np.ndarray, *, dt: float = 0.1, max_accel: float = 8.0, max_lat_accel: float = 8.0,
    a0: float | None = None, max_accel_step: float = 2.0, window_s: float = 0.3,
) -> np.ndarray:
    """[P] 1 if the world-frame plan (index 0 = current state) never exceeds ``max_accel`` longitudinally
    or ``max_lat_accel`` laterally (speed * yaw rate), else 0. Catches plans that start at a speed the
    ego is not at (a velocity jump between index 0 and 1). With ``a0`` (current acceleration), the
    plan's mean acceleration over the first ``window_s`` must also be within ``max_accel_step`` of it,
    so consecutive plans join without an acceleration step (jerk spike) at the replan."""
    speed = np.hypot(traj_pt5[..., 3], traj_pt5[..., 4])
    lon = np.abs(np.diff(speed, axis=1)) / dt
    yaw_rate = np.abs(np.angle(np.exp(1j * np.diff(traj_pt5[..., 2], axis=1)))) / dt
    lat = 0.5 * (speed[:, 1:] + speed[:, :-1]) * yaw_rate
    ok = (lon.max(axis=1) <= max_accel) & (lat.max(axis=1) <= max_lat_accel)
    if a0 is not None:
        k = max(1, int(round(window_s / dt)))
        a_start = (speed[:, k] - speed[:, 0]) / (k * dt)
        ok &= np.abs(a_start - a0) <= max_accel_step
    return ok.astype(np.float32)


def plan_jerk(traj_pt5: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """[P] mean squared jerk (longitudinal + lateral, m^2/s^6) of world-frame plans [P, T, 5]."""
    vel = traj_pt5[..., 3:5]
    acc = np.gradient(vel, dt, axis=1)
    jerk = np.gradient(acc, dt, axis=1)
    return (jerk**2).sum(-1).mean(-1)


def goal_progress(
    traj_pt5: np.ndarray, goal_xy: np.ndarray, timestep: int, goal_step: int = 90, dt: float = 0.1, reach_m: float = 2.0,
) -> np.ndarray:
    """Pace adherence toward the episode goal, averaged over 1 s checkpoints up to ``goal_step``.

    d_t = plan-to-goal distance, D_k = min_{t <= t_k} d_t at checkpoints t_k = 1 s, 2 s, ... (only up
    to the deadline: the episode ends at ``goal_step``). The pace needed to arrive on time covers
    d_0 t_k / (time left) by t_k, so
        g_k = 1 if D_k <= reach_m else clip((d_0 - D_k) / (d_0 t_k / time left), 0, 1),   G = mean_k g_k.
    Averaging over checkpoints rewards being on pace *now*: a plan that only speeds up late scores
    lower, which matters because only its first second is executed before the next replan. Going
    faster than needed earns nothing; when the deadline is inside the horizon the plan must get
    within ``reach_m`` (lateral offset included) for the last checkpoints to score 1.
    """
    xy = np.asarray(traj_pt5[..., :2], dtype=np.float64)
    d = np.linalg.norm(xy - np.asarray(goal_xy, np.float64).reshape(1, 1, 2), axis=-1)  # [P, T]
    steps_left = max(int(goal_step) - int(timestep), 1)
    n = min(xy.shape[1] - 1, steps_left)
    per = max(1, int(round(1.0 / dt)))
    ks = list(range(per, n + 1, per)) or [n]
    if ks[-1] != n:
        ks.append(n)
    d_cummin = np.minimum.accumulate(d, axis=1)
    g = []
    for k in ks:
        need = d[:, 0] * k / steps_left
        gk = np.clip((d[:, 0] - d_cummin[:, k]) / np.maximum(need, 1e-3), 0.0, 1.0)
        g.append(np.where(d_cummin[:, k] <= reach_m, 1.0, gk))
    return np.mean(g, axis=0).astype(np.float32)


def lane_keeping_cost(
    lane_points: Any, traj_pt5: np.ndarray, heading_w: float = 1.0, max_heading_rad: float = np.pi / 3,
    max_lat_m: float = 5.0, stride: int = 5,
) -> np.ndarray:
    """[P] mean over time of |lateral offset| (m) to the nearest direction-consistent lane centreline
    point + ``heading_w`` * |heading error| (rad). Offsets are capped at ``max_lat_m``."""
    traj = np.asarray(traj_pt5, dtype=np.float32)[:, ::stride]  # [P, T', 5]
    if lane_points is None or int(np.shape(lane_points)[0]) == 0:
        return np.zeros((traj.shape[0],), dtype=np.float32)
    lanes = np.asarray(lane_points, dtype=np.float32)
    lxy, ldir = lanes[:, :2], lanes[:, 2:4] / np.maximum(np.linalg.norm(lanes[:, 2:4], axis=-1, keepdims=True), 1e-6)
    lyaw = np.arctan2(ldir[:, 1], ldir[:, 0])
    d = traj[:, :, None, :2] - lxy[None, None]  # [P, T', L, 2]
    herr = np.abs(np.angle(np.exp(1j * (traj[:, :, None, 2] - lyaw[None, None]))))
    dist2 = np.where(herr <= max_heading_rad, (d**2).sum(-1), np.inf)
    j = dist2.argmin(-1)  # [P, T']
    dj = np.take_along_axis(d, j[..., None, None], axis=2)[:, :, 0]
    lat = np.abs(dj[..., 0] * ldir[j][..., 1] - dj[..., 1] * ldir[j][..., 0])
    lat = np.where(np.isfinite(np.take_along_axis(dist2, j[..., None], axis=2)[..., 0]), lat, max_lat_m)
    he = np.take_along_axis(herr, j[..., None], axis=2)[..., 0]
    return (np.minimum(lat, max_lat_m) + heading_w * he).mean(-1).astype(np.float32)


def make_ego_boxes(traj_pt5, sim_state, timestep: int, world_idx: int = 0) -> np.ndarray:
    """[P, T, 5] (x, y, yaw, length, width) with the ego's current length / width."""
    ego = scorer_helpers.get_ego_idx(sim_state, world_idx)
    length_nt = np.asarray(sim_state.log_trajectory.length[world_idx], dtype=np.float32)
    width_nt = np.asarray(sim_state.log_trajectory.width[world_idx], dtype=np.float32)
    t = int(np.clip(int(timestep), 0, length_nt.shape[1] - 1))
    traj = np.asarray(traj_pt5, dtype=np.float32)
    size = np.broadcast_to(np.array([length_nt[ego, t], width_nt[ego, t]], np.float32), traj.shape[:2] + (2,))
    return np.concatenate([traj[..., :3], size], axis=-1)


def wm_mode_hits(wm, ego_boxes: np.ndarray, sim_state, timestep: int, world_idx: int = 0):
    """Ego-box overlap with every WM mode: hits [P, N, K, T] (bool) and mode probabilities [N, K]
    (normalized per agent), or None when the forecaster has no modes."""
    from scores.polygon import pairwise_intersections, polygon_corners

    if wm is None or not hasattr(wm, "forecast_modes"):
        return None
    p, t = ego_boxes.shape[0], ego_boxes.shape[1]
    fc = wm.forecast_modes(sim_state, timestep=int(timestep), traj_length=t, world_idx=world_idx)
    traj_k = jnp.asarray(fc["traj_k"], dtype=jnp.float32)  # [N, K, T', 5]
    probs = np.asarray(fc["mode_probs"], dtype=np.float32)  # [N, K]
    masks = np.asarray(fc["masks"]).astype(bool)  # [N, T']
    n, k, t = int(traj_k.shape[0]), int(traj_k.shape[1]), min(t, int(traj_k.shape[2]))
    ego = jnp.asarray(ego_boxes[:, :t], dtype=jnp.float32)
    hits = jax.vmap(pairwise_intersections, in_axes=(1, 1), out_axes=0)(
        polygon_corners(ego), polygon_corners(traj_k[:, :, :t].reshape((n * k, t, 5)))
    )  # [T, P, N*K]
    hits = np.asarray(hits).reshape(t, p, n, k) & masks[:, :t].T[:, None, :, None]
    probs = probs / np.clip(probs.sum(-1, keepdims=True), 1e-8, None)
    return np.moveaxis(hits, 0, -1), probs


def compute_violations(
    scorer,
    traj_pt5,
    sim_state,
    timestep: int,
    goal_xy,
    world_idx: int = 0,
    *,
    near_s: float = 3.0,
    dt: float = 0.1,
    goal_step: int = 90,
    max_accel: float = 8.0,
    max_accel_step: float | None = 2.0,
    reach_m: float = 1.0,
) -> dict[str, np.ndarray]:
    """Per-plan rule violations ``VIOLATION_KEYS`` [P] for world-frame plans [P, T, 5] (x, y, yaw, vx, vy),
    plus ``collision`` / ``offroad`` / ``progress`` / ``feasible`` in the same sense as the other rewards."""
    from scores.scorer_lane_graph import OFFROAD_MARGIN_M

    if scorer.lane_points is None:
        scorer_helpers.update_lane_points(scorer, sim_state, world_idx)
    traj = np.asarray(traj_pt5, dtype=np.float32)
    p, t = traj.shape[0], traj.shape[1]
    near = min(int(round(near_s / dt)) + 1, t)
    ego_boxes = make_ego_boxes(traj, sim_state, timestep, world_idx)

    res = wm_mode_hits(scorer.world_model_forecaster, ego_boxes, sim_state, timestep, world_idx)
    if res is None:  # deterministic forecast: production binary collision over the horizon
        obj, mask, col_obj, col_mask = scorer_helpers.forecast_other_agents(
            sim_state, timestep=int(timestep), traj_length=t, world_idx=world_idx,
            mode=scorer.other_agent_forecast, world_model_forecaster=scorer.world_model_forecaster,
        )
        tl = int(col_obj.shape[1])
        p_all = 1.0 - np.asarray(scorer.compute_collision_score(ego_boxes[:, :tl], col_obj, col_mask), np.float32)
        p_near = p_all
        t_hit = np.full((p,), float(t))
    else:
        hits, probs = res  # [P, N, K, T], [N, K]

        def p_overlap(h):  # h [P, N, K] -> P(any agent overlaps), modes independent across agents
            return 1.0 - np.prod(1.0 - np.clip((probs[None] * h).sum(-1), 0.0, 1.0), axis=-1)

        p_near, p_all = p_overlap(hits[..., :near].any(-1)), p_overlap(hits.any(-1))
        hit_any = hits.any(-1)  # [P, N, K]
        first = np.where(hit_any, hits.argmax(-1), t).astype(np.float32)
        w = probs[None] * hit_any
        per_agent = np.where(w.sum(-1) > 0, (w * first).sum(-1) / np.clip(w.sum(-1), 1e-8, None), t)
        t_hit = per_agent.min(-1)  # earliest agent's expected first contact

    dist = scorer._offroad_corner_distances(jnp.asarray(ego_boxes), sim_state, timestep, world_idx)
    if dist is None:
        off = 1.0 - np.asarray(scorer.compute_offroad_score(jnp.asarray(ego_boxes)), np.float32)
        worst_t = np.repeat(off[:, None], t, axis=1) - 0.5  # > 0 = off-road
    else:
        worst_t = np.asarray(dist, np.float32).max(-1)  # [P, T], > 0 = past the road edge
    is_off = worst_t > 0
    t_off = np.where(is_off.any(-1), is_off.argmax(-1), t).astype(np.float32)

    a0 = None if max_accel_step is None else ego_kinematics(sim_state, timestep, world_idx)[1]
    feasible = kinematic_feasibility(traj, max_accel=max_accel, max_lat_accel=max_accel, a0=a0,
                                     max_accel_step=max_accel_step or 0.0)
    prog = goal_progress(traj, np.asarray(goal_xy), timestep, goal_step=goal_step, dt=dt, reach_m=reach_m)
    margin_m = -worst_t.max(-1)
    b = {
        "v_infeasible": 1.0 - feasible,
        "v_coll_near": p_near,
        "v_coll_all": p_all,
        "v_coll_time": -t_hit,
        "v_off_near": np.clip(worst_t[:, :near].max(-1), 0.0, None),
        "v_off_all": np.clip(worst_t.max(-1), 0.0, None),
        "v_off_time": -t_off,
        "v_goal": 1.0 - prog,
        "v_clearance": -np.minimum(margin_m, OFFROAD_MARGIN_M),
        "v_lane": lane_keeping_cost(scorer.lane_points, traj),
        "v_jerk": plan_jerk(traj),
    }
    b = {k: np.asarray(v, np.float32) for k, v in b.items()}
    b.update(
        collision=1.0 - b["v_coll_all"],
        offroad=(b["v_off_all"] == 0).astype(np.float32),
        progress=prog,
        feasible=feasible,
    )
    return b


def rank_violations(b: dict[str, np.ndarray], collision_slack: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """(scores [P] in [0, 1), depth [P] = number of keys passed) for violations ``b`` (see module doc)."""
    keys = [np.asarray(b[k], np.float32) for k in VIOLATION_KEYS]
    n = len(keys[0])
    alive = np.ones(n, dtype=bool)
    depth = np.zeros(n, dtype=np.int32)
    for name, v in zip(VIOLATION_KEYS[:-1], keys[:-1]):
        slack = collision_slack if name in SLACK_KEYS else 0.0
        keep = alive & (v <= v[alive].min() + max(slack, 1e-6))
        depth[keep] += 1
        alive = keep
    within = np.zeros(n, dtype=np.float32)
    for d in np.unique(depth):
        g = np.flatnonzero(depth == d)
        r = np.argsort(np.argsort(-keys[d][g], kind="stable"), kind="stable")  # lower violation -> higher
        within[g] = r / max(len(g) - 1, 1)
    scores = (depth + 0.99 * within) / len(keys)
    return scores.astype(np.float32), depth.astype(np.float32)


def concat_violations(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Merge per-plan violation dicts along the plan axis (elites + new samples)."""
    return {k: np.concatenate([np.asarray(p[k]) for p in parts], axis=0) for k in parts[0]}


def take_violations(b: dict[str, np.ndarray], idx) -> dict[str, np.ndarray]:
    idx = np.asarray(idx)
    return {k: np.asarray(v)[idx] for k, v in b.items()}
