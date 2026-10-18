"""What is LEFT after the safety fine-tune, step by step -- and whether the
barrier could ever have fixed it.

The residual is usually read from ``rows.jsonl``: three booleans per episode
(``reached``/``offroad``/``overlap``) which say *that* a scene failed and nothing
about *how*. That is enough to see the shape of the problem (see
``_reason_summary`` in :mod:`rl.diffusion_ft.tools.dump_reach_from_select`) and
not enough to act on it. Three questions need the geometry back:

1. **Is the no-goal residual a stall or a detour?** ``reached`` is a 3 m ball
   around the human's last logged pose within the episode's 8 s. An ego that
   drives the right line at 0.6x speed and an ego that turns down the wrong
   street both read ``reached=False``. Only the first is a *pace* problem, and
   the safety term -- a restriction on geometry -- cannot cause either.

2. **Is the overlap residual downstream of the stall?** Other agents are
   log-replayed and do not yield, so an ego that falls behind its own log is
   driven into by traffic that was never going to wait. If that is what the
   overlap bucket is, then shaping the agent barrier is shaping a *symptom*
   (the same finding ``probe_logreplay_collision`` reached for DPPO: a frozen
   car collides 0.719 of the time). The signature is specific and checkable:
   the collider approaches from BEHIND or from the side the ego should already
   have cleared, while the ego is metres behind its own log.

3. **Can the barrier lead the car out of the road edge it is in?** The edge term
   is a sum of per-corner barriers on a signed distance to the NEAREST road-edge
   sample. Its descent direction is therefore a local, per-corner thing, and two
   failure geometries break it:

   * **cancellation** -- corners on opposite sides of a narrow off-road strip
     (a median, a gore, a splitter island) push the box in opposite directions
     and the resultant is ~0, so the loss is large and its gradient says
     "nowhere";
   * **no exit** -- the whole neighbourhood is off-road, so there is a gradient
     but no nearby pose it can reach that satisfies the constraint.

   Both are measured here against ground truth, not asserted: at each excursion
   we sweep the ego box laterally through the pose and recompute the SAME
   clearance ``waymax.is_offroad`` thresholds, giving a 1-D profile of
   ``min_corner c_edge`` versus lateral offset. From it: whether any offset
   inside +-``--sweep_lat`` m is on-road, how far away the nearest one is, and
   whether the barrier's own gradient points at it.

NO POLICY, NO CHECKPOINT, NO GPU. The select run already dumped the executed
world path per arm and repeat (``<arm>/r<rep>/traj`` in ``feat/*.npz``, written
under ``--dump_traj``); the scenes reload on CPU through the same byte-offset
index (``SceneSampler.load_explicit``), and every geometric quantity is
recomputed with :mod:`rl.diffusion_ft.safety_loss` -- the same functions the
training loss is built from, so a disagreement here is a disagreement with the
objective and not with an approximation of it.

TWO ALIGNMENT FACTS, both checked at runtime rather than trusted:

* **The dump tiles the episode exactly.** Each replan dumps ``world[:, :n_exec]``
  -- the anchor plus the first ``n_exec - 1`` executed poses -- and the next
  replan's anchor IS the pose the previous one ended on, so concatenating the
  replans gives a contiguous 0.1 s path. Only the final executed pose of the
  last replan is missing.
* **Index i of that path is absolute log step ``t0 + i``**, with ``t0 =
  init_steps - 1 = 10`` (waymax's ``reset`` leaves ``state.timestep`` one below
  ``init_steps``). ``--align_tol`` bounds the comparison of the path's first pose against the
  ego's logged pose at ``t0``; they are the same pose by construction, so a
  non-zero residual means the alignment assumption is wrong and every per-step
  number below is off by a step. It is a hard failure, not a warning.

Usage (CPU)::

    python -m rl.diffusion_ft.tools.failure_anatomy \\
        --select_dir <select run with --dump_traj> --split failure_heldout \\
        --arms base,ft,c_r10 --chunk_index 0 --num_chunks 6 \\
        --out_dir OUT
    # then, once the array is done (CPU, seconds)
    python -m rl.diffusion_ft.tools.failure_anatomy --merge \
        --out_dir OUT --split failure_heldout
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

# Waymax replays `init_steps` of log before the policy takes over, and `reset`
# leaves `state.timestep` at `init_steps - 1`. The first dumped pose is that
# anchor, so dumped index i is absolute log step T0 + i.
T0 = 10
GOAL_RADIUS_M = 3.0
WORLD_DT = 0.1


# --------------------------------------------------------------------------- #
# Reading what the select run wrote
# --------------------------------------------------------------------------- #
def path_index(select_dir: Path) -> dict[int, tuple[str, int, str]]:
    """``scene_id -> (npz path, column, split)`` over every shard's feat/*.npz."""
    idx: dict[int, tuple[str, int, str]] = {}
    for f in sorted(glob.glob(str(select_dir / "tasks/*/feat/*.npz"))):
        split = Path(f).name.split("_s")[0]
        for j, sid in enumerate(np.load(f)["scene_ids"]):
            idx.setdefault(int(sid), (f, j, split))
    return idx


def load_rows(select_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(select_dir / "tasks/*/rows.jsonl"))):
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def episode_path(npz, arm: str, rep: int, col: int) -> np.ndarray:
    """The executed world path for one (arm, repeat, scene), ``[T, 5]``.

    Replans are concatenated in order and the NaN padding of a short final
    replan is dropped. See the module docstring for why this is contiguous.
    """
    key = f"{arm}/r{rep}/traj"
    tr = np.asarray(npz[key][col], np.float64)          # [R, pad, 5]
    flat = tr.reshape(-1, tr.shape[-1])
    return flat[~np.isnan(flat[:, 0])]


def episode_alive(npz, arm: str, rep: int, col: int) -> np.ndarray:
    return np.asarray(npz[f"{arm}/r{rep}/alive"][col], bool)


# --------------------------------------------------------------------------- #
# Geometry (all of it from safety_loss, so it is the objective's own view)
# --------------------------------------------------------------------------- #
def scene_geometry(state, jnp, sl):
    """Log geometry + goal for a loaded batch, as numpy."""
    from simulation.evaluation_utils import infer_goals_from_sim_state

    geo = sl.log_geometry(state)
    goal_xy, goal_t, ego_idx = infer_goals_from_sim_state(state)
    log = state.log_trajectory
    ego_b = np.asarray(ego_idx)
    vel = np.stack([np.asarray(log.vel_x), np.asarray(log.vel_y)], -1)   # [B,N,T,2]
    ego_vel = vel[np.arange(vel.shape[0]), ego_b]                        # [B,T,2]
    return {
        "ego_pose": np.asarray(geo["ego_pose"]),          # [B,T,3] log
        "ego_z": np.asarray(geo["ego_z"]),                # [B,T]
        "ego_len": np.asarray(geo["ego_len"]),            # [B]
        "ego_wid": np.asarray(geo["ego_wid"]),            # [B]
        "ego_valid": np.asarray(geo["ego_valid"]),        # [B,T]
        "ego_vel": ego_vel,                               # [B,T,2] log
        "other_box": np.asarray(geo["other_box"]),        # [B,T,N,5]
        "other_valid": np.asarray(geo["other_valid"]),    # [B,T,N]
        "goal_xy": np.asarray(goal_xy),                   # [B,2]
        "goal_t": np.asarray(goal_t),                     # [B]
    }


def step_clearances(ego_xyyaw, ego_z, ego_len, ego_wid, other_box, other_valid,
                    state, jnp, sl, cfg):
    """``(c_edge [B,T,4], c_agent [B,T,N])`` for an arbitrary ego path.

    ``c_edge`` is the quantity ``is_offroad`` thresholds and ``c_agent`` the one
    ``overlap`` thresholds, both positive == safe, so ``min(...) < 0`` reproduces
    the metric exactly rather than approximating it.
    """
    box = sl.ego_box_5dof(jnp.asarray(ego_xyyaw, jnp.float32),
                          jnp.asarray(ego_len, jnp.float32)[:, None],
                          jnp.asarray(ego_wid, jnp.float32)[:, None])
    c_edge, _ = sl.road_edge_clearance(
        box, jnp.asarray(ego_z, jnp.float32), state.roadgraph_points,
        z_stretch=cfg.z_stretch, time_chunk=cfg.time_chunk,
        edge_geometry=cfg.edge_geometry)
    gaps, gvalid = sl.agent_exact_gaps(
        box, jnp.asarray(other_box, jnp.float32), jnp.asarray(other_valid, bool))
    g = jnp.where(gvalid[..., 0, 0], gaps[..., 0, 0], jnp.inf)          # [B,T,N]
    return np.asarray(c_edge), np.asarray(g)


def lateral_sweep(scene_state, pose_xyyaw, z, length, width, offsets, jnp, sl, cfg):
    """``min_corner c_edge`` for the pose shifted along its own left normal.

    The ground truth the escape question needs: the clearance is recomputed from
    scratch at every offset (fresh nearest-edge search and fresh sign), so the
    profile is what ``is_offroad`` would say about each shifted pose, not a
    linearisation of the barrier around the current one.
    """
    x, y, yaw = float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2])
    nx, ny = -np.sin(yaw), np.cos(yaw)                   # left normal
    xs = x + offsets * nx
    ys = y + offsets * ny
    pose = np.stack([xs, ys, np.full_like(xs, yaw)], -1)[None]           # [1,G,3]
    box = sl.ego_box_5dof(jnp.asarray(pose, jnp.float32),
                          jnp.float32(length), jnp.float32(width))
    c_edge, _ = sl.road_edge_clearance(
        box, jnp.full((1, offsets.size), float(z), jnp.float32),
        scene_state.roadgraph_points,
        z_stretch=cfg.z_stretch, time_chunk=cfg.time_chunk,
        edge_geometry=cfg.edge_geometry)
    return np.asarray(jnp.min(c_edge, axis=-1))[0]                       # [G]


_GRAD_FNS: dict = {}


def _edge_grad_fn(jax, jnp, sl, margin, tau, shape, *, with_seg: bool):
    """``d/dd`` of the frozen-anchor edge barrier, built ONCE per configuration.

    Defining the differentiated function inside the caller would make a new
    Python function object on every excursion, and ``jax.grad`` keys its trace
    cache on that object -- so a few hundred sweeps accumulate a few hundred
    never-evicted cached traces and the job is killed for memory with nothing
    large anywhere in it. Hoisting it means one trace for the whole run.
    """
    key = (margin, tau, shape, with_seg)
    if key not in _GRAD_FNS:
        if with_seg:
            def loss(d, corners, normal, nearest_xy, sign, p_xy, n_xy, p_ok, n_ok):
                c = -sl.signed_distance_from_anchors(
                    corners + d * normal, nearest_xy, sign, (p_xy, n_xy, p_ok, n_ok))
                return jnp.sum(sl.barrier(c, margin, tau, shape))
        else:
            def loss(d, corners, normal, nearest_xy, sign):
                c = -sl.signed_distance_from_anchors(
                    corners + d * normal, nearest_xy, sign)
                return jnp.sum(sl.barrier(c, margin, tau, shape))
        _GRAD_FNS[key] = jax.jit(jax.grad(loss))
    return _GRAD_FNS[key]


def barrier_lateral_gradient(scene_state, pose_xyyaw, z, length, width,
                             jnp, jax, sl, cfg):
    """d/d(lateral offset) of the edge barrier, with the anchors held fixed.

    This is the trainer's own gradient restricted to a rigid lateral translation:
    :func:`road_edge_anchors` is piecewise constant and deliberately not
    differentiated (see ``safety_loss`` "Gradient structure"), so freezing it and
    differentiating :func:`signed_distance_from_anchors` gives the exact a.e.
    derivative the safety term would apply here. Negative == the loss wants the
    box to move LEFT.

    Returned for BOTH edge geometries, because they are not expected to agree and
    the difference is the actionable part. ``"point"`` (the default, and waymax's
    own definition) measures to the nearest roadgraph SAMPLE, so ``c_edge`` does
    not go to zero at the boundary -- the sign flips across the line through the
    sample while the magnitude stays at roughly half the sample spacing, a cliff
    ``SafetyLossConfig.edge_geometry`` documents at ~0.45 m. Frozen at one side
    of that cliff, the barrier can descend AWAY from a boundary 10 cm off.
    ``"segment"`` measures to the incident polyline segment instead, which moves
    no sign and therefore no zero level set, but makes the magnitude continuous
    through the boundary -- so it is the candidate fix, and ``descent_agrees_seg``
    is whether it would actually have fixed the direction.
    """
    from waymax.utils import geometry as wgeom

    yaw = float(pose_xyyaw[2])
    normal = jnp.asarray([-np.sin(yaw), np.cos(yaw)], jnp.float32)
    box = sl.ego_box_5dof(jnp.asarray(pose_xyyaw, jnp.float32)[None, None],
                          jnp.float32(length), jnp.float32(width))        # [1,1,5]
    corners = wgeom.corners_from_bboxes(box)                              # [1,1,4,2]
    zc = jnp.full(corners.shape[:-1] + (1,), float(z), jnp.float32)
    corners_xyz = jnp.concatenate([corners, zc], -1)
    anchors = sl.road_edge_anchors(corners_xyz, scene_state.roadgraph_points,
                                   z_stretch=cfg.z_stretch, time_chunk=1,
                                   with_segments=True)
    nearest_xy, sign, segments = anchors[0], anchors[1], anchors[2:]

    grad_fn = _edge_grad_fn(jax, jnp, sl, cfg.edge_margin_m, cfg.edge_tau_m,
                            cfg.barrier_shape, with_seg=False)
    grad_seg_fn = _edge_grad_fn(jax, jnp, sl, cfg.edge_margin_m, cfg.edge_tau_m,
                                cfg.barrier_shape, with_seg=True)
    g = float(grad_fn(jnp.float32(0.0), corners, normal, nearest_xy, sign))
    g_seg = float(grad_seg_fn(jnp.float32(0.0), corners, normal, nearest_xy, sign,
                              *segments))

    # Per-corner escape directions, to separate "the loss pushes nowhere"
    # (corners cancel) from "it pushes somewhere useless". The clearance is
    # `-dist * sign`, so its gradient wrt the corner is `-sign * unit(corner -
    # nearest)`: an ON-road corner (sign -1) gains clearance by moving AWAY from
    # the edge, an OFF-road one (sign +1) by moving back ACROSS it. Getting that
    # sign wrong would report a median-straddling box as perfectly coherent.
    c0 = np.asarray(-sl.signed_distance_from_anchors(corners, nearest_xy, sign))[0, 0]
    sgn = np.asarray(sign)[0, 0]                                          # [4]
    vec = np.asarray(corners)[0, 0] - np.asarray(nearest_xy)[0, 0]        # [4,2]
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    escape = np.where(n > 1e-6, vec / np.maximum(n, 1e-6), 0.0) * -sgn[:, None]
    # Weight each corner by how hard the barrier is pulling on it: the softplus
    # derivative, sigmoid((margin - c)/tau), clipped so a far corner underflows
    # to 0 instead of raising.
    w = 1.0 / (1.0 + np.exp(np.clip((c0 - cfg.edge_margin_m) / cfg.edge_tau_m,
                                    -60.0, 60.0)))
    denom = float(w.sum())
    # NaN, not 0, when every corner is far clear: 0 would read as "the corners
    # cancel" when the truth is that nothing is pulling at all. Dropped by the
    # percentile helper rather than counted as maximal disagreement.
    coherence = (float(np.linalg.norm((w[:, None] * escape).sum(0)) / denom)
                 if denom > 1e-9 else float("nan"))
    # The cliff, measured where it matters: with "point" geometry the reported
    # depth is the distance to a SAMPLE, not to the boundary, so it can say 0.46 m
    # of violation at a pose 0.1 m from the road.
    return g, g_seg, coherence, c0


# --------------------------------------------------------------------------- #
# Per-episode diagnostics
# --------------------------------------------------------------------------- #
def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as ``(start, length)``."""
    out, i, n = [], 0, mask.size
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def episode_record(path, c_edge, c_agent, g, b, arm, rep, sid, split):
    """Every scalar the three questions need, for one episode."""
    T = path.shape[0]
    steps = np.arange(T) + T0                               # absolute log steps
    own = np.arange(T) >= 1                                 # policy's own window

    ego_xy = path[:, :2]
    # TWO speeds, and they are not the same number. `speed_cmd` is the velocity
    # field the policy WROTE into the trajectory; `speed` is what the box
    # actually travelled between consecutive executed poses. Under StateDynamics
    # the pose is set exactly and the velocity is carried along unchecked, and
    # for this diffusion policy the two disagree by 13-23% (measured 2026-09-19,
    # base 1.13 / ft 1.15 / c_r10 1.23) while for the log-replayed agents they
    # agree to 1.000. Any pace statistic must use the realised one -- read off
    # the commanded field the ego looks FASTER than the human it is falling
    # behind, which inverts the conclusion.
    speed_cmd = np.hypot(path[:, 3], path[:, 4])
    speed = np.concatenate([[0.0], np.linalg.norm(np.diff(ego_xy, axis=0), axis=-1)
                            / WORLD_DT])
    c_edge_min = c_edge.min(-1)
    c_agent_min = c_agent.min(-1)
    collider = np.argmin(np.where(np.isfinite(c_agent), c_agent, np.inf), -1)

    off = (c_edge_min < 0) & own
    ovl = (c_agent_min < 0) & own

    # --- goal ---------------------------------------------------------- #
    goal_xy = g["goal_xy"][b]
    goal_d = np.linalg.norm(ego_xy - goal_xy[None], axis=-1)
    reach = goal_d <= GOAL_RADIUS_M
    t_reach = int(np.argmax(reach)) if reach.any() else -1

    # --- the log, at the same absolute steps ---------------------------- #
    log_xy = g["ego_pose"][b, steps, :2]
    log_valid = g["ego_valid"][b, steps].astype(bool)
    # Same treatment for the human, from ITS positions, so the ratio compares
    # like with like. (The log's own velocity field does agree with its
    # displacement; `log_speed_cmd_ratio` keeps that as a live check rather than
    # a remembered one.)
    log_speed = np.concatenate([[0.0], np.linalg.norm(np.diff(log_xy, axis=0),
                                                      axis=-1) / WORLD_DT])
    log_speed_cmd = np.linalg.norm(g["ego_vel"][b, steps], axis=-1)

    def arclen(p):
        return np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=-1))])

    s_ego = arclen(ego_xy)
    s_log = arclen(log_xy)
    # Along-path lag: how much of its own route the human had covered by now that
    # the ego has not. Positive == behind.
    lag_m = s_log - s_ego

    # Cross-track: match the ego to the CLOSEST point of the whole log path, so a
    # pure delay reads ~0 here while a detour does not. `lag_steps` is the same
    # match expressed in time.
    d2 = np.linalg.norm(ego_xy[:, None, :] - log_xy[None, :, :], axis=-1)   # [T,T]
    nearest = np.argmin(d2, axis=1)
    cross = d2[np.arange(T), nearest]
    lag_steps = np.arange(T) - nearest

    # When the EPISODE was over, by the env's own rule
    # (``RewardConfig.terminate_on_collision=True``, ``terminate_on_offroad=False``,
    # plus reaching the goal). The runner keeps physically driving a terminated
    # scene until every scene in the batch is done, and it ORs the raw metric
    # flags over that whole window -- which is how `solved` is defined, so it is
    # what is reproduced above. But an off-road excursion that begins AFTER the
    # car has already collided is not an independent failure, and counting it as
    # one would inflate the off-road bucket at the expense of the overlap one.
    term = ovl | reach
    t_term = int(np.argmax(term)) if term.any() else -1

    # Speed ratio, measured only where the HUMAN is still driving. A whole-window
    # mean is not a pace comparison: the log ends at the goal and sits there, so
    # its mean speed is dragged down by the stationary tail and an ego that never
    # got near the goal reads *faster* than the demonstration it is 8 m behind.
    # This is the ratio that can be read as "keeping up".
    moving = own & log_valid & (log_speed > 0.5)
    speed_ratio_moving = (float(speed[moving].mean() / log_speed[moving].mean())
                          if moving.any() else float("nan"))

    fin = T - 1
    rec = {
        "scene_id": int(sid), "split": split, "arm": arm, "repeat": int(rep),
        "T": int(T),
        "reached": bool(reach.any()), "t_reach": t_reach,
        "offroad": bool(off.any()), "overlap": bool(ovl.any()),
        "solved": bool(reach.any() and not off.any() and not ovl.any()),
        # --- pace / detour ------------------------------------------- #
        "goal_d_final": float(goal_d[fin]), "goal_d_min": float(goal_d.min()),
        "goal_d_start": float(goal_d[0]),
        "progress_frac": float(1.0 - goal_d.min() / max(goal_d[0], 1e-6)),
        "s_ego": float(s_ego[fin]), "s_log": float(s_log[fin]),
        "path_ratio": float(s_ego[fin] / max(s_log[fin], 1e-6)),
        "speed_mean": float(speed[own].mean()),
        "log_speed_mean": float(log_speed[own][log_valid[own]].mean()
                                if log_valid[own].any() else np.nan),
        "speed_ratio": float(speed[own].mean()
                             / max(log_speed[own][log_valid[own]].mean(), 1e-6)
                             if log_valid[own].any() else np.nan),
        "speed_ratio_moving": speed_ratio_moving,
        # Commanded / realised, for the ego and (as the control) for the log.
        # The log's must read ~1.000; the ego's is the defect.
        "speed_cmd_ratio": float(speed_cmd[own].mean()
                                 / max(speed[own].mean(), 1e-6)),
        "log_speed_cmd_ratio": float(log_speed_cmd[moving].mean()
                                     / max(log_speed[moving].mean(), 1e-6)
                                     if moving.any() else np.nan),
        "stall_frac": float((speed[own] < 0.5).mean()),
        "lag_m_final": float(lag_m[fin]), "lag_m_max": float(lag_m.max()),
        "lag_steps_final": int(lag_steps[fin]), "lag_steps_max": int(lag_steps.max()),
        "cross_final": float(cross[fin]), "cross_max": float(cross.max()),
    }

    # --- off-road anatomy ------------------------------------------------ #
    ex = runs_of(off)
    t_off = int(np.argmax(off)) if off.any() else -1
    rec.update({
        "t_term": t_term,
        "off_before_term": bool(off.any() and (t_term < 0 or t_off <= t_term)),
        "off_steps": int(off.sum()), "off_runs": len(ex),
        "off_first": t_off,
        "off_longest": int(max((L for _, L in ex), default=0)),
        "off_depth_max": float(-c_edge_min[off].min()) if off.any() else 0.0,
        "off_at_end": bool(off[fin]),
        "off_unrecovered": bool(ex and ex[-1][0] + ex[-1][1] == T),
        "edge_min": float(c_edge_min[own].min()),
    })

    # --- overlap anatomy ------------------------------------------------- #
    if ovl.any():
        t = int(np.argmax(ovl))
        k = int(collider[t])
        ob = g["other_box"][b, steps[t], k]                # x,y,len,wid,yaw
        rel = np.array([ob[0], ob[1]]) - ego_xy[t]
        yaw = path[t, 2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        rel_ego = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        # The collider's own motion, from its log (it is log-replayed).
        prev = max(steps[t] - 1, 0)
        ovel = (g["other_box"][b, steps[t], k, :2] - g["other_box"][b, prev, k, :2]) / WORLD_DT
        rec.update({
            "ovl_t": t, "ovl_step": int(steps[t]), "ovl_agent": k,
            "ovl_depth": float(-c_agent_min[t]),
            "ovl_rel_lon": float(rel_ego[0]), "ovl_rel_lat": float(rel_ego[1]),
            "ovl_bearing_deg": float(np.degrees(np.arctan2(rel_ego[1], rel_ego[0]))),
            "ovl_ego_speed": float(speed[t]),
            "ovl_log_speed": float(log_speed[t]) if log_valid[t] else float("nan"),
            "ovl_other_speed": float(np.linalg.norm(ovel)),
            "ovl_lag_m": float(lag_m[t]), "ovl_lag_steps": int(lag_steps[t]),
            "ovl_cross": float(cross[t]),
            "ovl_off": bool(off[t]),
        })
    else:
        rec["ovl_t"] = -1

    series = {
        "goal_d": goal_d.astype(np.float32), "speed": speed.astype(np.float32),
        "log_speed": log_speed.astype(np.float32), "lag_m": lag_m.astype(np.float32),
        "cross": cross.astype(np.float32),
        "c_edge_min": c_edge_min.astype(np.float32),
        "c_agent_min": np.where(np.isfinite(c_agent_min), c_agent_min, 1e3
                                ).astype(np.float32),
        "xy": ego_xy.astype(np.float32), "log_xy": log_xy.astype(np.float32),
    }
    return rec, series, off


# --------------------------------------------------------------------------- #
def run_chunk(args) -> None:
    import jax
    import jax.numpy as jnp

    from rl.diffusion_ft import safety_loss as sl
    from rl.diffusion_ft.scene_sampler import SceneSampler

    select_dir = Path(args.select_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = sl.SafetyLossConfig(edge_margin_m=args.edge_margin,
                              agent_margin_m=args.agent_margin,
                              edge_tau_m=args.edge_tau,
                              agent_tau_m=args.agent_tau,
                              edge_geometry=args.edge_geometry,
                              barrier_shape=args.barrier_shape)
    arms = [a for a in args.arms.split(",") if a]

    idx = path_index(select_dir)
    rows = load_rows(select_dir)
    row_key = {(r["scene_id"], r["arm"], r["repeat"]): r
               for r in rows if r["split"] == args.split}

    splits = json.loads(Path(args.splits_json).read_text())
    ids = [int(i) for i in splits["splits"][args.split]["scene_ids"]]
    ids = [i for i in ids if i in idx and idx[i][2] == args.split]
    if args.cap > 0:
        ids = ids[: args.cap]
    mine = ids[args.chunk_index :: args.num_chunks]
    print(f"[anatomy] split={args.split} total={len(ids)} "
          f"chunk {args.chunk_index}/{args.num_chunks} -> {len(mine)} scenes",
          flush=True)
    if not mine:
        return

    sampler = SceneSampler(args.shard_glob, batch_size=1, eval_size=0,
                           include_sdc_paths=False, num_paths=args.num_paths,
                           num_points_per_path=args.num_points_per_path,
                           prefetch=False, index_dir=args.index_dir)

    records, series_store, align_err = [], {}, 0.0
    sweeps = []
    offsets = np.arange(-args.sweep_lat, args.sweep_lat + 1e-9, args.sweep_step)

    for lo in range(0, len(mine), args.load_batch):
        batch_ids = mine[lo : lo + args.load_batch]
        state = sampler.load_explicit(batch_ids)
        g = scene_geometry(state, jnp, sl)
        print(f"[anatomy] loaded {len(batch_ids)} scenes "
              f"({lo + len(batch_ids)}/{len(mine)})", flush=True)

        # One open per npz file per load batch, not one per (scene, arm, repeat):
        # the same file holds every arm and repeat of 32 scenes.
        opened = {}
        for sid in batch_ids:
            f = idx[sid][0]
            if f not in opened:
                opened[f] = dict(np.load(f))

        for arm in arms:
            for rep in range(args.repeats):
                paths = []
                for sid in batch_ids:
                    f, col, _ = idx[sid]
                    z = opened[f]
                    paths.append(episode_path(z, arm, rep, col)
                                 if f"{arm}/r{rep}/traj" in z else None)
                if all(p is None for p in paths):
                    continue
                T = max(p.shape[0] for p in paths if p is not None)
                B = len(batch_ids)
                st = np.clip(np.arange(T) + T0, 0, g["ego_pose"].shape[1] - 1)
                pose = np.zeros((B, T, 3))
                zc = g["ego_z"][:, st]
                obox = g["other_box"][:, st]
                ovalid = g["other_valid"][:, st]
                for bi, p in enumerate(paths):
                    # A missing arm parks the ego on its own anchor pose: the row
                    # is dropped below, this only keeps the batch rectangular.
                    src = p[:, :3] if p is not None else g["ego_pose"][bi, st]
                    n = src.shape[0]
                    pose[bi, :n] = src
                    pose[bi, n:] = src[-1]

                c_edge, c_agent = step_clearances(
                    pose, zc, g["ego_len"], g["ego_wid"], obox, ovalid,
                    state, jnp, sl, cfg)

                for bi, sid in enumerate(batch_ids):
                    if paths[bi] is None:
                        continue
                    n = paths[bi].shape[0]
                    err = float(np.linalg.norm(
                        paths[bi][0, :2] - g["ego_pose"][bi, T0, :2]))
                    if err > args.align_tol:
                        # Fail on the first scene rather than after the array has
                        # spent an hour producing numbers that are off by a step.
                        raise SystemExit(
                            f"ALIGNMENT FAILED on scene {sid} ({arm}/r{rep}): the "
                            f"dumped path starts {err:.3f} m from the ego's logged "
                            f"pose at t={T0}, tol {args.align_tol}. The dump/log "
                            f"index mapping this tool assumes is wrong, so every "
                            f"per-step number below would be off by a step.")
                    align_err = max(align_err, err)
                    rec, ser, off = episode_record(
                        paths[bi], c_edge[bi, :n], c_agent[bi, :n], g, bi,
                        arm, rep, sid, args.split)
                    ref = row_key.get((sid, arm, rep))
                    if ref is not None:
                        rec["row_solved"] = bool(ref["solved"])
                        rec["row_reached"] = bool(ref["reached"])
                        rec["row_offroad"] = bool(ref["offroad"])
                        rec["row_overlap"] = bool(ref["overlap"])
                    records.append(rec)
                    if not rec["solved"]:
                        series_store[f"{arm}|{rep}|{sid}"] = ser

                    # Escape test: one sweep per excursion, on the chosen arm only.
                    if args.sweep_arm and arm == args.sweep_arm and rep == 0:
                        for k, (s0, L) in enumerate(runs_of(off)[: args.sweep_max]):
                            prof = lateral_sweep(
                                _one(state, bi), paths[bi][s0, :3],
                                g["ego_z"][bi, T0 + s0],
                                g["ego_len"][bi], g["ego_wid"][bi],
                                offsets, jnp, sl, cfg)
                            grad, grad_seg, coh, c0 = barrier_lateral_gradient(
                                _one(state, bi), paths[bi][s0, :3],
                                g["ego_z"][bi, T0 + s0],
                                g["ego_len"][bi], g["ego_wid"][bi],
                                jnp, jax, sl, cfg)
                            sweeps.append(_sweep_record(
                                sid, arm, s0, L, prof, offsets,
                                grad, grad_seg, coh, c0))
    print(f"[anatomy] worst alignment residual {align_err:.4f} m "
          f"(tol {args.align_tol})",
          flush=True)

    tag = f"{args.split}_{args.chunk_index:03d}"
    with open(out_dir / f"episodes_{tag}.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    if sweeps:
        with open(out_dir / f"sweeps_{tag}.jsonl", "w") as fh:
            for r in sweeps:
                fh.write(json.dumps(r) + "\n")
    flat = {}
    for k, ser in series_store.items():
        for name, arr in ser.items():
            flat[f"{k}|{name}"] = arr
    np.savez_compressed(out_dir / f"series_{tag}.npz", **flat)
    print(f"[anatomy] wrote {len(records)} episodes, {len(sweeps)} sweeps -> {out_dir}",
          flush=True)


def _one(state, b: int):
    """World ``b`` of a batched state, as a batch of one (roadgraph only is used)."""
    import jax
    return jax.tree_util.tree_map(
        lambda a: a[b : b + 1] if getattr(a, "ndim", 0) >= 1 else a, state)


def _sweep_record(sid, arm, s0, L, prof, offsets, grad, grad_seg, coh, c0) -> dict:
    """Read the escape verdict off a lateral clearance profile."""
    here = int(np.argmin(np.abs(offsets)))
    # "On road at this offset", excluding the pose itself: the sweep is taken at
    # an excursion ONSET, so offset 0 is off-road by construction and a
    # floating-point |offset| ~ 1e-15 masquerading as an exit would report a
    # 0 m escape for every trapped pose.
    on = (prof > 0.0) & (np.abs(offsets) > 0.5 * float(np.min(np.abs(np.diff(offsets)))))
    d_exit = float(np.min(np.abs(offsets[on]))) if on.any() else float("inf")
    # Which way the nearest exit lies; and which way the barrier pushes.
    side = float(offsets[on][np.argmin(np.abs(offsets[on]))]) if on.any() else float("nan")
    # loss gradient wrt +lateral; descent is -grad, so the loss pushes LEFT when
    # grad > 0. Agreement = the descent direction has the sign of the exit.
    def agreement(gr):
        return bool(on.any() and np.sign(-gr) == np.sign(side) and abs(gr) > 1e-9)

    return {
        "scene_id": int(sid), "arm": arm, "off_start": int(s0), "off_len": int(L),
        "c_here": float(prof[here]), "c_min_corner_now": float(c0.min()),
        "any_exit": bool(on.any()), "d_exit_m": d_exit, "exit_side_m": side,
        # How badly "point" geometry overstates the violation: the barrier is
        # charging |c_here| metres of depth at a pose d_exit metres from the road.
        "depth_overstated_m": (float(abs(prof[here]) - d_exit)
                               if np.isfinite(d_exit) else float("nan")),
        "grad_lat": float(grad), "descent_agrees": agreement(grad),
        "grad_lat_seg": float(grad_seg), "descent_agrees_seg": agreement(grad_seg),
        "coherence": float(coh),
        "prof_max": float(prof.max()), "prof_argmax_m": float(offsets[np.argmax(prof)]),
        "trapped_both_sides": bool(
            (not on[offsets > 0].any()) and (not on[offsets < 0].any())),
        "prof": [round(float(v), 4) for v in prof],
        "offsets": [round(float(v), 3) for v in offsets],
    }


# --------------------------------------------------------------------------- #
def merge(out_dir: Path, split: str) -> None:
    import collections

    eps = []
    for f in sorted(glob.glob(str(out_dir / f"episodes_{split}_*.jsonl"))):
        with open(f) as fh:
            eps.extend(json.loads(l) for l in fh if l.strip())
    sw = []
    for f in sorted(glob.glob(str(out_dir / f"sweeps_{split}_*.jsonl"))):
        with open(f) as fh:
            sw.extend(json.loads(l) for l in fh if l.strip())
    if not eps:
        raise SystemExit(f"no episodes_{split}_*.jsonl under {out_dir}")

    arms = sorted({e["arm"] for e in eps})
    rep = {"split": split, "n_episodes": len(eps), "arms": {}}

    def pct(v, q):
        v = [x for x in v if np.isfinite(x)]
        return float(np.percentile(v, q)) if v else float("nan")

    print("=" * 96)
    print(f"RECOMPUTED-vs-LOGGED AGREEMENT ({split})")
    for arm in arms:
        E = [e for e in eps if e["arm"] == arm]
        have = [e for e in E if "row_solved" in e]
        if not have:
            continue
        agree = {k: float(np.mean([e[k] == e[f"row_{k}"] for e in have]))
                 for k in ("solved", "reached", "offroad", "overlap")}
        print(f"  {arm:10s} n={len(have):5d} " +
              " ".join(f"{k}={v:.4f}" for k, v in agree.items()))
        rep["arms"].setdefault(arm, {})["agreement"] = agree

    for arm in arms:
        E = [e for e in eps if e["arm"] == arm]
        U = [e for e in E if not e["solved"]]
        nog = [e for e in U if not e["reached"]]
        ovl = [e for e in U if e["overlap"]]
        off = [e for e in U if e["offroad"]]
        a = rep["arms"].setdefault(arm, {})
        a["n"] = len(E)
        a["solved"] = float(np.mean([e["solved"] for e in E]))

        print("=" * 96)
        print(f"ARM {arm}   n={len(E)}  solved={a['solved']:.3f}  unsolved={len(U)}")

        # --- Q1 no-goal: pace or detour? -------------------------------- #
        if nog:
            a["no_goal"] = {
                "n": len(nog), "share_of_residual": len(nog) / len(U),
                "goal_d_final_p50": pct([e["goal_d_final"] for e in nog], 50),
                "goal_d_min_p50": pct([e["goal_d_min"] for e in nog], 50),
                "near_miss_frac": float(np.mean(
                    [e["goal_d_min"] < 2 * GOAL_RADIUS_M for e in nog])),
                "path_ratio_p50": pct([e["path_ratio"] for e in nog], 50),
                "speed_ratio_p50": pct([e["speed_ratio"] for e in nog], 50),
                "speed_ratio_moving_p50": pct(
                    [e.get("speed_ratio_moving", float("nan")) for e in nog], 50),
                "speed_cmd_ratio_p50": pct(
                    [e.get("speed_cmd_ratio", float("nan")) for e in nog], 50),
                "log_speed_cmd_ratio_p50": pct(
                    [e.get("log_speed_cmd_ratio", float("nan")) for e in nog], 50),
                "stall_frac_p50": pct([e["stall_frac"] for e in nog], 50),
                "lag_m_final_p50": pct([e["lag_m_final"] for e in nog], 50),
                "cross_final_p50": pct([e["cross_final"] for e in nog], 50),
                "cross_max_p50": pct([e["cross_max"] for e in nog], 50),
                "detour_frac": float(np.mean([e["cross_max"] > 5.0 for e in nog])),
                "terminated_early_frac": float(np.mean(
                    [e["overlap"] or e["offroad"] for e in nog])),
            }
            d = a["no_goal"]
            print(f"  NO-GOAL  {d['n']:4d} ({d['share_of_residual']:.1%} of residual)"
                  f"  final goal dist p50 {d['goal_d_final_p50']:6.1f} m"
                  f"  (min {d['goal_d_min_p50']:5.1f} m)")
            print(f"           speed ratio while the human drives p50 "
                  f"{d['speed_ratio_moving_p50']:.2f}"
                  f"   path-length ratio {d['path_ratio_p50']:.2f}"
                  f"   stalled steps {d['stall_frac_p50']:.2f}")
            print(f"           commanded/realised speed: ego "
                  f"{d['speed_cmd_ratio_p50']:.3f}   log "
                  f"{d['log_speed_cmd_ratio_p50']:.3f} (must be ~1.000)")
            print(f"           along-path lag p50 {d['lag_m_final_p50']:6.1f} m"
                  f"   cross-track p50 {d['cross_final_p50']:5.2f} m"
                  f"   detour(>5 m) {d['detour_frac']:.1%}"
                  f"   also violated {d['terminated_early_frac']:.1%}")

        # --- Q2 overlap: downstream of the stall? ----------------------- #
        O = [e for e in ovl if e.get("ovl_t", -1) >= 0]
        if O:
            beh = [e for e in O if abs(e["ovl_bearing_deg"]) > 90]
            a["overlap"] = {
                "n": len(O), "share_of_residual": len(O) / len(U),
                "from_behind_frac": len(beh) / len(O),
                "bearing_abs_p50": pct([abs(e["ovl_bearing_deg"]) for e in O], 50),
                "ego_speed_p50": pct([e["ovl_ego_speed"] for e in O], 50),
                "log_speed_p50": pct([e["ovl_log_speed"] for e in O], 50),
                "speed_deficit_p50": pct(
                    [e["ovl_log_speed"] - e["ovl_ego_speed"] for e in O], 50),
                "lag_m_p50": pct([e["ovl_lag_m"] for e in O], 50),
                "lag_steps_p50": pct([e["ovl_lag_steps"] for e in O], 50),
                "cross_p50": pct([e["ovl_cross"] for e in O], 50),
                "lagging_frac": float(np.mean([e["ovl_lag_m"] > 2.0 for e in O])),
                "on_line_frac": float(np.mean([e["ovl_cross"] < 1.0 for e in O])),
                "also_offroad_frac": float(np.mean([e["ovl_off"] for e in O])),
                "other_moving_frac": float(np.mean(
                    [e["ovl_other_speed"] > 1.0 for e in O])),
            }
            d = a["overlap"]
            print(f"  OVERLAP  {d['n']:4d} ({d['share_of_residual']:.1%} of residual)"
                  f"  collider from behind {d['from_behind_frac']:.1%}"
                  f"  |bearing| p50 {d['bearing_abs_p50']:.0f} deg")
            print(f"           ego speed p50 {d['ego_speed_p50']:.2f} vs log "
                  f"{d['log_speed_p50']:.2f} m/s  (deficit "
                  f"{d['speed_deficit_p50']:+.2f})   collider moving "
                  f"{d['other_moving_frac']:.1%}")
            print(f"           lag at contact p50 {d['lag_m_p50']:6.2f} m "
                  f"({d['lag_steps_p50']:.0f} steps)  lagging>2 m "
                  f"{d['lagging_frac']:.1%}   on the log line (<1 m) "
                  f"{d['on_line_frac']:.1%}")

        # --- Q3 offroad: does it ever come back? ------------------------- #
        if off:
            a["offroad"] = {
                "n": len(off), "share_of_residual": len(off) / len(U),
                "steps_p50": pct([e["off_steps"] for e in off], 50),
                "longest_p50": pct([e["off_longest"] for e in off], 50),
                "runs_p50": pct([e["off_runs"] for e in off], 50),
                "depth_p50": pct([e["off_depth_max"] for e in off], 50),
                "depth_p90": pct([e["off_depth_max"] for e in off], 90),
                "unrecovered_frac": float(np.mean([e["off_unrecovered"] for e in off])),
                "before_term_frac": float(np.mean(
                    [e.get("off_before_term", True) for e in off])),
            }
            d = a["offroad"]
            print(f"  OFFROAD  {d['n']:4d} ({d['share_of_residual']:.1%} of residual)"
                  f"  steps p50 {d['steps_p50']:.0f}  longest run p50 "
                  f"{d['longest_p50']:.0f}  excursions p50 {d['runs_p50']:.0f}")
            print(f"           depth p50 {d['depth_p50']:.2f} m  p90 "
                  f"{d['depth_p90']:.2f} m   never recovered "
                  f"{d['unrecovered_frac']:.1%}   began before the episode ended "
                  f"{d['before_term_frac']:.1%}")

    # --- the escape test ------------------------------------------------- #
    if sw:
        n = len(sw)
        no_exit = [s for s in sw if not s["any_exit"]]
        exits = [s for s in sw if s["any_exit"]]
        rep["escape"] = {
            "n": n,
            "no_exit_frac": len(no_exit) / n,
            "trapped_both_sides_frac": float(np.mean(
                [s["trapped_both_sides"] for s in sw])),
            "d_exit_p50": pct([s["d_exit_m"] for s in exits], 50),
            "d_exit_p90": pct([s["d_exit_m"] for s in exits], 90),
            "descent_agrees_frac": (float(np.mean([s["descent_agrees"] for s in exits]))
                                    if exits else float("nan")),
            "descent_agrees_seg_frac": (
                float(np.mean([s.get("descent_agrees_seg", False) for s in exits]))
                if exits else float("nan")),
            "depth_overstated_p50": pct(
                [s.get("depth_overstated_m", float("nan")) for s in exits], 50),
            "coherence_p50": pct([s["coherence"] for s in sw], 50),
            "coherence_lt_0.5_frac": float(np.mean(
                [s["coherence"] < 0.5 for s in sw if np.isfinite(s["coherence"])])),
            "grad_abs_p50": pct([abs(s["grad_lat"]) for s in sw], 50),
        }
        d = rep["escape"]
        print("=" * 96)
        print(f"ESCAPE TEST at {n} excursion onsets (lateral sweep, ground truth)")
        print(f"  no on-road offset anywhere in the sweep : {d['no_exit_frac']:.1%}"
              f"   (both sides blocked {d['trapped_both_sides_frac']:.1%})")
        print(f"  when an exit exists, |offset| p50 {d['d_exit_p50']:.2f} m "
              f"p90 {d['d_exit_p90']:.2f} m; the barrier calls it "
              f"{d['depth_overstated_p50']:+.2f} m deeper than it is")
        print(f"  barrier descent points at the nearer exit: "
              f"point geometry {d['descent_agrees_frac']:.1%}"
              f"   segment geometry {d['descent_agrees_seg_frac']:.1%}")
        print(f"  corner coherence p50 {d['coherence_p50']:.2f}  "
              f"(<0.5, i.e. corners disagree: {d['coherence_lt_0.5_frac']:.1%})")

    (out_dir / f"anatomy_{split}.json").write_text(json.dumps(rep, indent=2))
    print("=" * 96)
    print(f"wrote {out_dir / f'anatomy_{split}.json'}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select_dir", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="failure_heldout")
    p.add_argument("--arms", default="base,ft,c_r10")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--merge", action="store_true",
                   help="aggregate the chunk outputs already in --out_dir")
    p.add_argument("--chunk_index", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--cap", type=int, default=-1)
    p.add_argument("--load_batch", type=int, default=8)
    p.add_argument("--splits_json", default="")
    p.add_argument("--shard_glob",
                   default="/zfsauton/scratch/eshau/womd/tf_example/training/"
                           "training_tfexample.tfrecord-*")
    p.add_argument("--index_dir",
                   default="/zfsauton/scratch/yixiz/waymax_rs/womd_offsets")
    p.add_argument("--num_paths", type=int, default=45)
    p.add_argument("--num_points_per_path", type=int, default=800)
    p.add_argument("--edge_margin", type=float, default=0.3)
    p.add_argument("--agent_margin", type=float, default=0.75)
    p.add_argument("--edge_tau", type=float, default=0.15)
    p.add_argument("--agent_tau", type=float, default=0.25)
    p.add_argument("--edge_geometry", default="point")
    p.add_argument("--barrier_shape", default="softplus")
    p.add_argument("--align_tol", type=float, default=0.05)
    p.add_argument("--sweep_arm", default="ft",
                   help="arm whose excursions get the ground-truth lateral sweep "
                        "('' to skip)")
    p.add_argument("--sweep_max", type=int, default=3,
                   help="excursions swept per episode")
    p.add_argument("--sweep_lat", type=float, default=4.0)
    p.add_argument("--sweep_step", type=float, default=0.1)
    args = p.parse_args()

    if args.merge:
        merge(Path(args.out_dir), args.split)
        return
    if not args.select_dir or not args.splits_json:
        raise SystemExit("--select_dir and --splits_json are required without --merge")
    run_chunk(args)


if __name__ == "__main__":
    main()
