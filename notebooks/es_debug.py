"""Debug tooling for the diffusion-ES closed-loop pipeline.

Used by ``notebooks/diffusion_es_debug.ipynb``. Mirrors ``simulation/runner.py``
(periodic replan + Waymax rollout) and ``planner/diffusion_es_planner.py``
(sample -> score -> select elites -> resample) for a single scenario, but
records every ES iteration so it can be inspected, and makes the reward and
the elite-selection ("acquisition") step pluggable.

Shapes follow the repo convention: P = pool size, T = horizon (world steps),
N = objects, K = WM modes.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocess import preprocess_simulator_state  # noqa: E402
from data.scenario_loader import load_scenario_state_batch_fast  # noqa: E402
from lane_graph.lane_graph_utils import lane_graph_zip_path_for_tfrecord  # noqa: E402
from planner.abstract_planner import PlannerResult  # noqa: E402
from planner.diffusion_es_planner import DiffusionESPlanner  # noqa: E402
from scores import helpers as scorer_helpers  # noqa: E402
from scores.scorer_lane_graph import LaneGraphShardStore  # noqa: E402
from simulation.evaluation_utils import (  # noqa: E402
    _build_rollout_env_and_actor,
    _run_batched_rollout,
    check_goal_reaching,
    episode_failure_flags,
    infer_goals_from_sim_state,
)
from simulation.planning_utils import apply_ego_replacements_to_expanded_state  # noqa: E402
from viz import viz as viz_module  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_diffusion_policy(ckpt_path: str, *, use_ema: bool = True, seed: int = 0):
    """Returns (policy, preprocess_cfg) exactly as ``run_simulation.py --planner es``."""
    from flax import nnx

    from train_diffusion.utils.infer import load_model_for_inference

    inference = load_model_for_inference(
        checkpoint_path=ckpt_path,
        use_ema=use_ema,
        metadata_path=None,
        seed=seed,
        skip_checkpoint_load=False,
    )
    policy = nnx.merge(inference.graphdef, inference.params_state, inference.nonparam_state)
    return policy, inference.preprocess_cfg


def load_world_model(
    ckpt_path: str,
    *,
    backend: str = "multifuture",
    planning_mode: str | None = "expected_independent",
    device: str = "auto",
    refresh_steps: int = 10,
    num_collision_worlds: int = 1,
    collision_reduce: str = "any",
    collision_sample_seed: int = 0,
):
    """Loads the WM forecaster through the same code path as ``run_simulation.py``."""
    from simulation.run_simulation import _load_world_model_forecaster

    args = argparse.Namespace(
        other_agent_forecast="wm",
        world_model_ckpt=ckpt_path,
        world_model_device=device,
        wm_backend=backend,
        wm_planning_mode=planning_mode,
        wm_refresh_steps=refresh_steps,
        num_collision_worlds=num_collision_worlds,
        collision_reduce=collision_reduce,
        collision_sample_seed=collision_sample_seed,
    )
    return _load_world_model_forecaster(args)


@dataclasses.dataclass
class Scenario:
    sim_state: Any  # batched SimulatorState with B=1
    tfrecord_path: str
    scenario_idx: int
    lane_graph: Any
    goal_xy_b2: jax.Array
    goal_t: int
    ego_idx: int

    @property
    def name(self) -> str:
        return f"{Path(self.tfrecord_path).name}#{self.scenario_idx}"


def load_scenario(
    tfrecord_path: str,
    scenario_idx: int,
    *,
    lane_graph_dir: str | None,
    max_num_objects: int = 128,
) -> Scenario:
    from waymax import config as waymax_config

    if not Path(tfrecord_path).is_file():
        raise FileNotFoundError(f"tfrecord not found (expected a file): {tfrecord_path}")
    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord_path),
        max_num_objects=int(max_num_objects),
        batch_dims=(1,),
        shuffle_seed=0,
    )
    sim_state, _ = load_scenario_state_batch_fast(ds_cfg, [int(scenario_idx)])
    lane_graph = None
    if lane_graph_dir:
        zip_path = lane_graph_zip_path_for_tfrecord(str(tfrecord_path), lane_graph_dir)
        if not zip_path.exists():
            raise FileNotFoundError(f"Lane graph zip {zip_path} not found for {tfrecord_path}.")
        lane_graph = LaneGraphShardStore(zip_path.as_posix()).get_by_record_index(int(scenario_idx))
    goal_xy_b2, goal_t_b, ego_idx_b = infer_goals_from_sim_state(sim_state)
    return Scenario(
        sim_state=sim_state,
        tfrecord_path=str(tfrecord_path),
        scenario_idx=int(scenario_idx),
        lane_graph=lane_graph,
        goal_xy_b2=goal_xy_b2,
        goal_t=int(goal_t_b[0]),
        ego_idx=int(ego_idx_b[0]),
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RewardContext:
    """Everything a reward term may need for scoring one population at one replan step."""

    trajectories: jax.Array  # [P, T, 5] (x, y, yaw, vel_x, vel_y), world frame
    ego_boxes: jax.Array  # [P, T', 5] (x, y, yaw, length, width); T' = forecast horizon
    object_trajectories: jax.Array  # [N, T', 5] primary other-agent forecast
    object_masks: jax.Array  # [N, T']
    collision_object_trajectories: jax.Array
    collision_object_masks: jax.Array
    sim_state: Any
    timestep: int
    world_idx: int
    scorer: Any  # scores.scorer_lane_graph.Scorer with lane graph loaded
    world_model_forecaster: Any
    goal_xy: np.ndarray  # [2]


def build_reward_context(
    trajectories,
    *,
    sim_state,
    timestep: int,
    scorer,
    goal_xy,
    world_idx: int = 0,
) -> RewardContext:
    """Same preamble as ``Scorer.compute_score`` (forecast others, build ego boxes)."""
    trajectories = jnp.asarray(trajectories)
    if scorer.lane_points is None:
        scorer_helpers.update_lane_points(scorer, sim_state, world_idx)
    obj, obj_mask, col_obj, col_mask = scorer_helpers.forecast_other_agents(
        sim_state,
        timestep=int(timestep),
        traj_length=int(trajectories.shape[1]),
        world_idx=world_idx,
        mode=scorer.other_agent_forecast,
        world_model_forecaster=scorer.world_model_forecaster,
    )
    traj_length = int(obj.shape[1])
    ego_idx = scorer_helpers.get_ego_idx(sim_state, world_idx)
    length_nt = jnp.asarray(sim_state.log_trajectory.length[world_idx], dtype=jnp.float32)
    width_nt = jnp.asarray(sim_state.log_trajectory.width[world_idx], dtype=jnp.float32)
    t = int(np.clip(int(timestep), 0, int(length_nt.shape[1]) - 1))
    num = int(trajectories.shape[0])
    ego_boxes = jnp.stack(
        [
            trajectories[:, :traj_length, 0],
            trajectories[:, :traj_length, 1],
            trajectories[:, :traj_length, 2],
            jnp.broadcast_to(length_nt[ego_idx, t], (num, traj_length)),
            jnp.broadcast_to(width_nt[ego_idx, t], (num, traj_length)),
        ],
        axis=-1,
    )
    return RewardContext(
        trajectories=trajectories,
        ego_boxes=ego_boxes,
        object_trajectories=obj,
        object_masks=obj_mask,
        collision_object_trajectories=col_obj,
        collision_object_masks=col_mask,
        sim_state=sim_state,
        timestep=int(timestep),
        world_idx=int(world_idx),
        scorer=scorer,
        world_model_forecaster=scorer.world_model_forecaster,
        goal_xy=np.asarray(goal_xy, dtype=np.float32).reshape(2),
    )


# A reward term maps a RewardContext to a per-candidate score in [0, 1], shape [P].
RewardTerm = Callable[[RewardContext], np.ndarray]


def collision(ctx: RewardContext) -> np.ndarray:
    """Production collision term (same branch logic as ``Scorer.compute_score``)."""
    wm = ctx.world_model_forecaster
    if scorer_helpers.uses_occupancy_collision(wm):
        score, _ = wm.score_occupancy_risk(
            np.asarray(ctx.ego_boxes, dtype=np.float32), world_idx=ctx.world_idx, dt=0.1
        )
    elif scorer_helpers.uses_independent_mode_collision(wm):
        score, _ = wm.score_independent_mode_collision(
            np.asarray(ctx.ego_boxes, dtype=np.float32), world_idx=ctx.world_idx
        )
    else:
        score = ctx.scorer.compute_collision_score(
            ctx.ego_boxes, ctx.collision_object_trajectories, ctx.collision_object_masks
        )
    return np.asarray(score, dtype=np.float32)


def wm_collision(planning_mode: str) -> RewardTerm:
    """Multifuture-WM collision with a different planning mode than the loaded one.

    ``planning_mode``: top1 | expected_independent | worst_case_independent.
    """

    def term(ctx: RewardContext) -> np.ndarray:
        score, _ = ctx.world_model_forecaster.score_independent_mode_collision(
            np.asarray(ctx.ego_boxes, dtype=np.float32),
            world_idx=ctx.world_idx,
            planning_mode=planning_mode,
        )
        return np.asarray(score, dtype=np.float32)

    term.__name__ = f"wm_collision[{planning_mode}]"
    return term


def offroad(ctx: RewardContext) -> np.ndarray:
    return np.asarray(
        ctx.scorer.compute_offroad_score_v2(
            ctx.ego_boxes, sim_state=ctx.sim_state, timestep=ctx.timestep, world_idx=ctx.world_idx
        ),
        dtype=np.float32,
    )


def direction(ctx: RewardContext) -> np.ndarray:
    return np.asarray(ctx.scorer.compute_direction_score(ctx.ego_boxes), dtype=np.float32)


def tl_violation(ctx: RewardContext) -> np.ndarray:
    return np.asarray(
        ctx.scorer.compute_tl_violation_score(
            ctx.ego_boxes, sim_state=ctx.sim_state, timestep=ctx.timestep, world_idx=ctx.world_idx
        ),
        dtype=np.float32,
    )


def progress(ctx: RewardContext) -> np.ndarray:
    """Straight-line displacement over the horizon / 100 m (``Scorer.compute_progress_score``)."""
    return np.asarray(ctx.scorer.compute_progress_score(ctx.ego_boxes), dtype=np.float32)


def goal(max_dist_m: float = 50.0) -> RewardTerm:
    """1 when the mean distance to the episode goal is 0, 0 at ``max_dist_m`` or more."""

    def term(ctx: RewardContext) -> np.ndarray:
        xy = np.asarray(ctx.ego_boxes[..., :2])
        dist = np.linalg.norm(xy - ctx.goal_xy[None, None], axis=-1).mean(axis=1)
        return (np.clip(max_dist_m - dist, 0.0, max_dist_m) / max_dist_m).astype(np.float32)

    term.__name__ = f"goal[{max_dist_m:g}m]"
    return term


def clearance(max_dist_m: float = 10.0) -> RewardTerm:
    """Soft safety margin: min centre distance to any forecast agent over the horizon.

    Uses the primary (top-1 for WM) forecast. 1 at >= ``max_dist_m``, 0 at contact.
    """

    def term(ctx: RewardContext) -> np.ndarray:
        ego_xy = np.asarray(ctx.ego_boxes[..., :2])  # [P, T, 2]
        obj_xy = np.asarray(ctx.object_trajectories[..., :2])  # [N, T, 2]
        mask = np.asarray(ctx.object_masks).astype(bool)  # [N, T]
        if not mask.any():
            return np.ones((ego_xy.shape[0],), dtype=np.float32)
        d = np.linalg.norm(ego_xy[:, None] - obj_xy[None], axis=-1)  # [P, N, T]
        d = np.where(mask[None], d, np.inf).min(axis=(1, 2))
        return (np.clip(d, 0.0, max_dist_m) / max_dist_m).astype(np.float32)

    term.__name__ = f"clearance[{max_dist_m:g}m]"
    return term


def target_speed(speed_mps: float) -> RewardTerm:
    def term(ctx: RewardContext) -> np.ndarray:
        speed = np.linalg.norm(np.asarray(ctx.trajectories[..., 3:5]), axis=-1)
        return (1.0 - np.minimum(np.abs(speed - speed_mps) / 25.0, 1.0)).mean(axis=1).astype(np.float32)

    term.__name__ = f"target_speed[{speed_mps:g}]"
    return term


def comfort(max_accel_mps2: float = 4.0, dt: float = 0.1) -> RewardTerm:
    """1 - (peak |acceleration| / ``max_accel_mps2``), clipped to [0, 1]."""

    def term(ctx: RewardContext) -> np.ndarray:
        vel = np.asarray(ctx.trajectories[..., 3:5])
        acc = np.linalg.norm(np.diff(vel, axis=1), axis=-1) / dt
        return (1.0 - np.clip(acc.max(axis=1) / max_accel_mps2, 0.0, 1.0)).astype(np.float32)

    term.__name__ = f"comfort[{max_accel_mps2:g}]"
    return term


class Reward:
    """Composite reward with the same algebra as ``Scorer.compute_score``.

    score = (sum_i w_i * additive_i / sum_i w_i  or 1 if no additive terms)
            * prod_j multiplicative_j

    ``multiplicative``: {name: term} -- hard gates (collision, offroad, ...).
    ``additive``: {name: (term, weight)} -- shaped objectives (progress, goal, ...).
    """

    def __init__(
        self,
        multiplicative: dict[str, RewardTerm] | None = None,
        additive: dict[str, tuple[RewardTerm, float]] | None = None,
    ) -> None:
        self.multiplicative = dict(multiplicative or {})
        self.additive = dict(additive or {})

    def __call__(self, ctx: RewardContext) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        num = int(ctx.trajectories.shape[0])
        breakdown: dict[str, np.ndarray] = {}
        score = np.ones((num,), dtype=np.float32)
        if self.additive:
            total_w = sum(float(w) for _, w in self.additive.values())
            score = np.zeros((num,), dtype=np.float32)
            for name, (term, w) in self.additive.items():
                breakdown[name] = np.asarray(term(ctx), dtype=np.float32)
                score = score + float(w) * breakdown[name]
            score = score / total_w
        for name, term in self.multiplicative.items():
            breakdown[name] = np.asarray(term(ctx), dtype=np.float32)
            score = score * breakdown[name]
        return score.astype(np.float32), breakdown

    def __repr__(self) -> str:
        mul = list(self.multiplicative)
        add = {k: w for k, (_, w) in self.additive.items()}
        return f"Reward(multiplicative={mul}, additive={add})"


def production_reward() -> Reward:
    """What ``run_simulation.py --planner es`` uses: metrics=['collision', 'offroad']."""
    return Reward(multiplicative={"collision": collision, "offroad": offroad})


def collision_time_tiebreak(ctx: RewardContext) -> np.ndarray:
    """Graded collision signal for plans that all collide: 1 if no WM mode of any agent is
    hit; otherwise the earliest (over agents) mode-probability-weighted first-hit time,
    / horizon. Later contact -> higher. Uses the same WM futures as ``collision``."""
    from scores.polygon import pairwise_intersections, polygon_corners

    wm = ctx.world_model_forecaster
    ego = jnp.asarray(ctx.ego_boxes, dtype=jnp.float32)  # [P, T, 5]
    p, t = int(ego.shape[0]), int(ego.shape[1])
    if wm is None or not hasattr(wm, "forecast_modes"):
        return np.ones((p,), dtype=np.float32)
    fc = wm.forecast_modes(ctx.sim_state, timestep=int(ctx.timestep), traj_length=t, world_idx=ctx.world_idx)
    traj_k = jnp.asarray(fc["traj_k"], dtype=jnp.float32)  # [N, K, T, 5]
    probs = np.asarray(fc["mode_probs"], dtype=np.float32)  # [N, K]
    masks = np.asarray(fc["masks"]).astype(bool)  # [N, T]
    n, k = int(traj_k.shape[0]), int(traj_k.shape[1])
    obj_poly = polygon_corners(traj_k.reshape((n * k, t, 5)))
    ego_poly = polygon_corners(ego)
    hits = jax.vmap(pairwise_intersections, in_axes=(1, 1), out_axes=0)(ego_poly, obj_poly)  # [T, P, N*K]
    hits = np.asarray(hits).reshape(t, p, n, k) & np.repeat(masks.T[:, None, :, None], k, axis=3)
    any_hit = hits.any(axis=0)  # [P, N, K]
    first = np.where(any_hit, hits.argmax(axis=0), t).astype(np.float32)  # [P, N, K]
    w = probs / np.clip(probs.sum(-1, keepdims=True), 1e-8, None)
    expected_first = (first * w[None]).sum(-1)  # [P, N]
    expected_first = np.where(masks.any(-1)[None], expected_first, t)
    return (expected_first.min(-1) / t).astype(np.float32)


def goal_progress(ctx: RewardContext, goal_step: int = 90, dt: float = 0.1, reach_m: float = 2.0) -> np.ndarray:
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
    xy = np.asarray(ctx.trajectories[..., :2], dtype=np.float64)
    d = np.linalg.norm(xy - np.asarray(ctx.goal_xy, np.float64)[None, None], axis=-1)  # [P, T]
    steps_left = max(int(goal_step) - int(ctx.timestep), 1)
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


class MarginTiebreakReward:
    """``--offroad_mode margin_tiebreak``: collision * edge-margin + 1e-3 * offroad tiebreak.

    Same algebra as ``Scorer.compute_score`` in that mode. ``breakdown['offroad']`` is the
    margin score (0 if offroad, 0.5..1 by clearance) so ``> 0`` still means on-road.

    ``collision_eps > 0`` adds ``collision_eps * collision_time_tiebreak`` (not in
    production): among plans that all collide, the one that hits later ranks higher, so ES
    still has a direction when no candidate is collision-free.

    ``progress_weight = w > 0`` (not in production) multiplies the safety part by
    ``(1 - w) + w * goal_progress``: among safe plans the one keeping pace to the goal wins, while
    any collision-free plan (>= 0.5 (1 - w)) still beats a colliding one.
    """

    def __init__(self, collision_eps: float = 0.0, progress_weight: float = 0.0) -> None:
        self.collision_eps = float(collision_eps)
        self.progress_weight = float(progress_weight)

    def __call__(self, ctx: RewardContext) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        from scores.scorer_lane_graph import OFFROAD_TIEBREAK_EPS

        coll = np.asarray(collision(ctx), dtype=np.float32)
        margin, tiebreak = ctx.scorer.compute_offroad_score_margin_tiebreak(
            ctx.ego_boxes, sim_state=ctx.sim_state, timestep=ctx.timestep, world_idx=ctx.world_idx
        )
        margin = np.asarray(margin, dtype=np.float32)
        tiebreak = np.asarray(tiebreak, dtype=np.float32)
        safety = coll * margin
        breakdown = {"collision": coll, "offroad": margin, "offroad_tiebreak": tiebreak}
        if self.progress_weight > 0:
            prog = goal_progress(ctx)
            safety = safety * ((1.0 - self.progress_weight) + self.progress_weight * prog)
            breakdown["progress"] = prog
        score = safety + OFFROAD_TIEBREAK_EPS * tiebreak
        if self.collision_eps > 0:
            ctb = collision_time_tiebreak(ctx)
            score = score + self.collision_eps * ctb
            breakdown["collision_tiebreak"] = ctb
        return score.astype(np.float32), breakdown

    def __repr__(self) -> str:
        return f"MarginTiebreakReward(collision_eps={self.collision_eps:g}, progress_weight={self.progress_weight:g})"


def margin_tiebreak_reward(collision_eps: float = 0.0, progress_weight: float = 0.0) -> MarginTiebreakReward:
    return MarginTiebreakReward(collision_eps=collision_eps, progress_weight=progress_weight)


class LexicographicReward:
    """Safety first, then goal progress (ES reward for the MCMC pool).

    safety S = collision * 1[on-road] * 1[kinematically feasible] in [0, 1], quantized into ``1/bucket`` levels b = floor(S / bucket)
    (collision-probability differences below ``bucket`` count as equal). Within a level:
      b = 0 (colliding or off-road): (collision + collision_time_tiebreak + offroad_tiebreak) / 3,
             i.e. less likely / later contact, shallower / later off-road;
      b > 0: goal_progress, + ``jerk_eps`` * smoothness.
    score = (b + 0.99 * within) / (levels + 1), so a higher safety level always wins.
    """

    def __init__(self, bucket: float = 0.05, jerk_eps: float = 0.05, jerk_ref: float = 2.0, goal_step: int = 90,
                 max_accel: float = 8.0, max_accel_step: float | None = 2.0, reach_m: float = 1.0) -> None:
        self.bucket, self.jerk_eps, self.jerk_ref, self.goal_step = float(bucket), float(jerk_eps), float(jerk_ref), int(goal_step)
        self.max_accel, self.max_accel_step, self.reach_m = float(max_accel), max_accel_step, float(reach_m)

    def __call__(self, ctx: RewardContext) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        coll = np.asarray(collision(ctx), dtype=np.float32)
        margin, tiebreak = ctx.scorer.compute_offroad_score_margin_tiebreak(
            ctx.ego_boxes, sim_state=ctx.sim_state, timestep=ctx.timestep, world_idx=ctx.world_idx
        )
        margin, tiebreak = np.asarray(margin, np.float32), np.asarray(tiebreak, np.float32)
        ctb = collision_time_tiebreak(ctx)
        # reach_m below the 2 m success radius: aim at the goal, not its edge.
        prog = goal_progress(ctx, goal_step=self.goal_step, reach_m=self.reach_m)
        levels = int(round(1.0 / self.bucket))
        a0 = None if self.max_accel_step is None else _ego_kinematics(ctx.sim_state, ctx.timestep, ctx.world_idx)[1]
        feasible = kinematic_feasibility(np.asarray(ctx.trajectories), max_accel=self.max_accel,
                                         max_lat_accel=self.max_accel, a0=a0, max_accel_step=self.max_accel_step or 0.0)
        level = np.clip(np.floor(coll * (margin > 0) * feasible * levels + 1e-6), 0, levels)
        within = np.where(level == 0, (coll + ctb + tiebreak + feasible) / 4.0, prog)
        breakdown = {"collision": coll, "offroad": margin, "offroad_tiebreak": tiebreak,
                     "collision_tiebreak": ctb, "progress": prog, "feasible": feasible,
                     "safety_level": level.astype(np.float32)}
        if self.jerk_eps > 0:
            smooth = np.exp(-plan_jerk(np.asarray(ctx.trajectories)) / self.jerk_ref**2)
            within = np.where(level > 0, (1 - self.jerk_eps) * within + self.jerk_eps * smooth, within)
            breakdown["smoothness"] = smooth.astype(np.float32)
        score = (level + 0.99 * within) / (levels + 1)
        return score.astype(np.float32), breakdown

    def __repr__(self) -> str:
        return f"LexicographicReward(bucket={self.bucket:g}, jerk_eps={self.jerk_eps:g})"


def lexicographic_reward(bucket: float = 0.05, jerk_eps: float = 0.05, **kw) -> LexicographicReward:
    return LexicographicReward(bucket=bucket, jerk_eps=jerk_eps, **kw)


def wm_mode_hits(ctx: RewardContext) -> tuple[np.ndarray, np.ndarray] | None:
    """Ego-box overlap with every WM mode: hits [P, N, K, T] (bool) and mode probabilities [N, K]
    (normalized per agent), or None when the forecaster has no modes."""
    from scores.polygon import pairwise_intersections, polygon_corners

    wm = ctx.world_model_forecaster
    if wm is None or not hasattr(wm, "forecast_modes"):
        return None
    ego = jnp.asarray(ctx.ego_boxes, dtype=jnp.float32)  # [P, T, 5]
    p, t = int(ego.shape[0]), int(ego.shape[1])
    fc = wm.forecast_modes(ctx.sim_state, timestep=int(ctx.timestep), traj_length=t, world_idx=ctx.world_idx)
    traj_k = jnp.asarray(fc["traj_k"], dtype=jnp.float32)  # [N, K, T, 5]
    probs = np.asarray(fc["mode_probs"], dtype=np.float32)  # [N, K]
    masks = np.asarray(fc["masks"]).astype(bool)  # [N, T]
    n, k = int(traj_k.shape[0]), int(traj_k.shape[1])
    hits = jax.vmap(pairwise_intersections, in_axes=(1, 1), out_axes=0)(
        polygon_corners(ego), polygon_corners(traj_k.reshape((n * k, t, 5)))
    )  # [T, P, N*K]
    hits = np.asarray(hits).reshape(t, p, n, k) & masks.T[:, None, :, None]
    probs = probs / np.clip(probs.sum(-1, keepdims=True), 1e-8, None)
    return np.moveaxis(hits, 0, -1), probs


def lane_keeping_cost(ctx: RewardContext, heading_w: float = 1.0, max_heading_rad: float = np.pi / 3,
                      max_lat_m: float = 5.0, stride: int = 5) -> np.ndarray:
    """[P] mean over time of |lateral offset| (m) to the nearest direction-consistent lane centreline
    point + ``heading_w`` * |heading error| (rad). Offsets are capped at ``max_lat_m``."""
    lanes = getattr(ctx.scorer, "lane_points", None)
    traj = np.asarray(ctx.trajectories, dtype=np.float32)[:, ::stride]  # [P, T', 5]
    if lanes is None or int(np.shape(lanes)[0]) == 0:
        return np.zeros((traj.shape[0],), dtype=np.float32)
    lanes = np.asarray(lanes, dtype=np.float32)
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


class ViolationReward:
    """Rulebook-style ranking (Censi et al. 2019) with slack (Wray et al. 2015) on collision probability.

    Each rule is a violation >= 0 that is 0 when the rule holds, so plans satisfying a rule tie on it
    and only magnitudes of real violations are compared. Rules are checked over the executed window
    plus a lookahead (``near_s``) first, then over the whole horizon. Keys, in order (lower is better):
      feasibility    1 - kinematically feasible
      collision      P(overlap) within near_s, P(overlap) over the horizon     (slack ``collision_slack``)
                     -expected first-contact time                             (later contact first)
      off-road       depth past the road edge within near_s, over the horizon, -first off-road time
      goal           1 - goal_progress (0 = on pace to reach the goal by the deadline)
      tiebreak       -min(edge clearance, OFFROAD_MARGIN_M), lane-keeping cost, jerk
    The pool is filtered key by key: survivors are within the key's slack (1e-6 if none) of the best
    survivor. Score = (d + 0.99 r) / n_keys, d = keys passed, r = rank within the group by the key that
    failed, so top-k elites are the deepest survivors, then the nearest misses. The score is relative
    to the scored set, so the ES pool is re-ranked as a whole (``rank``) after each merge.
    """

    KEYS = ("v_infeasible", "v_coll_near", "v_coll_all", "v_coll_time", "v_off_near", "v_off_all",
            "v_off_time", "v_goal", "v_clearance", "v_lane", "v_jerk")

    def __init__(self, collision_slack: float = 0.02, near_s: float = 3.0, dt: float = 0.1, goal_step: int = 90,
                 max_accel: float = 8.0, max_accel_step: float | None = 2.0, reach_m: float = 1.0) -> None:
        self.collision_slack, self.near_s, self.dt = float(collision_slack), float(near_s), float(dt)
        self.goal_step, self.max_accel, self.max_accel_step, self.reach_m = int(goal_step), float(max_accel), max_accel_step, float(reach_m)

    def __call__(self, ctx: RewardContext) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        from scores.scorer_lane_graph import OFFROAD_MARGIN_M

        traj = np.asarray(ctx.trajectories)
        p, t = traj.shape[0], traj.shape[1]
        near = min(int(round(self.near_s / self.dt)) + 1, t)

        res = wm_mode_hits(ctx)
        if res is None:  # deterministic forecast: binary production collision term
            no_coll = np.asarray(collision(ctx), np.float32)
            p_near = p_all = 1.0 - no_coll
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

        dist = ctx.scorer._offroad_corner_distances(ctx.ego_boxes, ctx.sim_state, ctx.timestep, ctx.world_idx)
        if dist is None:
            off = 1.0 - np.asarray(ctx.scorer.compute_offroad_score(ctx.ego_boxes), np.float32)
            worst_t = np.repeat(off[:, None], t, axis=1) - 0.5  # > 0 = off-road
        else:
            worst_t = np.asarray(dist, np.float32).max(-1)  # [P, T], > 0 = past the road edge
        is_off = worst_t > 0
        t_off = np.where(is_off.any(-1), is_off.argmax(-1), t).astype(np.float32)

        a0 = None if self.max_accel_step is None else _ego_kinematics(ctx.sim_state, ctx.timestep, ctx.world_idx)[1]
        feasible = kinematic_feasibility(traj, max_accel=self.max_accel, max_lat_accel=self.max_accel, a0=a0,
                                         max_accel_step=self.max_accel_step or 0.0)
        prog = goal_progress(ctx, goal_step=self.goal_step, reach_m=self.reach_m)
        margin_m = -worst_t.max(-1)
        b = {
            "v_infeasible": 1.0 - feasible,
            "v_coll_near": p_near, "v_coll_all": p_all, "v_coll_time": -t_hit,
            "v_off_near": np.clip(worst_t[:, :near].max(-1), 0.0, None),
            "v_off_all": np.clip(worst_t.max(-1), 0.0, None), "v_off_time": -t_off,
            "v_goal": 1.0 - prog,
            "v_clearance": -np.minimum(margin_m, OFFROAD_MARGIN_M),
            "v_lane": lane_keeping_cost(ctx),
            "v_jerk": plan_jerk(traj),
        }
        b = {k: np.asarray(v, np.float32) for k, v in b.items()}
        # Same names as the other rewards, for the eval tables.
        b.update(collision=1.0 - b["v_coll_all"], offroad=(b["v_off_all"] == 0).astype(np.float32),
                 progress=prog.astype(np.float32), feasible=feasible.astype(np.float32))
        return self.rank(b), b

    def rank(self, b: dict[str, np.ndarray]) -> np.ndarray:
        keys = [b[k] for k in self.KEYS]
        slack = {"v_coll_near": self.collision_slack, "v_coll_all": self.collision_slack}
        n = len(keys[0])
        alive = np.ones(n, dtype=bool)
        depth = np.zeros(n, dtype=np.int32)
        for name, v in zip(self.KEYS[:-1], keys[:-1]):
            keep = alive & (v <= v[alive].min() + max(slack.get(name, 0.0), 1e-6))
            depth[keep] += 1
            alive = keep
        within = np.zeros(n, dtype=np.float32)
        for d in np.unique(depth):
            g = np.flatnonzero(depth == d)
            r = np.argsort(np.argsort(-keys[d][g], kind="stable"), kind="stable")  # lower violation -> higher
            within[g] = r / max(len(g) - 1, 1)
        b["rule_depth"] = depth.astype(np.float32)
        return ((depth + 0.99 * within) / len(keys)).astype(np.float32)

    def __repr__(self) -> str:
        return f"ViolationReward(collision_slack={self.collision_slack:g}, near_s={self.near_s:g})"


def violation_reward(**kw) -> ViolationReward:
    return ViolationReward(**kw)


def plan_jerk(traj_pt5: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """[P] mean squared jerk (longitudinal + lateral, m^2/s^6) of world-frame plans [P, T, 5]."""
    vel = traj_pt5[..., 3:5]
    acc = np.gradient(vel, dt, axis=1)
    jerk = np.gradient(acc, dt, axis=1)
    return (jerk**2).sum(-1).mean(-1)


# ---------------------------------------------------------------------------
# Acquisition (elite selection)
# ---------------------------------------------------------------------------

# An acquisition maps (scores [P], trajectories [P, T, 5], n_elite, rng) -> indices [n_elite].
Acquisition = Callable[[np.ndarray, np.ndarray, int, np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------------
# Initial pool (iteration-0 sampling)
# ---------------------------------------------------------------------------

# An initial-pool sampler maps (planner, cond_bf, inst_cond_bf, inst_cond_mask_bf, features,
# pre_batch, rng, sim_state, timestep) -> normalized proposals [1, P, T, D]. Later ES iterations are unchanged
# (elites are re-noised and denoised with the goal-conditioned cond_bf).
InitialPool = Callable[..., jax.Array]


def standard_pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng, **_) -> jax.Array:
    """Production behaviour: P i.i.d. diffusion samples."""
    del features, pre_batch
    return planner._sample_population_jit(
        cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=inst_cond_mask_bf, rng=rng
    )


def _farthest_points(feat_md: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point selection; starts from row 0."""
    chosen = [0]
    d = np.linalg.norm(feat_md - feat_md[0], axis=-1)
    for _ in range(1, n):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(feat_md - feat_md[i], axis=-1))
    return np.asarray(chosen, dtype=np.int32)


def oversample_fps(factor: int = 4, num_waypoints: int = 4) -> InitialPool:
    """Draw ``factor * P`` samples and keep P by farthest-point sampling on world xy at
    ``num_waypoints`` evenly spaced horizon steps. Same samples as the model would give,
    just spread over its modes instead of concentrated on the dominant one."""

    def pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng, **_):
        del features
        keys = jax.random.split(rng, factor)
        norm = jnp.concatenate(
            [
                planner._sample_population_jit(
                    cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=inst_cond_mask_bf, rng=k
                )
                for k in keys
            ],
            axis=1,
        )  # [1, factor*P, T, D]
        world = np.asarray(planner._postprocess_population(norm, pre_batch)[0][0])  # [M, T, 5]
        t_idx = np.linspace(world.shape[1] // num_waypoints, world.shape[1] - 1, num_waypoints).astype(int)
        feat = world[:, t_idx, :2].reshape(world.shape[0], -1)
        keep = _farthest_points(feat, planner.population_size)
        return norm[:, keep]

    pool.__name__ = f"oversample_fps[x{factor}]"
    return pool


def denoise_proposals(
    planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, proposals_bktd, rng, steps: int, noise_scale: float
) -> jax.Array:
    """SDEdit: re-noise ``proposals_bktd`` to diffusion step ``steps`` and denoise back.

    ``noise_scale`` multiplies the forward noise; the policy default (what production ES
    uses) is 2.0, i.e. twice the noise the schedule expects at that step, which few
    denoising steps cannot fully remove. 1.0 matches the schedule.
    """
    cache = planner.__dict__.setdefault("_denoise_jit_cache", {})
    key = (int(steps), float(noise_scale))
    if key not in cache:

        def f(cond_bf, inst_cond_bf, inst_cond_mask_bf, proposals_bktd, rng):
            b, k, t, d = proposals_bktd.shape
            out = planner.policy.resample_from_condition(
                jnp.repeat(cond_bf, k, axis=0),
                jnp.repeat(inst_cond_bf, k, axis=0),
                jnp.repeat(inst_cond_mask_bf, k, axis=0),
                proposals_bktd.reshape(b * k, t, d),
                key[0],
                rng=rng,
                noise_scale=key[1],
            )
            return out.reshape(b, k, t, d)

        cache[key] = jax.jit(f)
    return cache[key](cond_bf, inst_cond_bf, inst_cond_mask_bf, proposals_bktd, rng)


# ---------------------------------------------------------------------------
# MCMC proposal pool: sample smooth kinematic plans from p0(theta) * exp(-E(tau(theta)))
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KinematicPrior:
    """Prior over Bernstein control points of speed v(t) (degree n_v) and curvature kappa(t) (degree n_k).

    v(t) = sum_j w_j b_{j,n_v}(t/H): w_0 = v0, w_1 = v0 + a0 H / n_v (pinned, so speed AND acceleration
         continue the current motion), free w_j ~ N(v0, (accel_sigma * t_j)^2)
    k(t) = sum_j c_j b_{j,n_k}(t/H): c_0 = k0, free c_j ~ N(k0, (lat_accel_sigma / max(v0, v_floor)^2)^2)
    Feasible set (convex-hull property makes the first three exact for all t):
      -v_undershoot <= w_j <= v_max,  accel_min <= n_v (w_{j+1} - w_j) / H <= accel_max,  |c_j| <= kappa_max,
      |k(t)| v(t)^2 <= lat_accel_max on a dense grid.
    The executed speed is max(v(t), 0): a polynomial cannot reach 0 and stay there, so knots may
    undershoot and the clip turns that into a stop-and-hold.
    """

    speed_degree: int = 4
    curvature_degree: int = 3
    pin_accel: bool = True  # w_1 from the current acceleration (C1 speed across replans)
    accel_sigma: float = 2.0  # m/s^2: speed std grows as accel_sigma * t
    accel_min: float = -6.0  # m/s^2: hard braking
    accel_max: float = 3.0  # m/s^2
    lat_accel_sigma: float = 2.0  # m/s^2: curvature std = lat_accel_sigma / v0^2
    lat_accel_max: float = 5.0  # m/s^2
    kappa_max: float = 0.2  # 1/m: ~5 m turning radius
    v_floor: float = 3.0  # m/s: caps the curvature std at low speed
    v_max: float = 25.0  # m/s
    v_undershoot: float = 6.0  # m/s: speed knots may go this far below 0; the rollout clips v at 0 (stop and hold)


@dataclasses.dataclass(frozen=True)
class CostWeights:
    """E = collision (1 - C) + contact_time (1 - T_hit) + offroad (1 - M) + offroad_depth (1 - R)
         + progress (1 - G) + comfort * mean((j_lon^2 + j_lat^2) / jerk_ref^2)."""

    collision: float = 4.0
    contact_time: float = 2.0
    offroad: float = 4.0
    offroad_depth: float = 2.0
    progress: float = 0.0  # progress is ES's job (lexicographic_reward); MCMC only maps the safe region
    comfort: float = 0.2
    jerk_ref: float = 2.0  # m/s^3
    goal_step: int = 90  # the episode goal is the logged position at this step


def _bernstein(n: int, u: np.ndarray) -> np.ndarray:
    """[len(u), n + 1] Bernstein basis b_{j,n}(u)."""
    from math import comb

    j = np.arange(n + 1)
    return np.asarray([comb(n, k) for k in j], np.float64)[None] * u[:, None] ** j * (1.0 - u[:, None]) ** (n - j)


def _ego_kinematics(sim_state, timestep: int, world_idx: int = 0, lookback: int = 5) -> tuple[float, float, float]:
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


class KinematicPlanSpace:
    """theta = (free speed knots, free curvature knots) <-> smooth ego-frame plans; whitened z ~ N(0, I) under the prior."""

    def __init__(self, prior: KinematicPrior, *, v0: float, a0: float, kappa0: float, horizon_s: float, t_model: np.ndarray):
        self.prior, self.horizon_s = prior, float(horizon_s)
        self.nv, self.nk = prior.speed_degree, prior.curvature_degree
        self.v0 = float(np.clip(v0, 0.0, prior.v_max))
        # Pinned start values must themselves be feasible.
        a_lo = max(prior.accel_min, -self.v0 * self.nv / self.horizon_s)
        self.a0 = float(np.clip(a0, a_lo, prior.accel_max)) if prior.pin_accel else 0.0
        k_lat = prior.lat_accel_max / max(self.v0, 1e-3) ** 2
        self.kappa0 = float(np.clip(kappa0, -min(prior.kappa_max, k_lat), min(prior.kappa_max, k_lat)))
        self.w_fixed = [self.v0] + ([self.v0 + self.a0 * self.horizon_s / self.nv] if prior.pin_accel else [])
        j_free = np.arange(len(self.w_fixed), self.nv + 1)
        self.n_w = len(j_free)
        self.mu = np.concatenate([np.full(self.n_w, self.v0), np.full(self.nk, self.kappa0)])
        self.sigma = np.concatenate([
            prior.accel_sigma * j_free * self.horizon_s / self.nv,
            np.full(self.nk, prior.lat_accel_sigma / max(self.v0, prior.v_floor) ** 2),
        ])
        self.dim = self.n_w + self.nk
        # Fine grid for integration (0.05 s), sampled at the model timestamps.
        self.t_fine = np.linspace(0.0, self.horizon_s, int(round(self.horizon_s / 0.05)) + 1)
        self.basis_v = _bernstein(self.nv, self.t_fine / self.horizon_s)  # [F, nv+1]
        self.basis_k = _bernstein(self.nk, self.t_fine / self.horizon_s)  # [F, nk+1]
        self.t_model = np.asarray(t_model, np.float64)

    def theta(self, z: np.ndarray) -> np.ndarray:
        return self.mu + self.sigma * z

    def z_for_speed(self, v_target: np.ndarray, kappa_z: np.ndarray | None = None) -> np.ndarray:
        """z whose speed profile best fits ``v_target`` [F] (least squares on the free knots), curvature
        knots at ``kappa_z`` (whitened, default 0 = keep the current curvature)."""
        n_fixed = len(self.w_fixed)
        a = self.basis_v[:, n_fixed:]
        b = v_target - self.basis_v[:, :n_fixed] @ np.asarray(self.w_fixed)
        w_free = np.linalg.lstsq(a, b, rcond=None)[0]
        # Project onto the speed constraints: each knot step within the accel bounds, knots in range.
        pr, step = self.prior, self.horizon_s / self.nv
        w = list(self.w_fixed)
        for wj in w_free:
            w.append(float(np.clip(np.clip(wj, w[-1] + pr.accel_min * step, w[-1] + pr.accel_max * step),
                                   -pr.v_undershoot, pr.v_max)))
        z_w = (np.asarray(w[n_fixed:]) - self.mu[: self.n_w]) / self.sigma[: self.n_w]
        return np.concatenate([z_w, np.zeros(self.nk) if kappa_z is None else kappa_z])

    def braking_anchors(self, accels=(-6.0, -4.0, -2.0, 1.0, 2.0, 3.0), kappa_z=(0.0, 0.7, -0.7)) -> np.ndarray:
        """Feasible z for constant-acceleration speed profiles (negative = brake to a stop, positive =
        speed up to v_max), each with a few curvature offsets. Hard stops and hard speed-ups are
        ~3 sigma out in the speed prior, so random chains rarely start there."""
        out = []
        for acc in accels:
            # Fit the unclipped ramp (down to -v_undershoot); the rollout clip makes it a stop.
            v = np.clip(self.v0 + acc * self.t_fine, -self.prior.v_undershoot, self.prior.v_max)
            for kz in kappa_z:
                z = self.z_for_speed(v, np.full(self.nk, kz))
                # The least-squares fit can overshoot below 0 m/s or past the accel bounds: shrink
                # toward the prior mean until feasible.
                for scale in np.linspace(1.0, 0.3, 8):
                    if self.feasible((scale * z)[None])[0]:
                        out.append(scale * z)
                        break
        return np.asarray(out).reshape(-1, self.dim)

    def controls(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Control points: speed [P, nv+1], curvature [P, nk+1] (leading ones pinned to the current state)."""
        th = self.theta(z)
        p = th.shape[0]
        w = np.concatenate([np.tile(self.w_fixed, (p, 1)), th[:, : self.n_w]], axis=1)
        c = np.concatenate([np.full((p, 1), self.kappa0), th[:, self.n_w :]], axis=1)
        return w, c

    def profiles(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, c = self.controls(z)
        return np.maximum(w @ self.basis_v.T, 0.0), c @ self.basis_k.T  # [P, F] speed (clipped at 0), curvature

    def feasible(self, z: np.ndarray) -> np.ndarray:
        pr = self.prior
        w, c = self.controls(z)
        accel = self.nv * np.diff(w, axis=1) / self.horizon_s
        v, kappa = self.profiles(z)
        return (
            (w >= -pr.v_undershoot).all(1) & (w <= pr.v_max).all(1)
            & (accel >= pr.accel_min - 1e-6).all(1) & (accel <= pr.accel_max + 1e-6).all(1)
            & (np.abs(c) <= pr.kappa_max).all(1)
            & (np.abs(kappa) * v**2 <= pr.lat_accel_max + 1e-6).all(1)
        )

    def jerk_cost(self, z: np.ndarray, jerk_ref: float) -> np.ndarray:
        """mean over the horizon of (j_lon^2 + j_lat^2) / jerk_ref^2; j_lon = v'', j_lat = d(kappa v^2)/dt."""
        v, kappa = self.profiles(z)
        dt = self.t_fine[1] - self.t_fine[0]
        j_lon = np.gradient(np.gradient(v, dt, axis=1), dt, axis=1)
        j_lat = np.gradient(kappa * v**2, dt, axis=1)
        return ((j_lon**2 + j_lat**2) / jerk_ref**2).mean(axis=1)

    def rollout(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Integrate psi' = kappa v, x' = v cos psi, y' = v sin psi (trapezoid) -> ego-frame states at t_model."""
        v, kappa = self.profiles(z)  # [P, F]
        dt = np.diff(self.t_fine)[None]

        def cumtrapz(f):
            return np.concatenate([np.zeros_like(f[:, :1]), np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * dt, axis=1)], 1)

        psi = cumtrapz(kappa * v)
        x, y = cumtrapz(v * np.cos(psi)), cumtrapz(v * np.sin(psi))
        at = lambda f: np.stack([np.interp(self.t_model, self.t_fine, row) for row in f])  # noqa: E731
        x_m, y_m, psi_m, v_m = at(x), at(y), at(psi), at(v)
        return {"x": x_m, "y": y_m, "psi": psi_m, "v": v_m, "kappa": at(kappa),
                "vx": v_m * np.cos(psi_m), "vy": v_m * np.sin(psi_m)}

    def to_norm(self, z: np.ndarray, cfg, extra_ptd: np.ndarray) -> jax.Array:
        """Diffusion-normalized [1, P, T, D]: [x, y]/ego_range, [vx, vy]/max_velocity, psi/pi, extra dims copied."""
        r = self.rollout(z)
        core = np.stack([r["x"] / cfg.ego_range, r["y"] / cfg.ego_range,
                         r["vx"] / cfg.max_velocity, r["vy"] / cfg.max_velocity, r["psi"] / np.pi], axis=-1)
        return jnp.asarray(np.concatenate([core, extra_ptd], axis=-1), dtype=jnp.float32)[None]


def plan_energy(planner, norm_bktd, *, pre_batch, sim_state, timestep: int, goal_xy, weights: CostWeights):
    """Scene part of E [P] and its terms for normalized plans, scored with the planner's WM / lane graph (as ES)."""
    world = np.asarray(planner._postprocess_population(norm_bktd, pre_batch)[0][0])
    ctx = build_reward_context(world, sim_state=sim_state, timestep=timestep, scorer=planner.scorer, goal_xy=goal_xy)
    coll = np.asarray(collision(ctx), np.float32)
    hit_t = collision_time_tiebreak(ctx)
    margin, depth = ctx.scorer.compute_offroad_score_margin_tiebreak(
        ctx.ego_boxes, sim_state=sim_state, timestep=timestep, world_idx=0
    )
    margin, depth = np.asarray(margin, np.float32), np.asarray(depth, np.float32)
    prog = goal_progress(ctx, goal_step=weights.goal_step)
    parts = {"collision": coll, "contact_time": hit_t, "offroad_margin": margin, "offroad_depth": depth,
             "progress": prog}
    energy = (weights.collision * (1 - coll) + weights.contact_time * (1 - hit_t)
              + weights.offroad * (1 - margin) + weights.offroad_depth * (1 - depth)
              + weights.progress * (1 - prog))
    return energy.astype(np.float64), parts


def mcmc_pool(
    num_steps: int = 30,
    prior: KinematicPrior = KinematicPrior(),
    weights: CostWeights = CostWeights(),
    temperature: float = 1.0,
    anneal_frac: float = 0.5,
    step_size: float = 0.3,
    target_accept: float = 0.3,
    keep_frac: float = 0.25,
    select: str = "diverse_safe",
    safe_collision: float = 0.99,
    max_jerk: float = 3.0,
    braking_anchors: bool = True,
    denoise: bool = False,
    denoise_steps: int | None = None,
    noise_scale: float = 1.0,
) -> InitialPool:
    """Initial pool from P parallel pCN-MCMC chains over smooth kinematic plans.

    ``select='diverse_safe'``: every scored, feasible plan (initial states + all proposals) is a
    candidate; the pool is a farthest-point (diverse) subset of the safe ones (collision >=
    ``safe_collision``, on-road, jerk cost <= ``max_jerk``), topped up by the lowest-energy rest.
    ``select='final'``: the chain states after the last step. ``braking_anchors``: a few chains start
    from constant-acceleration profiles (-6..+3 m/s^2, straight and slightly curved) so emergency
    braking and hard speed-ups are explored.

    Target: pi(z) ∝ N(z; 0, I) 1[feasible] exp(-lambda_k E(tau(z)) / temperature), lambda_k
    ramping 0 -> 1 over the first ``anneal_frac`` of the steps (step size adapts there too).
    Chains start from the prior (symmetrized: z and -z, so left/right and faster/slower come in
    pairs). ``keep_frac`` of the pool are plain diffusion samples (pool slots 0..n_keep-1).
    ``denoise`` = SDEdit the chain states. ``num_steps=0`` = prior samples only. Diagnostics go to
    ``planner.last_pool_info``.
    """

    def pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng, sim_state, timestep,
             goal_xy, **_):
        del features
        k_sample, k_denoise, k_mcmc = jax.random.split(rng, 3)
        np_rng = np.random.default_rng(int(jax.random.randint(k_mcmc, (), 0, 2**31 - 1)))
        base = planner._sample_population_jit(
            cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=inst_cond_mask_bf, rng=k_sample
        )  # [1, P, T, D]
        n_keep = int(round(keep_frac * planner.population_size))
        n_chain = planner.population_size - n_keep
        horizon = int(base.shape[2])
        t_model = np.asarray(pre_batch.aux["model_t_seconds"][0], np.float64)
        t_model = t_model[1:] if t_model.shape[0] == horizon + 1 else t_model[:horizon]
        v0 = float(np.hypot(*np.asarray(pre_batch.aux["anchor_world_state"][0, 2:4])))
        _, a0, kappa0 = _ego_kinematics(sim_state, timestep)
        space = KinematicPlanSpace(
            dataclasses.replace(prior, v_max=min(prior.v_max, planner.preprocess_cfg.max_velocity)),
            v0=v0, a0=a0, kappa0=kappa0, horizon_s=float(t_model[-1]), t_model=t_model,
        )
        extra = np.asarray(base[0, n_keep:, :, 5:])

        # Initial states: feasible draws from the prior, symmetrized.
        half = np_rng.standard_normal((32 * n_chain, space.dim))
        cand = np.stack([half, -half], axis=1).reshape(-1, space.dim)  # z, -z, z', -z', ...
        cand = cand[space.feasible(cand)]
        if braking_anchors:
            cand = np.concatenate([space.braking_anchors(), cand])
        z = np.zeros((n_chain, space.dim))  # theta = mu (keep speed and curvature) is always feasible
        z[: min(len(cand), n_chain)] = cand[:n_chain]

        def energy(z_):
            e_scene, parts_ = plan_energy(
                planner, space.to_norm(z_, planner.preprocess_cfg, extra), pre_batch=pre_batch,
                sim_state=sim_state, timestep=timestep, goal_xy=goal_xy, weights=weights,
            )
            parts_["jerk"] = space.jerk_cost(z_, weights.jerk_ref)
            return e_scene + weights.comfort * parts_["jerk"], parts_

        e, parts = energy(z)
        e_hist, acc_hist, lam_hist = [e.copy()], [], []
        seen_z, seen_e, seen_parts = [z.copy()], [e.copy()], [parts]
        beta = float(step_size)
        n_anneal = max(1, int(round(anneal_frac * num_steps)))
        for k in range(num_steps):
            lam = min(1.0, (k + 1) / n_anneal) / temperature
            prop = np.sqrt(1.0 - beta**2) * z + beta * np_rng.standard_normal(z.shape)
            ok = space.feasible(prop)
            e_prop, parts_prop = energy(prop)
            accept = ok & (np.log(np_rng.uniform(size=n_chain)) < -lam * (e_prop - e))
            z = np.where(accept[:, None], prop, z)
            e = np.where(accept, e_prop, e)
            parts = {name: np.where(accept, parts_prop[name], parts[name]) for name in parts}
            rate = float(accept.mean())
            if k < n_anneal:
                beta = float(np.clip(beta * np.exp(rate - target_accept), 0.02, 0.95))
            e_hist.append(e.copy()); acc_hist.append(rate); lam_hist.append(lam)
            seen_z.append(prop[ok]); seen_e.append(e_prop[ok]); seen_parts.append({n: v[ok] for n, v in parts_prop.items()})

        all_z, all_e = np.concatenate(seen_z), np.concatenate(seen_e)
        all_parts = {n: np.concatenate([sp[n] for sp in seen_parts]) for n in parts}
        safe_all = (all_parts["collision"] >= safe_collision) & (all_parts["offroad_margin"] > 0)
        if select == "diverse_safe":
            pick = _select_diverse_safe(space, all_z, all_e, all_parts, n_chain,
                                        safe_collision=safe_collision, max_jerk=max_jerk)
            z, e, parts = all_z[pick], all_e[pick], {n: v[pick] for n, v in all_parts.items()}
        chains = space.to_norm(z, planner.preprocess_cfg, extra)
        if denoise:
            chains = denoise_proposals(
                planner, cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=inst_cond_mask_bf,
                proposals_bktd=chains, rng=k_denoise,
                steps=denoise_steps or planner.resample_timesteps, noise_scale=noise_scale,
            )
        planner.last_pool_info = {
            "v0": space.v0, "a0": space.a0, "kappa0": space.kappa0, "n_keep": n_keep, "z": z,
            "theta": space.theta(z), "rollout": space.rollout(z), "t_model": t_model, "energy": np.stack(e_hist),
            "accept": np.asarray(acc_hist), "lambda": np.asarray(lam_hist), "step_size": beta, "parts": parts,
            "n_feasible_init": int(min(len(cand), n_chain)),
            "n_scored": int(len(all_e)), "n_safe_found": int(safe_all.sum()),
            "n_safe_smooth_found": int((safe_all & (all_parts["jerk"] <= max_jerk)).sum()),
            "space": space,  # lets ES mutate these plans in z (es_mutation='hybrid')
        }
        return jnp.concatenate([base[:, :n_keep], chains], axis=1)

    pool.__name__ = (
        f"mcmc_pool[{num_steps} steps, T={temperature:g}"
        f"{'' if not denoise else f', dn{denoise_steps or 0}xσ{noise_scale:g}'}]"
    )
    return pool


def _plan_features(space: KinematicPlanSpace, z: np.ndarray, times=(1.25, 2.5, 3.75, 5.0)) -> np.ndarray:
    """Ego-frame (x, y) at a few horizon times: separates speed (along) and curvature (across)."""
    r = space.rollout(z)
    idx = [int(np.argmin(np.abs(space.t_model - t))) for t in times]
    return np.concatenate([r["x"][:, idx], r["y"][:, idx]], axis=1)


def _select_diverse_safe(space, z, e, parts, n, *, safe_collision, max_jerk) -> np.ndarray:
    """Indices of n plans: diverse (farthest-point) among safe & smooth ones, lowest energy first;
    if there are fewer than n of those, all of them plus the lowest-energy rest (also spread out)."""
    safe = (parts["collision"] >= safe_collision) & (parts["offroad_margin"] > 0)
    good = np.flatnonzero(safe & (parts["jerk"] <= max_jerk))
    good = good[np.argsort(e[good])]
    if len(good) >= n:
        return good[_farthest_points(_plan_features(space, z[good]), n)]
    rest = np.setdiff1d(np.arange(len(e)), good)
    rest = rest[np.argsort(e[rest])][: max(3 * (n - len(good)), n - len(good))]
    if len(good) == 0:
        return rest[_farthest_points(_plan_features(space, z[rest]), min(n, len(rest)))] if len(rest) >= n else rest
    feat = _plan_features(space, z[np.concatenate([good, rest])])
    chosen = list(range(len(good)))
    d = np.min(np.linalg.norm(feat[:, None] - feat[None, chosen], axis=-1), axis=1)
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(feat - feat[i], axis=-1))
    return np.concatenate([good, rest])[chosen]


def mcmc_summary(info: dict) -> dict[str, float]:
    """Final chain states: is anything collision-free / on-road, and when do the colliding ones hit?"""
    p, horizon = info["parts"], float(info["t_model"][-1])
    return {
        "E_min": float(info["energy"][-1].min()),
        "E_median": float(np.median(info["energy"][-1])),
        "frac_collision_free": float((p["collision"] >= 0.99).mean()),
        "frac_onroad": float((p["offroad_margin"] > 0).mean()),
        "latest_contact_s": float(p["contact_time"].max() * horizon),  # = horizon if some chain never hits
        "progress_median": float(np.median(p["progress"])),
        "jerk_median": float(np.median(p["jerk"])),
        "accept_final": float(info["accept"][-1]) if len(info["accept"]) else float("nan"),
        "n_scored": info.get("n_scored", 0),
        "n_safe_found": info.get("n_safe_found", 0),
        "n_safe_smooth_found": info.get("n_safe_smooth_found", 0),
    }


def plot_mcmc(info: dict, *, figsize=(15, 3.6)):
    """MCMC diagnostics for ``planner.last_pool_info``: energy quantiles and acceptance per step,
    speed and curvature profiles of the final chain states."""
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    e = info["energy"]
    steps = np.arange(len(e))
    ax[0].fill_between(steps, *np.percentile(e, [10, 90], axis=1), alpha=0.3, label="10-90%")
    ax[0].plot(steps, np.median(e, axis=1), label="median")
    ax[0].plot(steps, e.min(axis=1), label="min")
    ax[0].set(xlabel="MCMC step", ylabel="energy E", title="energy of the chains")
    ax[0].legend(fontsize=8)
    if len(info["accept"]):
        ax[1].plot(np.arange(1, len(steps)), info["accept"], label="accept rate")
        ax[1].plot(np.arange(1, len(steps)), info["lambda"], label="λ (anneal)")
    ax[1].set(xlabel="MCMC step", ylim=(0, 1.05), title="acceptance / annealing")
    ax[1].legend(fontsize=8)
    r, t = info["rollout"], info["t_model"]
    col = plt.cm.viridis((e[-1] - e[-1].min()) / max(np.ptp(e[-1]), 1e-6))
    for i in range(len(r["v"])):
        ax[2].plot(t, r["v"][i], color=col[i], lw=0.8)
        ax[3].plot(t, r["kappa"][i], color=col[i], lw=0.8)
    ax[2].set(xlabel="t [s]", ylabel="m/s", title=f"speed (v0={info['v0']:.1f}, a0={info.get('a0', 0.0):.1f})")
    ax[3].set(xlabel="t [s]", ylabel="1/m", title=f"curvature (κ0={info['kappa0']:.3f})")
    ax[3].axhline(0, color="k", lw=0.5)
    fig.suptitle("color = final energy (dark = low)", fontsize=9)
    fig.tight_layout()
    return fig


def topk(scores, trajectories, n_elite, rng) -> np.ndarray:
    """Production behaviour (``jax.lax.top_k``)."""
    del trajectories, rng
    return np.asarray(jax.lax.top_k(jnp.asarray(scores), n_elite)[1], dtype=np.int32)


def boltzmann(temperature: float = 0.1) -> Acquisition:
    """Sample elites without replacement with p ~ softmax(score / temperature)."""

    def acq(scores, trajectories, n_elite, rng):
        del trajectories
        z = (np.asarray(scores, dtype=np.float64) - np.max(scores)) / max(temperature, 1e-8)
        p = np.exp(z)
        p = p / p.sum()
        n_nonzero = int((p > 0).sum())
        if n_nonzero < n_elite:  # not enough mass: fall back to top-k for the tail
            return topk(scores, None, n_elite, rng)
        return rng.choice(len(scores), size=n_elite, replace=False, p=p).astype(np.int32)

    acq.__name__ = f"boltzmann[T={temperature:g}]"
    return acq


def diverse_topk(min_endpoint_dist_m: float = 3.0) -> Acquisition:
    """Greedy top-k that skips candidates whose endpoint is within ``min_endpoint_dist_m``
    of an already chosen elite (keeps distinct modes alive). Tops up with plain top-k."""

    def acq(scores, trajectories, n_elite, rng):
        del rng
        order = np.argsort(-np.asarray(scores), kind="stable")
        end_xy = np.asarray(trajectories)[:, -1, :2]
        chosen: list[int] = []
        for i in order:
            if all(np.linalg.norm(end_xy[i] - end_xy[j]) >= min_endpoint_dist_m for j in chosen):
                chosen.append(int(i))
            if len(chosen) == n_elite:
                break
        for i in order:
            if len(chosen) == n_elite:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
        return np.asarray(chosen, dtype=np.int32)

    acq.__name__ = f"diverse_topk[{min_endpoint_dist_m:g}m]"
    return acq


def epsilon_greedy(eps: float = 0.25) -> Acquisition:
    """Top-k, but each elite slot is replaced by a random non-elite with prob ``eps``."""

    def acq(scores, trajectories, n_elite, rng):
        idx = topk(scores, trajectories, n_elite, rng).copy()
        rest = np.setdiff1d(np.arange(len(scores)), idx)
        for s in range(n_elite):
            if rest.size and rng.random() < eps:
                pick = rng.integers(rest.size)
                idx[s] = rest[pick]
                rest = np.delete(rest, pick)
        return idx

    acq.__name__ = f"epsilon_greedy[{eps:g}]"
    return acq


# ---------------------------------------------------------------------------
# Instrumented ES planner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ESIteration:
    """One scored pool. Iteration 0 is the initial diffusion sample."""

    iteration: int
    trajectories: np.ndarray  # [P, T, 5] world frame
    scores: np.ndarray  # [P]
    breakdown: dict[str, np.ndarray]  # term name -> [P]
    is_carried_elite: np.ndarray  # [P] True for elites carried over from the previous iteration
    elite_indices: np.ndarray  # [E] indices into this pool chosen by the acquisition
    origin: np.ndarray | None = None  # [P] initial-pool slot each plan descends from (through elites)


@dataclasses.dataclass
class ESTrace:
    timestep: int
    iterations: list[ESIteration]
    best_index: int  # into iterations[-1]
    best_trajectory: np.ndarray  # [T, 5]
    best_score: float
    wm_forecast: dict[str, np.ndarray] | None  # traj_k [N,K,T,5], mode_probs [N,K], masks [N,T], top1_traj
    best_collision_debug: dict[str, Any] | None  # per-agent collision prob for the chosen plan
    label: str = ""
    n_diffusion_slots: int | None = None  # initial slots < this are diffusion samples, the rest from the sampler
    pool_info: dict[str, Any] | None = None  # initial-pool sampler diagnostics (mcmc_pool), if any

    def source(self, k: int | None = None, iteration: int = -1) -> str:
        """'diffusion' or 'sampler' (e.g. an MCMC chain) for plan ``k`` (default: the chosen plan)."""
        it = self.iterations[iteration]
        k = self.best_index if k is None else k
        if it.origin is None or self.n_diffusion_slots is None:
            return "diffusion"
        return "diffusion" if it.origin[k] < self.n_diffusion_slots else "sampler"

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for it in self.iterations:
            elite_scores = it.scores[it.elite_indices]
            row = {
                "iter": it.iteration,
                "pool": len(it.scores),
                "max": float(it.scores.max()),
                "mean": float(it.scores.mean()),
                "elite_mean": float(elite_scores.mean()),
                "frac_zero": float((it.scores <= 0).mean()),
            }
            for name, vals in it.breakdown.items():
                row[f"{name}_mean"] = float(vals.mean())
            rows.append(row)
        return rows


class DebugESPlanner(DiffusionESPlanner):
    """``DiffusionESPlanner`` (B=1) with pluggable reward/acquisition and full tracing.

    With ``reward=production_reward()`` and ``acquisition=topk`` the algorithm is the
    same as the production planner: elites are kept in the pool (elitism), the final
    plan is the argmax of the last pool.
    """

    def __init__(
        self,
        policy,
        preprocess_cfg,
        *,
        population_size: int = 64,
        elite_size: int = 8,
        num_iterations: int = 4,
        resample_timesteps: int = 5,
        other_agent_forecast: str = "wm",
        world_model_forecaster: Any = None,
        reward: Reward | None = None,
        acquisition: Acquisition = topk,
        initial_pool: InitialPool = standard_pool,
        resample_noise_scale: float | None = None,
        es_mutation: str = "diffusion",
        kinematic_step: float = 0.2,
        label: str = "baseline",
    ) -> None:
        super().__init__(
            policy=policy,
            preprocess_cfg=preprocess_cfg,
            population_size=population_size,
            num_worlds=1,
            metrics=["collision", "offroad"],
            resample_timesteps=resample_timesteps,
            elite_size=elite_size,
            num_iterations=num_iterations,
            other_agent_forecast=other_agent_forecast,
            world_model_forecaster=world_model_forecaster,
        )
        self.reward = reward if reward is not None else production_reward()
        self.acquisition = acquisition
        self.initial_pool = initial_pool
        # None = production (policy default noise_scale=2.0); e.g. 1.0 = the schedule's noise.
        self.resample_noise_scale = resample_noise_scale
        # 'diffusion' = production (re-noise + denoise every elite); 'hybrid' = elites that come from a
        # kinematic sampler (mcmc_pool) are mutated with a pCN step of size kinematic_step in its z-space.
        self.es_mutation = es_mutation
        self.kinematic_step = float(kinematic_step)
        self.label = label
        self.last_trace: ESTrace | None = None
        self.last_pool_info: dict[str, Any] | None = None  # set by samplers that have diagnostics (mcmc_pool)
        self._scenario: Scenario | None = None
        # Kept so variant() can rebuild when jitted closures must change.
        self._init_kwargs = dict(
            population_size=population_size,
            elite_size=elite_size,
            num_iterations=num_iterations,
            resample_timesteps=resample_timesteps,
            other_agent_forecast=other_agent_forecast,
            world_model_forecaster=world_model_forecaster,
            reward=self.reward,
            acquisition=acquisition,
            initial_pool=initial_pool,
            resample_noise_scale=resample_noise_scale,
            es_mutation=es_mutation,
            kinematic_step=kinematic_step,
            label=label,
        )

    @property
    def scorer(self):
        return self.scorers[0]

    def prepare(self, scenario: Scenario) -> None:
        """Attach the scenario's lane graph to the scorer (as ``runner.py`` does)."""
        if self._scenario is scenario:
            return
        self.scorer.reset()
        if scenario.lane_graph is not None:
            self.scorer.set_lane_graph(scenario.lane_graph)
            scorer_helpers.update_lane_points(self.scorer, scenario.sim_state, 0)
        self._scenario = scenario

    def variant(self, **overrides) -> "DebugESPlanner":
        """Copy with some settings changed, e.g. ``variant(reward=..., acquisition=..., initial_pool=..., label=...)``.

        Changing ``population_size`` or ``resample_timesteps`` rebuilds (and re-jits);
        everything else reuses the compiled functions.
        """
        unknown = set(overrides) - set(self._init_kwargs)
        if unknown:
            raise TypeError(f"Unknown overrides: {sorted(unknown)}")
        kwargs = {**self._init_kwargs, **overrides}
        needs_rebuild = any(
            k in overrides and overrides[k] != self._init_kwargs[k]
            for k in ("population_size", "resample_timesteps", "other_agent_forecast", "world_model_forecaster")
        )
        if needs_rebuild:
            new = DebugESPlanner(self.policy, self.preprocess_cfg, **kwargs)
            if self._scenario is not None:
                new.prepare(self._scenario)
            return new
        new = copy.copy(self)
        new.elite_size = int(kwargs["elite_size"])
        new.num_iterations = int(kwargs["num_iterations"])
        new.reward = kwargs["reward"]
        new.acquisition = kwargs["acquisition"]
        new.initial_pool = kwargs["initial_pool"]
        new.resample_noise_scale = kwargs["resample_noise_scale"]
        new.es_mutation = kwargs["es_mutation"]
        new.kinematic_step = float(kwargs["kinematic_step"])
        new.label = kwargs["label"]
        new.last_trace = None
        new._init_kwargs = kwargs
        return new

    def _mutate_kinematic(self, space, z, np_rng, tries: int = 8):
        """pCN step z' = sqrt(1 - b^2) z + b xi, redrawn until feasible (else keep z)."""
        b = self.kinematic_step
        out = z.copy()
        todo = np.ones(len(z), dtype=bool)
        for _ in range(tries):
            prop = np.sqrt(1.0 - b**2) * z[todo] + b * np_rng.standard_normal((int(todo.sum()), z.shape[1]))
            ok = space.feasible(prop)
            idx = np.flatnonzero(todo)
            out[idx[ok]] = prop[ok]
            todo[idx[ok]] = False
            if not todo.any():
                break
        return out

    def _score(self, traj_pt5, sim_state, timestep, goal_xy):
        ctx = build_reward_context(
            traj_pt5, sim_state=sim_state, timestep=timestep, scorer=self.scorer, goal_xy=goal_xy
        )
        return self.reward(ctx)

    def plan_trajectory(
        self,
        sim_state,
        goal,
        *,
        rng: jax.Array,
        instruction=None,
        instruction_mask=None,
        timestep: int = 0,
        mask_goal: bool = False,
        **kwargs: Any,
    ) -> PlannerResult:
        del instruction, instruction_mask, kwargs
        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            anchor_step_override=timestep,
            goal_step_override=90,
            goal_xy_override=goal,
        )
        features = pre_batch.features
        instruction_mask = jnp.zeros((1,), dtype=jnp.float32)
        features["inst_features"] = jnp.zeros((1, self.preprocess_cfg.inst_dim), dtype=jnp.float32)
        features["inst_valid"] = instruction_mask
        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
        features["subgoal_xy"] = jnp.zeros((1, 2), dtype=jnp.float32)
        features["subgoal_valid"] = jnp.zeros((1,), dtype=jnp.float32)
        cond_bf, inst_cond_bf = self._compute_condition_jit(features)

        goal_xy = np.asarray(goal).reshape(-1, 2)[0]
        np_rng = np.random.default_rng(int(timestep))

        self.last_pool_info = None
        norm_bktd = self.initial_pool(
            self,
            cond_bf=cond_bf,
            inst_cond_bf=inst_cond_bf,
            inst_cond_mask_bf=instruction_mask,
            features=features,
            pre_batch=pre_batch,
            rng=key_sample,
            sim_state=sim_state,
            timestep=int(timestep),
            goal_xy=goal_xy,
        )
        world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(norm_bktd, pre_batch)
        pool_norm = norm_bktd[0]
        pool_world = np.asarray(world_bkt5[0])
        pool_scores, pool_breakdown = self._score(pool_world, sim_state, timestep, goal_xy)
        carried = np.zeros((len(pool_scores),), dtype=bool)
        origin = np.arange(len(pool_scores))
        info = self.last_pool_info or {}
        space = info.get("space") if self.es_mutation == "hybrid" else None
        if space is not None:  # z of every pool member, NaN for diffusion samples
            pool_z = np.full((len(pool_scores), space.dim), np.nan)
            pool_z[info["n_keep"]:] = info["z"]
        # Same elite -> slot mapping as _replicate_elites.
        n_el, n_pop = self.elite_size, self.population_size
        replica_of = np.repeat(np.arange(n_el), [n_pop // n_el + (i < n_pop % n_el) for i in range(n_el)])

        iterations: list[ESIteration] = []
        for it in range(self.num_iterations + 1):
            elite_idx = np.asarray(
                self.acquisition(pool_scores, pool_world, self.elite_size, np_rng), dtype=np.int32
            )
            iterations.append(
                ESIteration(
                    iteration=it,
                    trajectories=pool_world,
                    scores=pool_scores,
                    breakdown=pool_breakdown,
                    is_carried_elite=carried,
                    elite_indices=elite_idx,
                    origin=origin,
                )
            )
            if it == self.num_iterations:
                break

            elite_norm = pool_norm[elite_idx][None]  # [1, E, T, D]
            replicated = self._replicate_elites(elite_norm, population_size=self.population_size)
            rng, key_resample = jax.random.split(rng)
            resample_kw = dict(
                cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=instruction_mask,
                proposals_bktd=replicated, rng=key_resample,
            )
            if self.resample_noise_scale is None:
                new_norm = self._resample_population_jit(**resample_kw)
            else:
                new_norm = denoise_proposals(
                    self, **resample_kw, steps=self.resample_timesteps, noise_scale=self.resample_noise_scale
                )
            if space is not None:
                child_z = pool_z[elite_idx][replica_of]
                kin = ~np.isnan(child_z[:, 0])
                if kin.any():
                    child_z[kin] = self._mutate_kinematic(space, child_z[kin], np_rng)
                    kin_norm = space.to_norm(child_z[kin], self.preprocess_cfg, np.asarray(replicated[0, kin, :, 5:]))
                    new_norm = new_norm.at[0, np.flatnonzero(kin)].set(kin_norm[0])
                pool_z = np.concatenate([pool_z[elite_idx], child_z], axis=0)
            new_world = np.asarray(self._postprocess_population(new_norm, pre_batch)[0][0])
            new_scores, new_breakdown = self._score(new_world, sim_state, timestep, goal_xy)

            pool_norm = jnp.concatenate([pool_norm[elite_idx], new_norm[0]], axis=0)
            pool_world = np.concatenate([pool_world[elite_idx], new_world], axis=0)
            pool_scores = np.concatenate([pool_scores[elite_idx], new_scores], axis=0)
            pool_breakdown = {
                k: np.concatenate([pool_breakdown[k][elite_idx], new_breakdown[k]], axis=0)
                for k in new_breakdown
            }
            if hasattr(self.reward, "rank"):  # set-relative rewards: re-rank the merged pool
                pool_scores = np.asarray(self.reward.rank(pool_breakdown), dtype=np.float32)
            carried = np.concatenate(
                [np.ones((len(elite_idx),), dtype=bool), np.zeros((len(new_scores),), dtype=bool)]
            )
            origin = np.concatenate([origin[elite_idx], origin[elite_idx][replica_of]])

        best = int(np.argmax(pool_scores))
        best_world = pool_world[best]

        wm_forecast, best_debug = None, None
        wm = self.world_model_forecaster
        if wm is not None and hasattr(wm, "forecast_modes"):
            wm_forecast = wm.forecast_modes(
                sim_state, timestep=int(timestep), traj_length=int(best_world.shape[0]), world_idx=0
            )
            wm_forecast = {k: np.asarray(v) for k, v in wm_forecast.items()}
            ctx = build_reward_context(
                best_world[None], sim_state=sim_state, timestep=timestep, scorer=self.scorer, goal_xy=goal_xy
            )
            _, best_debug = wm.score_independent_mode_collision(
                np.asarray(ctx.ego_boxes, dtype=np.float32), world_idx=0
            )
            # Strip the leading E=1 candidate axis; per-agent fields like top1_mode have none.
            best_debug = {
                k: (np.asarray(v)[0] if np.ndim(v) and np.shape(v)[0] == 1 else v)
                for k, v in best_debug.items()
            }

        self.last_trace = ESTrace(
            timestep=int(timestep),
            iterations=iterations,
            best_index=best,
            best_trajectory=best_world,
            best_score=float(pool_scores[best]),
            wm_forecast=wm_forecast,
            best_collision_debug=best_debug,
            label=self.label,
            n_diffusion_slots=(self.last_pool_info or {}).get("n_keep", self.population_size),
            pool_info=self.last_pool_info,
        )
        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=pool_norm[best][None],
            trajectory_world_bt5=jnp.asarray(best_world)[None],
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=dict(pre_batch.aux),
        )


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ClosedLoopResult:
    scenario: Scenario
    label: str
    replan_steps: list[int]
    snapshots: dict[int, Any]  # replan step -> sim_state the planner saw at that step
    traces: dict[int, ESTrace]
    plans: dict[int, np.ndarray]  # replan step -> chosen [T, 5] plan
    planned_state: Any  # log with ego replaced by executed plans (B=1)
    rollout_traj: Any  # Waymax Trajectory [N, T] after the evaluation rollout (log or IDM agents)
    overlap_t: np.ndarray  # [T_total] bool, from rollout metrics
    offroad_t: np.ndarray
    goal_reached: bool
    reached_step: int
    min_goal_distance_m: float
    overlap: bool
    offroad: bool
    success: bool
    start_timestep: int

    def report(self) -> str:
        first = lambda a: int(np.flatnonzero(a)[0]) if a.any() else None  # noqa: E731
        return (
            f"[{self.label}] {self.scenario.name}: success={self.success} "
            f"goal_reached={self.goal_reached} (step {self.reached_step}, min dist {self.min_goal_distance_m:.1f} m) "
            f"overlap={self.overlap} (first t={first(self.overlap_t)}) "
            f"offroad={self.offroad} (first t={first(self.offroad_t)})"
        )

    def anchor_for(self, t: int) -> int:
        """Latest replan step <= t (the plan that was being executed at time t)."""
        steps = [s for s in self.replan_steps if s <= int(t)]
        if not steps:
            raise ValueError(f"No replan step <= {t}; first replan is {self.replan_steps[0]}.")
        return steps[-1]


def run_closed_loop(
    planner: DebugESPlanner,
    scenario: Scenario,
    *,
    start_timestep: int = 10,
    replan_interval_steps: int = 10,
    sim_agents: str = "log",
    seed: int = 0,
    resume_from: ClosedLoopResult | None = None,
    resume_at: int | None = None,
    verbose: bool = True,
) -> ClosedLoopResult:
    """Periodic-replan rollout identical to ``runner.run`` for one scenario.

    ``resume_from`` / ``resume_at``: keep the plans of an earlier result before
    ``resume_at`` (a replan step) and only use ``planner`` from there on -- answers
    "would a different ES config have recovered from this state?".
    """
    planner.prepare(scenario)
    sim_state = scenario.sim_state
    goal_xy_b2 = scenario.goal_xy_b2
    num_steps = int(sim_state.log_trajectory.x.shape[-1])
    rng_key = jax.random.PRNGKey(int(seed))
    all_steps = list(range(start_timestep, num_steps - 1, replan_interval_steps))

    snapshots: dict[int, Any] = {}
    traces: dict[int, ESTrace] = {}
    plans: dict[int, np.ndarray] = {}
    state = sim_state
    steps = all_steps
    if resume_from is not None:
        if resume_at not in resume_from.snapshots:
            raise ValueError(f"resume_at must be one of {resume_from.replan_steps}, got {resume_at}.")
        for s in resume_from.replan_steps:
            if s < resume_at:
                snapshots[s] = resume_from.snapshots[s]
                traces[s] = resume_from.traces[s]
                plans[s] = resume_from.plans[s]
        state = resume_from.snapshots[resume_at]
        steps = [s for s in all_steps if s >= resume_at]

    plan_key = rng_key
    for s in all_steps:
        plan_key, key_plan = jax.random.split(plan_key)
        if s not in steps:
            continue  # keep the RNG stream aligned with a from-scratch run
        snapshots[s] = state
        result = planner.plan_trajectory(state, goal_xy_b2, rng=key_plan, timestep=s, mask_goal=False)
        traces[s] = planner.last_trace
        plans[s] = np.asarray(result.trajectory_world_bt5[0])
        state = apply_ego_replacements_to_expanded_state(
            state, start_t_b=s, traj_bt5=np.asarray(result.trajectory_world_bt5, dtype=np.float32)
        )
        if verbose:
            print(f"  t={s:3d}  best score={traces[s].best_score:.3f}", flush=True)

    # Evaluation, mirroring runner.run.
    env_obj, actor = _build_rollout_env_and_actor(
        ("overlap", "offroad"), int(state.log_trajectory.x.shape[1]), sim_agents
    )
    rollout_out = _run_batched_rollout(state, env_obj, actor, rng_key, num_steps - 1)
    overlap_full = np.asarray(rollout_out.metrics["overlap"].value)[:, 0] > 0  # [rollout steps]
    offroad_full = np.asarray(rollout_out.metrics["offroad"].value)[:, 0] > 0
    rollout_t = np.asarray(rollout_out.state.timestep)
    rollout_t = rollout_t.reshape(rollout_t.shape[0], -1)[:, 0]
    n_aligned = min(len(rollout_t), len(overlap_full))
    rollout_t, overlap_al, offroad_al = rollout_t[:n_aligned], overlap_full[:n_aligned], offroad_full[:n_aligned]
    final_state = jax.tree_util.tree_map(lambda x: x[-1], rollout_out.state)
    rollout_traj = viz_module._index_pytree(final_state.sim_trajectory, 0)

    goal_eval = check_goal_reaching(state, goal_xy_b2, start_timestep=start_timestep)
    reached = bool(goal_eval["reached"][0])
    reached_step = int(goal_eval["reached_step"][0])
    # Same episode rule as runner.run: failures in [start_timestep, goal step] by absolute timestep.
    overlap, offroad_flag = (
        bool(
            episode_failure_flags(
                flags[:, None], rollout_t,
                start_timestep=start_timestep, goal_reached_b=[reached], goal_reached_step_b=[reached_step],
                num_log_steps=num_steps,
            )[0]
        )
        for flags in (overlap_al, offroad_al)
    )

    # Per absolute sim timestep, for plotting (rollout index may be offset by env init steps).
    overlap_t = np.zeros((num_steps,), dtype=bool)
    offroad_t = np.zeros((num_steps,), dtype=bool)
    in_range = (rollout_t >= 0) & (rollout_t < num_steps)
    overlap_t[rollout_t[in_range]] = overlap_al[in_range]
    offroad_t[rollout_t[in_range]] = offroad_al[in_range]
    overlap_t[:start_timestep] = False
    offroad_t[:start_timestep] = False

    out = ClosedLoopResult(
        scenario=scenario,
        label=planner.label,
        replan_steps=sorted(snapshots),
        snapshots=snapshots,
        traces=traces,
        plans=plans,
        planned_state=state,
        rollout_traj=rollout_traj,
        overlap_t=overlap_t,
        offroad_t=offroad_t,
        goal_reached=reached,
        reached_step=reached_step,
        min_goal_distance_m=float(goal_eval["min_goal_distance_m"][0]),
        overlap=overlap,
        offroad=offroad_flag,
        success=reached and not overlap and not offroad_flag,
        start_timestep=start_timestep,
    )
    if verbose:
        print(out.report())
    return out


def replan_at(planner: DebugESPlanner, result: ClosedLoopResult, anchor: int, *, seed: int = 0) -> ESTrace:
    """Re-run one ES planning call on the exact state the baseline saw at ``anchor``."""
    planner.prepare(result.scenario)
    planner.plan_trajectory(
        result.snapshots[anchor],
        result.scenario.goal_xy_b2,
        rng=jax.random.PRNGKey(int(seed) * 1000 + int(anchor)),
        timestep=int(anchor),
    )
    return planner.last_trace


# ---------------------------------------------------------------------------
# Pool diversity diagnostics
# ---------------------------------------------------------------------------


def first_failure_t(result: ClosedLoopResult) -> int | None:
    """First absolute timestep with overlap or offroad in the evaluation rollout."""
    bad = np.flatnonzero(result.overlap_t | result.offroad_t)
    return int(bad[0]) if bad.size else None


def pool_stats(trace: ESTrace, iteration: int = 0, *, safe_collision: float = 0.99) -> dict[str, float]:
    """Is there a safe plan in the pool, and how spread out is it?

    safe = collision term >= ``safe_collision`` (no WM mode hits) and on-road (offroad term > 0).
    Spread is over world-frame trajectories: mean pairwise endpoint distance, endpoint
    std across / along the chosen plan's final heading, and final-speed std.
    """
    it = trace.iterations[iteration]
    coll = it.breakdown.get("collision", np.ones_like(it.scores))
    off = it.breakdown.get("offroad", np.ones_like(it.scores))
    safe = (coll >= safe_collision) & (off > 0)
    traj = np.asarray(it.trajectories)
    end = traj[:, -1, :2]
    pair = np.linalg.norm(end[:, None] - end[None], axis=-1)
    yaw = float(trace.best_trajectory[-1, 2])
    along = (end - end.mean(0)) @ np.array([np.cos(yaw), np.sin(yaw)])
    across = (end - end.mean(0)) @ np.array([-np.sin(yaw), np.cos(yaw)])
    speed = np.linalg.norm(traj[:, -1, 3:5], axis=-1)
    dyaw = np.rad2deg(np.angle(np.exp(1j * (traj[:, -1, 2] - traj[:, 0, 2]))))
    return {
        "iter": it.iteration,
        "pool": len(it.scores),
        "n_safe": int(safe.sum()),
        "frac_safe": float(safe.mean()),
        "frac_collision_free": float((coll >= safe_collision).mean()),
        "frac_onroad": float((off > 0).mean()),
        "best_score": float(it.scores.max()),
        "endpoint_pairwise_m": float(pair[np.triu_indices(len(end), 1)].mean()),
        "endpoint_std_across_m": float(across.std()),
        "endpoint_std_along_m": float(along.std()),
        "final_speed_std_mps": float(speed.std()),
        "heading_change_std_deg": float(dyaw.std()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

EGO_COLOR = "#1a9850"
AGENT_COLOR = "#8c8c8c"
HIT_COLOR = "#d73027"
PLAN_COLOR = "#2c7fb8"


def _box_corners(x, y, yaw, length, width) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([1, 1, -1, -1]) * length / 2
    dy = np.array([1, -1, -1, 1]) * width / 2
    return np.stack([x + dx * c - dy * s, y + dx * s + dy * c], axis=-1)


def _draw_boxes(ax, traj, t, ego_idx, *, highlight=(), ego_hit=False, labels=False, alpha=1.0):
    """traj: Waymax Trajectory [N, T] (single scenario). Returns created artists."""
    artists = []
    valid = np.asarray(traj.valid[:, t])
    x, y, yaw = np.asarray(traj.x[:, t]), np.asarray(traj.y[:, t]), np.asarray(traj.yaw[:, t])
    length, width = np.asarray(traj.length[:, t]), np.asarray(traj.width[:, t])
    for i in np.flatnonzero(valid):
        if i == ego_idx:
            color = HIT_COLOR if ego_hit else EGO_COLOR
        elif i in highlight:
            color = "#fc8d59"
        else:
            color = AGENT_COLOR
        poly = Polygon(
            _box_corners(x[i], y[i], yaw[i], length[i], width[i]),
            closed=True, fc=color, ec="k", lw=0.5, alpha=alpha, zorder=6,
        )
        ax.add_patch(poly)
        artists.append(poly)
        if labels and i != ego_idx:
            artists.append(ax.text(x[i], y[i], str(i), fontsize=6, ha="center", va="center", zorder=7))
    return artists


def _single_state(sim_state):
    return viz_module._index_pytree(sim_state, 0)


def _ego_xy(traj, ego_idx, t):
    return float(traj.x[ego_idx, t]), float(traj.y[ego_idx, t])


def _set_view(ax, center, view_m):
    ax.set_xlim(center[0] - view_m, center[0] + view_m)
    ax.set_ylim(center[1] - view_m, center[1] + view_m)
    ax.set_aspect("equal")


def _base_axes(ax, sim_state):
    viz_module.plot_roadgraph_points(ax, _single_state(sim_state).roadgraph_points)
    ax.set_xticks([])
    ax.set_yticks([])


def animate_closed_loop(
    result: ClosedLoopResult,
    *,
    view_m: float = 40.0,
    show_plans: bool = True,
    show_elites: bool = False,
    show_ids: bool = False,
    t_range: tuple[int, int] | None = None,
    fps: int = 10,
    figsize=(7, 7),
):
    """Returns an ``IPython.display.HTML`` scrub-able animation of the rollout."""
    from IPython.display import HTML

    scen = result.scenario
    traj = result.rollout_traj
    ego = scen.ego_idx
    num_steps = int(traj.x.shape[-1])
    t0, t1 = t_range if t_range is not None else (0, num_steps - 1)
    ego_speed = np.hypot(np.asarray(traj.vel_x[ego]), np.asarray(traj.vel_y[ego]))

    fig, ax = plt.subplots(figsize=figsize)
    _base_axes(ax, scen.sim_state)
    gx, gy = np.asarray(scen.goal_xy_b2[0])
    ax.plot(gx, gy, marker="*", ms=14, color="gold", mec="k", zorder=8)
    dynamic: list[Any] = []

    def draw(t):
        for a in dynamic:
            a.remove()
        dynamic.clear()
        hit = bool(result.overlap_t[t] or result.offroad_t[t])
        dynamic.extend(_draw_boxes(ax, traj, t, ego, ego_hit=hit, labels=show_ids))
        ex = np.asarray(traj.x[ego, : t + 1])
        ey = np.asarray(traj.y[ego, : t + 1])
        dynamic.extend(ax.plot(ex, ey, color=EGO_COLOR, lw=1.5, zorder=5))
        if show_plans and t >= result.replan_steps[0]:
            anchor = result.anchor_for(t)
            if show_elites:
                last = result.traces[anchor].iterations[-1]
                for k in last.elite_indices:
                    xy = last.trajectories[k]
                    dynamic.extend(ax.plot(xy[:, 0], xy[:, 1], color=PLAN_COLOR, lw=0.6, alpha=0.35, zorder=4))
            plan = result.plans[anchor]
            dynamic.extend(ax.plot(plan[:, 0], plan[:, 1], color=PLAN_COLOR, lw=2.0, ls="--", zorder=5))
        _set_view(ax, _ego_xy(traj, ego, t), view_m)
        flags = []
        if result.overlap_t[t]:
            flags.append("OVERLAP")
        if result.offroad_t[t]:
            flags.append("OFFROAD")
        anchor_txt = f" plan@{result.anchor_for(t)}" if t >= result.replan_steps[0] else ""
        ax.set_title(
            f"[{result.label}] t={t} ({t / 10:.1f}s) v={ego_speed[t]:.1f} m/s{anchor_txt} {' '.join(flags)}",
            color=HIT_COLOR if flags else "k",
            fontsize=10,
        )
        return dynamic

    anim = animation.FuncAnimation(fig, draw, frames=range(t0, t1 + 1), interval=1000 / fps, blit=False)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def plot_closed_loop_replans(result: ClosedLoopResult, *, view_m: float = 40.0, cols: int = 4, size: float = 4.0):
    """One panel per replan: agents (evaluation rollout) at the replan time, the ego's executed path
    over the next replan interval (green), the chosen plan (red), the last ES elites (blue) and
    the goal (star). Title: speed, plan source, chosen plan's safety level / progress, failures in the interval."""
    steps = result.replan_steps
    rows = int(np.ceil(len(steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    traj, ego = result.rollout_traj, result.scenario.ego_idx
    v = np.hypot(np.asarray(traj.vel_x[ego]), np.asarray(traj.vel_y[ego]))
    gx, gy = np.asarray(result.scenario.goal_xy_b2[0])
    for n, (ax, s) in enumerate(zip(axes.ravel(), steps)):
        nxt = steps[n + 1] if n + 1 < len(steps) else len(v) - 1
        _base_axes(ax, result.scenario.sim_state)
        _draw_boxes(ax, traj, s, ego, ego_hit=bool((result.overlap_t[s:nxt + 1] | result.offroad_t[s:nxt + 1]).any()))
        tr = result.traces[s]
        last = tr.iterations[-1]
        for k in last.elite_indices:
            ax.plot(last.trajectories[k][:, 0], last.trajectories[k][:, 1], color=PLAN_COLOR, lw=0.7, alpha=0.5, zorder=4)
        plan = result.plans[s]
        ax.plot(plan[:, 0], plan[:, 1], color=HIT_COLOR, lw=1.8, zorder=6)
        ax.plot(np.asarray(traj.x[ego, s:nxt + 1]), np.asarray(traj.y[ego, s:nxt + 1]), color=EGO_COLOR, lw=2.5, zorder=7)
        ax.plot(gx, gy, marker="*", ms=12, color="gold", mec="k", zorder=8)
        _set_view(ax, _ego_xy(traj, ego, s), view_m)
        b = last.breakdown
        terms = " ".join(f"{k[:4]}={float(b[k][tr.best_index]):.2f}" for k in ("collision", "offroad", "progress", "safety_level") if k in b)
        flags = [f for f, a in (("OVERLAP", result.overlap_t), ("OFFROAD", result.offroad_t)) if a[s:nxt + 1].any()]
        ax.set_title(f"t={s} v={v[s]:.1f} [{tr.source()}] {' '.join(flags)}\n{terms}", fontsize=8,
                     color=HIT_COLOR if flags else "k")
    for ax in axes.ravel()[len(steps):]:
        ax.axis("off")
    fig.suptitle(f"[{result.label}] {result.report()}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_speed_profile(results: list[ClosedLoopResult], *, t_range: tuple[int, int] | None = None, figsize=(12, 7)):
    """Executed ego speed / acceleration / jerk per run (0.1 s steps), with each replan's planned speed
    (thin) and replan markers: o = chosen plan descends from a diffusion sample, ^ = from the sampler
    (MCMC chain). A speed jump at a replan where the planned profile itself jumps = plan switch."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    for n, r in enumerate(results):
        color = plt.cm.tab10(n % 10)
        ego = r.scenario.ego_idx
        v = np.hypot(np.asarray(r.rollout_traj.vel_x[ego]), np.asarray(r.rollout_traj.vel_y[ego]))
        valid = np.asarray(r.rollout_traj.valid[ego])
        t = np.arange(len(v))
        sel = valid & (t >= r.start_timestep)
        a = np.gradient(v, 0.1)
        j = np.gradient(a, 0.1)
        axes[0].plot(t[sel], v[sel], color=color, lw=1.8, label=r.label)
        axes[1].plot(t[sel], a[sel], color=color, lw=1.2)
        axes[2].plot(t[sel], j[sel], color=color, lw=1.0)
        for s in r.replan_steps:
            plan = r.plans[s]
            pv = np.hypot(plan[:, 3], plan[:, 4])
            axes[0].plot(s + np.arange(len(pv)), pv, color=color, lw=0.6, alpha=0.5)
            mk = "^" if r.traces[s].source() == "sampler" else "o"
            axes[0].plot(s, v[s], mk, color=color, ms=6, mec="k", mew=0.5)
        fail = r.overlap_t | r.offroad_t
        if fail.any():
            axes[0].axvline(int(np.flatnonzero(fail)[0]), color=color, ls="--", lw=1)
    axes[0].set(ylabel="speed [m/s]", title="executed (thick) and planned at each replan (thin); o diffusion / ^ sampler; dashed = first failure")
    axes[1].set(ylabel="accel [m/s²]")
    axes[2].set(ylabel="jerk [m/s³]", xlabel="timestep (0.1 s)")
    for ax in axes:
        ax.grid(alpha=0.3)
        for s in results[0].replan_steps:
            ax.axvline(s, color="k", lw=0.3, alpha=0.4)
        if t_range is not None:
            ax.set_xlim(*t_range)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_timeline(result: ClosedLoopResult, *, compare: list[ClosedLoopResult] = (), figsize=(11, 5.5)):
    """Ego speed, failure flags and ES best/mean score per replan step -- for picking an anchor."""
    results = [result, *compare]
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for r in results:
        traj = r.rollout_traj
        ego = r.scenario.ego_idx
        speed = np.hypot(np.asarray(traj.vel_x[ego]), np.asarray(traj.vel_y[ego]))
        axes[0].plot(speed, label=f"{r.label} speed")
        steps = r.replan_steps
        axes[1].plot(steps, [r.traces[s].best_score for s in steps], marker="o", label=f"{r.label} best")
        axes[1].plot(
            steps, [r.traces[s].iterations[-1].scores.mean() for s in steps],
            ls=":", marker=".", label=f"{r.label} pool mean",
        )
    for ax in axes:
        for t in np.flatnonzero(result.overlap_t):
            ax.axvspan(t - 0.5, t + 0.5, color=HIT_COLOR, alpha=0.15, lw=0)
        for t in np.flatnonzero(result.offroad_t):
            ax.axvspan(t - 0.5, t + 0.5, color="orange", alpha=0.15, lw=0)
        for s in result.replan_steps:
            ax.axvline(s, color="k", alpha=0.08)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ego speed (m/s)")
    axes[1].set_ylabel("ES score")
    axes[1].set_xlabel("timestep (red = overlap, orange = offroad; vertical lines = replans)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_es_iterations(
    trace: ESTrace,
    sim_state,
    *,
    ego_idx: int,
    view_m: float = 35.0,
    show_wm_top1: bool = True,
    risk_threshold: float = 0.05,
    figsize_per=(4.2, 4.2),
):
    """One panel per ES iteration: pool coloured by score, elites bold, final best red.

    Agents shown at the replan time (the state the planner scored against). Orange = agents with
    WM collision probability >= ``risk_threshold`` against the chosen plan; their WM top-1 future is
    drawn solid orange, the dots on it and on the chosen plan mark the same instants (every 1 s).
    """
    n = len(trace.iterations)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    traj = _single_state(sim_state).log_trajectory
    t = trace.timestep
    dbg = trace.best_collision_debug or {}
    p_hit = np.asarray(dbg.get("p_collision_i", np.zeros(0)))
    risky = [int(i) for i in np.flatnonzero(p_hit >= risk_threshold)]
    for ax, it in zip(axes.ravel(), trace.iterations):
        _base_axes(ax, sim_state)
        _draw_boxes(ax, traj, t, ego_idx, highlight=risky)
        if show_wm_top1 and trace.wm_forecast is not None:
            top1 = trace.wm_forecast["top1_traj"]
            mask = trace.wm_forecast["masks"][:, 0]
            for i in np.flatnonzero(mask):
                if i in risky:
                    ax.plot(top1[i, :, 0], top1[i, :, 1], color="#fc8d59", lw=1.5, zorder=7)
                    ax.plot(top1[i, ::10, 0], top1[i, ::10, 1], "o", color="#fc8d59", ms=3, zorder=7)
                else:
                    ax.plot(top1[i, :, 0], top1[i, :, 1], color=AGENT_COLOR, lw=0.8, ls=":", zorder=3)
        order = np.argsort(it.scores)
        for k in order:
            xy = it.trajectories[k]
            ax.plot(xy[:, 0], xy[:, 1], color=cmap(float(np.clip(it.scores[k], 0, 1))), lw=0.7, alpha=0.6, zorder=4)
        for k in it.elite_indices:
            xy = it.trajectories[k]
            ax.plot(xy[:, 0], xy[:, 1], color=cmap(float(np.clip(it.scores[k], 0, 1))), lw=2.2, zorder=5)
        if it.iteration == trace.iterations[-1].iteration:
            b = trace.best_trajectory
            ax.plot(b[:, 0], b[:, 1], color=HIT_COLOR, lw=2.5, zorder=6)
            ax.plot(b[::10, 0], b[::10, 1], "o", color=HIT_COLOR, ms=3, zorder=7)
        _set_view(ax, _ego_xy(traj, ego_idx, t), view_m)
        ax.set_title(
            f"iter {it.iteration}: max={it.scores.max():.2f} mean={it.scores.mean():.2f} "
            f"zero={100 * (it.scores <= 0).mean():.0f}%",
            fontsize=9,
        )
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="score")
    last = trace.iterations[-1]
    terms = " ".join(f"{k}={float(v[trace.best_index]):.2f}" for k, v in last.breakdown.items())
    risk = ", ".join(f"{i}:{p_hit[i]:.2f}" for i in sorted(risky, key=lambda i: -p_hit[i])[:4]) or "none"
    fig.suptitle(
        f"[{trace.label}] ES at t={t} (bold = elites, red = chosen plan, dotted = WM top-1)\n"
        f"chosen ({trace.source()}): {terms} | P(hit) per agent: {risk}",
        fontsize=10,
    )
    return fig


def plot_score_evolution(traces: list[ESTrace], *, figsize=(12, 4)):
    """Score distribution per iteration (left) and per-term mean of elites (right)."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    width = 0.8 / max(len(traces), 1)
    for j, tr in enumerate(traces):
        xs = np.array([it.iteration for it in tr.iterations], dtype=float) + (j - (len(traces) - 1) / 2) * width
        for x, it in zip(xs, tr.iterations):
            jitter = np.random.default_rng(0).uniform(-width / 3, width / 3, size=len(it.scores))
            ax0.scatter(x + jitter, it.scores, s=6, alpha=0.4, color=f"C{j}")
        ax0.plot(xs, [it.scores.max() for it in tr.iterations], color=f"C{j}", marker="o", label=f"{tr.label} max")
        ax0.plot(
            xs, [it.scores[it.elite_indices].mean() for it in tr.iterations],
            color=f"C{j}", ls="--", label=f"{tr.label} elite mean",
        )
        for name in tr.iterations[0].breakdown:
            ax1.plot(
                [it.iteration for it in tr.iterations],
                [it.breakdown[name][it.elite_indices].mean() for it in tr.iterations],
                marker=".", label=f"{tr.label}: {name}",
                color=f"C{j}", ls=["-", "--", ":", "-."][list(tr.iterations[0].breakdown).index(name) % 4],
            )
    ax0.set_xlabel("ES iteration")
    ax0.set_ylabel("score")
    ax0.set_title("pool scores (dots), max and elite mean")
    ax0.legend(fontsize=7)
    ax1.set_xlabel("ES iteration")
    ax1.set_ylabel("term value (elite mean)")
    ax1.set_title("reward terms over iterations")
    ax1.legend(fontsize=7)
    for ax in (ax0, ax1):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def candidates_table(trace: ESTrace, iteration: int = -1, top: int | None = 20):
    """pandas DataFrame of one pool, sorted by score, with each reward term."""
    import pandas as pd

    it = trace.iterations[iteration]
    speed = np.hypot(it.trajectories[..., 3], it.trajectories[..., 4])
    df = pd.DataFrame(
        {
            "score": it.scores,
            **{k: v for k, v in it.breakdown.items()},
            "carried_elite": it.is_carried_elite,
            "selected_as_elite": np.isin(np.arange(len(it.scores)), it.elite_indices),
            "path_len_m": np.linalg.norm(np.diff(it.trajectories[:, :, :2], axis=1), axis=-1).sum(axis=1),
            "end_speed": speed[:, -1],
            "min_speed": speed.min(axis=1),
        }
    )
    df = df.sort_values("score", ascending=False)
    return df.head(top) if top else df


def risky_agents(trace: ESTrace, top: int = 8):
    """Agents with the highest WM collision probability against the chosen plan."""
    import pandas as pd

    dbg = trace.best_collision_debug
    if dbg is None or "p_collision_i" not in dbg:
        return None
    p = np.asarray(dbg["p_collision_i"])
    order = np.argsort(-p)[:top]
    probs = trace.wm_forecast["mode_probs"] if trace.wm_forecast is not None else None
    return pd.DataFrame(
        {
            "agent": order,
            "p_collision": p[order],
            "top1_mode": [int(np.argmax(probs[i])) if probs is not None else -1 for i in order],
            "top1_prob": [float(np.max(probs[i])) if probs is not None else np.nan for i in order],
        }
    )


def plot_wm_forecast(
    trace: ESTrace,
    sim_state,
    *,
    ego_idx: int,
    view_m: float = 35.0,
    min_mode_prob: float = 0.05,
    agents: list[int] | None = None,
    show_log_future: bool = True,
    figsize=(8, 8),
):
    """All K WM modes per agent (alpha ~ mode prob), logged future (black dashed),
    chosen ego plan. Agents ranked risky for the plan are highlighted with ids."""
    if trace.wm_forecast is None:
        raise ValueError("No WM forecast on this trace (other_agent_forecast != 'wm' or backend lacks forecast_modes).")
    wm = trace.wm_forecast
    traj_k, probs, masks = wm["traj_k"], wm["mode_probs"], wm["masks"]
    t = trace.timestep
    horizon = traj_k.shape[2]
    log = _single_state(sim_state).log_trajectory
    risk = None
    if trace.best_collision_debug is not None and "p_collision_i" in trace.best_collision_debug:
        risk = np.asarray(trace.best_collision_debug["p_collision_i"])
    risky = set(np.flatnonzero(risk > 0).tolist()) if risk is not None else set()

    fig, ax = plt.subplots(figsize=figsize)
    _base_axes(ax, sim_state)
    _draw_boxes(ax, log, t, ego_idx, highlight=risky, labels=True)
    candidates = np.flatnonzero(masks[:, 0]) if agents is None else np.asarray(agents)
    cmap = plt.get_cmap("tab10")
    for n_i, i in enumerate(candidates):
        color = cmap(n_i % 10)
        for k in range(traj_k.shape[1]):
            p = float(probs[i, k])
            if p < min_mode_prob:
                continue
            xy = traj_k[i, k]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.6 + 2.5 * p, alpha=0.25 + 0.75 * p, zorder=4)
        if show_log_future:
            end = min(t + horizon, int(log.x.shape[-1]))
            v = np.asarray(log.valid[i, t:end])
            ax.plot(np.asarray(log.x[i, t:end])[v], np.asarray(log.y[i, t:end])[v], "k--", lw=0.8, zorder=4)
        if i in risky:
            ax.annotate(
                f"p={risk[i]:.2f}", (float(log.x[i, t]), float(log.y[i, t])),
                xytext=(6, 6), textcoords="offset points", fontsize=8, color=HIT_COLOR, zorder=9,
            )
    b = trace.best_trajectory
    ax.plot(b[:, 0], b[:, 1], color=EGO_COLOR, lw=2.5, zorder=6, label="chosen ego plan")
    _set_view(ax, _ego_xy(log, ego_idx, t), view_m)
    ax.set_title(
        f"[{trace.label}] WM modes at t={t} (width/alpha ~ mode prob, black dashed = logged future, "
        f"orange = agents with p_collision>0)",
        fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=8)
    return fig


def animate_wm_forecast(
    trace: ESTrace,
    sim_state,
    *,
    ego_idx: int,
    candidate: np.ndarray | None = None,
    view_m: float = 35.0,
    stride: int = 2,
    figsize=(7, 7),
):
    """Animate over the planning horizon: ego candidate box vs WM top-1 boxes (coloured)
    vs logged agents (grey outline). Shows where/when the WM thinks a conflict is."""
    from IPython.display import HTML

    if trace.wm_forecast is None:
        raise ValueError("No WM forecast on this trace.")
    wm = trace.wm_forecast
    top1, masks = wm["top1_traj"], wm["masks"]
    ego_plan = trace.best_trajectory if candidate is None else np.asarray(candidate)
    log = _single_state(sim_state).log_trajectory
    t0 = trace.timestep
    horizon = min(top1.shape[1], ego_plan.shape[0])
    num_log = int(log.x.shape[-1])
    ego_len = float(log.length[ego_idx, t0])
    ego_wid = float(log.width[ego_idx, t0])

    fig, ax = plt.subplots(figsize=figsize)
    _base_axes(ax, sim_state)
    ax.plot(ego_plan[:horizon, 0], ego_plan[:horizon, 1], color=EGO_COLOR, lw=1.0, ls="--")
    _set_view(ax, _ego_xy(log, ego_idx, t0), view_m)
    dynamic: list[Any] = []

    def draw(h):
        for a in dynamic:
            a.remove()
        dynamic.clear()
        ego_poly = _box_corners(ego_plan[h, 0], ego_plan[h, 1], ego_plan[h, 2], ego_len, ego_wid)
        dynamic.append(ax.add_patch(Polygon(ego_poly, fc=EGO_COLOR, ec="k", lw=0.8, zorder=7)))
        for i in np.flatnonzero(masks[:, h]):
            x, y, yaw, length, width = top1[i, h]
            dynamic.append(ax.add_patch(Polygon(
                _box_corners(x, y, yaw, length, width), fc="#74add1", ec="k", lw=0.4, alpha=0.8, zorder=6,
            )))
            tl = t0 + h
            if tl < num_log and bool(log.valid[i, tl]):
                dynamic.append(ax.add_patch(Polygon(
                    _box_corners(float(log.x[i, tl]), float(log.y[i, tl]), float(log.yaw[i, tl]),
                                 float(log.length[i, tl]), float(log.width[i, tl])),
                    fill=False, ec="k", lw=0.8, ls="--", zorder=6,
                )))
        ax.set_title(f"[{trace.label}] t={t0}+{h} (blue = WM top-1, dashed = logged, green = ego plan)", fontsize=9)
        return dynamic

    anim = animation.FuncAnimation(fig, draw, frames=range(0, horizon, stride), interval=150, blit=False)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html
