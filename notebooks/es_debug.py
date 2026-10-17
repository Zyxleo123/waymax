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
import inspect
import os
import sys
import types
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
    road_tracked_offroad,
)
from simulation.planning_utils import apply_ego_replacements_to_expanded_state  # noqa: E402
from viz import viz as viz_module  # noqa: E402


def enable_compilation_cache(cache_dir: str | os.PathLike | None = None, min_compile_secs: float = 0.2) -> str:
    """Persist XLA compilations on disk so a restarted kernel / new job skips most of the ~2 min
    warm-up. Call before the first jitted call. Default dir: ``$JAX_CACHE_DIR`` or ``<repo>/../jax_cache``."""
    path = str(cache_dir or os.environ.get("JAX_CACHE_DIR") or REPO_ROOT.parent / "jax_cache")
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_secs))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return path


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
    mode_temperature: float = 1.0,
):
    """Loads the WM forecaster through the same code path as ``run_simulation.py``.

    ``mode_temperature``: scoring uses softmax(mode_logits / T) (multifuture only). It can
    also be changed later with ``wm.mode_temperature = T``; ``wm.mode_prob_stats()`` gives
    the mean / std of the mode probabilities seen so far.
    """
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
        wm_mode_temperature=mode_temperature,
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
    forecast: tuple | None = None,
) -> RewardContext:
    """Same preamble as ``Scorer.compute_score`` (forecast others, build ego boxes).

    ``forecast``: the ``scores.helpers.forecast_other_agents`` output for this state, timestep and
    horizon, if already computed (the planner reuses it across ES iterations)."""
    trajectories = jnp.asarray(trajectories)
    if scorer.lane_points is None:
        scorer_helpers.update_lane_points(scorer, sim_state, world_idx)
    if forecast is None:
        forecast = forecast_others(sim_state, timestep, int(trajectories.shape[1]), scorer, world_idx)
    obj, obj_mask, col_obj, col_mask = forecast
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


def forecast_others(sim_state, timestep: int, traj_length: int, scorer, world_idx: int = 0) -> tuple:
    """Other-agent forecast (obj, obj_mask, col_obj, col_mask) as the scorer would compute it."""
    return scorer_helpers.forecast_other_agents(
        sim_state,
        timestep=int(timestep),
        traj_length=int(traj_length),
        world_idx=world_idx,
        mode=scorer.other_agent_forecast,
        world_model_forecaster=scorer.world_model_forecaster,
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


def plan_jerk(traj_pt5: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """[P] mean squared jerk (longitudinal + lateral, m^2/s^6) of world-frame plans [P, T, 5]."""
    vel = traj_pt5[..., 3:5]
    acc = np.gradient(vel, dt, axis=1)
    jerk = np.gradient(acc, dt, axis=1)
    return (jerk**2).sum(-1).mean(-1)


# ---------------------------------------------------------------------------
# Acquisition (elite selection)
# ---------------------------------------------------------------------------

# An acquisition maps (scores [P], trajectories [P, T, 5], n_elite, rng) -> elite indices into the pool.
# If it also takes an ``info`` keyword it gets an AcquisitionInfo (reward terms, iteration, scene, ...).
# It may return any number >= 1 of indices; each elite is replicated over the next population of P.
Acquisition = Callable[..., np.ndarray]


@dataclasses.dataclass
class AcquisitionInfo:
    """Context passed as ``info=`` to acquisitions that accept it.

    Example::

        def lateral_spread(scores, trajectories, n_elite, rng, *, info):
            ok = info.breakdown["collision"] >= 0.99          # plans no WM mode hits
            ...
    """

    iteration: int  # pool being selected from: 0 = initial pool, ..., num_iterations = final pool
    num_iterations: int
    breakdown: dict[str, np.ndarray]  # reward term -> [P] (e.g. collision, offroad, offroad_tiebreak)
    is_carried_elite: np.ndarray  # [P] True for elites carried over from the previous iteration
    timestep: int
    goal_xy: np.ndarray  # [2] world frame
    sim_state: Any  # state the planner scores against (batched, B=1)
    ego_idx: int
    planner: Any  # the DebugESPlanner (e.g. planner.reward, planner.scorer)

    def ego_state(self) -> dict[str, float]:
        """x, y, yaw, speed of the ego at the replan time (world frame)."""
        log = _single_state(self.sim_state).log_trajectory
        i, t = self.ego_idx, self.timestep
        return {"x": float(log.x[i, t]), "y": float(log.y[i, t]), "yaw": float(log.yaw[i, t]),
                "speed": float(np.hypot(log.vel_x[i, t], log.vel_y[i, t]))}

    def to_ego_frame(self, trajectories: np.ndarray) -> np.ndarray:
        """[P, T, 2] xy in the ego frame at the replan time (x forward, y left)."""
        e = self.ego_state()
        d = np.asarray(trajectories)[..., :2] - np.asarray([e["x"], e["y"]])
        c, s_ = np.cos(e["yaw"]), np.sin(e["yaw"])
        return np.stack([d[..., 0] * c + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c], axis=-1)

    def agent_forecast(self) -> tuple[np.ndarray, np.ndarray]:
        """Other agents' forecast used for scoring: trajectories [N, T, 5] (x, y, yaw, length, width)
        and masks [N, T]. With the WM this is its top-1 future."""
        for (t, _), fc in self.planner._forecast.items():
            if t == self.timestep:
                return np.asarray(fc[0]), np.asarray(fc[1])
        raise RuntimeError("No forecast cached for this replan.")


def _takes_info(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "info" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _select_elites(acq, scores, trajectories, n_elite, rng, info: AcquisitionInfo) -> np.ndarray:
    out = acq(scores, trajectories, n_elite, rng, info=info) if _takes_info(acq) else acq(scores, trajectories, n_elite, rng)
    idx = np.asarray(out).astype(np.int64).reshape(-1)
    name = getattr(acq, "__name__", repr(acq))
    if idx.size == 0 or idx.min() < 0 or idx.max() >= len(scores):
        raise ValueError(f"acquisition {name} returned {idx.size} indices outside [0, {len(scores)})")
    return idx.astype(np.int32)


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


def _sample_many(planner, m: int):
    """Jitted: ``m`` i.i.d. diffusion samples for one scenario -> [1, m, T, D]."""
    cache = planner.__dict__.setdefault("_sample_many_jit_cache", {})
    if m not in cache:
        policy = planner.policy

        def f(cond_bf, inst_cond_bf, inst_cond_mask_bf, rng):
            out = policy.sample_from_condition(
                jnp.repeat(cond_bf, m, axis=0),
                jnp.repeat(inst_cond_bf, m, axis=0),
                jnp.repeat(inst_cond_mask_bf, m, axis=0),
                rng=rng,
            )
            return out[None]

        cache[m] = jax.jit(f)
    return cache[m]


def oversample_fps(factor: int = 4, num_waypoints: int = 4) -> InitialPool:
    """Draw ``factor * P`` samples and keep P by farthest-point sampling on world xy at
    ``num_waypoints`` evenly spaced horizon steps. Same samples as the model would give,
    just spread over its modes instead of concentrated on the dominant one."""

    def pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng, **_):
        del features
        norm = _sample_many(planner, factor * planner.population_size)(
            cond_bf, inst_cond_bf, inst_cond_mask_bf, rng
        )  # [1, factor*P, T, D], one batched call (the GPU is latency-bound at P=64)
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
        self.label = label
        self.last_trace: ESTrace | None = None
        self._scenario: Scenario | None = None
        self._forecast: dict[tuple, tuple] = {}  # (timestep, horizon) -> other-agent forecast, per replan
        cfg = preprocess_cfg
        # Preprocessing / postprocessing are pure JAX; jitted they take ~ms instead of ~300 / ~30 ms eager.
        self._preprocess_jit = jax.jit(
            lambda state, key, t, goal: preprocess_simulator_state(
                state, key, cfg, anchor_step_override=t, goal_step_override=90, goal_xy_override=goal,
                world_dt_method="median",
            )
        )
        self._postprocess_jit = jax.jit(
            lambda norm, aux: DiffusionESPlanner._postprocess_population(self, norm, types.SimpleNamespace(aux=aux))
        )
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
        new.label = kwargs["label"]
        new.last_trace = None
        new._init_kwargs = kwargs
        return new

    def _postprocess_population(self, proposals_bktd, pre_batch):
        return self._postprocess_jit(proposals_bktd, pre_batch.aux)

    def _context(self, traj_pt5, sim_state, timestep, goal_xy) -> RewardContext:
        key = (int(timestep), int(np.shape(traj_pt5)[1]))
        if key not in self._forecast:
            self._forecast[key] = forecast_others(sim_state, timestep, key[1], self.scorer)
        return build_reward_context(
            traj_pt5, sim_state=sim_state, timestep=timestep, scorer=self.scorer, goal_xy=goal_xy,
            forecast=self._forecast[key],
        )

    def _score(self, traj_pt5, sim_state, timestep, goal_xy):
        return self.reward(self._context(traj_pt5, sim_state, timestep, goal_xy))

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
        self._forecast = {}  # new state: forecasts are reused only within this call
        pre_batch, _ = self._preprocess_jit(sim_state, key_pre, jnp.int32(timestep), jnp.asarray(goal))
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
        ego_idx = int(scorer_helpers.get_ego_idx(sim_state, 0))

        iterations: list[ESIteration] = []
        for it in range(self.num_iterations + 1):
            info = AcquisitionInfo(
                iteration=it, num_iterations=self.num_iterations, breakdown=pool_breakdown,
                is_carried_elite=carried, timestep=int(timestep), goal_xy=goal_xy, sim_state=sim_state,
                ego_idx=ego_idx, planner=self,
            )
            elite_idx = _select_elites(self.acquisition, pool_scores, pool_world, self.elite_size, np_rng, info)
            iterations.append(
                ESIteration(
                    iteration=it,
                    trajectories=pool_world,
                    scores=pool_scores,
                    breakdown=pool_breakdown,
                    is_carried_elite=carried,
                    elite_indices=elite_idx,
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

        best = int(np.argmax(pool_scores))
        best_world = pool_world[best]

        wm_forecast, best_debug = None, None
        wm = self.world_model_forecaster
        if wm is not None and hasattr(wm, "forecast_modes"):
            wm_forecast = wm.forecast_modes(
                sim_state, timestep=int(timestep), traj_length=int(best_world.shape[0]), world_idx=0
            )
            wm_forecast = {k: np.asarray(v) for k, v in wm_forecast.items()}
            ctx = self._context(best_world[None], sim_state, timestep, goal_xy)
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
    offroad_t: np.ndarray  # ego box height following the road (as in rollout_predicted_trajectories_with_metrics)
    goal_reached: bool
    reached_step: int
    min_goal_distance_m: float
    overlap: bool
    offroad: bool
    success: bool
    start_timestep: int
    offroad_t_waymax: np.ndarray | None = None  # Waymax's own offroad flags (frozen ego height), for comparison

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
    offroad_waymax_al = offroad_al
    offroad_al = road_tracked_offroad(rollout_out, rollout_t)[:, 0]

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
    offroad_t_waymax = np.zeros((num_steps,), dtype=bool)
    in_range = (rollout_t >= 0) & (rollout_t < num_steps)
    overlap_t[rollout_t[in_range]] = overlap_al[in_range]
    offroad_t[rollout_t[in_range]] = offroad_al[in_range]
    offroad_t_waymax[rollout_t[in_range]] = offroad_waymax_al[in_range]
    overlap_t[:start_timestep] = False
    offroad_t[:start_timestep] = False
    offroad_t_waymax[:start_timestep] = False

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
        offroad_t_waymax=offroad_t_waymax,
    )
    if verbose:
        print(out.report())
    return out


def replan_at(planner: DebugESPlanner, result: ClosedLoopResult, anchor: int, *, seed: int = 0) -> ESTrace:
    """Re-run one ES planning call on the exact state the baseline saw at ``anchor``.

    Uses the RNG key ``run_closed_loop`` used at that replan (same ``seed``), so the result's own
    planner reproduces ``result.traces[anchor]`` and other planners start from the same random draws.
    """
    planner.prepare(result.scenario)
    key = jax.random.PRNGKey(int(seed))
    for s in result.replan_steps:  # same split sequence as run_closed_loop
        key, key_plan = jax.random.split(key)
        if s == anchor:
            break
    else:
        raise ValueError(f"anchor must be one of {result.replan_steps}, got {anchor}")
    planner.plan_trajectory(
        result.snapshots[anchor],
        result.scenario.goal_xy_b2,
        rng=key_plan,
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
        if labels:
            artists.append(ax.text(x[i], y[i], f"ego {i}" if i == ego_idx else str(i), fontsize=6,
                                   ha="center", va="center", zorder=7, clip_on=True))
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
    the goal (star). Title: speed, chosen plan's safety level / progress, failures in the interval."""
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
        ax.set_title(f"t={s} v={v[s]:.1f} {' '.join(flags)}\n{terms}", fontsize=8,
                     color=HIT_COLOR if flags else "k")
    for ax in axes.ravel()[len(steps):]:
        ax.axis("off")
    fig.suptitle(f"[{result.label}] {result.report()}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_speed_profile(results: list[ClosedLoopResult], *, t_range: tuple[int, int] | None = None, figsize=(12, 7)):
    """Executed ego speed / acceleration / jerk per run (0.1 s steps), with each replan's planned speed
    (thin) and replan markers (o). A speed jump at a replan where the planned profile itself jumps = plan switch."""
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
            axes[0].plot(s, v[s], "o", color=color, ms=6, mec="k", mew=0.5)
        fail = r.overlap_t | r.offroad_t
        if fail.any():
            axes[0].axvline(int(np.flatnonzero(fail)[0]), color=color, ls="--", lw=1)
    axes[0].set(ylabel="speed [m/s]", title="executed (thick) and planned at each replan (thin); dashed = first failure")
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


def _draw_es_iteration(ax, trace: ESTrace, it: ESIteration, sim_state, *, ego_idx: int, view_m: float,
                       show_wm_top1: bool, risky: list[int], cmap, labels: bool = False) -> None:
    """One ES pool on ``ax``: plans coloured by score, elites bold, chosen plan red (final pool only)."""
    traj = _single_state(sim_state).log_trajectory
    t = trace.timestep
    _base_axes(ax, sim_state)
    _draw_boxes(ax, traj, t, ego_idx, highlight=risky, labels=labels)
    if show_wm_top1 and trace.wm_forecast is not None:
        top1 = trace.wm_forecast["top1_traj"]
        mask = trace.wm_forecast["masks"][:, 0]
        for i in np.flatnonzero(mask):
            if i in risky:
                ax.plot(top1[i, :, 0], top1[i, :, 1], color="#fc8d59", lw=1.5, zorder=7)
                ax.plot(top1[i, ::10, 0], top1[i, ::10, 1], "o", color="#fc8d59", ms=3, zorder=7)
            else:
                ax.plot(top1[i, :, 0], top1[i, :, 1], color=AGENT_COLOR, lw=0.8, ls=":", zorder=3)
    for k in np.argsort(it.scores):
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


def _risky_agents(trace: ESTrace, risk_threshold: float) -> tuple[list[int], np.ndarray]:
    dbg = trace.best_collision_debug or {}
    p_hit = np.asarray(dbg.get("p_collision_i", np.zeros(0)))
    return [int(i) for i in np.flatnonzero(p_hit >= risk_threshold)], p_hit


def plot_es_iterations(
    trace: ESTrace,
    sim_state,
    *,
    ego_idx: int,
    view_m: float = 35.0,
    show_wm_top1: bool = True,
    risk_threshold: float = 0.05,
    labels: bool = False,
    figsize_per=(4.2, 4.2),
):
    """One panel per ES iteration: pool coloured by score, elites bold, final best red.

    Agents shown at the replan time (the state the planner scored against). Orange = agents with
    WM collision probability >= ``risk_threshold`` against the chosen plan; their WM top-1 future is
    drawn solid orange, the dots on it and on the chosen plan mark the same instants (every 1 s).
    ``labels=True`` writes agent ids on the boxes.
    """
    n = len(trace.iterations)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    risky, p_hit = _risky_agents(trace, risk_threshold)
    for ax, it in zip(axes.ravel(), trace.iterations):
        _draw_es_iteration(ax, trace, it, sim_state, ego_idx=ego_idx, view_m=view_m, show_wm_top1=show_wm_top1,
                           risky=risky, cmap=cmap, labels=labels)
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
        f"[{trace.label}] ES at t={trace.timestep} (bold = elites, red = chosen plan, dotted = WM top-1)\n"
        f"chosen: {terms} | P(hit) per agent: {risk}",
        fontsize=10,
    )
    return fig


def compare_acquisitions(
    planner: DebugESPlanner,
    result: ClosedLoopResult,
    anchor: int,
    acquisitions: dict[str, Acquisition],
    *,
    seed: int = 0,
    **overrides,
) -> dict[str, ESTrace]:
    """Re-plan at ``anchor`` once per acquisition, everything else equal (same initial pool, same RNG).

    ``overrides`` go to ``planner.variant`` (e.g. ``num_iterations=6``). Returns {name: ESTrace}; view
    with ``plot_acquisition_compare`` and ``acquisition_table``.
    """
    return {
        name: replan_at(planner.variant(label=name, acquisition=acq, **overrides), result, anchor, seed=seed)
        for name, acq in acquisitions.items()
    }


def acquisition_table(traces: dict[str, ESTrace], *, safe_collision: float = 0.99):
    """Per acquisition and iteration: best / elite scores, safe plans in the pool, elites that are safe,
    elite spread (mean pairwise endpoint distance, m) and number of distinct elites."""
    import pandas as pd

    rows = []
    for name, tr in traces.items():
        for it in tr.iterations:
            b = it.breakdown
            safe = (b["collision"] >= safe_collision) & (b.get("offroad", np.ones_like(it.scores)) > 0)
            e = np.unique(it.elite_indices)
            end = it.trajectories[e, -1, :2]
            pair = np.linalg.norm(end[:, None] - end[None], axis=-1)
            rows.append({
                "acquisition": name, "iter": it.iteration,
                "best": float(it.scores.max()), "elite_mean": float(it.scores[it.elite_indices].mean()),
                "n_safe_pool": int(safe.sum()), "n_safe_elites": int(safe[e].sum()),
                "elite_spread_m": float(pair[np.triu_indices(len(e), 1)].mean()) if len(e) > 1 else 0.0,
                "n_distinct_elites": int(len(e)),
            })
    return pd.DataFrame(rows).set_index(["acquisition", "iter"])


def plot_acquisition_compare(
    traces: dict[str, ESTrace],
    sim_state,
    *,
    ego_idx: int,
    iterations: list[int] | None = None,
    view_m: float = 35.0,
    show_wm_top1: bool = True,
    risk_threshold: float = 0.05,
    size: float = 3.6,
):
    """Grid: one row per acquisition, one column per ES iteration (default: all). Same drawing as
    ``plot_es_iterations``; the row label gives the chosen plan's terms."""
    names = list(traces)
    its = iterations if iterations is not None else [it.iteration for it in traces[names[0]].iterations]
    fig, axes = plt.subplots(len(names), len(its), figsize=(size * len(its), size * len(names)), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for r, name in enumerate(names):
        tr = traces[name]
        risky, _ = _risky_agents(tr, risk_threshold)
        by_iter = {it.iteration: it for it in tr.iterations}
        for c, k in enumerate(its):
            ax = axes[r, c]
            it = by_iter[k if k >= 0 else tr.iterations[k].iteration]
            _draw_es_iteration(ax, tr, it, sim_state, ego_idx=ego_idx, view_m=view_m, show_wm_top1=show_wm_top1,
                               risky=risky, cmap=cmap)
            ax.set_title(f"iter {it.iteration}: max={it.scores.max():.2f} elites={len(np.unique(it.elite_indices))}",
                         fontsize=8)
        last = tr.iterations[-1]
        terms = " ".join(f"{k[:5]}={float(v[tr.best_index]):.2f}" for k, v in last.breakdown.items())
        axes[r, 0].set_ylabel(f"{name}\n{terms}", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="score")
    fig.suptitle(f"Acquisitions at t={traces[names[0]].timestep} (bold = elites, red = chosen plan)", fontsize=10)
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
                annotation_clip=True,
            )
    b = trace.best_trajectory
    ax.plot(b[:, 0], b[:, 1], color=EGO_COLOR, lw=2.5, zorder=6, label="chosen ego plan")
    _set_view(ax, _ego_xy(log, ego_idx, t), view_m)
    ax.set_title(
        f"[{trace.label}] WM modes at t={t}: width/alpha ~ mode prob, black dashed = logged future,\n"
        f"orange = agents with p_collision > 0. Agents predicted to stay put show as a dot.",
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
