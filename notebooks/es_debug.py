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


class MarginTiebreakReward:
    """``--offroad_mode margin_tiebreak``: collision * edge-margin + 1e-3 * offroad tiebreak.

    Same algebra as ``Scorer.compute_score`` in that mode. ``breakdown['offroad']`` is the
    margin score (0 if offroad, 0.5..1 by clearance) so ``> 0`` still means on-road.
    """

    def __call__(self, ctx: RewardContext) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        from scores.scorer_lane_graph import OFFROAD_TIEBREAK_EPS

        coll = np.asarray(collision(ctx), dtype=np.float32)
        margin, tiebreak = ctx.scorer.compute_offroad_score_margin_tiebreak(
            ctx.ego_boxes, sim_state=ctx.sim_state, timestep=ctx.timestep, world_idx=ctx.world_idx
        )
        margin = np.asarray(margin, dtype=np.float32)
        tiebreak = np.asarray(tiebreak, dtype=np.float32)
        score = coll * margin + OFFROAD_TIEBREAK_EPS * tiebreak
        return score.astype(np.float32), {"collision": coll, "offroad": margin, "offroad_tiebreak": tiebreak}

    def __repr__(self) -> str:
        return "MarginTiebreakReward()"


def margin_tiebreak_reward() -> MarginTiebreakReward:
    return MarginTiebreakReward()


# ---------------------------------------------------------------------------
# Acquisition (elite selection)
# ---------------------------------------------------------------------------

# An acquisition maps (scores [P], trajectories [P, T, 5], n_elite, rng) -> indices [n_elite].
Acquisition = Callable[[np.ndarray, np.ndarray, int, np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------------
# Initial pool (iteration-0 sampling)
# ---------------------------------------------------------------------------

# An initial-pool sampler maps (planner, cond_bf, inst_cond_bf, inst_cond_mask_bf, features,
# pre_batch, rng) -> normalized proposals [1, P, T, D]. Later ES iterations are unchanged
# (elites are re-noised and denoised with the goal-conditioned cond_bf).
InitialPool = Callable[..., jax.Array]


def standard_pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng) -> jax.Array:
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

    def pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng):
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


def goal_mix(frac_unconditioned: float = 0.25) -> InitialPool:
    """Replace a fraction of the pool with goal-masked samples (goal_xy and
    remaining_timesteps zeroed, as ``mask_goal=True``): proposals not pulled toward the
    single logged route."""

    def pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng):
        del pre_batch
        k_goal, k_free = jax.random.split(rng)
        masked = dict(features)
        masked["goal_xy"] = jnp.zeros_like(features["goal_xy"])
        masked["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
        cond_free, inst_free = planner._compute_condition_jit(masked)
        n_free = int(round(frac_unconditioned * planner.population_size))
        goal_pool = planner._sample_population_jit(
            cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=inst_cond_mask_bf, rng=k_goal
        )
        free_pool = planner._sample_population_jit(
            cond_bf=cond_free, inst_cond_bf=inst_free, inst_cond_mask_bf=inst_cond_mask_bf, rng=k_free
        )
        return jnp.concatenate([goal_pool[:, : planner.population_size - n_free], free_pool[:, :n_free]], axis=1)

    pool.__name__ = f"goal_mix[{frac_unconditioned:g}]"
    return pool


def _rotate_scale_norm(norm_ktd: jax.Array, delta_rad: jax.Array, scale: jax.Array) -> jax.Array:
    """Rotate ego-normalized [x, y, vx, vy, yaw/pi] plans about the ego (origin) by ``delta_rad``
    and scale distance/speed by ``scale`` (per plan, shape [K])."""
    c, s = jnp.cos(delta_rad)[:, None], jnp.sin(delta_rad)[:, None]
    x, y, vx, vy, yaw = (norm_ktd[..., i] for i in range(5))
    k = scale[:, None]
    out = jnp.stack(
        [
            k * (c * x - s * y),
            k * (s * x + c * y),
            k * (c * vx - s * vy),
            k * (s * vx + c * vy),
            yaw + delta_rad[:, None] / jnp.pi,
        ],
        axis=-1,
    )
    return jnp.concatenate([out, norm_ktd[..., 5:]], axis=-1)


def heading_fan(
    max_deg: float = 30.0,
    num_headings: int = 12,
    keep_frac: float = 0.25,
    speed_scales: tuple[float, ...] = (1.0,),
    denoise: bool = True,
) -> InitialPool:
    """Fan the diffusion samples out in heading, then re-denoise them (SDEdit).

    ``keep_frac`` of the pool stays as plain samples; each remaining sample is rotated about
    the ego by one of ``num_headings`` angles in [-max_deg, max_deg] (and optionally scaled
    by one of ``speed_scales``, < 1 = slower), then partially noised and denoised with the
    planner's ``resample_timesteps`` so the result is back on the model's manifold.
    ``denoise=False`` keeps the raw rotated plans (useful to see what the fan looks like).
    """

    def pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng):
        del features, pre_batch
        k_sample, k_denoise = jax.random.split(rng)
        base = planner._sample_population_jit(
            cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=inst_cond_mask_bf, rng=k_sample
        )  # [1, P, T, D]
        n_keep = int(round(keep_frac * planner.population_size))
        combos = [(d, k) for k in speed_scales for d in np.deg2rad(np.linspace(-max_deg, max_deg, num_headings))]
        n_fan = planner.population_size - n_keep
        delta = jnp.asarray([combos[i % len(combos)][0] for i in range(n_fan)], dtype=jnp.float32)
        scale = jnp.asarray([combos[i % len(combos)][1] for i in range(n_fan)], dtype=jnp.float32)
        fan = _rotate_scale_norm(base[0, n_keep:], delta, scale)[None]  # [1, n_fan, T, D]
        if denoise:
            fan = planner._resample_population_jit(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=inst_cond_mask_bf,
                proposals_bktd=fan,
                rng=k_denoise,
            )
        return jnp.concatenate([base[:, :n_keep], fan], axis=1)

    speeds = "" if tuple(speed_scales) == (1.0,) else f",v{'/'.join(f'{k:g}' for k in speed_scales)}"
    pool.__name__ = f"heading_fan[±{max_deg:g}°x{num_headings}{speeds}{'' if denoise else ',raw'}]"
    return pool


# Heading-offset profiles over normalized time v in [0, 1]; all start at 0 (no instant turn).
#   arc:         a*v            keeps turning one way
#   lane_change: a*sin(pi v)    out, then straightens: ends shifted sideways
#   swerve:      a*sin(2 pi v)  out, then back the other way: returns to the original line
#   reverse:     a*sin(1.5pi v) out, then curves past straight to the opposite side
SHAPE_PROFILES = {
    "arc": lambda v: v,
    "lane_change": lambda v: np.sin(np.pi * v),
    "swerve": lambda v: np.sin(2.0 * np.pi * v),
    "reverse": lambda v: np.sin(1.5 * np.pi * v),
}


def _heading_profile(name: str, amp_rad: float, duration_frac: float, horizon: int) -> np.ndarray:
    """[T] heading offset; the manoeuvre takes ``duration_frac`` of the horizon, then holds."""
    u = np.linspace(0.0, 1.0, horizon)
    v = np.clip(u / max(duration_frac, 1e-6), 0.0, 1.0)
    return (amp_rad * SHAPE_PROFILES[name](v)).astype(np.float32)


def _reshape_heading(norm_ktd: jax.Array, dpsi_kt: jax.Array, scale_k: jax.Array) -> jax.Array:
    """Re-integrate ego-normalized [x, y, vx, vy, yaw/pi] plans with each step's displacement
    rotated by ``dpsi_kt`` and scaled by ``scale_k`` (0 = stop). Keeps the sample's own speed
    profile; changes its curvature."""
    k = scale_k[:, None]
    pos = norm_ktd[..., :2] * k[..., None]
    step = jnp.diff(jnp.concatenate([jnp.zeros_like(pos[:, :1]), pos], axis=1), axis=1)
    c, s = jnp.cos(dpsi_kt), jnp.sin(dpsi_kt)
    new_pos = jnp.cumsum(jnp.stack([c * step[..., 0] - s * step[..., 1], s * step[..., 0] + c * step[..., 1]], -1), axis=1)
    vx, vy = k * norm_ktd[..., 2], k * norm_ktd[..., 3]
    out = jnp.stack(
        [
            new_pos[..., 0],
            new_pos[..., 1],
            c * vx - s * vy,
            s * vx + c * vy,
            norm_ktd[..., 4] + dpsi_kt / jnp.pi,
        ],
        axis=-1,
    )
    return jnp.concatenate([out, norm_ktd[..., 5:]], axis=-1)


def shape_fan(
    profiles: tuple[str, ...] = ("arc", "lane_change", "swerve", "reverse"),
    amplitudes_deg: tuple[float, ...] = (15.0, 30.0),
    durations: tuple[float, ...] = (0.5, 1.0),
    speed_scales: tuple[float, ...] = (1.0,),
    include_stop: bool = True,
    keep_frac: float = 0.25,
    denoise: bool = True,
) -> InitialPool:
    """Fan samples out in *curvature*: each fan member gets a heading-offset profile
    (``SHAPE_PROFILES``: arc, lane_change, swerve = out-and-back, reverse = out then the
    opposite way), both signs, each amplitude and manoeuvre duration (fraction of the
    horizon), optionally a speed scale, plus one stopped plan. The sample is re-integrated
    with the rotated step displacements (smooth, same speed profile), then re-denoised
    (SDEdit) with the planner's ``resample_timesteps``.
    """

    def pool(planner, *, cond_bf, inst_cond_bf, inst_cond_mask_bf, features, pre_batch, rng):
        del features, pre_batch
        k_sample, k_denoise = jax.random.split(rng)
        base = planner._sample_population_jit(
            cond_bf=cond_bf, inst_cond_bf=inst_cond_bf, inst_cond_mask_bf=inst_cond_mask_bf, rng=k_sample
        )  # [1, P, T, D]
        horizon = int(base.shape[2])
        n_keep = int(round(keep_frac * planner.population_size))
        n_fan = planner.population_size - n_keep
        combos = [(np.zeros((horizon,), np.float32), 0.0)] if include_stop else []
        combos += [
            (_heading_profile(name, sign * np.deg2rad(a), d, horizon), k)
            for k in speed_scales
            for name in profiles
            for a in amplitudes_deg
            for d in durations
            for sign in (1.0, -1.0)
        ]
        if len(combos) > n_fan:
            print(f"shape_fan: {len(combos)} shapes > {n_fan} slots; dropping the last {len(combos) - n_fan}")
        pick = [combos[i % len(combos)] for i in range(n_fan)]
        dpsi = jnp.asarray(np.stack([p[0] for p in pick]))
        scale = jnp.asarray([p[1] for p in pick], dtype=jnp.float32)
        fan = _reshape_heading(base[0, n_keep:], dpsi, scale)[None]
        if denoise:
            fan = planner._resample_population_jit(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=inst_cond_mask_bf,
                proposals_bktd=fan,
                rng=k_denoise,
            )
        return jnp.concatenate([base[:, :n_keep], fan], axis=1)

    pool.__name__ = (
        f"shape_fan[{'/'.join(profiles)} ±{'/'.join(f'{a:g}' for a in amplitudes_deg)}°"
        f"{'' if tuple(speed_scales) == (1.0,) else ' v' + '/'.join(f'{k:g}' for k in speed_scales)}"
        f"{'+stop' if include_stop else ''}{'' if denoise else ',raw'}]"
    )
    return pool


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
        self.label = label
        self.last_trace: ESTrace | None = None
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
        new.label = kwargs["label"]
        new.last_trace = None
        new._init_kwargs = kwargs
        return new

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

        norm_bktd = self.initial_pool(
            self,
            cond_bf=cond_bf,
            inst_cond_bf=inst_cond_bf,
            inst_cond_mask_bf=instruction_mask,
            features=features,
            pre_batch=pre_batch,
            rng=key_sample,
        )
        world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(norm_bktd, pre_batch)
        pool_norm = norm_bktd[0]
        pool_world = np.asarray(world_bkt5[0])
        pool_scores, pool_breakdown = self._score(pool_world, sim_state, timestep, goal_xy)
        carried = np.zeros((len(pool_scores),), dtype=bool)

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
                )
            )
            if it == self.num_iterations:
                break

            elite_norm = pool_norm[elite_idx][None]  # [1, E, T, D]
            replicated = self._replicate_elites(elite_norm, population_size=self.population_size)
            rng, key_resample = jax.random.split(rng)
            new_norm = self._resample_population_jit(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=instruction_mask,
                proposals_bktd=replicated,
                rng=key_resample,
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
    figsize_per=(4.2, 4.2),
):
    """One panel per ES iteration: pool coloured by score, elites bold, final best red."""
    n = len(trace.iterations)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    traj = _single_state(sim_state).log_trajectory
    t = trace.timestep
    for ax, it in zip(axes.ravel(), trace.iterations):
        _base_axes(ax, sim_state)
        _draw_boxes(ax, traj, t, ego_idx)
        if show_wm_top1 and trace.wm_forecast is not None:
            top1 = trace.wm_forecast["top1_traj"]
            mask = trace.wm_forecast["masks"][:, 0]
            for i in np.flatnonzero(mask):
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
    fig.suptitle(f"[{trace.label}] ES at t={t} (bold = elites, red = chosen plan, dotted = WM top-1)")
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
