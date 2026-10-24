"""The single, batched simulator-reward function every fine-tuning method uses.

Phase 1, item 7: "Refactor the reward calculation from ``waymax_env.py`` into a
common batched function. Every method must consume the same scalar returned by
this environment."

The reward computed here reproduces the per-step arithmetic of
``rl.waymax_env.WaymaxGymEnv.step`` (lines ~331-367) *exactly*, but vectorized
over a batch of worlds and expressed as a pure function of arrays so it JITs and
so batched/single-scene rewards are provably identical (Gate 1).

The reward is state-dependent across a rollout (progress is measured against the
previous distance; collision/offroad are charged at most once per episode). That
running state is carried explicitly in :class:`RewardState` -- there is no hidden
per-instance mutation -- so a batched rollout, a permuted batch, and a single
scene all thread the same recurrence.

Route projection (``s``, ``lateral``) is *not* computed here: it is host-side
polyline math (see ``rl.waymax_env._project_to_route``) that the caller performs
per world and passes in. This module owns only the scalar reward algebra, which
is what has to match across methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["RewardConfig", "RewardState", "init_reward_state", "reward_step"]


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the shaped simulator reward.

    Field-for-field identical to ``rl.waymax_env.RewardConfig`` (kept as a
    separate, dependency-light dataclass so this module -- and the diffusion
    environment -- do not import ``rl.waymax_env``, which pulls in ``gymnasium``
    and the whole SB3 stack). ``reward_step`` below is the shared arithmetic; if
    a weight is added here, mirror it there. See ``waymax_env`` for the long-form
    rationale on each choice (one-time collision/offroad charges, bounded route
    indicators, etc.).

    Two of the fields here exist to fix a *ranking* defect rather than to retune
    a magnitude, and are worth reading together (see
    ``tests/test_reward_ranking.py``):

    * ``offroad_entry`` / ``collision_entry`` -- an extra charge levied on each
      on-road -> off-road (or no-contact -> contact) **transition**. A per-step
      charge (``offroad_per_step``) is paid for making an excursion *shorter*;
      only an edge charge is paid for not taking it at all. The three modes are
      the three ways to price the same event: charged-once (an entry toll on the
      *first* excursion only -- farmable, since the second is free), per-step
      (dwell only -- the marginal value of the last off-road step equals that of
      any interior one, so the optimizer buys shortening, never avoidance), and
      entry+dwell, which prices both the decision to leave the road and the time
      spent off it.
    * ``clean_goal_bonus`` -- pay ``goal_bonus`` only to an episode that reaches
      the goal having never gone off-road or collided. This is what puts a
      discontinuity at "zero violations", which is where the solve criterion
      (reached AND never offroad AND never collided) has its own. Without it the
      reward is a smooth function of a quantity whose objective value is a step
      function, and no coefficient can fix that.
    """

    progress: float = 1.0
    step_penalty: float = 0.0
    action_penalty: float = 0.01
    collision: float = -10.0
    offroad: float = -5.0
    goal_bonus: float = 10.0
    goal_threshold_m: float = 3.0
    terminate_on_collision: bool = True
    terminate_on_offroad: bool = False
    offroad_per_step: bool = False
    collision_per_step: bool = False
    offroad_entry: float = 0.0
    collision_entry: float = 0.0
    clean_goal_bonus: bool = False
    route_reward: bool = False
    lateral_penalty: float = 0.5
    off_route_threshold_m: float | None = None
    off_route_penalty: float = -0.2
    progression_indicator: bool = False


@dataclass(frozen=True)
class RewardState:
    """Per-world running reward state, each field a ``[B]`` array.

    * ``prev_dist``          -- straight-line goal distance at the previous step.
    * ``prev_s``             -- route arclength at the previous step (route mode).
    * ``charged_collision``  -- whether the one-time collision penalty was paid.
      Still tracked under ``collision_per_step`` (see ``charged_offroad``).
    * ``charged_offroad``    -- whether the one-time offroad penalty was paid.
      Still tracked under ``offroad_per_step`` (the field keeps the pytree shape
      constant so the JITted rollout does not retrace) but unused there.
    * ``was_offroad``        -- the *instantaneous* offroad flag at the previous
      step (not the monotone ``charged_offroad``). Their difference is the
      rising edge that ``offroad_entry`` charges: ``off_b & ~was_offroad``
      fires once per excursion, however long each excursion runs.
    * ``was_collision``      -- ditto for contact.
    * ``done``               -- whether the episode has already terminated. Once
      set, subsequent steps contribute zero reward (an ended world is frozen),
      so per-step rewards still sum to the episode return.
    """

    prev_dist: jax.Array
    prev_s: jax.Array
    charged_collision: jax.Array
    charged_offroad: jax.Array
    was_collision: jax.Array
    was_offroad: jax.Array
    done: jax.Array


def init_reward_state(
    *,
    init_goal_dist_b: jax.Array,
    init_s_b: jax.Array,
) -> RewardState:
    """Build the reset :class:`RewardState` for a batch of worlds."""
    init_goal_dist_b = jnp.asarray(init_goal_dist_b, dtype=jnp.float32)
    init_s_b = jnp.asarray(init_s_b, dtype=jnp.float32)
    zeros_bool = jnp.zeros_like(init_goal_dist_b, dtype=bool)
    return RewardState(
        prev_dist=init_goal_dist_b,
        prev_s=init_s_b,
        charged_collision=zeros_bool,
        charged_offroad=zeros_bool,
        # The ego starts on-road and out of contact, so the first off-road /
        # colliding step is genuinely a rising edge and pays the entry charge.
        was_collision=zeros_bool,
        was_offroad=zeros_bool,
        done=zeros_bool,
    )


def reward_step(
    cfg: RewardConfig,
    state: RewardState,
    *,
    goal_dist_b: jax.Array,
    overlap_b: jax.Array,
    offroad_b: jax.Array,
    s_b: jax.Array | None = None,
    lateral_b: jax.Array | None = None,
    action_sq_b: jax.Array | None = None,
) -> tuple[jax.Array, RewardState, jax.Array, dict[str, jax.Array]]:
    """One batched reward step.

    Args:
      cfg: reward weights (shared with the single-scene env).
      state: running :class:`RewardState` from the previous step (or reset).
      goal_dist_b: ``[B]`` straight-line distance from ego to goal *after* the step.
      overlap_b: ``[B]`` collision metric value (>0.5 == colliding).
      offroad_b: ``[B]`` offroad metric value (>0.5 == off road).
      s_b, lateral_b: ``[B]`` route arclength / lateral deviation (route mode only).
      action_sq_b: ``[B]`` sum of squared low-level action (control-space envs);
        the trajectory-action diffusion env passes ``None`` (no per-control cost).

    Returns:
      ``(reward_b, new_state, terminated_b, components)`` where ``terminated_b``
      marks worlds that terminate *on this step*, and ``components`` breaks the
      reward into named additive parts for logging / the sum-to-return test.
    """
    goal_dist_b = jnp.asarray(goal_dist_b, dtype=jnp.float32)
    overlap_b = jnp.asarray(overlap_b, dtype=jnp.float32)
    offroad_b = jnp.asarray(offroad_b, dtype=jnp.float32)

    collision_b = overlap_b > 0.5
    off_b = offroad_b > 0.5
    reached_b = goal_dist_b <= cfg.goal_threshold_m

    # --- progress term ---------------------------------------------------- #
    if cfg.route_reward:
        if s_b is None or lateral_b is None:
            raise ValueError("route_reward=True requires s_b and lateral_b.")
        s_b = jnp.asarray(s_b, dtype=jnp.float32)
        lateral_b = jnp.asarray(lateral_b, dtype=jnp.float32)
        if cfg.progression_indicator:
            progress_b = cfg.progress * (s_b > state.prev_s).astype(jnp.float32)
        else:
            progress_b = cfg.progress * (s_b - state.prev_s)
        if cfg.off_route_threshold_m is None:
            route_pen_b = -cfg.lateral_penalty * lateral_b
        else:
            route_pen_b = cfg.off_route_penalty * (
                lateral_b > cfg.off_route_threshold_m
            ).astype(jnp.float32)
        progress_b = progress_b + route_pen_b
        new_prev_s = s_b
    else:
        progress_b = cfg.progress * (state.prev_dist - goal_dist_b)
        new_prev_s = state.prev_s

    # --- constant / action penalties -------------------------------------- #
    step_pen_b = jnp.full_like(goal_dist_b, -cfg.step_penalty)
    if action_sq_b is None:
        action_pen_b = jnp.zeros_like(goal_dist_b)
    else:
        action_pen_b = -cfg.action_penalty * jnp.asarray(action_sq_b, dtype=jnp.float32)

    # --- one-time event terms --------------------------------------------- #
    # Collision mirrors offroad below: charged once per episode by default, or on
    # every colliding world step under `collision_per_step`. Pair the per-step
    # form with `terminate_on_collision=False` -- otherwise the episode ends on
    # the first contact and "per step" only ever bills one step.
    if cfg.collision_per_step:
        pay_collision_b = collision_b
    else:
        pay_collision_b = collision_b & (~state.charged_collision)
    # Offroad is charged once per episode by default. Under `offroad_per_step`
    # every off-road world step pays: the cost then scales with dwell time
    # instead of being a flat toll, which is what makes a non-terminating
    # offroad penalty still unprofitable to farm. Note the effective magnitude
    # multiplies by the excursion length -- at `offroad`=-1.0 a 24-step
    # excursion costs -24, so pair this flag with a smaller coefficient.
    if cfg.offroad_per_step:
        pay_offroad_b = off_b
    else:
        pay_offroad_b = off_b & (~state.charged_offroad)
    # Rising-edge (entry) charges. `charged_*` is monotone, so it cannot express
    # "started a *new* excursion"; `was_*` is the previous step's instantaneous
    # flag, and `flag & ~was_flag` fires exactly once per excursion. This is the
    # only term whose value depends on the excursion COUNT rather than on the
    # total dwell, and so the only one an optimizer can collect by avoiding an
    # excursion rather than by shortening one.
    entry_offroad_b = off_b & (~state.was_offroad)
    entry_collision_b = collision_b & (~state.was_collision)
    collision_term_b = (
        cfg.collision * pay_collision_b.astype(jnp.float32)
        + cfg.collision_entry * entry_collision_b.astype(jnp.float32)
    )
    offroad_term_b = (
        cfg.offroad * pay_offroad_b.astype(jnp.float32)
        + cfg.offroad_entry * entry_offroad_b.astype(jnp.float32)
    )

    # Clean-gated goal bonus. `charged_*` carries "ever violated *before* this
    # step"; OR in the current step's flag so arriving at the goal while off-road
    # does not count as clean. Gating here (rather than adding a separate terminal
    # term) keeps the bonus inside the existing `goal_bonus` component, so the
    # six-way reward decomposition and its sum-to-return invariant are unchanged.
    ever_offroad_b = state.charged_offroad | off_b
    ever_collision_b = state.charged_collision | collision_b
    clean_b = ~(ever_offroad_b | ever_collision_b)
    if cfg.clean_goal_bonus:
        pay_goal_b = reached_b & clean_b
    else:
        pay_goal_b = reached_b
    goal_term_b = cfg.goal_bonus * pay_goal_b.astype(jnp.float32)

    raw_reward_b = (
        progress_b
        + step_pen_b
        + action_pen_b
        + collision_term_b
        + offroad_term_b
        + goal_term_b
    )

    # A world that already terminated contributes nothing further, so the sum of
    # per-step rewards equals the episode return regardless of padding steps.
    active_b = ~state.done
    reward_b = jnp.where(active_b, raw_reward_b, 0.0)

    # --- termination ------------------------------------------------------ #
    term_now_b = jnp.zeros_like(goal_dist_b, dtype=bool)
    if cfg.terminate_on_collision:
        term_now_b = term_now_b | collision_b
    if cfg.terminate_on_offroad:
        term_now_b = term_now_b | off_b
    term_now_b = term_now_b | reached_b
    terminated_b = active_b & term_now_b

    new_state = RewardState(
        prev_dist=goal_dist_b,
        prev_s=new_prev_s,
        charged_collision=ever_collision_b,
        charged_offroad=ever_offroad_b,
        was_collision=collision_b,
        was_offroad=off_b,
        done=state.done | terminated_b,
    )

    components = {
        "progress": jnp.where(active_b, progress_b, 0.0),
        "step_penalty": jnp.where(active_b, step_pen_b, 0.0),
        "action_penalty": jnp.where(active_b, action_pen_b, 0.0),
        "collision": jnp.where(active_b, collision_term_b, 0.0),
        "offroad": jnp.where(active_b, offroad_term_b, 0.0),
        "goal_bonus": jnp.where(active_b, goal_term_b, 0.0),
        "collision_flag": collision_b,
        "offroad_flag": off_b,
        "reached_flag": reached_b,
        # Rising edges and the running cleanliness bit, for diagnostics. All are
        # `_flag` keys, so callers that accumulate components OR them (rather
        # than summing them into the return) and the reward decomposition above
        # stays exactly six additive terms.
        "offroad_entry_flag": entry_offroad_b,
        "collision_entry_flag": entry_collision_b,
        "clean_flag": clean_b,
    }
    return reward_b, new_state, terminated_b, components
