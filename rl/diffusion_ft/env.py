"""``DiffusionWaymaxEnv``: the one trustworthy batched environment (Phase 1).

The outer action of this MDP is *one sampled diffusion trajectory*. A step:

1. builds features from the current *simulated* ego state (executed motion, not
   the human log -- see :func:`_splice_sim_into_log_batched`),
2. samples a trajectory from the policy (or accepts one passed in, for
   best-of-N / ES search),
3. postprocesses it to world coordinates,
4. executes only the configured prefix through ``StateDynamics`` (the backend
   that matches the state-trajectory action representation and the existing
   diffusion evaluation),
5. returns the batched simulator reward (from :mod:`rl.diffusion_ft.reward`, the
   one shared reward function), the next state, termination flags, and the
   per-step reward components.

Design invariants (Phase 1):

* **``log_trajectory`` is immutable.** Executed motion lives in
  ``sim_trajectory`` (Waymax writes it there as we step ``StateDynamics``). The
  goal and the expert route are read from the untouched log.
* **Real goal timestep per scene.** No ``goal_step_override=90``; the goal step
  is each scene's last valid ego log step (:func:`infer_goals_from_sim_state`).
* **Batched == single.** Every quantity is computed over the batch axis with the
  same recurrence a single scene would use, so a permuted batch permutes the
  outputs identically (Gate 1).

This env owns the policy for on-policy collection, but ``step`` also accepts an
externally supplied ``trajectory_world_bt5`` so the search baselines
(best-of-N, ES) can score their own candidates through the identical executor
and reward.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from waymax import config as waymax_config
from waymax import datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

from data.postprocess import postprocess_predictions
from data.preprocess import (WORLD_DT_METHODS, compute_world_dt_seconds,
                             preprocess_simulator_state)
from data.types import PreprocessBatch
from simulation.evaluation_utils import infer_goals_from_sim_state

from rl.diffusion_ft.checkpoint import DiffusionCheckpoint
from rl.diffusion_ft import reward as reward_mod
from rl.diffusion_ft.sim_agents import SIM_AGENT_CHOICES, build_sim_agents
from rl.diffusion_ft.reward import RewardConfig, RewardState


# --------------------------------------------------------------------------- #
# Batched helpers
# --------------------------------------------------------------------------- #
def _ego_indices_b(state: Any) -> jax.Array:
    """One SDC index per world, ``[B]``."""
    is_sdc_bn = jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32)
    return jnp.argmax(is_sdc_bn, axis=1)


# Which velocity the EGO's executed rows carry into the observation. See
# `_splice_sim_into_log_batched`.
EGO_VEL_SOURCES = ("commanded", "realised")


def _splice_sim_into_log_batched(state: Any, *, ego_vel_source: str = "commanded",
                                 first_policy_step: Any = None,
                                 dt_b: jax.Array | None = None) -> Any:
    """Batched analog of ``rl.waymax_env._splice_sdc_sim_into_log``.

    Returns a *scratch* copy of ``state`` in which **every** object's
    ``log_trajectory`` rows are overwritten by its ``sim_trajectory`` for every
    already-simulated, valid step (``step <= timestep``). This is used **only**
    to feed the preprocessor the executed history; the caller's real state keeps
    its ``log_trajectory`` untouched.

    **Why every object and not just the ego.** The preprocessor reads the other
    agents' poses off ``log_trajectory`` (``data/preprocess.py``, the ``_bn``
    gathers at the anchor step). While this spliced the ego alone, a reactive
    NPC (``sim_agents="idm_brake"``) moved in the simulator but stayed at its
    logged pose in the observation -- so the policy was scored against traffic it
    could not see. That is the
    ``project_sb3_route_reward_unobservable`` failure mode, and its signature was
    visible in the 2026-09-21 A/B: ``reached_rate`` came back bit-identical
    across the two simulators on all three splits (0.7327 / 0.9883 / 1.0000),
    because the ego's rollout was literally unchanged and only the scoring moved.

    **This is a no-op under log replay**, exactly and not approximately. Where
    ``sim_valid`` holds, a log-replayed object's simulated pose IS its logged
    pose -- measured at ``max |sim - log| = 0.00e+00`` over 19529 poses by gate 1
    of ``tests/test_reactive_npc.py`` -- and where it does not hold the log is
    kept. So the blend writes back the same numbers and every result taken under
    ``sim_agents="none"`` is undisturbed. Gate 1b asserts that rather than
    trusting it.

    **``ego_vel_source``: which velocity the EGO's executed rows carry.** The
    diffusion model's position and velocity heads are denormalized by two
    independent constants (``data/postprocess.py``: ``pos * ego_range``,
    ``vel * max_velocity``) and nothing couples them; ``StateDynamics`` sets the
    pose from the positions and carries the velocity along unchecked. Measured
    2026-09-19 over 753 episodes, the velocity the policy WRITES overstates the
    displacement it DRIVES by 13-23%, while the log's two agree to 1.000 -- and
    ``data/preprocess.py`` builds the policy's own ``ego_state`` from exactly
    this field. So every replan conditions the model on a speed it is not doing,
    after it was pretrained where the two agree.

    * ``"commanded"`` (the default, and every result before 2026-09-23) copies
      the written velocity, as above.
    * ``"realised"`` replaces the ego's ``vel_x``/``vel_y`` at every step the
      POLICY executed (``first_policy_step <= t <= timestep``) with the
      backward difference of the executed positions over ``dt_b``. Backward
      because it is the only causal one: at the anchor the next pose has not
      been driven yet, and the blend holds the HUMAN's future there, so a
      central difference would put the demonstration into the observation.
      Everything else is untouched -- positions, yaw, every non-ego row, and
      the replayed history before ``first_policy_step`` (whose logged velocity
      is what pretraining saw) -- so the first replan's observation is
      bit-identical under both settings. ``tests/test_ego_vel_splice.py``
      asserts that, and that the estimator reproduces the human's own logged
      velocity from the human's positions.
    """
    sim = state.sim_trajectory
    log = state.log_trajectory
    num_t = log.x.shape[-1]
    step_idx = jnp.arange(num_t)                           # [T]
    # Simulated & valid, per object per step: [B, N, T]. `sim_trajectory` is
    # filled invalid at reset and written one world step at a time, so the
    # `<= timestep` bound is what keeps the unwritten future out of the blend.
    use_bnt = (step_idx[None, None, :] <= state.timestep) & jnp.asarray(sim.valid)

    def blend(sim_f: jax.Array, log_f: jax.Array) -> jax.Array:
        return jnp.where(use_bnt, sim_f, log_f)

    x, y = blend(sim.x, log.x), blend(sim.y, log.y)
    vel_x, vel_y = blend(sim.vel_x, log.vel_x), blend(sim.vel_y, log.vel_y)
    if ego_vel_source == "realised":
        if first_policy_step is None or dt_b is None:
            raise ValueError("ego_vel_source='realised' needs first_policy_step and dt_b")
        ego_bn = (jnp.arange(log.x.shape[-2])[None, :]
                  == _ego_indices_b(state)[:, None])                  # [B, N]
        # Step t-1 must be a real pose too; at the first policy step it is the
        # last replayed log pose, which waymax `reset` copied into the sim.
        prev_ok = jnp.concatenate(
            [jnp.zeros_like(use_bnt[..., :1]), use_bnt[..., :-1]], axis=-1)
        own = ego_bn[:, :, None] & use_bnt & prev_ok & (step_idx >= first_policy_step)
        dt = jnp.asarray(dt_b, x.dtype)[:, None, None]

        def backward(p):
            return jnp.concatenate([jnp.zeros_like(p[..., :1]), jnp.diff(p, axis=-1)],
                                   axis=-1) / dt

        vel_x = jnp.where(own, backward(x), vel_x)
        vel_y = jnp.where(own, backward(y), vel_y)
    elif ego_vel_source != "commanded":
        raise ValueError(f"ego_vel_source must be one of {EGO_VEL_SOURCES}, "
                         f"got {ego_vel_source!r}")

    new_log = log.replace(
        x=x,
        y=y,
        yaw=blend(sim.yaw, log.yaw),
        vel_x=vel_x,
        vel_y=vel_y,
    )
    return state.replace(log_trajectory=new_log)


def _ego_pose_now_b(state: Any) -> jax.Array:
    """Ego ``[x, y]`` at the current ``sim_trajectory`` timestep, ``[B, 2]``."""
    ego_b = _ego_indices_b(state)
    cur = datatypes.dynamic_slice(state.sim_trajectory, state.timestep, 1, axis=-1)
    x_b = jnp.take_along_axis(cur.x[..., 0], ego_b[:, None], axis=1)[:, 0]
    y_b = jnp.take_along_axis(cur.y[..., 0], ego_b[:, None], axis=1)[:, 0]
    return jnp.stack([x_b, y_b], axis=-1)


# --------------------------------------------------------------------------- #
@dataclass
class StepOutput:
    """One outer (replan) step's result."""

    features: dict[str, jax.Array]          # observation for the next replan
    reward_b: jax.Array                     # [B] summed over executed world steps
    terminated_b: jax.Array                 # [B]
    truncated_b: jax.Array                  # [B]
    per_step_reward_bk: jax.Array           # [B, prefix] atomic world-step rewards
    components: dict[str, jax.Array]        # summed reward components, each [B]
    trajectory_world_bt5: jax.Array         # the executed candidate (world coords)
    # Per-*world-step* reward components, each ``[B, n_exec]`` (flags as bool).
    # `components` above ORs/sums these over the executed prefix, which is all the
    # trainer needs; the diagnostic needs the un-collapsed version to say *which*
    # 0.1 s of a 0.5 s prefix went offroad. Defaults to empty so existing
    # constructions and consumers are unaffected.
    per_step_components: dict[str, jax.Array] = dataclasses.field(default_factory=dict)


class DiffusionWaymaxEnv:
    """Batched diffusion-policy environment; outer action = one trajectory."""

    def __init__(
        self,
        ckpt: DiffusionCheckpoint,
        *,
        reward_config: RewardConfig | None = None,
        max_num_objects: int = 128,
        start_timestep: int | None = None,
        replan_interval_steps: int = 10,
        use_ema: bool = False,
        seed: int = 0,
        max_replans: int | None = None,
        goal_shaping_coef: float = 0.0,
        goal_shaping_discount: float = 0.99,
        goal_shaping_normalize: bool = True,
        jit_replan: bool = True,
        sim_agents: str = "none",
        ego_vel_source: str = "commanded",
        world_dt_method: str = "median",
    ):
        self._ckpt = ckpt
        # Compile each replan's world steps + reward + next-observation preprocess
        # into one XLA program. Eager, the preprocessor's vmapped lexsort/unique/
        # cond roadgraph build dispatched op-by-op and was ~0.5 s per replan --
        # about half of every collection and eval. False keeps the original eager
        # path for A/B (tests/bench_env_jit.py).
        self._jit_replan_enabled = bool(jit_replan)
        self._cfg = ckpt.preprocess_cfg
        self._reward_cfg = reward_config or RewardConfig()
        # Route-based reward (V-Max `progression` / `off_route`) needs the SDC's
        # candidate paths and the two waymax route metrics; both are off unless the
        # reward config asks for them, so existing (goal-distance) callers are
        # unaffected and never require ``sdc_paths`` in the loaded state.
        self._need_route = bool(self._reward_cfg.route_reward)
        # Potential-based goal shaping (V-Max BraxWrapper). Applied at the OUTER
        # replan step so it telescopes against the trainer's per-replan GAE
        # discount. coef=0 (default) is a no-op, keeping every existing rollout
        # -- and the Gate 3 sum-to-return invariant -- byte-identical.
        self._goal_shaping_coef = float(goal_shaping_coef)
        self._goal_shaping_discount = float(goal_shaping_discount)
        self._goal_shaping_normalize = bool(goal_shaping_normalize)
        self._init_goal_dist_b: jax.Array | None = None
        self._replan = int(replan_interval_steps)
        # Optional cap on the number of outer replan steps per episode (bounds
        # cost for CPU smoke tests). None == run to the scenario horizon.
        self._max_replans = None if max_replans is None else int(max_replans)
        self._max_num_objects = int(max_num_objects)
        # How the NON-ego agents move -- see `rl.diffusion_ft.sim_agents`, which
        # owns the policies and the measurement that motivates them.
        #
        # "none" (the default, and every result before 2026-09-19) log-replays
        # them: each repeats its recorded pose and cannot see the ego.
        # "idm_brake" makes them reactive in the one direction that is well
        # posed: on their own logged path, at their own logged speed, except
        # slower when something is in the way.
        # "idm" is waymax stock and is kept only to reproduce jobs 58075/58076 --
        # its free-road term accelerates every unobstructed car toward 108 km/h,
        # which is not the recorded scene and measured strictly worse.
        #
        # Why this axis at all: the log-replay probe
        # (tests/probe_logreplay_collision.py, 2026-08-30) found that rails are
        # cheap when the ego is in the wrong PLACE (1 m lateral: collision 0.031
        # -> 0.062) and expensive when it is there at the wrong TIME (5-step
        # delay 0.125, 40-step -- a car that never moves -- 0.719). "There at the
        # wrong time" is exactly what a slow policy is.
        if sim_agents not in SIM_AGENT_CHOICES:
            raise ValueError(
                f"sim_agents must be one of {SIM_AGENT_CHOICES}, got {sim_agents!r}")
        self._sim_agents = str(sim_agents)
        # Which velocity the ego's executed history shows the policy -- see
        # `_splice_sim_into_log_batched`. Read at TRACE time by `_features_core`,
        # so it is fixed for the env's lifetime.
        if ego_vel_source not in EGO_VEL_SOURCES:
            raise ValueError(
                f"ego_vel_source must be one of {EGO_VEL_SOURCES}, got {ego_vel_source!r}")
        self._ego_vel_source = str(ego_vel_source)
        # How each scene's world step is read off its timestamps -- the grid the
        # plan is resampled onto, one point per executed log step. "first" (every
        # result before 2026-09-24) takes the first timestamp pair, which is ~0.075
        # s in 44/173 failure_heldout scenes and runs their plan at ~0.75x;
        # "median" (the default since the 2026-09-24 A/B) is the scene's median
        # step. See compute_world_dt_seconds.
        if world_dt_method not in WORLD_DT_METHODS:
            raise ValueError(
                f"world_dt_method must be one of {WORLD_DT_METHODS}, got {world_dt_method!r}")
        self._world_dt_method = str(world_dt_method)
        self._rng = jax.random.PRNGKey(int(seed))

        self._policy = ckpt.merge(use_ema=use_ema)

        metrics_to_run: tuple[str, ...] = ("overlap", "offroad")
        if self._need_route:
            # ``sdc_progression`` is the route fraction NORMALIZED to [0, 1]
            # (waymax `metrics/route.py`, FULL_PROGRESS_VALUE): the arclength
            # from the logged start to the ego's projection, over the arclength
            # from the logged start to the logged final position. It is NOT
            # metres -- a `progress` coefficient tuned as if it were per-metre is
            # off by the route length. (V-Max `progression`.)
            # ``sdc_off_route`` is 0 on-route else the lateral distance to it
            # (V-Max `off_route`). Both are SDC-only metrics (not in the planning
            # env's multi-agent one-hot list) so they come back shaped ``[B]``.
            metrics_to_run = metrics_to_run + ("sdc_progression", "sdc_off_route")
        env_cfg = waymax_config.EnvironmentConfig(
            max_num_objects=self._max_num_objects,
            controlled_object=waymax_config.ObjectType.SDC,
            compute_reward=False,
            metrics=waymax_config.MetricsConfig(metrics_to_run=metrics_to_run),
        )
        sim_actors, sim_params = build_sim_agents(self._sim_agents)
        self._env = waymax_env.PlanningAgentEnvironment(
            dynamics_model=waymax_dynamics.StateDynamics(),
            config=env_cfg,
            sim_agent_actors=sim_actors,
            sim_agent_params=sim_params,
        )
        self._start_timestep = (
            int(start_timestep) if start_timestep is not None
            else int(self._env.config.init_steps)
        )
        # The first world step the POLICY writes: waymax `reset` copies
        # `init_steps` of log into `sim_trajectory` and leaves `timestep` at
        # `init_steps - 1`, so everything before this is the replayed human.
        self._first_policy_step = int(self._env.config.init_steps)

        env_obj = self._env

        need_route = self._need_route

        def _metric_values(state):
            m = env_obj.metrics(state)
            if need_route:
                return (m["overlap"].value, m["offroad"].value,
                        m["sdc_progression"].value, m["sdc_off_route"].value)
            return m["overlap"].value, m["offroad"].value

        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)
        self._jit_metrics = jax.jit(_metric_values)
        self._jit_condition = jax.jit(self._policy.compute_condition)
        self._jit_features = jax.jit(self._features_core)
        self._jit_prefix = jax.jit(self._prefix_core, static_argnames=("n_exec",))
        cfg = self._cfg
        self._jit_post = jax.jit(
            lambda norm, aux: postprocess_predictions(norm, aux, cfg)[
                "trajectory_world_world_dt"].astype(jnp.float32))

        # Per-episode batched state.
        self._state: Any = None
        self._goal_xy_b2: jax.Array | None = None
        self._goal_t_b: jax.Array | None = None
        self._reward_state: RewardState | None = None
        self._batch_size: int = 0

    # ------------------------------------------------------------------ #
    @property
    def sim_state(self) -> Any:
        return self._state

    @property
    def sim_agents(self) -> str:
        """Which non-ego agent model this env is running.

        Read by anything that reports a metric whose meaning depends on it --
        ``overlap`` under reactive agents is not the same number as ``overlap``
        under log replay, and a run that does not say which it used cannot be
        compared to one that does.
        """
        return self._sim_agents

    @property
    def ego_vel_source(self) -> str:
        """Which velocity the observation carries for the ego's executed steps:
        ``"commanded"`` (what the policy wrote) or ``"realised"`` (the executed
        positions' backward difference). Reported for the same reason as
        :attr:`sim_agents`: it changes what the policy is conditioned on."""
        return self._ego_vel_source

    @property
    def world_dt_method(self) -> str:
        """``"first"`` or ``"median"``: how the plan's world time grid is read off
        the log's timestamps. Reported because it sets the ego's executed pace."""
        return self._world_dt_method

    @property
    def timestep(self) -> int:
        # The JAX simulator state is the single source of truth for the current
        # timestep. Tracking a separate Python counter drifted from it (Waymax
        # reset sets state.timestep = init_steps - 1, not init_steps), which
        # anchored every replan observation one step ahead of the executed motion.
        return int(self._state.timestep)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # ------------------------------------------------------------------ #
    def reset(self, scenarios: Any, *, seed: int | None = None) -> dict[str, jax.Array]:
        """Reset from a batched (prefix ``[B]``) Waymax scenario pytree.

        ``scenarios`` is a device/host pytree with a leading batch axis (stack B
        unbatched scenarios, e.g. from :meth:`build_batched_scenarios`).
        Returns the first observation (preprocessed feature dict).
        """
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        scen = jax.tree_util.tree_map(jnp.asarray, scenarios)
        self._state = self._jit_reset(scen)
        self._batch_size = int(jnp.asarray(scen.log_trajectory.x).shape[0])

        goal_xy_b2, goal_t_b, _ = infer_goals_from_sim_state(self._state)
        self._goal_xy_b2 = jnp.asarray(goal_xy_b2, dtype=jnp.float32)
        self._goal_t_b = jnp.asarray(goal_t_b, dtype=jnp.int32)

        init_goal_dist_b = jnp.linalg.norm(
            _ego_pose_now_b(self._state) - self._goal_xy_b2, axis=-1
        )
        # Clamp away from 0 so potential-based shaping's normalization (Phi =
        # -dist / dist0) never divides by zero on a scene that starts at the goal.
        self._init_goal_dist_b = jnp.maximum(init_goal_dist_b, 1e-6)
        if self._need_route:
            # Seed prev_s with the actual initial arclength so the first step's
            # progression indicator (s > prev_s) is measured against the start,
            # not a spurious 0 that would always fire on step 1.
            metric_out = self._jit_metrics(self._state)
            init_s_b = jnp.reshape(
                jnp.asarray(metric_out[2], dtype=jnp.float32), (self._batch_size,)
            )
        else:
            init_s_b = jnp.zeros_like(init_goal_dist_b)
        self._reward_state = reward_mod.init_reward_state(
            init_goal_dist_b=init_goal_dist_b,
            init_s_b=init_s_b,
        )
        return self._build_features()

    # ------------------------------------------------------------------ #
    def _build_features(self) -> dict[str, jax.Array]:
        """Preprocess the current *executed* state into policy features.

        Uses a scratch state whose ego log rows are the executed sim motion (so
        the preprocessor observes what the policy actually drove), anchored at
        the current timestep, with each scene's real goal step and goal xy.
        """
        if self._jit_replan_enabled:
            raw, aux, self._rng = self._jit_features(
                self._state, self._rng, self._goal_t_b, self._goal_xy_b2)
            pre_batch = PreprocessBatch(features=raw, aux=aux)
        else:
            scratch = self._splice(self._state)
            self._rng, key = jax.random.split(self._rng)
            pre_batch, _ = preprocess_simulator_state(
                scratch,
                key,
                self._cfg,
                anchor_step_override=self.timestep,
                goal_step_override=self._goal_t_b,   # array: real per-scene goal step
                goal_xy_override=self._goal_xy_b2,
                world_dt_method=self._world_dt_method,
            )
        features = dict(pre_batch.features)
        bsz = self._batch_size
        features["inst_features"] = jnp.zeros((bsz, self._cfg.inst_dim), dtype=jnp.float32)
        features["inst_valid"] = jnp.zeros((bsz,), dtype=jnp.float32)
        features["subgoal_xy"] = jnp.zeros((bsz, 2), dtype=jnp.float32)
        features["subgoal_valid"] = jnp.zeros((bsz,), dtype=jnp.float32)
        self._last_pre_batch = pre_batch
        return features

    def _splice(self, state):
        """The observation's scratch state, under this env's ``ego_vel_source``.
        ``dt_b`` is the preprocessor's own world dt, so the derived velocity is
        on the time base the executed trajectory was resampled onto."""
        dt_b = None
        if self._ego_vel_source == "realised":
            dt_b = compute_world_dt_seconds(state, self._cfg.world_dt_fallback,
                                            self._world_dt_method)
        return _splice_sim_into_log_batched(
            state, ego_vel_source=self._ego_vel_source,
            first_policy_step=self._first_policy_step, dt_b=dt_b)

    def _features_core(self, state, rng, goal_t_b, goal_xy_b2):
        """Traced twin of the eager branch of :meth:`_build_features`.

        The anchor step is ``state.timestep`` as a traced scalar rather than the
        host int, so one executable serves every replan instead of one per t0.
        Returns ``(raw_features, aux, rng)`` with the same key split as eager."""
        scratch = self._splice(state)
        rng, key = jax.random.split(rng)
        pre_batch, _ = preprocess_simulator_state(
            scratch, key, self._cfg,
            anchor_step_override=state.timestep,
            goal_step_override=goal_t_b,
            goal_xy_override=goal_xy_b2,
            world_dt_method=self._world_dt_method,
        )
        return pre_batch.features, pre_batch.aux, rng

    def _prefix_core(self, state, traj, reward_state, goal_xy_b2, *, n_exec):
        """Traced twin of the eager world-step loop in :meth:`_execute_prefix`.

        ``lax.scan`` over the ``n_exec`` executed steps (static), so the waymax
        step + metrics + reward compile once rather than ten times unrolled.
        ``reward_state`` travels as a tuple (``RewardState`` is not a pytree).
        Returns ``(state, reward_state, per_step [B,k], comps {key: [B,k]},
        term_any [B])``."""
        bsz = traj.shape[0]
        need_route = self._need_route
        rcfg = self._reward_cfg

        def body(carry, pose_b5):
            st, rs = carry
            action = datatypes.Action(data=pose_b5,
                                      valid=jnp.ones((bsz, 1), dtype=jnp.bool_))
            st = self._env.step(st, action)
            m = self._env.metrics(st)
            s_b = lateral_b = None
            if need_route:
                s_b = jnp.reshape(m["sdc_progression"].value, (bsz,))
                lateral_b = jnp.reshape(m["sdc_off_route"].value, (bsz,))
            goal_dist_b = jnp.linalg.norm(_ego_pose_now_b(st) - goal_xy_b2, axis=-1)
            reward_b, rs_new, term_b, comps = reward_mod.reward_step(
                rcfg, RewardState(*rs),
                goal_dist_b=goal_dist_b,
                overlap_b=jnp.reshape(m["overlap"].value, (bsz,)),
                offroad_b=jnp.reshape(m["offroad"].value, (bsz,)),
                s_b=s_b, lateral_b=lateral_b)
            rs_new = tuple(getattr(rs_new, f.name)
                           for f in dataclasses.fields(RewardState))
            return (st, rs_new), (reward_b, term_b, comps)

        xs = jnp.transpose(traj[:, 1:n_exec + 1, :], (1, 0, 2))     # [k, B, 5]
        (state, reward_state), (rew_kb, term_kb, comps_kb) = jax.lax.scan(
            body, (state, reward_state), xs)
        per_step_bk = jnp.transpose(rew_kb, (1, 0))
        comps_bk = {k: jnp.moveaxis(v, 0, -1) for k, v in comps_kb.items()}
        return state, reward_state, per_step_bk, comps_bk, jnp.any(term_kb, axis=0)

    # ------------------------------------------------------------------ #
    def sample_trajectory(
        self, features: dict[str, jax.Array], *, rng: jax.Array | None = None
    ) -> jax.Array:
        """Sample one trajectory per world and postprocess to world coords.

        Returns ``trajectory_world_bt5`` of shape ``[B, world_steps, 5]``
        (``x, y, yaw, vel_x, vel_y``), aligned to ``world_t_seconds`` with the
        anchor (current) state as its first element.
        """
        if rng is None:
            self._rng, rng = jax.random.split(self._rng)
        cond_bf, inst_cond_bf = self._jit_condition(features)
        norm_btd = self._policy.sample_from_condition(
            cond_bf, inst_cond_bf, features["inst_valid"], rng=rng
        )
        post = postprocess_predictions(norm_btd, self._last_pre_batch.aux, self._cfg)
        return jnp.asarray(post["trajectory_world_world_dt"], dtype=jnp.float32)

    # ------------------------------------------------------------------ #
    # Glue for external samplers (DPPO / search): expose the frozen condition
    # and the norm->world postprocess so a trainer can sample trajectories with
    # its own (frozen-early / trainable-late) denoiser and feed them to step().
    # ------------------------------------------------------------------ #
    def compute_condition(self, features: dict[str, jax.Array]):
        """Frozen scene/instruction condition ``(cond1, cond2)`` for ``features``.

        The scene tokenizer + instruction/subgoal encoders are frozen, so this is
        the fixed conditioning the DPPO-trainable denoiser is applied on top of.
        """
        return self._jit_condition(features)

    def norm_to_world(self, norm_btd: jax.Array) -> jax.Array:
        """Postprocess a normalized ``[B, T, 5]`` trajectory to world ``[B, W, 5]``
        using the current replan's aux (call after :meth:`_build_features` /
        :meth:`reset`, i.e. once per replan)."""
        post = postprocess_predictions(jnp.asarray(norm_btd), self._last_pre_batch.aux, self._cfg)
        return jnp.asarray(post["trajectory_world_world_dt"], dtype=jnp.float32)

    @property
    def last_aux(self):
        """The current replan's postprocess aux (ego frame, time base, extents).

        Constant for one replan and needed to map a normalized candidate back to
        world coordinates *later* -- the analytic safety fine-tune collects replan
        contexts in a no-grad rollout and differentiates them after the env has
        moved on, so it cannot rely on ``_last_pre_batch``.
        """
        return self._last_pre_batch.aux

    def norm_to_world_with(self, norm_btd: jax.Array, aux) -> jax.Array:
        """:meth:`norm_to_world` against an explicitly supplied ``aux``."""
        if self._jit_replan_enabled:
            # Nested inside an outer jit/grad this simply inlines.
            return self._jit_post(jnp.asarray(norm_btd), aux)
        post = postprocess_predictions(jnp.asarray(norm_btd), aux, self._cfg)
        return jnp.asarray(post["trajectory_world_world_dt"], dtype=jnp.float32)

    @property
    def policy(self):
        """The frozen policy (scene tokenizer + base denoiser)."""
        return self._policy

    # ------------------------------------------------------------------ #
    def _execute_prefix(self, trajectory_world_bt5: jax.Array) -> StepOutput:
        """Execute the first ``replan`` world steps of a world-coord trajectory.

        ``trajectory_world_bt5[:, 0]`` is the anchor (current) state; steps
        ``1..replan`` are applied as ``StateDynamics`` actions, advancing the
        simulated trajectory one world step at a time and accumulating reward.
        """
        traj = jnp.asarray(trajectory_world_bt5, dtype=jnp.float32)
        world_steps = int(traj.shape[1])
        bsz = self._batch_size
        # Number of world steps we can still execute: bounded by the prefix, the
        # candidate length, and the scenario horizon.
        horizon = int(jnp.asarray(self._state.log_trajectory.x).shape[-1])
        max_by_horizon = horizon - 1 - self.timestep
        n_exec = int(min(self._replan, world_steps - 1, max_by_horizon))

        per_step = []
        comp_accum: dict[str, jax.Array] = {}
        comp_steps: dict[str, list[jax.Array]] = {}
        term_any_b = jnp.zeros((bsz,), dtype=bool)
        # Snapshot for the outer-step potential-based shaping (below): which worlds
        # were already absorbed before this prefix, and the goal distance at the
        # prefix anchor (``prev_dist`` is updated every world step by reward_step).
        done_before_b = self._reward_state.done
        phi_anchor_dist_b = self._reward_state.prev_dist

        if self._jit_replan_enabled and n_exec > 0:
            rs = tuple(getattr(self._reward_state, f.name)
                       for f in dataclasses.fields(RewardState))
            self._state, rs, per_step_bk, per_step_comps, term_any_b = self._jit_prefix(
                self._state, traj, rs, self._goal_xy_b2, n_exec=n_exec)
            self._reward_state = RewardState(*rs)
            comp_accum = {k: (jnp.any(v, axis=-1) if k.endswith("_flag")
                              else jnp.sum(v, axis=-1))
                          for k, v in per_step_comps.items()}
            reward_sum_b = jnp.sum(per_step_bk, axis=-1)
        else:
            for k in range(1, n_exec + 1):
                pose_b5 = traj[:, k, :]  # target [x, y, yaw, vel_x, vel_y]
                # env.step -> PlanningAgentDynamics.compute_update tiles BOTH data and
                # valid over the object axis with the same `x[..., newaxis, :]`, so
                # valid must carry a trailing axis (``[B, 1]``) to tile consistently
                # with data (``[B, 5]``); a bare ``[B]`` mis-tiles to ``[num_objects, B]``.
                action = datatypes.Action(
                    data=pose_b5,
                    valid=jnp.ones((bsz, 1), dtype=jnp.bool_),
                )
                self._state = self._jit_step(self._state, action)

                # Single-agent current-step metrics are per world; normalize any
                # trailing singleton (``[B,1]``) to ``[B]`` so they broadcast cleanly
                # against goal_dist_b in the reward.
                s_b = lateral_b = None
                if self._need_route:
                    overlap_b, offroad_b, s_b, lateral_b = self._jit_metrics(self._state)
                    s_b = jnp.reshape(s_b, (bsz,))          # on-route arclength
                    lateral_b = jnp.reshape(lateral_b, (bsz,))  # 0 on-route else dist
                else:
                    overlap_b, offroad_b = self._jit_metrics(self._state)
                overlap_b = jnp.reshape(overlap_b, (bsz,))
                offroad_b = jnp.reshape(offroad_b, (bsz,))
                goal_dist_b = jnp.linalg.norm(
                    _ego_pose_now_b(self._state) - self._goal_xy_b2, axis=-1
                )
                reward_b, self._reward_state, term_b, comps = reward_mod.reward_step(
                    self._reward_cfg,
                    self._reward_state,
                    goal_dist_b=goal_dist_b,
                    overlap_b=overlap_b,
                    offroad_b=offroad_b,
                    s_b=s_b,
                    lateral_b=lateral_b,
                )
                per_step.append(reward_b)
                term_any_b = term_any_b | term_b
                for key, val in comps.items():
                    if key.endswith("_flag"):
                        comp_accum[key] = comp_accum.get(key, jnp.zeros((bsz,), bool)) | val
                    else:
                        comp_accum[key] = comp_accum.get(key, jnp.zeros((bsz,))) + val
                    comp_steps.setdefault(key, []).append(val)

            per_step_comps = {k: jnp.stack(v, axis=-1) for k, v in comp_steps.items()}
            if per_step:
                per_step_bk = jnp.stack(per_step, axis=-1)  # [B, n_exec]
                reward_sum_b = jnp.sum(per_step_bk, axis=-1)
            else:
                per_step_bk = jnp.zeros((bsz, 0), dtype=jnp.float32)
                reward_sum_b = jnp.zeros((bsz,), dtype=jnp.float32)

        # --- potential-based goal shaping (V-Max BraxWrapper parity) --------- #
        # F = coef * (gamma * Phi(s') * flag - Phi(s)),  Phi(s) = -dist / dist0.
        # Added ONCE per outer replan step (not per world step) so it telescopes
        # against the trainer's per-replan GAE discount. `flag` zeroes Phi(s') at
        # an absorbing terminal (collision/offroad/goal reached this prefix) and
        # keeps it at a horizon truncation -- V-Max's exact convention. Worlds
        # already absorbed before this prefix get no shaping. It lives in
        # `reward_sum_b` only, deliberately NOT in `per_step_reward_bk`: with
        # coef=0 (default) it is a no-op and the Gate 3 sum-to-return invariant
        # (outer == sum(per_step) == return) is preserved exactly.
        shaping_b = jnp.zeros((bsz,), dtype=jnp.float32)
        if self._goal_shaping_coef != 0.0 and n_exec > 0:
            denom = (
                self._init_goal_dist_b if self._goal_shaping_normalize
                else jnp.ones((bsz,), dtype=jnp.float32)
            )
            phi_before = -phi_anchor_dist_b / denom
            phi_after = -self._reward_state.prev_dist / denom
            flag = (~term_any_b).astype(jnp.float32)       # 0 at absorbing terminal
            alive_prefix = (~done_before_b).astype(jnp.float32)
            shaping_b = (
                self._goal_shaping_coef
                * (self._goal_shaping_discount * phi_after * flag - phi_before)
                * alive_prefix
            )
            reward_sum_b = reward_sum_b + shaping_b
        comp_accum["shaping"] = shaping_b

        # Truncate any world that ran out of horizon (or hit the prefix cap at
        # the scenario end) without terminating.
        out_of_horizon = (self.timestep >= horizon - 1)
        truncated_b = jnp.full((bsz,), bool(out_of_horizon)) & (~self._reward_state.done)

        next_features = self._build_features()
        return StepOutput(
            features=next_features,
            reward_b=reward_sum_b,
            terminated_b=term_any_b,
            truncated_b=truncated_b,
            per_step_reward_bk=per_step_bk,
            components=comp_accum,
            trajectory_world_bt5=traj,
            per_step_components=per_step_comps,
        )

    # ------------------------------------------------------------------ #
    def step(
        self,
        *,
        trajectory_world_bt5: jax.Array | None = None,
        features: dict[str, jax.Array] | None = None,
        rng: jax.Array | None = None,
    ) -> StepOutput:
        """One outer replan step.

        If ``trajectory_world_bt5`` is given (best-of-N / ES), it is executed
        directly. Otherwise a trajectory is sampled from the current policy using
        ``features`` (defaults to freshly built features for the current state).
        """
        if trajectory_world_bt5 is None:
            if features is None:
                features = self._build_features()
            trajectory_world_bt5 = self.sample_trajectory(features, rng=rng)
        return self._execute_prefix(trajectory_world_bt5)

    # ------------------------------------------------------------------ #
    def ego_log_world_trajectory(self) -> jax.Array:
        """The ego's *logged* world trajectory ``[B, T, 5]`` (for expert replay).

        Layout matches a sampled candidate: ``x, y, yaw, vel_x, vel_y``. Executing
        this from ``start_timestep`` reproduces the human demonstration and must
        yield clean metrics (Gate 1 expert-replay test).
        """
        ego_b = _ego_indices_b(self._state)
        log = self._state.log_trajectory

        def ego_row(field: jax.Array) -> jax.Array:
            return jnp.take_along_axis(field, ego_b[:, None, None], axis=1)[:, 0]

        return jnp.stack(
            [ego_row(log.x), ego_row(log.y), ego_row(log.yaw),
             ego_row(log.vel_x), ego_row(log.vel_y)],
            axis=-1,
        )

    def ego_sim_world_trajectory(self) -> jax.Array:
        """The ego's *simulated* world trajectory ``[B, T, 5]``, same layout as
        :meth:`ego_log_world_trajectory`.

        Pairs with the log one to measure how far the driven rollout has wandered
        from the demonstration -- the degeneracy diagnostic a single-term safety
        objective needs, since falling geometric loss and rising drift together
        mean clearance is being bought by leaving the demonstrated line.
        """
        ego_b = _ego_indices_b(self._state)
        sim = self._state.sim_trajectory

        def ego_row(field: jax.Array) -> jax.Array:
            return jnp.take_along_axis(field, ego_b[:, None, None], axis=1)[:, 0]

        return jnp.stack(
            [ego_row(sim.x), ego_row(sim.y), ego_row(sim.yaw),
             ego_row(sim.vel_x), ego_row(sim.vel_y)],
            axis=-1,
        )

    # ------------------------------------------------------------------ #
    def expert_replay_episode(self, log_override_bt5: jax.Array | None = None) -> dict[str, Any]:
        """Roll the whole episode by feeding the ego's *logged* poses as actions.

        Deterministic (no sampling): at each replan the candidate is the log
        window starting at the current timestep, so the ego tracks the human
        demonstration exactly. Used by the Gate 1 expert-replay / parity /
        permutation tests. Returns aggregated metrics, per-step rewards, and the
        episode return.
        """
        # `log_override_bt5` replaces the human log with a caller-supplied [B,T,5]
        # in the same `x, y, yaw, vel_x, vel_y` layout. Used by the log-replay
        # collision probe to re-run the demonstration under a small perturbation:
        # every non-ego agent is on rails (controlled_object=SDC), so this measures
        # how much of the collision rate is caused by the SDC deviating from the
        # path the other agents were recorded around, rather than by bad driving.
        log_world_bt5 = (self.ego_log_world_trajectory() if log_override_bt5 is None
                         else jnp.asarray(log_override_bt5, dtype=jnp.float32))  # [B, T, 5]
        horizon = int(log_world_bt5.shape[1])
        bsz = self._batch_size

        per_step_chunks: list[jax.Array] = []
        # Per-world-step flag chunks, so a caller can ask *when* inside a prefix
        # the expert first trips a terminating condition -- the control for
        # "does the human log itself absorb in the first replan?".
        flag_chunks: dict[str, list[jax.Array]] = {
            "offroad_flag": [], "collision_flag": [], "reached_flag": []}
        outer_reward_b = jnp.zeros((bsz,), dtype=jnp.float32)
        overlap_any_b = jnp.zeros((bsz,), dtype=bool)
        offroad_any_b = jnp.zeros((bsz,), dtype=bool)
        reached_any_b = jnp.zeros((bsz,), dtype=bool)
        # 1-based replan index at which each world absorbed; 0 == never absorbed.
        absorb_replan_b = jnp.zeros((bsz,), dtype=jnp.int32)

        n_replans = 0
        while True:
            t = self.timestep
            if t >= horizon - 1:
                break
            if self._max_replans is not None and n_replans >= self._max_replans:
                break
            n_replans += 1
            window = log_world_bt5[:, t : t + self._replan + 1, :]
            if int(window.shape[1]) < 2:
                break
            out = self._execute_prefix(window)
            per_step_chunks.append(out.per_step_reward_bk)
            for key, chunks in flag_chunks.items():
                got = out.per_step_components.get(key)
                if got is not None:
                    chunks.append(got)
            newly_b = (absorb_replan_b == 0) & (out.terminated_b | out.truncated_b)
            absorb_replan_b = jnp.where(newly_b, n_replans, absorb_replan_b)
            outer_reward_b = outer_reward_b + out.reward_b
            overlap_any_b = overlap_any_b | out.components.get(
                "collision_flag", jnp.zeros((bsz,), bool)
            )
            offroad_any_b = offroad_any_b | out.components.get(
                "offroad_flag", jnp.zeros((bsz,), bool)
            )
            reached_any_b = reached_any_b | out.components.get(
                "reached_flag", jnp.zeros((bsz,), bool)
            )
            if bool(jnp.all(self._reward_state.done)):
                break

        per_step_bk = (
            jnp.concatenate(per_step_chunks, axis=-1)
            if per_step_chunks
            else jnp.zeros((bsz, 0), dtype=jnp.float32)
        )
        def _cat(key):
            chunks = flag_chunks[key]
            return (jnp.concatenate(chunks, axis=-1) if chunks
                    else jnp.zeros((bsz, 0), dtype=bool))

        return {
            "per_step_reward_bk": per_step_bk,
            "episode_return_b": jnp.sum(per_step_bk, axis=-1),
            "outer_reward_b": outer_reward_b,
            "overlap_any_b": overlap_any_b,
            "offroad_any_b": offroad_any_b,
            "reached_any_b": reached_any_b,
            "n_replans": n_replans,
            "absorb_replan_b": absorb_replan_b,
            "per_step_offroad_bk": _cat("offroad_flag"),
            "per_step_collision_bk": _cat("collision_flag"),
            "per_step_reached_bk": _cat("reached_flag"),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_batched_scenarios(source: Any, indices: list[int]) -> Any:
        """Stack ``len(indices)`` unbatched scenarios from a ScenarioSource into
        a leading-batch-axis pytree suitable for :meth:`reset`.
        """
        scens = [source.get(int(i)) for i in indices]
        return jax.tree_util.tree_map(
            lambda *leaves: jnp.stack([jnp.asarray(x) for x in leaves], axis=0), *scens
        )
