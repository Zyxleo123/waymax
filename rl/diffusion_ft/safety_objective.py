"""The safety loss as a *trainable* objective on a sampled diffusion candidate.

:mod:`rl.diffusion_ft.safety_loss` scores geometry that already happened. This
module closes the loop: it scores the trajectory the policy is *proposing*, as a
differentiable function of the denoiser parameters, so the fine-tune gets an
analytic gradient instead of a score-function estimate. That is the entire point
of the exercise -- waymax's ``offroad``/``overlap`` are indicators with no
gradient, and buying that signal back through RL is what has been costing
variance and sample count.

The differentiable chain is::

    params -> sample_with_trace -> normalized [B, C, T]
           -> postprocess_predictions -> world [B, W, 5]
           -> ego bbox corners -> safety_terms -> scalar

Everything after the sampler is affine un-normalization, a rotation and a linear
resample, all differentiable; ``aux`` is a constant of the replan. The expensive
roadgraph nearest-point search inside ``safety_terms`` is already stop-gradiented
and stays that way.

**A safety term alone has a degenerate minimum.** Nothing here rewards progress,
so the cheapest trajectory is to stop in the middle of the widest open space, or
to leave the road network for an empty field. The objective is therefore always
``safety + anchor``: the anchor is the policy's own pretraining loss on the human
demonstration, and it is what keeps "safe" from collapsing into "motionless". Do
not run this with ``anchor_weight=0`` except as a deliberate ablation -- the
degeneracy check in ``tests/gate_safety_grad.py`` exists because of it.

**What gets scored.** Only the window the environment will actually execute
(``window`` world steps, defaulting to the replan interval), not the full 25-step
candidate. Steps past the replan are re-planned before they are ever driven, so
charging them prices a trajectory that will never exist. Neighbours over that
window are read from the log at the matching absolute timesteps -- under the
default log-replayed NPCs they are driven from that same log, so this is not an
approximation at all, it is the identical motion. Under reactive NPCs
(``DiffusionWaymaxEnv(sim_agents="idm")``) it becomes one, and
``candidate_safety(..., neighbor_state=...)`` reads the executed NPC motion
instead. See :func:`neighbor_window`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from rl.diffusion_ft.safety_loss import SafetyLossConfig, ego_box_5dof, safety_terms

__all__ = [
    "SafetyObjectiveConfig",
    "neighbor_window",
    "_window_slice",
    "ego_window_boxes",
    "candidate_safety",
]


@dataclass(frozen=True)
class SafetyObjectiveConfig:
    """Weights and window for the trainable objective."""

    safety: SafetyLossConfig = field(default_factory=SafetyLossConfig)
    # Scale on the geometric term. The safety loss is in metres-equivalent and a
    # violating frame runs ~1-20, so this multiplies something already O(1).
    safety_weight: float = 1.0
    # Scale on the expert denoising anchor. Non-zero is not optional (see the
    # module docstring): it is the only thing in the objective that knows the car
    # is supposed to go somewhere.
    anchor_weight: float = 1.0
    # World steps of the candidate to score. None == the env's replan interval,
    # which is exactly the part that gets driven.
    window: int | None = None


def _ego_index_b(state) -> jax.Array:
    return jnp.argmax(jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32), axis=1)


def _concrete_int(v) -> int | None:
    """``int(v)`` when ``v`` is a host-readable scalar, else ``None``.

    Used only by preconditions: a traced or batched value simply skips the
    check rather than failing the trace.
    """
    try:
        if jnp.ndim(v) != 0:
            return None
        return int(v)
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return None


def _window_slice(arr, t0, n_steps: int, axis: int):
    """``arr`` sliced to ``n_steps`` steps starting at ``t0+1`` along ``axis``.

    ``t0`` may be a scalar (one absolute timestep for the whole batch -- every
    on-policy replan context, where the batch shares a replan boundary) or a
    ``[B]`` array (a replayed minibatch, whose rows were recorded at different
    replans of different episodes). The scalar path is the original
    ``dynamic_slice_in_dim`` and stays bit-for-bit what it was; the per-row path
    vmaps the same slice over the leading batch axis, which is why ``axis`` is
    given relative to ``arr`` and decremented inside the map.
    """
    n = int(n_steps)
    if jnp.ndim(t0) == 0:
        return jax.lax.dynamic_slice_in_dim(jnp.asarray(arr), t0 + 1, n, axis=axis)
    return jax.vmap(
        lambda a, s: jax.lax.dynamic_slice_in_dim(a, s + 1, n, axis=axis - 1)
    )(jnp.asarray(arr), jnp.asarray(t0))


def neighbor_window(state, t0, n_steps: int, *, neighbor_state=None):
    """Other agents' boxes over absolute world steps ``t0+1 .. t0+n_steps``.

    **Where the neighbours come from is the whole question.** With log-replayed
    NPCs (``DiffusionWaymaxEnv(sim_agents="none")``, the default) the log *is*
    what the candidate will be driven against -- each agent repeats its recorded
    pose no matter what the ego does -- so reading ``log_trajectory`` is exact,
    not an approximation. Under reactive NPCs (``sim_agents="idm"``) it stops
    being exact: an IDM car brakes for the ego, so the box the candidate will
    actually meet is *behind* the logged one, and charging the candidate for
    the logged one prices a hazard that reacted its way out of existence.

    ``neighbor_state`` switches the source to that state's ``sim_trajectory``,
    i.e. where the NPCs *actually were* when this window was driven. It is the
    state the episode ended in, held by the caller; the gradient pass runs after
    the rollout, so those steps are already written (see the precondition check
    below). Pass ``None`` -- the default -- and this is byte-for-byte the
    log-reading function it has always been.

    The ego is masked out of its own neighbour set either way: left in, it
    overlaps itself at ``gap = -2r`` on every step and swamps the term.

    Returns ``(box_btn5, valid_btn)`` with a time axis of length ``n_steps``.
    """
    if neighbor_state is None:
        log = state.log_trajectory
    else:
        # `sim_trajectory` is filled invalid at reset and written one world step
        # at a time, so a window past `timestep` reads x=-1, valid=False -- and
        # `safety_terms` masks invalid pairs to zero. The agent term would go
        # SILENTLY to 0 rather than fail. Check it wherever both numbers are
        # concrete (every on-policy context; the replayed-minibatch path passes a
        # traced [B] t0 and is checked by its caller instead).
        t_last, t_now = _concrete_int(t0), _concrete_int(neighbor_state.timestep)
        if t_last is not None and t_now is not None:
            t_last += int(n_steps)
            if t_last > t_now:
                raise ValueError(
                    f"neighbor_state has only been simulated to timestep {t_now} "
                    f"but the scored window ends at {t_last}. Reading "
                    f"sim_trajectory there returns invalid placeholders and the "
                    f"agent term would silently read 0.")
        log = neighbor_state.sim_trajectory
    ego_b = _ego_index_b(state)
    n_obj = jnp.asarray(log.x).shape[1]
    # dynamic_slice, not a Python slice, so ``t0`` may be traced: one compiled
    # gradient then serves every replan. Identical for a Python int because the
    # caller clips ``n_steps`` to the horizon (no start-index clamping kicks in).
    def to_btn(f):
        w = _window_slice(f, t0, n_steps, axis=2)
        return jnp.transpose(w, (0, 2, 1))                            # [B,N,t]->[B,t,N]

    not_ego = ~jax.nn.one_hot(ego_b, n_obj, dtype=bool)               # [B, N]
    box = jnp.stack(
        [to_btn(log.x), to_btn(log.y), to_btn(log.length), to_btn(log.width),
         to_btn(log.yaw)], axis=-1)                                   # [B, t, N, 5]
    return box, to_btn(log.valid) & not_ego[:, None, :]


# Where the other agents' FUTURE boxes come from when scoring a candidate. See
# `predicted_neighbor_window`; the wm_* sources are learned world models whose
# boxes are computed outside the jit (rl.diffusion_ft.world_model_forecast).
WM_PREDICTIONS = ("wm_det", "wm_mf", "wm_mf_top1")
NEIGHBOR_PREDICTIONS = ("log", "cv", "none") + WM_PREDICTIONS
# Sources whose boxes the CALLER supplies (``candidate_safety(neighbor_boxes=)``):
# the wm_* ones, and "external" -- any predictor outside this package, e.g. an ES
# planner handing over the very forecast it ranks with. Not a CLI choice.
BOX_PREDICTIONS = WM_PREDICTIONS + ("external",)


def predicted_neighbor_window(state, t0, n_steps: int, *, mode: str, dt_b=None):
    """Other agents' boxes over ``t0+1 .. t0+n_steps`` WITHOUT reading the future.

    :func:`neighbor_window` reads the recorded future. Under log replay that is
    exactly where each car will be, so a selector scored on it knows the other
    cars' next 2 s -- privileged information no deployed planner has (the policy
    itself only sees their CURRENT state). This is the honest counterpart:

    * ``"cv"``   -- constant velocity: each agent's pose, velocity and extent at
      ``t0`` (read from ``sim_trajectory``, i.e. where it actually IS, reactive or
      not), extrapolated ``x + v * k * dt`` with the heading held. The standard
      baseline predictor. An agent invalid at ``t0`` is invisible for the whole
      window -- a car that has not appeared yet cannot be predicted.
    * ``"none"`` -- no agents at all: road edges only. The floor.

    ``dt_b`` ``[B]``: seconds per world step (the scene's clock); 0.1 if None.
    ``t0`` must be a scalar (plan-time scoring). Same return contract as
    :func:`neighbor_window`: ``(box_btn5, valid_btn)``, ego masked out.
    """
    if mode not in ("cv", "none"):
        raise ValueError(f"predicted_neighbor_window mode must be cv|none, got {mode!r}")
    sim = state.sim_trajectory
    ego_b = _ego_index_b(state)
    n_obj = jnp.asarray(sim.x).shape[1]
    n = int(n_steps)

    def now(f):                                                       # [B, N] at t0
        return _window_slice(f, t0 - 1, 1, axis=2)[..., 0]

    not_ego = ~jax.nn.one_hot(ego_b, n_obj, dtype=bool)               # [B, N]
    valid_bn = now(sim.valid) & not_ego
    if mode == "none":
        valid_bn = jnp.zeros_like(valid_bn)
    dt = (jnp.full((jnp.asarray(sim.x).shape[0],), 0.1, jnp.float32) if dt_b is None
          else jnp.asarray(dt_b, jnp.float32))
    k = jnp.arange(1, n + 1, dtype=jnp.float32)                       # [t]
    step = k[None, :, None] * dt[:, None, None]                       # [B, t, 1]
    x = now(sim.x)[:, None, :] + now(sim.vel_x)[:, None, :] * step    # [B, t, N]
    y = now(sim.y)[:, None, :] + now(sim.vel_y)[:, None, :] * step

    def hold(f):
        return jnp.broadcast_to(now(f)[:, None, :], x.shape)

    box = jnp.stack([x, y, hold(sim.length), hold(sim.width), hold(sim.yaw)], axis=-1)
    return box, jnp.broadcast_to(valid_bn[:, None, :], x.shape)


def ego_window_boxes(world_bt5: jax.Array, length_b, width_b) -> jax.Array:
    """Ego 5-dof boxes from a world-coordinate candidate window.

    ``world_bt5`` is ``[B, t, 5] = (x, y, yaw, vel_x, vel_y)`` -- already sliced
    to the scored window, so index 0 here is the first *executed* step, not the
    anchor. Velocity columns are dropped; the geometry only needs the pose.
    """
    pose = jnp.stack([world_bt5[..., 0], world_bt5[..., 1], world_bt5[..., 2]], axis=-1)
    return ego_box_5dof(pose, length_b[:, None], width_b[:, None])


def candidate_safety(
    cfg: SafetyObjectiveConfig,
    world_traj_bw5: jax.Array,
    *,
    state,
    t0,
    n_steps: int,
    alive_b: jax.Array | None = None,
    neighbor_state=None,
    neighbor_pred: str = "log",
    neighbor_dt_b: jax.Array | None = None,
    neighbor_boxes=None,
) -> dict[str, jax.Array]:
    """Safety loss of a proposed trajectory, differentiable in ``world_traj_bw5``.

    Args:
      world_traj_bw5: ``[B, W, 5]`` postprocessed candidate; index 0 is the anchor
        (the current state), so the scored window is ``1 .. n_steps``.
      state: the batched simulator state at ``t0`` (for the roadgraph, the ego
        extents and the log-replayed neighbours).
      t0: current absolute world timestep -- ``state.timestep``. A scalar (the
        whole batch is at one replan boundary, which is every on-policy context)
        or ``[B]`` (a replayed minibatch whose rows come from different replans
        of different episodes; see :mod:`rl.diffusion_ft.transition_buffer`).
      n_steps: world steps to score (bounded by the candidate and the horizon).
      alive_b: ``[B]`` mask; a world that has already absorbed contributes no
        gradient, so a terminated scene cannot keep steering the update.
      neighbor_state: optional state whose ``sim_trajectory`` supplies the other
        agents' boxes instead of ``state.log_trajectory``. Required to be
        meaningful only when the NPCs are reactive
        (``DiffusionWaymaxEnv(sim_agents="idm")``) -- under log replay the two
        sources are the same motion, so passing it there changes nothing but
        costs a slice. See :func:`neighbor_window`.
      neighbor_pred: ``"log"`` (default: the recorded future, i.e. an oracle under
        log replay), or ``"cv"`` / ``"none"``, which never read past ``t0`` -- see
        :func:`predicted_neighbor_window`. ``neighbor_dt_b`` is its per-scene dt.
        A ``wm_*`` source takes its forecast from ``neighbor_boxes`` =
        ``(box [B, n, M, 5], valid [B, n, M], weight [B, n, M])``, already decoded
        on the scene's clock (``world_model_forecast.WMNeighborForecaster``).

    Returns the ``safety_terms`` dict plus ``total`` (the scalar to differentiate)
    and ``n_neighbors`` (mean valid ego-neighbour pairs per scored step). Read
    ``n_neighbors``: it is what goes to 0 if the neighbour source is empty, which
    is otherwise indistinguishable from a scene with a clear road.
    """
    log = state.log_trajectory
    ego_b = _ego_index_b(state)

    def ego_row(f):
        return jnp.take_along_axis(jnp.asarray(f), ego_b[:, None, None], axis=1)[:, 0]

    length_b = ego_row(log.length)[:, 0]
    width_b = ego_row(log.width)[:, 0]
    # z only steers which roadgraph points are associated (z_stretch) and carries
    # no gradient; take it from the log over the same window.
    z_bt = _window_slice(ego_row(log.z), t0, n_steps, axis=1)

    window = world_traj_bw5[:, 1 : 1 + n_steps, :]
    ego_box = ego_window_boxes(window, length_b, width_b)
    other_weight = None
    if (neighbor_boxes is not None) != (neighbor_pred in BOX_PREDICTIONS):
        raise ValueError(f"neighbor_boxes is required by, and only by, {BOX_PREDICTIONS} "
                         f"(neighbor_pred={neighbor_pred!r})")
    if neighbor_pred == "log":
        other_box, other_valid = neighbor_window(
            state, t0, n_steps, neighbor_state=neighbor_state)
    elif neighbor_state is not None:
        raise ValueError("neighbor_state applies to neighbor_pred='log' only")
    elif neighbor_boxes is not None:
        other_box, other_valid, other_weight = neighbor_boxes
    else:
        other_box, other_valid = predicted_neighbor_window(
            state, t0, n_steps, mode=neighbor_pred, dt_b=neighbor_dt_b)

    if alive_b is None:
        step_valid = jnp.ones(ego_box.shape[:2], jnp.float32)
    else:
        step_valid = jnp.broadcast_to(
            jnp.asarray(alive_b, jnp.float32)[:, None], ego_box.shape[:2])

    terms = safety_terms(
        cfg.safety,
        ego_box_bt5=ego_box,
        ego_z_bt=z_bt,
        roadgraph_points=state.roadgraph_points,
        other_box_btn5=other_box,
        other_valid_btn=other_valid,
        step_valid_bt=step_valid,
        other_weight_btn=other_weight,
    )
    # Mean over scored steps, not sum: the window length varies at the end of a
    # scenario, and a scalar whose scale depends on how many steps were left would
    # silently reweight the last replan of every episode.
    n = max(int(n_steps), 1)
    terms["total"] = terms["total"] / n
    # How many neighbours the term was actually charged against. Zero here means
    # the agent barrier contributed nothing -- which happens both on an empty
    # road and on a misaddressed neighbour window, and those must not look alike.
    # Weighted, so a K-mode forecast counts each agent once (its modes sum to 1).
    nb_valid = jnp.asarray(other_valid, jnp.float32)
    if other_weight is not None:
        nb_valid = nb_valid * jnp.asarray(other_weight, jnp.float32)
    terms["n_neighbors"] = jnp.mean(jnp.sum(nb_valid, axis=-1))
    return terms
