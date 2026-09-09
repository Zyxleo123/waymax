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
window are read from the log at the matching absolute timesteps -- they are
log-replayed in the simulator too, so this is the same approximation the reward
already makes, not a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from rl.diffusion_ft.safety_loss import SafetyLossConfig, ego_box_5dof, safety_terms

__all__ = [
    "SafetyObjectiveConfig",
    "neighbor_window",
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


def neighbor_window(state, t0: int, n_steps: int):
    """Other agents' boxes over absolute world steps ``t0+1 .. t0+n_steps``.

    Read from ``log_trajectory``: every non-ego object is log-replayed by the
    simulator, so the log *is* what the candidate will be driven against. The ego
    is masked out of its own neighbour set -- left in, it overlaps itself at
    ``gap = -2r`` on every step and swamps the term.

    Returns ``(box_btn5, valid_btn)`` with a time axis of length ``n_steps``.
    """
    log = state.log_trajectory
    ego_b = _ego_index_b(state)
    n_obj = jnp.asarray(log.x).shape[1]
    sl = slice(t0 + 1, t0 + 1 + n_steps)

    def to_btn(f):
        return jnp.transpose(jnp.asarray(f)[:, :, sl], (0, 2, 1))     # [B,N,t]->[B,t,N]

    not_ego = ~jax.nn.one_hot(ego_b, n_obj, dtype=bool)               # [B, N]
    box = jnp.stack(
        [to_btn(log.x), to_btn(log.y), to_btn(log.length), to_btn(log.width),
         to_btn(log.yaw)], axis=-1)                                   # [B, t, N, 5]
    return box, to_btn(log.valid) & not_ego[:, None, :]


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
    t0: int,
    n_steps: int,
    alive_b: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Safety loss of a proposed trajectory, differentiable in ``world_traj_bw5``.

    Args:
      world_traj_bw5: ``[B, W, 5]`` postprocessed candidate; index 0 is the anchor
        (the current state), so the scored window is ``1 .. n_steps``.
      state: the batched simulator state at ``t0`` (for the roadgraph, the ego
        extents and the log-replayed neighbours).
      t0: current absolute world timestep -- ``state.timestep``.
      n_steps: world steps to score (bounded by the candidate and the horizon).
      alive_b: ``[B]`` mask; a world that has already absorbed contributes no
        gradient, so a terminated scene cannot keep steering the update.

    Returns the ``safety_terms`` dict plus ``total`` (the scalar to differentiate).
    """
    log = state.log_trajectory
    ego_b = _ego_index_b(state)

    def ego_row(f):
        return jnp.take_along_axis(jnp.asarray(f), ego_b[:, None, None], axis=1)[:, 0]

    length_b = ego_row(log.length)[:, 0]
    width_b = ego_row(log.width)[:, 0]
    # z only steers which roadgraph points are associated (z_stretch) and carries
    # no gradient; take it from the log over the same window.
    z_bt = ego_row(log.z)[:, t0 + 1 : t0 + 1 + n_steps]

    window = world_traj_bw5[:, 1 : 1 + n_steps, :]
    ego_box = ego_window_boxes(window, length_b, width_b)
    other_box, other_valid = neighbor_window(state, t0, n_steps)

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
    )
    # Mean over scored steps, not sum: the window length varies at the end of a
    # scenario, and a scalar whose scale depends on how many steps were left would
    # silently reweight the last replan of every episode.
    n = max(int(n_steps), 1)
    terms["total"] = terms["total"] / n
    return terms
