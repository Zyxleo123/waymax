"""Non-ego agent models for the closed-loop and evaluation-only IDM rollouts.

``idm`` is Waymax's stock ``IDMRoutePolicy``: each NPC is held on its own logged
path and IDM sets its speed. Its free-road term is

    accel = max_accel * (1 - (v / desired_vel)**delta - (s* / s)**2)

with ``max_accel=2``, ``desired_vel=30`` m/s, ``delta=4``, so an unobstructed car
at 10 m/s accelerates at ~2 m/s^2 toward 108 km/h whatever the human did, races
through its logged route, runs out of waypoints and (``invalidate_on_end=False``)
is extrapolated straight past the end. Measured in yixiz/waymax_rs (jobs 58075/
58076): NPC same-timestep deviation p99 = 53.7 m, and on the human's own log the
collision rate doubled. It is kept only so earlier ``idm`` results stay
reproducible.

``idm_brake`` is :class:`BrakingIDMPolicy`, vendored from yixiz/waymax_rs@af0b80b
``rl/diffusion_ft/sim_agents.py`` (rationale and measurements are there), plus
one change for this pipeline -- see :func:`build_sim_agents` on late entrants.
"""

from __future__ import annotations

import numpy as np
from jax import numpy as jnp
from waymax import datatypes
from waymax.agents.waypoint_following_agent import (
    _DEFAULT_LEAD_DISTANCE,
    _DEFAULT_LEAD_VELOCITY,
    _MINIMUM_LEAD_DISTANCE,
    IDMRoutePolicy,
    _add_headway_waypoints,
    _find_reference_traj_from_log_traj,
)
from waymax.utils import geometry

REACTIVE_CHOICES = ("idm", "idm_brake")


class BrakingIDMPolicy(IDMRoutePolicy):
    """IDM that may only ever go **slower** than the human it is replaying.

    * Speed is IDM's, capped at the agent's own logged speed for the step being
      produced: ``v = min(idm(v, lead), v_log[t+1])``. With nothing ahead the cap
      binds and the agent replays its recording; with the ego ahead the IDM term
      drops below the cap and it brakes. Since ``v <= v_log`` at every step, its
      arclength never exceeds the human's, so it can never get AHEAD of the log
      (it may catch up after yielding). Time-indexed on purpose: a position-indexed
      cap reads ~0 in the waypoint cluster where the human stopped and deadlocks
      there (yixiz/waymax_rs job 59778).
    * Only the SDC (the ego) counts as a leader. Stock IDM also brakes for other
      NPCs with generic spacing (2 m + 2 s headway) that logged traffic violates,
      inventing stops the human never made that the ego then rear-ends: the base
      policy's clean-scene success went 0.9968 -> 0.9357 from that alone (jobs
      59781/59812). With no NPC leader the NPCs replay their mutually consistent
      recorded motion; the ego is the only thing the recording did not contain.
    * ``invalidate_on_end=True``: an agent that finishes its logged route is
      invalidated, as log replay does, instead of extrapolated past its end.

    Projection onto the logged path, lead search and IDM constants are waymax's.
    """

    def __init__(self, *args, invalidate_on_end: bool = True, **kwargs):
        super().__init__(*args, invalidate_on_end=invalidate_on_end, **kwargs)

    def update_speed(self, state, dt: float = 0.1):
        speed, valid = super().update_speed(state, dt)
        # The speed solved for is the one at timestep + 1; clipped because the last
        # step of an episode asks for one past the log.
        log = state.log_trajectory
        n_t = jnp.asarray(log.x).shape[-1]
        idx = jnp.clip(state.timestep + 1, 0, n_t - 1)
        nxt = datatypes.dynamic_slice(log, idx, 1, axis=-1)
        # Trajectory.speed is the invalid sentinel (-1) on an invalid step: no cap
        # there, or every agent whose log has dropped out would freeze.
        cap = jnp.where(nxt.valid[..., 0], nxt.speed[..., 0], jnp.inf)
        return jnp.minimum(speed, cap), valid

    def _get_accel(self, log_waypoints, cur_position, cur_speed, obj_curr_traj):
        """Waymax's ``IDMRoutePolicy._get_accel`` with the SDC as the only possible leader."""
        num_obj = obj_curr_traj.num_objects
        prefix_shape = obj_curr_traj.shape[:-2]

        if self.lookahead_from_current_position:
            traj = _find_reference_traj_from_log_traj(cur_position, obj_curr_traj, 1)
            total_lookahead = 1 + self.additional_headway_points
        else:
            traj = _find_reference_traj_from_log_traj(cur_position, log_waypoints, self.max_lookahead)
            total_lookahead = self.max_lookahead + self.additional_headway_points
        if self.additional_headway_points > 0:
            traj = _add_headway_waypoints(
                traj, distance=self.additional_lookahead_distance, num_points=self.additional_headway_points
            )

        broadcast_shape = prefix_shape + (num_obj, num_obj, total_lookahead, 5)
        traj_bbox = jnp.broadcast_to(
            jnp.expand_dims(traj.stack_fields(["x", "y", "length", "width", "yaw"]), axis=-3), broadcast_shape
        )
        obj_bbox = jnp.broadcast_to(
            jnp.expand_dims(obj_curr_traj.stack_fields(["x", "y", "length", "width", "yaw"]), axis=-4),
            broadcast_shape,
        )
        # collision[..., i, j, t]: does obj i's lookahead overlap obj j now.
        collision_pairwise = geometry.has_overlap(traj_bbox, obj_bbox)
        self_mask = jnp.expand_dims(
            jnp.eye(num_obj, dtype=jnp.bool_)[..., jnp.newaxis], axis=np.arange(len(prefix_shape))
        )
        collision_valid = (
            traj.valid[..., jnp.newaxis, :] & obj_curr_traj.valid[..., jnp.newaxis, :, :] & ~self_mask
        )
        # The one change vs waymax: axis -2 is j, the obstacle; keep only j == SDC.
        collision_valid = collision_valid & self._is_sdc_j[..., jnp.newaxis]
        collision_pairwise = jnp.where(collision_valid, collision_pairwise, False)

        obj_speed_tiled = jnp.broadcast_to(jnp.expand_dims(obj_curr_traj.speed, axis=-3), collision_pairwise.shape)
        lead_vel = self._compute_lead_velocity(
            obj_speed_tiled, collision_pairwise, obj_curr_traj.valid[..., jnp.newaxis, :, :]
        )
        lead_dist = self._compute_lead_distance(
            agent_future=traj.xyz,
            collision_indicator=collision_pairwise.any(axis=-2),
            current_position=obj_curr_traj.xyz,
            agent_future_valid=traj.valid,
        )

        s_star = self.min_spacing_s0 + jnp.maximum(
            0,
            cur_speed * self.safe_time_headway
            + cur_speed * (cur_speed - lead_vel) / (2 * jnp.sqrt(self.max_accel * self.max_decel)),
        )
        s_star = jnp.where((lead_dist == _DEFAULT_LEAD_DISTANCE) | (lead_vel == _DEFAULT_LEAD_VELOCITY), 0, s_star)
        lead_dist = jnp.where(lead_dist == 0, _MINIMUM_LEAD_DISTANCE, lead_dist)
        return self.max_accel * (1 - (cur_speed / self.desired_vel) ** self.delta - (s_star / lead_dist) ** 2)

    def update_trajectory(self, state):
        # _get_accel's signature is fixed by waymax, so the SDC mask rides on self for
        # the duration of this call (traced once per jit; the actor stays stateless).
        self._is_sdc_j = jnp.asarray(state.object_metadata.is_sdc)[..., None, :]
        return super().update_trajectory(state)


def build_sim_agents(kind: str):
    """``(actors, params)`` for ``PlanningAgentEnvironment``: ``log`` | ``idm`` | ``idm_brake``.

    ``idm_brake`` controls only NPCs that exist in the simulation NOW. Waymax's
    waypoint policies AND their update's validity with the current one, so an object
    that enters the scene after reset (out of occlusion, into sensor range) and is
    controlled from the start is injected by the environment at the invalid
    sentinel pose (-1, -1) instead of its logged one -- it never really appears,
    though log replay shows it. Leaving it uncontrolled while invalid lets the
    environment inject it at its logged pose; IDM takes it over from the next step.
    An agent the policy invalidates (end of its route) is re-injected only if its
    log is still valid there, which a policy that cannot get ahead of the log
    reaches only at the log's own end. ``idm`` keeps waymax's mask, unchanged.
    """
    if kind == "log":
        return [], []
    if kind == "idm":
        return [IDMRoutePolicy(is_controlled_func=lambda st: ~st.object_metadata.is_sdc)], [None]
    if kind == "idm_brake":
        def present_npc(st):
            return ~st.object_metadata.is_sdc & st.current_sim_trajectory.valid[..., 0]

        return [BrakingIDMPolicy(is_controlled_func=present_npc)], [None]
    raise ValueError(f"Unknown non-ego agent model {kind!r}; expected log | {' | '.join(REACTIVE_CHOICES)}.")
