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

import jax
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
    _project_to_a_trajectory,
)
from waymax.utils import geometry

REACTIVE_CHOICES = ("idm", "idm_brake", "idm_faithful")


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
        return self._accel_and_lead_dist(log_waypoints, cur_position, cur_speed, obj_curr_traj)[0]

    def _accel_and_lead_dist(self, log_waypoints, cur_position, cur_speed, obj_curr_traj):
        """``(accel, lead_dist)``; ``lead_dist`` is ``_DEFAULT_LEAD_DISTANCE`` when no leader is in the lookahead."""
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
        found = lead_dist
        lead_dist = jnp.where(lead_dist == 0, _MINIMUM_LEAD_DISTANCE, lead_dist)
        accel = self.max_accel * (1 - (cur_speed / self.desired_vel) ** self.delta - (s_star / lead_dist) ** 2)
        return accel, found

    def update_trajectory(self, state):
        # _get_accel's signature is fixed by waymax, so the SDC mask rides on self for
        # the duration of this call (traced once per jit; the actor stays stateless).
        self._is_sdc_j = jnp.asarray(state.object_metadata.is_sdc)[..., None, :]
        return super().update_trajectory(state)


class FaithfulIDMPolicy(BrakingIDMPolicy):
    """``idm_brake`` with the two defects the expert replay exposed removed.

    scripts/diagnose_idm_gap.py replays the human's own log (val10): stock ``idm`` makes
    it collide in ~6.5% of scenes (log replay 0.2%), and ``idm_brake`` still invents
    collisions of two kinds, both fixed here:

    * **Lag.** IDM's free-road acceleration is at most ``max_accel`` = 2 m/s^2, so an NPC
      whose human pulled away harder falls behind its recording with nothing in its way,
      and ``v <= v_log`` keeps the lag forever; the ego, on the human's path, rear-ends it.
      An NPC with no leader here accelerates at ``max(IDM, a_log)`` (``a_log`` = its human's
      own acceleration at this step), so on schedule it reproduces ``v_log`` exactly, and
      after braking for the ego it recovers at >= ``max_accel``. With the ego ahead it is
      plain IDM, still capped at ``v_log``.
    * **Existence.** waymax's ``apply_trajectory_update_to_state`` keeps a controlled object
      that exists valid whatever the policy says (``next_valid = current.valid`` for
      ``exist_and_controlled``). So ``invalidate_on_end`` does not remove a finished NPC: it
      parks it at the (-1, -1) sentinel, still valid, and the next projection drops it back
      onto its path; and an NPC whose log has ended is driven on past it. Here an NPC is
      controlled only while it exists AND its log is valid at t+1 (``build_sim_agents``);
      otherwise the environment takes it from the log, which removes it when the log does
      and injects late entrants at their logged pose -- log replay's existence exactly.
      ``invalidate_on_end`` is off, so the policy never writes the sentinel.
    * **Run-away.** Once an agent is at the end of its logged path or > 5 m off it, waymax's
      projection (``invalidate_on_end=False``) freezes its velocity and extrapolates it in a
      straight line for good: a jittery parked car whose log reads 27 m/s for one step leaves
      at 27 m/s and never comes back (val10 shard 5 scene 169; stock idm's whole ``off_path``
      class, p50 16 m off). Here such an agent is snapped to its nearest logged waypoint with
      zero velocity and stays valid -- waymax's ``invalidate_on_end=True`` branch without the
      invalidation that the environment turns into the sentinel teleport.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("invalidate_on_end", False)
        super().__init__(*args, **kwargs)

    def update_speed(self, state, dt: float = 0.1):
        cur = state.current_sim_trajectory
        cur_speed = cur.speed[..., 0]
        accel, lead_dist = self._accel_and_lead_dist(state.log_trajectory, cur.xyz[..., 0, :], cur_speed, cur)
        log = state.log_trajectory
        n_t = jnp.asarray(log.x).shape[-1]
        now = datatypes.dynamic_slice(log, jnp.clip(state.timestep, 0, n_t - 1), 1, axis=-1)
        nxt = datatypes.dynamic_slice(log, jnp.clip(state.timestep + 1, 0, n_t - 1), 1, axis=-1)
        v_now, v_next = now.speed[..., 0], nxt.speed[..., 0]
        a_log = jnp.where(now.valid[..., 0] & nxt.valid[..., 0], (v_next - v_now) / dt, 0.0)
        accel = jnp.where(lead_dist == _DEFAULT_LEAD_DISTANCE, jnp.maximum(accel, a_log), accel)
        speed = jnp.maximum(cur_speed + dt * accel, 0.0)
        cap = jnp.where(nxt.valid[..., 0], v_next, jnp.inf)
        return jnp.minimum(speed, cap), jnp.ones_like(cur_speed, dtype=jnp.bool_)

    def _get_next_trajectory_by_projection(self, log_traj, cur_sim_traj, new_speed, new_speed_valid, dt=0.1):
        """waymax's ``WaypointFollowingPolicy._get_next_trajectory_by_projection``, minus the run-away:
        an agent at its path's end or > 5 m off it snaps to the nearest waypoint at rest (no extrapolation,
        no frozen velocity) and keeps its validity."""
        cur_speed = cur_sim_traj.speed
        valid = new_speed_valid[..., jnp.newaxis]
        new_speed = jnp.where(valid, new_speed[..., jnp.newaxis], cur_speed)
        dist_travel = (new_speed + cur_speed) / 2 * dt
        next_x = cur_sim_traj.x + dist_travel * jnp.cos(cur_sim_traj.yaw)
        next_y = cur_sim_traj.y + dist_travel * jnp.sin(cur_sim_traj.yaw)
        next_xy, next_yaw, reached = _project_to_a_trajectory(
            jnp.stack([next_x, next_y], axis=-1), log_traj, extrapolate_traj=False
        )
        next_traj = cur_sim_traj.replace(
            x=next_xy[..., 0],
            y=next_xy[..., 1],
            yaw=next_yaw,
            vel_x=jnp.where(reached, 0.0, new_speed * jnp.cos(cur_sim_traj.yaw)),
            vel_y=jnp.where(reached, 0.0, new_speed * jnp.sin(cur_sim_traj.yaw)),
            valid=valid & cur_sim_traj.valid,
        )
        next_traj.validate()
        invalid_traj = jax.tree_util.tree_map(datatypes.make_invalid_data, next_traj)
        return jax.tree_util.tree_map(lambda x, y: jnp.where(next_traj.valid, x, y), next_traj, invalid_traj)


def _exists_with_log_next(st):
    """NPCs that exist now and whose log is valid at t+1; the environment moves the rest by the log."""
    n_t = jnp.asarray(st.log_trajectory.x).shape[-1]
    nxt = datatypes.dynamic_slice(st.log_trajectory, jnp.clip(st.timestep + 1, 0, n_t - 1), 1, axis=-1)
    return ~st.object_metadata.is_sdc & st.current_sim_trajectory.valid[..., 0] & nxt.valid[..., 0]


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
    if kind == "idm_faithful":
        return [FaithfulIDMPolicy(is_controlled_func=_exists_with_log_next)], [None]
    raise ValueError(f"Unknown non-ego agent model {kind!r}; expected log | {' | '.join(REACTIVE_CHOICES)}.")
