"""Non-ego agent models for :mod:`rl.diffusion_ft.env`.

Waymax ships one reactive sim agent, ``IDMRoutePolicy``: each agent is held on
its **own logged path** by projection and only its speed is free, set by IDM
against whatever box is ahead of it. That is the right skeleton -- longitudinal
reactivity, no invented steering -- but its stock free-road term is not a
traffic model of the recorded scene:

    accel = max_accel * (1 - (v / desired_vel)**delta - (s* / s)**2)

with waymax's defaults ``max_accel=2.0``, ``desired_vel=30.0`` m/s and
``delta=4``. An agent with nothing ahead of it therefore accelerates at
``2 * (1 - (v/30)^4)`` -- 1.98 m/s^2 for a car doing 10 m/s -- toward **108
km/h**, whatever the human did. Over an 8 s episode that is ~16 m/s and ~64 m of
extra distance.

Measured, job 58075 (8 failure scenes, expert replay, so the ego is on the log
and every difference belongs to the NPCs): same-timestep deviation from the log
p99 = **53.7 m**, and 5% of poses end up **7.75 m off the agent's own logged
polyline** -- the latter because an agent that races through its route runs out
of waypoints and ``invalidate_on_end=False`` then extrapolates it straight past
the end. The paired collision probe (job 58076, 32 scenes) shows what that costs:
the human demonstration itself goes from 0.031 to 0.062 collision, 1 m of lateral
offset from 0.062 to 0.188, and the delay column -- the entire reason to want
reactive agents -- does not move at all.

So the stock policy does not measure reactivity. It measures every car flooring
it, which both adds collisions and removes the braking that was supposed to
absorb a late ego (a speeding follower does not brake for you, it arrives
sooner).

:class:`BrakingIDMPolicy` keeps the reactive half and deletes the free-road half.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.agents.waypoint_following_agent import (
    _DEFAULT_LEAD_DISTANCE, _DEFAULT_LEAD_VELOCITY, _MINIMUM_LEAD_DISTANCE,
    IDMRoutePolicy, _add_headway_waypoints, _find_reference_traj_from_log_traj,
)

__all__ = ["BrakingIDMPolicy", "build_sim_agents", "SIM_AGENT_CHOICES"]

SIM_AGENT_CHOICES = ("none", "idm", "idm_brake")


class BrakingIDMPolicy(IDMRoutePolicy):
    """IDM that may only ever go **slower** than the human it is replaying.

    The speed is IDM's, capped at the agent's own logged speed for the timestep
    being produced::

        v = min(idm(v, lead_dist, lead_vel), v_log[t+1])

    With nothing ahead, IDM's free-road term is above the logged speed almost
    everywhere, the cap binds, and the agent reproduces its recording -- so an
    undisturbed scene is log replay and the ``d=0`` control is a control again.
    With the ego ahead, IDM's interaction term drops below the cap, the cap stops
    binding, and the agent brakes. Reactivity is exactly the part that survives.

    **Why this cannot race.** If ``v(tau) <= v_log(tau)`` at every step, then
    ``int v <= int v_log``, so the agent's arclength never exceeds the human's
    and it can never pass the position the recording had at that time. It may
    *catch up* after yielding -- a lagging agent is allowed the current time's
    logged speed, which may be higher than the speed the human had at the point
    the agent is currently at -- and that is fine; catching up is not passing.

    An earlier version indexed the cap by POSITION instead (``v_log[j]`` at the
    nearest waypoint to where the agent is), on the theory that time-indexing
    would let a lagging agent sprint. It does not, by the integral argument
    above, and position-indexing is strictly worse: wherever the human stopped,
    the logged path has a dense cluster of near-identical waypoints, so the
    nearest-waypoint index is ill-conditioned, and an agent that halts in that
    cluster reads a cap of ~0 and **deadlocks there forever**. Measured worse
    too (job 59778: 5.8% of poses "ahead" against the time-indexed version's
    2.4%, both of which are mostly the same stop-cluster artifact in the gate).

    Two further notes:

    * **Degenerate paths.** A parked agent's logged path is effectively one
      point, so ``j`` is arbitrary among identical waypoints and the cap is
      ~0 -- which is correct (it stays parked), though it makes ``j`` unusable
      as a progress measure for such agents. The gate masks them out for that
      reason; the policy does not need to.
    * **``invalidate_on_end=True``**, unlike the waymax default. An agent that
      reaches the end of its logged route is invalidated rather than extrapolated
      straight past the last waypoint -- which is what log replay does at that
      point, and it is what removes the 7.75 m off-polyline tail measured above.

    Everything else -- the projection onto the logged path, the lead-vehicle
    search, the IDM constants -- is waymax's, untouched.
    """

    def __init__(self, *args, invalidate_on_end: bool = True, **kwargs):
        super().__init__(*args, invalidate_on_end=invalidate_on_end, **kwargs)

    def update_speed(self, state, dt: float = 0.1):
        speed, valid = super().update_speed(state, dt)

        # The speed being solved for is the one at `timestep + 1` (see
        # `_get_next_trajectory_by_projection`: dist_travel averages it with the
        # current speed over one dt), so the cap is read there. Clipped because
        # the last replan of an episode asks for a step past the log.
        log = state.log_trajectory
        n_t = jnp.asarray(log.x).shape[-1]
        idx = jnp.clip(state.timestep + 1, 0, n_t - 1)
        nxt = datatypes.dynamic_slice(log, idx, 1, axis=-1)
        # `Trajectory.speed` returns the invalid sentinel (-1), not a speed, on
        # an invalid step. Capping at -1 would freeze every agent whose log has
        # dropped out, so an invalid step means NO cap and IDM is left alone.
        cap = jnp.where(nxt.valid[..., 0], nxt.speed[..., 0], jnp.inf)
        return jnp.minimum(speed, cap), valid

    def _get_accel(self, log_waypoints, cur_position, cur_speed, obj_curr_traj):
        """IDM acceleration with the **ego as the only possible leader**.

        Vendored from `waymax.agents.waypoint_following_agent.IDMRoutePolicy`
        with one change, marked below: the pairwise collision indicator is masked
        so that only the SDC counts as an obstacle. Everything else -- the
        lookahead construction, `geometry.has_overlap`, the lead distance and
        velocity reductions, the IDM formula -- is waymax's.

        **Why.** The purpose of this policy is "replay the recording, except
        brake for the ego". Letting an NPC brake for *another NPC* is not that,
        and it is not free: IDM's spacing constants (`min_spacing` 2 m,
        `safe_time_headway` 2 s) are generic, and real logged traffic follows
        closer than that. So stock IDM invents braking in dense traffic where the
        human did not brake, the ego -- which is executing a perfectly good plan
        against the recorded motion -- drives into the back of it, and the
        collision is charged to the ego.

        That is exactly what job 59781/59812 measured. On `success_heldout`
        (1000 scenes the base solves cleanly) the BASE policy went
        **0.9968 -> 0.9357** under `idm_brake`, entirely through overlap
        (0.0000 -> 0.0610), with `reached` and `offroad` unchanged and a bank
        switch rate of 0.001 -- so it was the simulator, not the intervention.
        A 6% spurious collision floor is the same size as every effect this
        project measures.

        NPC-NPC interaction is not lost by this, it is *restored*: with no NPC
        leader the free-road term is above the logged speed almost everywhere,
        the cap binds, and the agents replay exactly the mutually-consistent
        motion that was recorded. The ego is the only thing the recording did
        not contain, so it is the only thing worth reacting to.
        """
        import numpy as _np
        from waymax.utils import geometry

        num_obj = obj_curr_traj.num_objects
        prefix_shape = obj_curr_traj.shape[:-2]

        if self.lookahead_from_current_position:
            traj = _find_reference_traj_from_log_traj(cur_position, obj_curr_traj, 1)
            total_lookahead = 1 + self.additional_headway_points
        else:
            traj = _find_reference_traj_from_log_traj(
                cur_position, log_waypoints, self.max_lookahead)
            total_lookahead = self.max_lookahead + self.additional_headway_points
        if self.additional_headway_points > 0:
            traj = _add_headway_waypoints(
                traj, distance=self.additional_lookahead_distance,
                num_points=self.additional_headway_points)

        broadcast_shape = prefix_shape + (num_obj, num_obj, total_lookahead, 5)
        traj_5dof = traj.stack_fields(["x", "y", "length", "width", "yaw"])
        traj_bbox = jnp.broadcast_to(
            jnp.expand_dims(traj_5dof, axis=-3), broadcast_shape)
        obj_curr_5dof = obj_curr_traj.stack_fields(
            ["x", "y", "length", "width", "yaw"])
        obj_bbox = jnp.broadcast_to(
            jnp.expand_dims(obj_curr_5dof, axis=-4), broadcast_shape)
        # collision[..., i, j, t]: does obj i's future overlap obj j now.
        collision_pairwise = geometry.has_overlap(traj_bbox, obj_bbox)
        self_mask = jnp.expand_dims(
            jnp.eye(num_obj, dtype=jnp.bool_)[..., jnp.newaxis],
            axis=_np.arange(len(prefix_shape)))
        collision_valid = (
            traj.valid[..., jnp.newaxis, :]
            & obj_curr_traj.valid[..., jnp.newaxis, :, :]
            & ~self_mask
        )
        # ---- THE ONE CHANGE vs waymax ------------------------------------- #
        # Axis -2 is j, the obstacle. Keep only j == the SDC.
        collision_valid = collision_valid & self._is_sdc_j[..., jnp.newaxis]
        # ------------------------------------------------------------------- #
        collision_pairwise = jnp.where(collision_valid, collision_pairwise, False)

        obj_speed_tiled = jnp.broadcast_to(
            jnp.expand_dims(obj_curr_traj.speed, axis=-3), collision_pairwise.shape)
        lead_vel = self._compute_lead_velocity(
            obj_speed_tiled, collision_pairwise,
            obj_curr_traj.valid[..., jnp.newaxis, :, :])
        lead_dist = self._compute_lead_distance(
            agent_future=traj.xyz,
            collision_indicator=collision_pairwise.any(axis=-2),
            current_position=obj_curr_traj.xyz,
            agent_future_valid=traj.valid)

        s_star = self.min_spacing_s0 + jnp.maximum(
            0,
            cur_speed * self.safe_time_headway
            + cur_speed * (cur_speed - lead_vel)
            / (2 * jnp.sqrt(self.max_accel * self.max_decel)))
        s_star = jnp.where(
            (lead_dist == _DEFAULT_LEAD_DISTANCE)
            | (lead_vel == _DEFAULT_LEAD_VELOCITY), 0, s_star)
        lead_dist = jnp.where(lead_dist == 0, _MINIMUM_LEAD_DISTANCE, lead_dist)
        return self.max_accel * (
            1 - (cur_speed / self.desired_vel) ** self.delta
            - (s_star / lead_dist) ** 2)

    def update_trajectory(self, state):
        """Stash the SDC mask for :meth:`_get_accel`, then defer to waymax.

        ``_get_accel``'s signature is fixed by the base class and carries no
        state, so the one extra array it needs rides on ``self`` for the duration
        of the call. Set per call, never read outside it; the policy stays
        functionally stateless from waymax's point of view (``actor_state`` is
        still ``None``), which is what keeps the jitted scan's carry structure
        unchanged.
        """
        self._is_sdc_j = jnp.asarray(state.object_metadata.is_sdc)[..., None, :]
        return super().update_trajectory(state)


def build_sim_agents(kind: str):
    """``(actors, params)`` for ``PlanningAgentEnvironment``, by name.

    ``none`` is the empty pair, i.e. log replay: every non-ego object repeats its
    recorded pose and cannot see the ego. ``idm`` is waymax stock (kept so the
    58075/58076 measurement stays reproducible, NOT because it is a good model --
    read this module's docstring first). ``idm_brake`` is
    :class:`BrakingIDMPolicy`.

    The ego is excluded from `is_controlled_func` because it is driven by the
    planning action; waymax merges the two by that mask inside ``step``. Both
    policies are stateless (``SimAgentActor.init`` returns ``None``), so the
    environment carries ``sim_agent_actor_states = [None]`` -- an empty pytree
    node, free through the jitted scan.
    """
    if kind not in SIM_AGENT_CHOICES:
        raise ValueError(
            f"sim_agents must be one of {SIM_AGENT_CHOICES}, got {kind!r}")
    if kind == "none":
        return (), ()
    non_sdc = lambda st: ~st.object_metadata.is_sdc          # noqa: E731
    cls = IDMRoutePolicy if kind == "idm" else BrakingIDMPolicy
    return (cls(is_controlled_func=non_sdc),), (None,)
