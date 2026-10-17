"""Closed-loop simulation with reactive (IDM) non-ego agents.

The default pipeline (``predict_planner_trajectories_with_periodic_replan``)
plans every ``replan_interval_steps`` against log-replayed agents and only
replays the stitched ego plan afterwards. With reactive agents that is not
equivalent to closed-loop simulation, so here planning and stepping are
interleaved:

    for each replan step t:
        planner observes the *simulated* past (sim_trajectory[:t+1] copied into
        log_trajectory[:t+1]; the logged future of other agents is kept, so the
        ``gt`` forecast becomes "logged human future", not an oracle);
        ego plan is written into the ego's log from t onwards;
        the env steps ``replan_interval_steps`` times: ego follows its plan
        (expert actor on the rewritten log), every non-SDC object is driven by
        Waymax's default ``IDMRoutePolicy`` (SimAgentConfig IDM / NON_SDC).

The returned state has ``log_trajectory`` equal to the simulated trajectory of
every object, so the existing metric/goal code (which replays the log) scores
the IDM rollout unchanged.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from planner.abstract_planner import AbstractPlanner
from simulation.planning_utils import (
    PredictionBatch,
    apply_ego_replacements_to_expanded_state,
    sim_state_to_ego_trajectory,
)

_TRAJ_FIELDS = ("x", "y", "z", "yaw", "vel_x", "vel_y", "valid", "length", "width", "height", "timestamp_micros")


def _build_idm_env_and_ego_actor(max_num_objects: int, init_steps: int):
    from waymax import agents
    from waymax import config as waymax_config
    from waymax import dynamics as waymax_dynamics
    from waymax import env as waymax_env

    state_dynamics = waymax_dynamics.StateDynamics()
    env_cfg = waymax_config.EnvironmentConfig(
        max_num_objects=int(max_num_objects),
        init_steps=int(init_steps),
        controlled_object=waymax_config.ObjectType.SDC,
    )
    idm = agents.create_sim_agents_from_config(
        waymax_config.SimAgentConfig(
            agent_type=waymax_config.SimAgentType.IDM,
            controlled_objects=waymax_config.ObjectType.NON_SDC,
        )
    )
    env_obj = waymax_env.PlanningAgentEnvironment(
        dynamics_model=state_dynamics,
        config=env_cfg,
        sim_agent_actors=[idm],
        sim_agent_params=[{}],
    )
    ego_actor = agents.create_expert_actor(
        dynamics_model=waymax_env.PlanningAgentDynamics(state_dynamics),
        is_controlled_func=lambda state: state.object_metadata.is_sdc,
    )
    return env_obj, ego_actor


def _overwrite_log_with_sim(base_state, env_state, upto_t: int | None):
    """``base_state`` with log_trajectory[..., :upto_t+1] (or all of it) taken from the simulation.

    Steps after ``upto_t`` keep ``env_state.log_trajectory`` (others: logged future, ego: current plan).
    """
    log = env_state.log_trajectory
    sim = env_state.sim_trajectory
    updates = {}
    for field in _TRAJ_FIELDS:
        if not hasattr(log, field):
            continue
        log_arr = np.array(getattr(log, field), copy=True)
        sim_arr = np.asarray(getattr(sim, field))
        if upto_t is None:
            log_arr[...] = sim_arr
        else:
            log_arr[..., : upto_t + 1] = sim_arr[..., : upto_t + 1]
        updates[field] = log_arr
    return dataclasses.replace(base_state, log_trajectory=log.replace(**updates))


def predict_planner_trajectories_closed_loop_idm(
    cfg,
    sim_state,
    goal,
    planner: AbstractPlanner,
    replan_interval_steps: int,
    instruction_interval_steps: int,
    start_timestep: int,
    use_subgoal: bool,
    rng_key: jax.Array,
):
    num_worlds = int(sim_state.log_trajectory.x.shape[0])
    num_objects = int(sim_state.log_trajectory.x.shape[1])
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    last_t = episode_num_steps - 1

    env_obj, ego_actor = _build_idm_env_and_ego_actor(num_objects, init_steps=start_timestep + 1)

    @jax.jit
    def _env_step(state):
        ego_out = ego_actor.select_action({}, state, None, None)
        return env_obj.step(state, ego_out.action)

    env_state = env_obj.reset(sim_state)
    instructions = [dict() for _ in range(num_worlds)]
    subgoals = [dict() for _ in range(num_worlds)]
    reward_functions = [dict() for _ in range(num_worlds)]
    target_lane_ids = [dict() for _ in range(num_worlds)]

    plan_result = None
    prev_instruction_texts = None
    for step_offset in tqdm(range(start_timestep, last_t, replan_interval_steps)):
        rng_key, key_plan = jax.random.split(rng_key)
        observed_state = _overwrite_log_with_sim(sim_state, env_state, upto_t=step_offset)
        instruction_texts = (
            None if (step_offset - start_timestep) % instruction_interval_steps == 0 else prev_instruction_texts
        )
        plan_result = planner.plan_trajectory(
            observed_state,
            goal,
            rng=key_plan,
            timestep=step_offset,
            mask_goal=cfg.mask_goal,
            instruction_texts=instruction_texts,
            use_subgoal=cfg.use_subgoal,
        )
        instruction_t = getattr(plan_result, "instruction_texts", None)
        subgoal_t = getattr(plan_result, "subgoal_preds", None)
        reward_functions_t = getattr(plan_result, "reward_texts", None)
        target_lane_ids_t = getattr(plan_result, "target_lane_ids", None)
        prev_instruction_texts = instruction_t
        for b in range(num_worlds):
            instructions[b][f"t{step_offset}"] = instruction_t[b] if instruction_t is not None else ""
            subgoals[b][f"t{step_offset}"] = subgoal_t[b] if subgoal_t is not None else None
            reward_functions[b][f"t{step_offset}"] = reward_functions_t[b] if reward_functions_t is not None else None
            target_lane_ids[b][f"t{step_offset}"] = target_lane_ids_t[b] if target_lane_ids_t is not None else None

        plan_traj_bt5 = np.asarray(plan_result.trajectory_world_bt5, dtype=np.float32)
        env_state = apply_ego_replacements_to_expanded_state(
            env_state,
            start_t_b=step_offset,
            traj_bt5=plan_traj_bt5,
        )
        num_steps = min(int(replan_interval_steps), last_t - step_offset)
        for _ in range(num_steps):
            env_state = _env_step(env_state)

    while int(np.asarray(env_state.timestep).reshape(-1)[0]) < last_t:
        env_state = _env_step(env_state)

    final_state = _overwrite_log_with_sim(sim_state, env_state, upto_t=None)

    start_t_b1 = np.full([num_worlds], start_timestep, dtype=np.int32)
    world_dt = float(np.asarray(plan_result.aux["world_dt_seconds"], dtype=np.float32).reshape(-1)[0])
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt
    world_valid = np.ones((num_worlds, episode_num_steps), dtype=bool)
    traj_bt5 = sim_state_to_ego_trajectory(final_state)

    pred = PredictionBatch(
        start_t_b=jnp.asarray(start_t_b1, dtype=jnp.int32),
        trajectory_world_bt5=jnp.asarray(traj_bt5, dtype=jnp.float32),
        world_t_seconds_bt=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bt=jnp.asarray(world_valid),
        aux=plan_result.aux,
    )
    info = {
        "instructions": instructions,
        "subgoals": subgoals,
        "reward_functions": reward_functions,
        "target_lane_ids": target_lane_ids,
        "target_lane_xys": None,
    }
    return pred, final_state, info
