from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import re
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from glob import glob

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scenario_loader import load_scenario_state_batch_fast
from lane_graph.lane_graph_utils import LaneGraphLoader, lane_graph_zip_path_for_tfrecord
from scores import helpers as scorer_helpers
from scores.scorer_lane_graph import LaneGraphShardStore
from simulation.evaluation_utils import (
    rollout_predicted_trajectories_with_metrics,
    infer_goals_from_sim_state, 
    check_goal_reaching, 
    check_traffic_light_violation
)
from simulation.planning_utils import (
    predict_planner_trajectories_with_periodic_replan,
    get_sdc_indices_for_batched_state,
    sim_state_to_ego_trajectory,
)
from planner.abstract_planner import AbstractPlanner, PlannerResult

from simulation.utils import _save_json  # noqa: E402
from viz.render import render_videos_batched
from waymax import config as waymax_config
from tqdm import tqdm
try:
    # reuse extraction logic from scripts/build_cache if available
    from scripts import build_cache as _build_cache
except Exception:
    _build_cache = None


_SCENARIO_RESULT_RE = re.compile(r"^(?P<tfrecord_name>.+)\.scenario_(?P<scenario_idx>\d+)\.json$")


def _load_completed_scenarios(start_from: str | None) -> set[tuple[str, int]]:
    if not start_from:
        return set()

    completed: set[tuple[str, int]] = set()
    start_from_path = Path(start_from)
    if not start_from_path.exists():
        print(f"start_from path does not exist: {start_from_path}")
        return completed

    for result_path in start_from_path.rglob("*.json"):
        match = _SCENARIO_RESULT_RE.match(result_path.name)
        if match is None:
            continue
        completed.add((match.group("tfrecord_name"), int(match.group("scenario_idx"))))

    if completed:
        print(f"Resuming from {start_from_path}; found {len(completed)} completed scenario results.")
    return completed


def run(args, planner: AbstractPlanner, log_replay=False) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_scenario_path is not None:
        with open(args.target_scenario_path, "r") as f:
            target_scenarios = json.load(f)
    else:
        target_scenarios = None

    completed_scenarios = _load_completed_scenarios(getattr(args, "start_from", None))
    tested_scenarios, curr_scenario_index = 0, 0
    target_scenario_cursor = 0
    reached_all, success_all = 0, 0
    tfrecord_paths = sorted(glob(str(Path(args.tfrecord_dir) / "*")))
    if not tfrecord_paths:
        raise FileNotFoundError(f"No tfrecords found under {args.tfrecord_dir}")

    def _advance_tfrecord() -> bool:
        """Move to next shard. Returns False when this split is exhausted (do not pop empty)."""
        nonlocal tfrecord_path, curr_scenario_index, ds_cfg, target_scenario_cursor
        if not tfrecord_paths:
            print(
                f"Scenarios in {os.path.basename(tfrecord_path)} are all tested; "
                "no more tfrecords left — finishing this split."
            )
            return False
        print(
            f"Scenarios in {os.path.basename(tfrecord_path)} are all tested, "
            "moving to next tfrecord."
        )
        tfrecord_path = tfrecord_paths.pop(0)
        curr_scenario_index = 0
        target_scenario_cursor = 0
        ds_cfg = dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING,
            path=str(tfrecord_path),
            max_num_objects=int(args.max_num_objects),
            batch_dims=(args.num_worlds,),
            shuffle_seed=0,
        )
        return True

    tfrecord_path = tfrecord_paths.pop(0)
    # lane_graph_loader = LaneGraphLoader(args.lane_graph_dir)
    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(args.num_worlds,),
        shuffle_seed=0,
    )
    if args.scenario_indices is not None:
        target_scenario_indices = [int(idx) for idx in args.scenario_indices.split(",")]
    else:
        target_scenario_indices = None

    lane_graph_zip_path = None
    while tested_scenarios < int(args.num_scenarios):
        if target_scenarios is not None:
            scenario_base_name = os.path.basename(tfrecord_path)
            # Skip shards that are not in the target map (or have an empty list).
            if scenario_base_name not in target_scenarios or not target_scenarios[scenario_base_name]:
                if not _advance_tfrecord():
                    break
                continue
            scenario_indices = []
            while (
                len(scenario_indices) < int(args.num_worlds)
                and target_scenario_cursor < len(target_scenarios[scenario_base_name])
            ):
                scenario_idx = int(target_scenarios[scenario_base_name][target_scenario_cursor])
                target_scenario_cursor += 1
                if (scenario_base_name, scenario_idx) in completed_scenarios:
                    continue
                scenario_indices.append(scenario_idx)
            if not scenario_indices:
                # Finished all targets on this shard (or all remaining were already done).
                if not _advance_tfrecord():
                    break
                continue
            # Partial last batch on a shard: planner/batch_dims expect num_worlds.
            # Drop leftovers rather than crash; prefer target lists divisible by num_worlds.
            if len(scenario_indices) < int(args.num_worlds):
                print(
                    f"[target] dropping incomplete last batch of {len(scenario_indices)} "
                    f"on {scenario_base_name} (need num_worlds={args.num_worlds})."
                )
                if not _advance_tfrecord():
                    break
                continue
        elif target_scenario_indices is not None:
            scenario_indices = []
            while len(scenario_indices) < int(args.num_worlds) and target_scenario_cursor < len(target_scenario_indices):
                scenario_idx = int(target_scenario_indices[target_scenario_cursor])
                target_scenario_cursor += 1
                if (os.path.basename(tfrecord_path), scenario_idx) in completed_scenarios:
                    continue
                scenario_indices.append(scenario_idx)
            if not scenario_indices:
                break
        else:
            scenario_indices = []
            while len(scenario_indices) < int(args.num_worlds) and tested_scenarios + len(scenario_indices) < int(args.num_scenarios):
                scenario_idx = int(curr_scenario_index)
                curr_scenario_index += 1
                if (os.path.basename(tfrecord_path), scenario_idx) in completed_scenarios:
                    continue
                scenario_indices.append(scenario_idx)
            if not scenario_indices:
                break
        try:
            sim_state, _ = load_scenario_state_batch_fast(ds_cfg, scenario_indices)
        except Exception as e:
            print(f"Error loading scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}: {e}")
            if not _advance_tfrecord():
                break
            continue

        print(f"Testing scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}...")

        # lane_graphs = lane_graph_loader.get_lane_graph_for_scenarios(tfrecord_path, scenario_indices)
        rng_key = jax.random.PRNGKey(int(args.seed))
        goal_xy_b2, goal_t_b, ego_idx_b = infer_goals_from_sim_state(sim_state)

        if log_replay:
            replaced_state = sim_state
            pred_traj = np.asarray(sim_state_to_ego_trajectory(sim_state))
            start_t = np.full((len(scenario_indices),), int(args.start_timestep), dtype=np.int32)
            instructions = [dict() for _ in range(len(scenario_indices))]
            subgoals = [dict() for _ in range(len(scenario_indices))]
            reward_functions = [dict() for _ in range(len(scenario_indices))]
            target_lane_ids = [None for _ in range(len(scenario_indices))]
            target_lane_xys = [None for _ in range(len(scenario_indices))]
        else:
            if args.planner in ("es", "es_fm", "temporal-vla"):
                if args.lane_graph_dir:
                    requested_zip_path = lane_graph_zip_path_for_tfrecord(str(tfrecord_path), args.lane_graph_dir)
                    if lane_graph_zip_path != requested_zip_path:
                        lane_graph_zip_path = requested_zip_path
                        if lane_graph_zip_path.exists():
                            lane_graph_store = LaneGraphShardStore(lane_graph_zip_path.as_posix())
                        else:
                            lane_graph_store = None
                            raise FileNotFoundError(f"Lane graph zip file {lane_graph_zip_path} not found for tfrecord {tfrecord_path}.")
                lane_graphs = []
                for world_idx in range(len(scenario_indices)):
                    graph = lane_graph_store.get_by_record_index(int(scenario_indices[world_idx]))
                    planner.scorers[world_idx].set_lane_graph(graph)
                    lane_graphs.append(graph)
                    scorer_helpers.update_lane_points(
                        planner.scorers[world_idx], sim_state, world_idx
                    )
            pred, replaced_state, info = predict_planner_trajectories_with_periodic_replan(
                args,
                sim_state,
                goal_xy_b2,
                planner,
                start_timestep=int(args.start_timestep),
                replan_interval_steps=int(args.replan_interval_steps),
                instruction_interval_steps=int(args.instruction_interval_steps),
                use_subgoal=args.use_subgoal,
                rng_key=rng_key,
            )
            pred_traj = np.asarray(pred.trajectory_world_bt5)
            start_t = np.asarray(pred.start_t_b, dtype=np.int32)
            instructions = info["instructions"]
            subgoals = info["subgoals"]
            reward_functions = info["reward_functions"]
            target_lane_ids = info["target_lane_ids"]
            

        rollout = rollout_predicted_trajectories_with_metrics(
            replaced_state,
            rng_key=rng_key,
        )
        # tl_violation_eval = check_traffic_light_violation(
        #     replaced_state,
        #     lane_graphs,
        # )
        # tl_violation = np.asarray(tl_violation_eval["violation"])
        # tl_violation_step = np.asarray(tl_violation_eval["violation_step"])
        goal_eval = check_goal_reaching(
            replaced_state,
            goal_xy_b2,
            start_timestep=int(args.start_timestep),
        )
        goal_reached = np.asarray(goal_eval["reached"])
        goal_reached_step = np.asarray(goal_eval["reached_step"])

        overlap_timeseries = np.asarray(rollout["metric_timeseries"]["overlap"][int(args.start_timestep):])
        offroad_timeseries = np.asarray(rollout["metric_timeseries"]["offroad"][int(args.start_timestep):])

        overlap = np.zeros((len(scenario_indices),), dtype=bool)
        offroad = np.zeros((len(scenario_indices),), dtype=bool)
        for world_idx in range(len(scenario_indices)):
            episode_length = goal_reached_step[world_idx] + 1 if goal_reached[world_idx] else overlap_timeseries.shape[0]
            if "overlap" in rollout["metric_timeseries"]:
                overlap[world_idx] = overlap_timeseries[:episode_length, world_idx].sum() > 0
            if "offroad" in rollout["metric_timeseries"]:
                offroad[world_idx] = offroad_timeseries[:episode_length, world_idx].sum() > 0
        # tl_violation = tl_violation & (tl_violation_step < goal_reached_step)

        success = goal_reached & (~overlap) & (~offroad)
        reached_all += int(np.sum(goal_eval["reached"]))
        success_all += int(np.sum(success))

        if args.save_trajectory:
            for i, scenario_idx in enumerate(scenario_indices):
                traj_data = {
                    "scenario_idx": int(scenario_idx),
                    "ego_idx": int(ego_idx_b[i]),
                    "predicted_trajectory": pred_traj[i],
                    "start_timestep": int(start_t[i]),
                    "goal_xy": np.asarray(goal_xy_b2[i]),
                }
                traj_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.trajectory.npz"
                np.savez(traj_path, **traj_data)
        # Save per-episode cache from the planner-updated episode state.
        save_cache_mode = getattr(args, "save_cache", None)
        if save_cache_mode is not None and save_cache_mode != "none":
            for i, scenario_idx in enumerate(scenario_indices):
                should_save = False
                if save_cache_mode == "all":
                    should_save = True
                elif save_cache_mode == "success" and bool(success[i]):
                    should_save = True
                elif save_cache_mode == "failure" and not bool(success[i]):
                    should_save = True

                if not should_save:
                    continue

                try:
                    if _build_cache is None:
                        # fallback: replicate minimal extraction for common fields
                        # We attempt to save a small set of useful arrays from the updated episode state.
                        world_idx = int(i)
                        packed = {
                            "tfrecord_path": np.asarray(str(tfrecord_path), dtype=np.str_),
                            "tfrecord_name": np.asarray(os.path.basename(tfrecord_path), dtype=np.str_),
                            "scenario_idx": np.asarray(int(scenario_idx), dtype=np.int32),
                            "world_idx": np.asarray(world_idx, dtype=np.int32),
                            "log_trajectory_x": np.asarray(replaced_state.log_trajectory.x[world_idx]),
                            "log_trajectory_y": np.asarray(replaced_state.log_trajectory.y[world_idx]),
                            "log_trajectory_yaw": np.asarray(replaced_state.log_trajectory.yaw[world_idx]),
                            "log_trajectory_valid": np.asarray(replaced_state.log_trajectory.valid[world_idx]),
                        }
                    else:
                        # reuse build_cache extraction (returns dict[str, np.ndarray])
                        world_idx = int(i)
                        extracted = _build_cache._extract_used_sim_state_fields(replaced_state, world_idx)
                        packed = {
                            "tfrecord_path": np.asarray(str(tfrecord_path), dtype=np.str_),
                            "tfrecord_name": np.asarray(os.path.basename(tfrecord_path), dtype=np.str_),
                            "scenario_idx": np.asarray(int(scenario_idx), dtype=np.int32),
                            "world_idx": np.asarray(world_idx, dtype=np.int32),
                        }
                        packed.update(extracted)

                    cache_dir = output_dir / "sim_state_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / f"{os.path.basename(tfrecord_path)}.scenario_{int(scenario_idx):03d}.sim_state_cache.npz"
                    np.savez_compressed(cache_path, **packed)
                except Exception as e:
                    print(f"Failed to save sim_state cache for scenario {scenario_idx} (world {i}): {e}")
        if args.save_instruction:
            for i, scenario_idx in enumerate(scenario_indices):
                instruction_data = {
                    "scenario_idx": int(scenario_idx),
                    "ego_idx": int(ego_idx_b[i]),
                    "instructions": instructions[i],
                    "subgoals": subgoals[i] if args.use_subgoal else None,
                    "reward_functions": reward_functions[i],
                    "target_lane_ids": target_lane_ids[i]
                }
                instruction_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.instructions.json"
                _save_json(instruction_path, instruction_data)

        if args.visualize_mode == "all":
            visualize_indices = np.arange(len(scenario_indices))
            video_requests = [
                (str(tfrecord_path), int(idx)) for idx in scenario_indices
            ]
        elif args.visualize_mode == "success":
            visualize_indices = np.where(success)[0]
            video_requests = [
                (str(tfrecord_path), int(scenario_indices[i]))
                for i in range(len(scenario_indices)) if success[i]
            ]
        elif args.visualize_mode == "failure":
            visualize_indices = np.where(~success)[0]
            video_requests = [
                (str(tfrecord_path), int(scenario_indices[i]))
                for i in range(len(scenario_indices)) if not success[i]
            ]
        else:
            visualize_indices = []
        video_requests = [
            (str(tfrecord_path), int(scenario_indices[i]))
            for i in visualize_indices
        ]
        ego_start_times = [0 for i in visualize_indices]
        ego_trajectories = [np.asarray(pred_traj[i]) for i in visualize_indices]
        # ego_indices = get_sdc_indices_for_batched_state(replaced_state)
        # ego_trajectories = [
        #     np.asarray(
        #         jnp.stack(
        #             [
        #                 replaced_state.log_trajectory.x[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.y[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.yaw[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.vel_x[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.vel_y[i, int(ego_indices[i]), :],
        #             ], axis=-1
        #         )
        #     ) for i in visualize_indices
        # ]
        goal_xys = [np.asarray(goal_xy_b2[i]) for i in visualize_indices]
        if video_requests:
            video_paths = render_videos_batched(
                tfrecord_scenarios=video_requests,
                target_vehicles=[None] * len(scenario_indices),
                output_dir=output_dir.as_posix(),
                ego_start_times=ego_start_times,
                ego_trajectories=ego_trajectories,
                goal_xy=goal_xys,
            )
        for i, scenario_idx in enumerate(scenario_indices):
            result = {
                "scenario_idx": int(scenario_idx),
                "ego_idx": int(ego_idx_b[i]),
                "goal_xy": [float(goal_xy_b2[i, 0]), float(goal_xy_b2[i, 1])],
                "goal_timestep": int(goal_t_b[i]),
                "goal_reached": bool(goal_eval["reached"][i]),
                "reached_timestep": int(goal_eval["reached_step"][i]),
                "min_goal_distance_m": float(goal_eval["min_goal_distance_m"][i]),
                "final_goal_distance_m": float(goal_eval["final_goal_distance_m"][i]),
                "overlap": bool(overlap[i]),
                "offroad": bool(offroad[i]),
                "success": bool(success[i]),
                "tfrecord": str(tfrecord_path),
            }
            result_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.json"
            _save_json(result_path, result)
            completed_scenarios.add((os.path.basename(tfrecord_path), int(scenario_idx)))
        tested_scenarios += len(scenario_indices)
        print(f"Finished testing scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}.")
        print(f"Current success rate: {success_all}/{tested_scenarios} = {success_all / tested_scenarios:.4f}")
        print("" + "-" * 50)

    summary = {
        "num_scenarios": tested_scenarios,
        "num_goal_reached": int(reached_all),
        "num_success": int(success_all),
        "goal_reach_rate": float(reached_all / tested_scenarios) if tested_scenarios > 0 else 0.0,
        "success_rate": float(success_all / tested_scenarios) if tested_scenarios > 0 else 0.0,
    }
    _save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
