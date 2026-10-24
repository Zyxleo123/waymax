from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import sys
import re
import struct
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

from data.scenario_loader import (
    ScenarioIndexOutOfRange,
    infer_route_dims,
    load_scenario_state_batch_fast,
)
from lane_graph.lane_graph_utils import lane_graph_zip_path_for_tfrecord
from scores import helpers as scorer_helpers
from scores.scorer_lane_graph import LaneGraphShardStore
from simulation.evaluation_utils import (
    rollout_predicted_trajectories_with_metrics,
    infer_goals_from_sim_state, 
    check_goal_reaching,
    check_traffic_light_violation,
    episode_failure_flags,
    episode_failure_step,
)
from simulation.idm_closed_loop import predict_planner_trajectories_closed_loop_idm
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


@functools.lru_cache(maxsize=None)
def _num_tfrecord_records(path: str) -> int:
    """Records in an uncompressed TFRecord file (each: 8-byte length, 4-byte CRC, data, 4-byte CRC)."""
    count = 0
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return count
            f.seek(struct.unpack("<Q", header)[0] + 8, os.SEEK_CUR)
            count += 1


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


# Sidecar files that live next to the records in some eval dirs (e.g. the
# failure set ships excluded_overpass.json). A bare glob("*") hands these to the
# TFRecord reader, which fails with "corrupted record at 0".
_NON_TFRECORD_SUFFIXES = frozenset({
    ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".csv",
    ".npz", ".npy", ".pkl", ".png", ".jpg", ".mp4", ".log",
})


def _list_tfrecord_paths(tfrecord_dir: str) -> list[str]:
    paths = [
        p
        for p in sorted(glob(str(Path(tfrecord_dir) / "*")))
        if not os.path.isdir(p) and Path(p).suffix.lower() not in _NON_TFRECORD_SUFFIXES
    ]
    if not paths:
        raise FileNotFoundError(f"no tfrecords under {tfrecord_dir}")
    return paths


def _dataset_config_for(tfrecord_path: str, args) -> waymax_config.DatasetConfig:
    """Dataset config matched to *this* tfrecord's route budget.

    The route dims are read off the file rather than inherited from
    WOD_1_3_1_TRAINING's 45x800, so eval sets exported with a different budget
    (failures.tfrecord is 10x300) parse instead of dying at ParseExampleV2.
    """
    num_paths, num_points_per_path = infer_route_dims(tfrecord_path)
    return dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(args.num_worlds,),
        shuffle_seed=0,
        num_paths=num_paths,
        num_points_per_path=num_points_per_path,
    )


def run(args, planner: AbstractPlanner, log_replay=False) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "target_scenario_path", None) is not None:
        with open(args.target_scenario_path, "r") as f:
            target_scenarios = json.load(f)
    else:
        target_scenarios = None

    completed_scenarios = _load_completed_scenarios(getattr(args, "start_from", None))
    tested_scenarios, curr_scenario_index = 0, 0
    target_scenario_cursor = 0
    reached_all, success_all = 0, 0
    results: list[dict[str, Any]] = []
    tfrecord_paths = _list_tfrecord_paths(args.tfrecord_dir)

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
        ds_cfg = _dataset_config_for(tfrecord_path, args)
        return True

    tfrecord_path = tfrecord_paths.pop(0)
    ds_cfg = _dataset_config_for(tfrecord_path, args)
    if args.scenario_indices is not None:
        target_scenario_indices = [int(idx) for idx in args.scenario_indices.split(",")]
    else:
        target_scenario_indices = None

    candidate_recorder = None
    if getattr(args, "save_es_candidates", False) and not log_replay:
        if not hasattr(planner, "candidate_recorder"):
            raise ValueError("--save_es_candidates requires a DiffusionESPlanner (--planner es).")
        from es_discriminator.es_candidate_recorder import ESCandidateRecorder

        candidate_recorder = ESCandidateRecorder(
            output_dir / "es_candidates",
            population_size=int(args.population_size),
            num_iterations=int(args.num_es_iterations),
        )
        candidate_recorder.write_metadata(args)
        planner.candidate_recorder = candidate_recorder

    # --other_agent idm (new_model): non-ego agents are IDM-driven during planning too (closed loop);
    # --sim_agents idm only swaps them in for the evaluation rollout.
    if getattr(args, "other_agent", "log") == "idm":
        replan_fn = predict_planner_trajectories_closed_loop_idm
    else:
        replan_fn = predict_planner_trajectories_with_periodic_replan

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
            # A partial last batch on a shard is padded after loading (below), not dropped.
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
            num_records = _num_tfrecord_records(str(tfrecord_path))
            scenario_indices = []
            while (
                len(scenario_indices) < int(args.num_worlds)
                and tested_scenarios + len(scenario_indices) < int(args.num_scenarios)
                and curr_scenario_index < num_records
            ):
                scenario_idx = int(curr_scenario_index)
                curr_scenario_index += 1
                if (os.path.basename(tfrecord_path), scenario_idx) in completed_scenarios:
                    continue
                scenario_indices.append(scenario_idx)
            if not scenario_indices:
                if curr_scenario_index >= num_records and _advance_tfrecord():
                    continue
                break
        # A shard's last batch can be short; planner/batch_dims expect num_worlds, so pad the loaded
        # state with copies of its last scenario and keep only the first num_real results. (The loader
        # deduplicates indices, so the padding has to happen after loading.)
        num_real = len(scenario_indices)
        try:
            sim_state, _, scenario_ids = load_scenario_state_batch_fast(
                ds_cfg, scenario_indices, return_scenario_ids=True
            )
            batch_state = sim_state  # one world per real scenario (--label_in_loop labels against it)
            num_pad = int(args.num_worlds) - num_real
            if num_pad > 0:
                sim_state = jax.tree_util.tree_map(
                    lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], num_pad, axis=0)], axis=0)
                    if getattr(x, "ndim", 0) > 0 and x.shape[0] == num_real else x,
                    sim_state,
                )
                scenario_indices = scenario_indices + [scenario_indices[-1]] * num_pad
                scenario_ids = list(scenario_ids) + [scenario_ids[-1]] * num_pad
        except ScenarioIndexOutOfRange:
            # Only "ran off the end of this file" advances to the next tfrecord. Every other load error
            # (unreadable file, route-dim mismatch, truncated record) is a real fault and is raised rather
            # than laundered into a quietly skipped shard.
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
            if getattr(args, "planner", None) in ("es", "es_fm", "temporal-vla", "pdm_closed", "diffusion_planner"):
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
            if candidate_recorder is not None:
                candidate_recorder.start_batch(
                    sim_state, os.path.basename(tfrecord_path), scenario_indices, scenario_ids, goal_xy_b2
                )
            pred, replaced_state, info = replan_fn(
                args,
                sim_state,
                goal_xy_b2,
                planner,
                start_timestep=int(args.start_timestep),
                replan_interval_steps=int(args.replan_interval_steps),
                instruction_interval_steps=int(args.instruction_interval_steps),
                use_subgoal=getattr(args, "use_subgoal", False),
                rng_key=rng_key,
                goal_timestep=goal_t_b,  # real per-scene goal step (posthoc-opt-es); ES planners keep 90
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
            sim_agents=getattr(args, "sim_agents", "log"),
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

        # Count failures only in [start_timestep, goal step], indexed by absolute rollout timestep.
        overlap, offroad = (
            episode_failure_flags(
                rollout["metric_timeseries"][name],
                rollout["timestep"],
                start_timestep=int(args.start_timestep),
                goal_reached_b=goal_reached,
                goal_reached_step_b=goal_reached_step,
                num_log_steps=rollout["num_log_steps"],
            )
            for name in ("overlap", "offroad")
        )
        overlap_step, offroad_step = (
            episode_failure_step(
                rollout["metric_timeseries"][name],
                rollout["timestep"],
                start_timestep=int(args.start_timestep),
                goal_reached_b=goal_reached,
                goal_reached_step_b=goal_reached_step,
                num_log_steps=rollout["num_log_steps"],
            )
            for name in ("overlap", "offroad")
        )
        # tl_violation = tl_violation & (tl_violation_step < goal_reached_step)

        success = goal_reached & (~overlap) & (~offroad)
        if candidate_recorder is not None:
            batch_path = candidate_recorder.finish_batch(replaced_state, {
                "overlap": overlap,
                "offroad": offroad,
                "overlap_step": overlap_step,
                "offroad_step": offroad_step,
                "goal_reached": goal_reached,
                "reached_step": goal_reached_step,
                "success": success,
                "min_goal_dist": np.asarray(goal_eval["min_goal_distance_m"]),
                "final_goal_dist": np.asarray(goal_eval["final_goal_distance_m"]),
            }, num_real=num_real)
            if batch_path is not None and getattr(args, "label_in_loop", False):
                from anchor_wm.label_in_loop import label_and_cache

                info = label_and_cache(str(batch_path), batch_state, str(output_dir / "anchor_cache"))
                print(f"[label_in_loop] {batch_path.name}: {json.dumps(info)}", flush=True)
        if num_real < len(scenario_indices):
            # Padded last batch: drop the padding copies.
            scenario_indices = scenario_indices[:num_real]
            overlap, offroad, success = overlap[:num_real], offroad[:num_real], success[:num_real]
            overlap_step, offroad_step = overlap_step[:num_real], offroad_step[:num_real]
            goal_eval = {key: value[:num_real] for key, value in goal_eval.items()}
        reached_all += int(np.sum(goal_eval["reached"]))
        success_all += int(np.sum(success))

        if getattr(args, "save_trajectory", False):
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
        if getattr(args, "save_instruction", False):
            for i, scenario_idx in enumerate(scenario_indices):
                instruction_data = {
                    "scenario_idx": int(scenario_idx),
                    "ego_idx": int(ego_idx_b[i]),
                    "instructions": instructions[i],
                    "subgoals": subgoals[i] if getattr(args, "use_subgoal", False) else None,
                    "reward_functions": reward_functions[i],
                    "target_lane_ids": target_lane_ids[i]
                }
                instruction_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.instructions.json"
                _save_json(instruction_path, instruction_data)

        if args.visualize_mode == "all":
            visualize_indices = list(range(len(scenario_indices)))
        elif args.visualize_mode == "success":
            visualize_indices = [i for i in range(len(scenario_indices)) if success[i]]
        elif args.visualize_mode == "failure":
            visualize_indices = [i for i in range(len(scenario_indices)) if not success[i]]
        else:  # "none" (or anything else): no videos
            visualize_indices = []
        video_requests = [
            (str(tfrecord_path), int(scenario_indices[i]))
            for i in visualize_indices
        ]
        # Render the evaluated state itself (ego plan and, with --other_agent idm, the simulated agents)
        # instead of reloading the logged scenario, so the video matches the metrics (new_model).
        render_states = [(replaced_state, int(i)) for i in visualize_indices]
        goal_xys = [np.asarray(goal_xy_b2[i]) for i in visualize_indices]

        # Map every scenario position to its rendered video path (None when
        # visualization is disabled or this scenario was not requested).
        # render_videos_batched returns one path per *request*, aligned to
        # `visualize_indices` -- not one per scenario -- so indexing it directly
        # by the scenario loop counter was wrong whenever a subset was rendered
        # and crashed outright (NameError) when nothing was rendered.
        video_path_by_pos: dict[int, Any] = {i: None for i in range(len(scenario_indices))}
        if video_requests:
            rendered = render_videos_batched(
                tfrecord_scenarios=video_requests,
                target_vehicles=[None] * len(video_requests),
                output_dir=output_dir.as_posix(),
                goal_xy=goal_xys,
                states=render_states,
            )
            for pos, path in zip(visualize_indices, rendered):
                video_path_by_pos[pos] = path

        for i, scenario_idx in enumerate(scenario_indices):
            vp = video_path_by_pos.get(i)
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
                "overlap_timestep": int(overlap_step[i]),
                "offroad_timestep": int(offroad_step[i]),
                "success": bool(success[i]),
                "video_path": vp.as_posix() if vp is not None else None,
                "tfrecord": str(tfrecord_path),
            }
            result_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.json"
            _save_json(result_path, result)
            results.append(result)
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
    return results
