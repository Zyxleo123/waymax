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
    check_traffic_light_violation,
    episode_failure_flags,
)
from simulation.planning_utils import (
    predict_planner_trajectories_with_periodic_replan,
    get_sdc_indices_for_batched_state,
    sim_state_to_ego_trajectory,
)
from simulation.idm_closed_loop import predict_planner_trajectories_closed_loop_idm
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


def _count_completed_results(start_from: str | None) -> tuple[int, int, int]:
    """(tested, goal_reached, success) over per-scenario JSONs already in start_from."""
    if not start_from or not Path(start_from).exists():
        return 0, 0, 0
    tested = reached = success = 0
    for result_path in Path(start_from).rglob("*.json"):
        if _SCENARIO_RESULT_RE.match(result_path.name) is None:
            continue
        with open(result_path, "r") as f:
            result = json.load(f)
        tested += 1
        reached += int(bool(result.get("goal_reached", False)))
        success += int(bool(result.get("success", False)))
    return tested, reached, success


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
    if completed_scenarios:
        tested_scenarios, reached_all, success_all = _count_completed_results(getattr(args, "start_from", None))
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
            sim_state, _, scenario_ids = load_scenario_state_batch_fast(
                ds_cfg, scenario_indices, return_scenario_ids=True
            )
            # Which scenes the first-pair clock would have slowed (logged, not used).
            from data.preprocess import compute_world_dt_seconds
            _dt_first = np.asarray(compute_world_dt_seconds(sim_state, 0.1, "first"))
            _dt_used = np.asarray(compute_world_dt_seconds(sim_state, 0.1))
            print(f"[world_dt] first-pair {np.round(_dt_first, 3).tolist()} used {np.round(_dt_used, 3).tolist()}", flush=True)
            wm_forecaster = getattr(planner, "world_model_forecaster", None)
            if hasattr(wm_forecaster, "set_scene_ids"):
                wm_forecaster.set_scene_ids(scenario_ids)
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
            if getattr(args, "other_agents", "log") == "idm":
                replan_fn = predict_planner_trajectories_closed_loop_idm
            else:
                replan_fn = predict_planner_trajectories_with_periodic_replan
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

        # Count failures only in [start_timestep, goal step], indexed by the rollout's
        # absolute timesteps (Ishan's 4219b45: the rollout starts at t=10, and the old
        # [start_timestep:] slice counted t=20..goal+20).
        def _window(ro):
            return tuple(
                episode_failure_flags(
                    ro["metric_timeseries"][name],
                    ro["timestep"],
                    start_timestep=int(args.start_timestep),
                    goal_reached_b=goal_reached,
                    goal_reached_step_b=goal_reached_step,
                    num_log_steps=ro["num_log_steps"],
                )
                for name in ("overlap", "offroad")
            )

        overlap, offroad = _window(rollout)
        # tl_violation = tl_violation & (tl_violation_step < goal_reached_step)

        success = goal_reached & (~overlap) & (~offroad)

        # The pre-4219b45 count that es_WM's evaluator still uses: metric ROWS
        # [start_timestep:] up to goal step + 1 (absolute t=20..goal+20). Recorded
        # alongside, so a run can be reported under either window.
        def _old_window(ro):
            out = []
            for name in ("overlap", "offroad"):
                ts = np.asarray(ro["metric_timeseries"][name][int(args.start_timestep):])
                flag = np.zeros((len(scenario_indices),), dtype=bool)
                for w in range(len(scenario_indices)):
                    ep_len = goal_reached_step[w] + 1 if goal_reached[w] else ts.shape[0]
                    flag[w] = ts[:ep_len, w].sum() > 0
                out.append(flag)
            return tuple(out)

        overlap_old, offroad_old = _old_window(rollout)
        success_old = goal_reached & (~overlap_old) & (~offroad_old)
        if candidate_recorder is not None:
            candidate_recorder.finish_batch(replaced_state, {
                "overlap": overlap,
                "offroad": offroad,
                "goal_reached": goal_reached,
                "reached_step": goal_reached_step,
                "success": success,
                "min_goal_dist": np.asarray(goal_eval["min_goal_distance_m"]),
                "final_goal_dist": np.asarray(goal_eval["final_goal_distance_m"]),
            })
        # Also score the SAME executed ego trajectory with other non-ego agent models
        # (e.g. --also_eval_sim_agents idm). Only the evaluation rollout changes --
        # planning saw logged agents either way -- so one planning run serves both.
        extra_eval = {}
        for sa in [x for x in str(getattr(args, "also_eval_sim_agents", "") or "").split(",")
                   if x and x != getattr(args, "sim_agents", "log")]:
            ro = rollout_predicted_trajectories_with_metrics(replaced_state, rng_key=rng_key, sim_agents=sa)
            ovl, off = _window(ro)
            extra_eval[sa] = (ovl, off, goal_reached & (~ovl) & (~off))
        reached_all += int(np.sum(goal_eval["reached"]))
        success_all += int(np.sum(success))

        if args.save_trajectory:
            metric_t = np.asarray(rollout["timestep"])
            for i, scenario_idx in enumerate(scenario_indices):
                traj_data = {
                    "scenario_idx": int(scenario_idx),
                    "ego_idx": int(ego_idx_b[i]),
                    "predicted_trajectory": pred_traj[i],
                    "start_timestep": int(start_t[i]),
                    "goal_xy": np.asarray(goal_xy_b2[i]),
                    # Per-step metrics of the primary rollout (row k = absolute timestep
                    # metric_timestep[k]), so any failure window can be re-applied offline;
                    # under --other_agents idm the NPCs here are the closed-loop IDM ones.
                    "metric_timestep": metric_t,
                    "overlap_t": np.asarray(rollout["metric_timeseries"]["overlap"])[:, i] > 0,
                    "offroad_t": np.asarray(rollout["metric_timeseries"]["offroad"])[:, i] > 0,
                    "goal_reached": bool(goal_reached[i]),
                    "goal_reached_step": int(goal_reached_step[i]),
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
                # Failures counted over [start_timestep, goal step] by absolute rollout timestep (4219b45).
                "eval_window": "abs_t10_to_goal",
                "world_dt_method": str(getattr(args, "world_dt_method", "first")),
                # The same rollout under es_WM's original window (metric rows [start_timestep:]).
                "overlap_oldwin": bool(overlap_old[i]),
                "offroad_oldwin": bool(offroad_old[i]),
                "success_oldwin": bool(success_old[i]),
                # WOMD scenario/id: joins against es_discriminator's scenario-hash train/val/test split.
                "scenario_id": str(scenario_ids[i]) if scenario_ids is not None else None,
                # log = NPCs replay the log; idm = closed-loop IDM (simulation/idm_closed_loop.py).
                "other_agents": str(getattr(args, "other_agents", "log")),
                "discriminator": bool(getattr(args, "discriminator_ckpt", None)),
            }
            for sa, (ovl, off, suc) in extra_eval.items():
                result[f"{sa}_overlap"] = bool(ovl[i])
                result[f"{sa}_offroad"] = bool(off[i])
                result[f"{sa}_success"] = bool(suc[i])
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
