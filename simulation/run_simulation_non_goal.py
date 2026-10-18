"""Simulation + WOMD-style evaluation for non-goal-conditioned ego policies.

Evaluates policies trained with `train_diffusion/train_diffusion_policy.py` or
`train_diffusion/train_fm_policy.py` (with `--no-goal_conditioned`).

For every scenario two evaluations are run:
  1. Open-loop (WOMD motion prediction style): K trajectories are sampled once at
     `--start_timestep` and compared with the logged ego future
     (minADE / minFDE / miss rate / overlap / offroad).
  2. Closed-loop (Waymax style): the ego re-plans every `--replan_interval_steps`
     and executes sample 0 of each plan while every other agent replays its log
     (collision / offroad / log divergence / route progress).

Example:
  python -m simulation.run_simulation_non_goal --planner fm \
      --ckpt_path /path/to/train_flow_matching/latest --split validation \
      --num_scenarios 500 --num_samples 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocess import compute_world_dt_seconds, preprocess_simulator_state
from data.scenario_loader import load_scenario_state_batch_fast
from simulation.planning_utils import (
    predict_planner_trajectories_with_periodic_replan,
    sim_state_to_ego_trajectory,
)
from simulation.run_simulation import _load_fm_model_for_inference
from simulation.runner import _load_completed_scenarios
from simulation.utils import _save_json
from simulation.womd_metrics import (
    evaluate_closed_loop,
    evaluate_open_loop,
    extract_log_arrays,
    summarize,
)
from train_diffusion.utils.checkpoints import resolve_orbax_checkpoint_path
from train_diffusion.utils.infer import load_model_for_inference
from waymax import config as waymax_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", type=str, default="fm", choices=["diffusion", "fm", "log-replay"])
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Checkpoint dir (run dir, 'latest' or 'epoch_xxxx').")
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_denoising_steps", type=int, default=None,
                        help="FM only: number of ODE steps (default: metadata).")
    parser.add_argument("--fm_solver", type=str, default="euler")

    parser.add_argument("--tfrecord_dir", type=str, default="/zfsauton/scratch/mineuih/womd/tf_example/")
    parser.add_argument("--split", type=str, default="validation", choices=["training", "validation"])
    parser.add_argument("--exp_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/exp")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--start_from", type=str, default=None,
                        help="Existing experiment dir to resume into (relative to --exp_dir allowed).")

    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--num_worlds", type=int, default=5)
    parser.add_argument("--num_scenarios", type=int, default=100000)
    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--scenario_indices", type=str, default=None,
                        help="Comma-separated scenario indices in the first tfrecord.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--start_timestep", type=int, default=10,
                        help="WOMD current-time index (1.1s of history).")
    parser.add_argument("--replan_interval_steps", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=6,
                        help="K for open-loop minADE/minFDE (WOMD uses 6).")
    parser.add_argument("--ol_horizons", type=str, default="3,5",
                        help="Open-loop horizons in seconds (<= model horizon).")
    parser.add_argument("--cl_horizons", type=str, default="3,5,8",
                        help="Closed-loop log-divergence horizons in seconds.")
    parser.add_argument("--skip_open_loop", action="store_true", default=True)
    parser.add_argument("--include_open_loop", action="store_false", dest="skip_open_loop", default=False)

    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--visualize_mode", type=str, default="none",
                        choices=["all", "success", "failure", "none"])
    return parser.parse_args()


def _parse_float_csv(raw: str) -> list[float]:
    return [float(v) for v in raw.split(",") if v.strip()]


def build_planner(args: argparse.Namespace):
    """Returns (planner, mask_goal). planner is None for log replay."""
    if args.planner == "log-replay":
        return None, False
    if args.ckpt_path is None:
        raise ValueError(f"--ckpt_path is required for --planner {args.planner}.")
    # Accept a run dir as well as its 'latest' / 'epoch_xxxx' subdir (metadata.json lives there).
    args.ckpt_path = resolve_orbax_checkpoint_path(args.ckpt_path)

    if args.planner == "diffusion":
        inference = load_model_for_inference(
            checkpoint_path=args.ckpt_path,
            use_ema=args.use_ema,
            seed=args.seed,
        )
        policy = nnx.merge(inference.graphdef, inference.params_state, inference.nonparam_state)
        from planner.diffusion_planner import DiffusionPlanner

        planner = DiffusionPlanner(
            policy=policy,
            preprocess_cfg=inference.preprocess_cfg,
            population_size=args.num_samples,
            num_worlds=args.num_worlds,
        )
    else:
        policy, preprocess_cfg = _load_fm_model_for_inference(
            checkpoint_path=args.ckpt_path,
            use_ema=args.use_ema,
            seed=args.seed,
            num_denoising_steps=args.num_denoising_steps,
        )
        from planner.fm_planner import FMPlanner

        planner = FMPlanner(
            policy=policy,
            preprocess_cfg=preprocess_cfg,
            population_size=args.num_samples,
            num_worlds=args.num_worlds,
            num_steps=args.num_denoising_steps,
            solver=args.fm_solver,
        )

    # A goal-conditioned checkpoint can still be evaluated without a goal by
    # zeroing the goal features (the unconditional branch of goal masking).
    mask_goal = bool(policy.goal_conditioned)
    if mask_goal:
        warnings.warn(
            "Checkpoint is goal-conditioned; goal features will be zeroed during simulation.",
            RuntimeWarning,
        )
    return planner, mask_goal


def sample_open_loop(planner, sim_state, rng: jax.Array, timestep: int, mask_goal: bool):
    """K samples from the planner at a single anchor step (no ego replacement).

    Returns (world_bkt5, world_valid_bt, world_dt) with index 0 = anchor step.
    """
    key_pre, key_sample = jax.random.split(rng)
    num_steps = int(sim_state.log_trajectory.x.shape[-1])
    pre_batch, _ = preprocess_simulator_state(
        sim_state,
        key_pre,
        planner.preprocess_cfg,
        anchor_step_override=timestep,
        goal_step_override=num_steps - 1,
    )
    features = pre_batch.features
    if mask_goal:
        features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
        features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
    cond_bf = planner._compute_condition_jit(features)
    norm_bktd = planner._sample_population_jit(cond_bf=cond_bf, rng=key_sample)
    world_bkt5, _, world_valid_bt = planner._postprocess_population(norm_bktd, pre_batch)
    world_dt = float(np.asarray(pre_batch.aux["world_dt_seconds"]).reshape(-1)[0])
    return np.asarray(world_bkt5), np.asarray(world_valid_bt), world_dt


def _count_records(tfrecord_path: str) -> int:
    import tensorflow as tf

    return sum(1 for _ in tf.data.TFRecordDataset([tfrecord_path]))


def _iter_scenario_batches(args: argparse.Namespace, completed: set[tuple[str, int]]):
    """Yields (tfrecord_path, sorted scenario indices) batches of size <= num_worlds."""
    tfrecord_paths = sorted(glob(str(Path(args.tfrecord_dir) / "*")))
    num_shards = int(args.num_shards) if args.num_shards is not None else len(tfrecord_paths)
    if not tfrecord_paths:
        raise FileNotFoundError(f"No tfrecords found in {args.tfrecord_dir}.")

    if args.scenario_indices is not None:
        explicit = [int(i) for i in args.scenario_indices.split(",") if i.strip()]
        sources = [(tfrecord_paths[0], explicit)]
    else:
        sources = ((p, None) for p in tfrecord_paths[:num_shards])

    remaining = int(args.num_scenarios)
    for tfrecord_path, indices in sources:
        if remaining <= 0:
            return
        if indices is None:
            indices = range(_count_records(tfrecord_path))
        name = os.path.basename(tfrecord_path)
        todo = [int(i) for i in indices if (name, int(i)) not in completed][:remaining]
        remaining -= len(todo)
        for start in range(0, len(todo), int(args.num_worlds)):
            yield tfrecord_path, sorted(todo[start:start + int(args.num_worlds)])


def _make_ds_cfg(args: argparse.Namespace, tfrecord_path: str, batch_size: int):
    return dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(batch_size,),
        shuffle_seed=0,
    )


def _to_python(value: Any) -> Any:
    value = np.asarray(value).item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _print_running(summary: dict[str, float], keys: list[str]) -> None:
    shown = [k for k in keys if k in summary]
    print("  " + " | ".join(f"{k}={summary[k]:.4f}" for k in shown))


def run(args: argparse.Namespace, planner, mask_goal: bool) -> dict[str, float]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_replay = planner is None
    ol_horizons = _parse_float_csv(args.ol_horizons)
    cl_horizons = _parse_float_csv(args.cl_horizons)
    # Arguments read by predict_planner_trajectories_with_periodic_replan.
    args.mask_goal = mask_goal
    args.use_subgoal = False

    completed = _load_completed_scenarios(args.output_dir if args.start_from else None)
    base_key = jax.random.PRNGKey(int(args.seed))
    metric_keys: list[str] = []
    results: list[dict[str, Any]] = []

    for batch_idx, (tfrecord_path, scenario_indices) in enumerate(_iter_scenario_batches(args, completed)):
        tfrecord_name = os.path.basename(tfrecord_path)
        ds_cfg = _make_ds_cfg(args, tfrecord_path, len(scenario_indices))
        try:
            sim_state, _ = load_scenario_state_batch_fast(ds_cfg, scenario_indices)
        except Exception as e:
            print(f"Error loading scenarios {scenario_indices} from {tfrecord_name}: {e}")
            continue
        print(f"[batch {batch_idx}] scenarios {scenario_indices} from {tfrecord_name}")

        rng = jax.random.fold_in(base_key, batch_idx)
        rng_ol, rng_cl = jax.random.split(rng)
        log = extract_log_arrays(sim_state)
        # Per-scenario dt estimated from the log timestamps, as used by pre/postprocess.
        dt_fallback = planner.preprocess_cfg.world_dt_fallback if planner is not None else 0.1
        scenario_world_dt = np.asarray(compute_world_dt_seconds(sim_state, dt_fallback))
        metrics: dict[str, np.ndarray] = {}
        world_dt = 0.1
        ol_world = None

        # 1) Open-loop: K samples at the WOMD current time.
        if not log_replay and not args.skip_open_loop:
            ol_world, ol_valid, world_dt = sample_open_loop(
                planner, sim_state, rng_ol, int(args.start_timestep), mask_goal
            )
            metrics.update(
                evaluate_open_loop(
                    ol_world, ol_valid, log, sim_state.roadgraph_points,
                    int(args.start_timestep), world_dt, ol_horizons,
                )
            )

        # 2) Closed-loop: periodic re-planning, non-reactive log-replay agents.
        if log_replay:
            sim_world = np.asarray(sim_state_to_ego_trajectory(sim_state))
        else:
            pred, _, _ = predict_planner_trajectories_with_periodic_replan(
                args,
                sim_state,
                None,
                planner,
                start_timestep=int(args.start_timestep),
                replan_interval_steps=int(args.replan_interval_steps),
                instruction_interval_steps=int(args.replan_interval_steps),
                use_subgoal=False,
                rng_key=rng_cl,
            )
            sim_world = np.asarray(pred.trajectory_world_bt5)
            world_dt = float(np.asarray(pred.aux["world_dt_seconds"]).reshape(-1)[0])
        metrics.update(
            evaluate_closed_loop(
                sim_world, log, sim_state.roadgraph_points,
                int(args.start_timestep), world_dt, cl_horizons,
            )
        )
        for key in metrics:
            if key not in metric_keys:
                metric_keys.append(key)

        for i, scenario_idx in enumerate(scenario_indices):
            result = {
                "scenario_idx": int(scenario_idx),
                "ego_idx": int(log["ego_idx"][i]),
                "tfrecord": str(tfrecord_path),
                # dt used to convert metric horizons (seconds -> steps) for this batch.
                "world_dt_seconds": float(world_dt),
                # dt estimated for this scenario from log timestamps (pre/postprocess).
                "world_dt_seconds_estimated": float(scenario_world_dt[i]),
                **{k: _to_python(v[i]) for k, v in metrics.items()},
            }
            results.append(result)
            _save_json(output_dir / f"{tfrecord_name}.scenario_{scenario_idx:03d}.json", result)
            if args.save_trajectory:
                traj = {
                    "scenario_idx": int(scenario_idx),
                    "ego_idx": int(log["ego_idx"][i]),
                    "start_timestep": int(args.start_timestep),
                    "closed_loop_trajectory": sim_world[i],
                    "log_trajectory_xy": log["ego_xy"][i],
                    "log_valid": log["ego_valid"][i],
                }
                if ol_world is not None:
                    traj["open_loop_samples"] = ol_world[i]
                np.savez(output_dir / f"{tfrecord_name}.scenario_{scenario_idx:03d}.trajectory.npz", **traj)

        if args.visualize_mode != "none":
            success = metrics["cl_success"] > 0
            if args.visualize_mode == "all":
                vis = np.arange(len(scenario_indices))
            elif args.visualize_mode == "success":
                vis = np.flatnonzero(success)
            else:
                vis = np.flatnonzero(~success)
            if len(vis) > 0:
                from viz.render import render_videos_batched

                render_videos_batched(
                    tfrecord_scenarios=[(str(tfrecord_path), int(scenario_indices[i])) for i in vis],
                    target_vehicles=[None] * len(vis),
                    output_dir=output_dir.as_posix(),
                    ego_start_times=[0 for _ in vis],
                    ego_trajectories=[sim_world[i] for i in vis],
                    # Ego ground-truth log as a light-blue line (invalid steps -> NaN gaps).
                    reference_trajectories=[
                        np.where(log["ego_valid"][i][:, None], log["ego_xy"][i], np.nan)
                        for i in vis
                    ],
                )

        running = summarize(results, metric_keys)
        print(f"  evaluated {len(results)} scenarios")
        _print_running(running, [
            "cl_collision", "cl_offroad", "cl_success", "cl_progress_ratio",
            "cl_ADE@full", "ol_minADE@5s", "ol_minFDE@5s", "ol_miss@5s",
        ])

    # Final summary over every scenario json in the output dir (includes resumed runs).
    all_results = []
    for path in sorted(output_dir.glob("*.scenario_*.json")):
        if path.name.endswith(".instructions.json"):
            continue
        all_results.append(json.loads(path.read_text(encoding="utf-8")))
    all_keys = sorted({k for r in all_results for k in r} - {"scenario_idx", "ego_idx", "tfrecord", "world_dt_seconds", "world_dt_seconds_estimated"})
    summary = {
        "num_scenarios": len(all_results),
        "planner": args.planner,
        "ckpt_path": args.ckpt_path,
        "num_samples": int(args.num_samples),
        "start_timestep": int(args.start_timestep),
        "replan_interval_steps": int(args.replan_interval_steps),
        **summarize(all_results, all_keys),
    }
    _save_json(output_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("/count")}, indent=2))
    return summary


def main(args: argparse.Namespace) -> None:
    import tensorflow as tf

    # TF is only used for reading tfrecords; keep it off the GPU.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    if args.start_from:
        start_dir = os.path.expanduser(args.start_from)
        if not os.path.isabs(start_dir):
            start_dir = os.path.join(args.exp_dir, start_dir)
        args.output_dir = start_dir
    else:
        exp_name = f"simulation_nongoal_{args.planner}_{args.split}"
        if args.tag is not None:
            exp_name += f"_{args.tag}"
        exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output_dir = os.path.join(args.exp_dir, exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    _save_json(Path(args.output_dir) / "args.json", vars(args))

    planner, mask_goal = build_planner(args)
    run(args, planner, mask_goal)


if __name__ == "__main__":
    main(_parse_args())
