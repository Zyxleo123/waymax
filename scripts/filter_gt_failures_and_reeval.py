#!/usr/bin/env python3
"""Filter scenarios whose logged GT fails the same success criteria as closed-loop sim,
then recompute Diffusion-ES / FM-ES (cv|gt) metrics on the remaining scenes.

GT failure uses the same path as ``simulation/runner.py``:
  rollout overlap/offroad + infer_goals + check_goal_reaching
  success = goal_reached & ~overlap & ~offroad

Usage (from repo root, waymax_rs conda env)::

  python scripts/filter_gt_failures_and_reeval.py

Outputs under::

  /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/gt_filtered_es_metrics_<stamp>/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymax import config as waymax_config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scenario_loader import load_scenario_state_batch_fast
from simulation.evaluation_utils import (
    check_goal_reaching,
    infer_goals_from_sim_state,
    rollout_predicted_trajectories_with_metrics,
)

DEFAULT_TF_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example"
DEFAULT_EXP_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp"
DEFAULT_OUT_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp"

EXPERIMENT_RUNS = {
    "diffusion_es_cv": "es_diffusion_cv",
    "diffusion_es_gt": "es_diffusion_gt",
    "fm_es_cv": "es_fm_cv",
    "fm_es_gt": "es_fm_gt",
}

SHARDS = {
    "training": [
        "training/training_tfexample.tfrecord-00000-of-01000",
        "training/training_tfexample.tfrecord-00001-of-01000",
    ],
    "validation": [
        "validation/validation_tfexample.tfrecord-00000-of-00150",
        "validation/validation_tfexample.tfrecord-00001-of-00150",
    ],
}


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _count_tfrecord_scenarios(tfrecord_path: str) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset([str(tfrecord_path)]))


def evaluate_log_trajectory_like_runner(
    sim_state,
    *,
    start_timestep: int = 10,
    goal_threshold_m: float = 2.0,
    rng_key: jax.Array | None = None,
) -> dict[str, Any]:
    """Same metric path as ``simulation/runner.py`` on unmodified log trajectories."""
    goal_xy, goal_t_b, ego_idx_b = infer_goals_from_sim_state(sim_state)
    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    rollout = rollout_predicted_trajectories_with_metrics(sim_state, rng_key=key)
    goal_eval = check_goal_reaching(
        sim_state,
        goal_xy,
        start_timestep=int(start_timestep),
        goal_threshold_m=float(goal_threshold_m),
    )

    batch_size = int(sim_state.log_trajectory.x.shape[0])
    goal_reached = np.asarray(goal_eval["reached"])
    goal_reached_step = np.asarray(goal_eval["reached_step"])
    overlap_timeseries = np.asarray(
        rollout["metric_timeseries"]["overlap"][int(start_timestep) :]
    )
    offroad_timeseries = np.asarray(
        rollout["metric_timeseries"]["offroad"][int(start_timestep) :]
    )

    overlap = np.zeros((batch_size,), dtype=bool)
    offroad = np.zeros((batch_size,), dtype=bool)
    for world_idx in range(batch_size):
        episode_length = (
            int(goal_reached_step[world_idx]) + 1
            if bool(goal_reached[world_idx])
            else int(overlap_timeseries.shape[0])
        )
        overlap[world_idx] = bool(overlap_timeseries[:episode_length, world_idx].sum() > 0)
        offroad[world_idx] = bool(offroad_timeseries[:episode_length, world_idx].sum() > 0)

    success = goal_reached & (~overlap) & (~offroad)

    per_world: list[dict[str, Any]] = []
    for world_idx in range(batch_size):
        fail_reasons: list[str] = []
        if bool(overlap[world_idx]):
            fail_reasons.append("overlap")
        if bool(offroad[world_idx]):
            fail_reasons.append("offroad")
        if not bool(goal_reached[world_idx]):
            fail_reasons.append("goal")

        per_world.append(
            {
                "ego_idx": int(ego_idx_b[world_idx]),
                "goal_xy": [float(goal_xy[world_idx, 0]), float(goal_xy[world_idx, 1])],
                "goal_timestep": int(goal_t_b[world_idx]),
                "goal_reached": bool(goal_reached[world_idx]),
                "reached_timestep": int(goal_reached_step[world_idx]),
                "min_goal_distance_m": float(goal_eval["min_goal_distance_m"][world_idx]),
                "final_goal_distance_m": float(goal_eval["final_goal_distance_m"][world_idx]),
                "overlap": bool(overlap[world_idx]),
                "offroad": bool(offroad[world_idx]),
                "success": bool(success[world_idx]),
                "fail_reasons": fail_reasons,
            }
        )

    return {"per_world": per_world}


def build_gt_filter(
    *,
    tf_root: Path,
    split: str,
    start_timestep: int,
    max_num_objects: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    fail_counter = Counter()
    n_success = 0
    n_total = 0

    for rel in SHARDS[split]:
        tfrecord_path = str(tf_root / rel)
        if not Path(tfrecord_path).exists():
            raise FileNotFoundError(tfrecord_path)
        n_scenarios = _count_tfrecord_scenarios(tfrecord_path)
        tfrecord_name = os.path.basename(tfrecord_path)
        print(f"[gt-eval] {split} {tfrecord_name}: {n_scenarios} scenarios")

        for start in tqdm(range(0, n_scenarios, batch_size), desc=f"GT {tfrecord_name}"):
            indices = list(range(start, min(start + batch_size, n_scenarios)))
            # batch_dims must match the number of distinct indices we load.
            cfg = dataclasses.replace(
                waymax_config.WOD_1_3_1_TRAINING,
                path=tfrecord_path,
                max_num_objects=int(max_num_objects),
                batch_dims=(len(indices),),
                shuffle_seed=0,
            )
            try:
                sim_state, scenario_to_batch_idx = load_scenario_state_batch_fast(cfg, indices)
            except Exception as e:
                print(f"  ERROR loading {indices} from {tfrecord_name}: {e}")
                for scenario_idx in indices:
                    records.append(
                        {
                            "tfrecord": tfrecord_name,
                            "scenario_idx": int(scenario_idx),
                            "success": False,
                            "overlap": False,
                            "offroad": False,
                            "goal_reached": False,
                            "fail_reasons": ["load_error"],
                            "error": str(e),
                        }
                    )
                    fail_counter["load_error"] += 1
                    n_total += 1
                continue

            rng = jax.random.PRNGKey(int(seed) + start)
            eval_out = evaluate_log_trajectory_like_runner(
                sim_state,
                start_timestep=start_timestep,
                rng_key=rng,
            )
            for scenario_idx in indices:
                local_i = int(scenario_to_batch_idx[int(scenario_idx)])
                row = eval_out["per_world"][local_i]
                entry = {
                    "tfrecord": tfrecord_name,
                    "scenario_idx": int(scenario_idx),
                    "success": bool(row["success"]),
                    "overlap": bool(row["overlap"]),
                    "offroad": bool(row["offroad"]),
                    "goal_reached": bool(row["goal_reached"]),
                    "fail_reasons": list(row["fail_reasons"]),
                    "min_goal_distance_m": float(row["min_goal_distance_m"]),
                    "reached_timestep": int(row["reached_timestep"]),
                }
                records.append(entry)
                n_total += 1
                if row["success"]:
                    n_success += 1
                else:
                    for reason in row["fail_reasons"]:
                        fail_counter[reason] += 1
                    if not row["fail_reasons"]:
                        fail_counter["unknown"] += 1

    failing_keys = [
        {"tfrecord": r["tfrecord"], "scenario_idx": r["scenario_idx"], "fail_reasons": r["fail_reasons"]}
        for r in records
        if not r["success"]
    ]
    failing_set = {(r["tfrecord"], int(r["scenario_idx"])) for r in failing_keys}

    # Mutual-exclusion style counts for reporting
    exclusive = Counter()
    for r in records:
        if r["success"]:
            continue
        reasons = set(r["fail_reasons"])
        if reasons == {"overlap"}:
            exclusive["overlap_only"] += 1
        elif reasons == {"offroad"}:
            exclusive["offroad_only"] += 1
        elif reasons == {"goal"}:
            exclusive["goal_only"] += 1
        elif reasons == {"overlap", "offroad"}:
            exclusive["overlap+offroad"] += 1
        elif "load_error" in reasons:
            exclusive["load_error"] += 1
        else:
            exclusive["other_combo"] += 1

    return {
        "split": split,
        "start_timestep": int(start_timestep),
        "success_definition": "goal_reached & ~overlap & ~offroad (same as simulation/runner.py)",
        "num_scenarios": n_total,
        "num_gt_success": n_success,
        "num_gt_fail": n_total - n_success,
        "fail_reason_counts_any": dict(fail_counter),
        "fail_reason_counts_exclusive": dict(exclusive),
        "failing_keys": failing_keys,
        "records": records,
        "_failing_set": failing_set,
    }


def _latest_run_dir(exp_root: Path, run_folder: str, split: str) -> Path | None:
    root = exp_root / run_folder
    if not root.exists():
        return None
    cands = sorted(root.glob(f"simulation_*_{split}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_experiment_results(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for p in run_dir.glob("*.json"):
        if p.name == "summary.json" or "instruction" in p.name:
            continue
        data = json.loads(p.read_text())
        tfrecord_name = Path(data.get("tfrecord", p.name)).name
        if ".scenario_" in tfrecord_name:
            # fallback from filename
            tfrecord_name = p.name.split(".scenario_")[0]
        key = (tfrecord_name, int(data["scenario_idx"]))
        out[key] = data
    return out


def classify_planner_failure(row: dict[str, Any]) -> str:
    if bool(row.get("success")):
        return "success"
    ov = bool(row.get("overlap"))
    off = bool(row.get("offroad"))
    gr = bool(row.get("goal_reached"))
    if ov and off:
        return "overlap+offroad"
    if ov:
        return "overlap"
    if off:
        return "offroad"
    if not gr:
        return "goal"
    return "other"


def recompute_metrics(
    results: dict[tuple[str, int], dict[str, Any]],
    failing_gt: set[tuple[str, int]],
) -> dict[str, Any]:
    kept = {k: v for k, v in results.items() if k not in failing_gt}
    dropped = {k: v for k, v in results.items() if k in failing_gt}

    fail_counts = Counter()
    for row in kept.values():
        fail_counts[classify_planner_failure(row)] += 1

    n = len(kept)
    n_success = fail_counts["success"]
    n_goal = sum(1 for row in kept.values() if bool(row.get("goal_reached")))
    return {
        "num_scenarios_raw": len(results),
        "num_dropped_gt_fail": len(dropped),
        "num_scenarios_filtered": n,
        "num_success": n_success,
        "num_goal_reached": n_goal,
        "success_rate": float(n_success / n) if n else 0.0,
        "goal_reach_rate": float(n_goal / n) if n else 0.0,
        "failure_breakdown": {
            "overlap": int(fail_counts["overlap"]),
            "offroad": int(fail_counts["offroad"]),
            "overlap+offroad": int(fail_counts["overlap+offroad"]),
            "goal": int(fail_counts["goal"]),
            "other": int(fail_counts["other"]),
        },
        "dropped_keys": [
            {"tfrecord": k[0], "scenario_idx": k[1]} for k in sorted(dropped.keys())
        ],
    }


def compare_cv_gt(
    cv_results: dict[tuple[str, int], dict[str, Any]],
    gt_results: dict[tuple[str, int], dict[str, Any]],
    failing_gt: set[tuple[str, int]],
) -> dict[str, Any]:
    keys = sorted((set(cv_results) & set(gt_results)) - failing_gt)
    same = flips = cv_only = gt_only = 0
    gt_fixes = Counter()
    gt_only_scenes: list[dict[str, Any]] = []
    for k in keys:
        a, b = cv_results[k], gt_results[k]
        sa, sb = bool(a["success"]), bool(b["success"])
        if (
            sa == sb
            and bool(a["overlap"]) == bool(b["overlap"])
            and bool(a["offroad"]) == bool(b["offroad"])
            and bool(a["goal_reached"]) == bool(b["goal_reached"])
        ):
            same += 1
        if sa != sb:
            flips += 1
            if sa and not sb:
                cv_only += 1
            if (not sa) and sb:
                gt_only += 1
                reason = classify_planner_failure(a)
                gt_fixes[reason] += 1
                gt_only_scenes.append(
                    {
                        "tfrecord": k[0],
                        "scenario_idx": k[1],
                        "cv_failure": reason,
                        "cv": {
                            "success": sa,
                            "overlap": bool(a["overlap"]),
                            "offroad": bool(a["offroad"]),
                            "goal_reached": bool(a["goal_reached"]),
                        },
                        "gt": {
                            "success": sb,
                            "overlap": bool(b["overlap"]),
                            "offroad": bool(b["offroad"]),
                            "goal_reached": bool(b["goal_reached"]),
                        },
                    }
                )
    return {
        "shared_filtered": len(keys),
        "same_outcome": same,
        "success_flips": flips,
        "cv_only_wins": cv_only,
        "gt_only_wins": gt_only,
        "net_gt_minus_cv": gt_only - cv_only,
        "gt_only_wins_fix_cv_failure": dict(gt_fixes),
        "gt_only_win_scenes": gt_only_scenes,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tf_root", type=str, default=DEFAULT_TF_ROOT)
    p.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--start_timestep", type=int, default=10)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--splits",
        type=str,
        default="training,validation",
        help="Comma-separated splits to process.",
    )
    p.add_argument(
        "--skip_gt_eval",
        action="store_true",
        help="Reuse an existing gt_filter dir (pass --gt_filter_dir).",
    )
    p.add_argument("--gt_filter_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"gt_filtered_es_metrics_{stamp}"
    filter_dir = out_dir / "gt_filter"
    metrics_dir = out_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    gt_filters: dict[str, dict[str, Any]] = {}

    if args.skip_gt_eval:
        if not args.gt_filter_dir:
            raise ValueError("--gt_filter_dir required with --skip_gt_eval")
        src = Path(args.gt_filter_dir)
        for split in splits:
            summary = json.loads((src / f"{split}_gt_filter_summary.json").read_text())
            failing = {
                (x["tfrecord"], int(x["scenario_idx"])) for x in summary["failing_keys"]
            }
            summary["_failing_set"] = failing
            gt_filters[split] = summary
            # copy into new out dir for self-contained results
            _save_json(filter_dir / f"{split}_gt_filter_summary.json", {k: v for k, v in summary.items() if k != "_failing_set"})
    else:
        for split in splits:
            print(f"\n========== Building GT filter: {split} ==========")
            filt = build_gt_filter(
                tf_root=Path(args.tf_root),
                split=split,
                start_timestep=args.start_timestep,
                max_num_objects=args.max_num_objects,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            gt_filters[split] = filt
            records = filt.pop("records")
            failing_set = filt["_failing_set"]
            summary_public = {k: v for k, v in filt.items() if k != "_failing_set"}
            _save_json(filter_dir / f"{split}_gt_filter_summary.json", summary_public)
            _save_json(filter_dir / f"{split}_gt_eval_all.json", records)
            # restore for in-memory use
            filt["records"] = records
            filt["_failing_set"] = failing_set
            print(
                f"[gt-eval] {split}: total={filt['num_scenarios']} "
                f"gt_ok={filt['num_gt_success']} gt_fail={filt['num_gt_fail']} "
                f"exclusive={filt['fail_reason_counts_exclusive']}"
            )

    master: dict[str, Any] = {
        "output_dir": str(out_dir),
        "start_timestep": args.start_timestep,
        "success_definition": "goal_reached & ~overlap & ~offroad",
        "gt_filter_policy": "drop scenarios where logged GT fails that success definition",
        "splits": {},
    }

    for split in splits:
        failing_gt = gt_filters[split]["_failing_set"]
        split_out: dict[str, Any] = {
            "gt_filter": {
                "num_scenarios": gt_filters[split]["num_scenarios"],
                "num_gt_success": gt_filters[split]["num_gt_success"],
                "num_gt_fail": gt_filters[split]["num_gt_fail"],
                "fail_reason_counts_any": gt_filters[split]["fail_reason_counts_any"],
                "fail_reason_counts_exclusive": gt_filters[split]["fail_reason_counts_exclusive"],
            },
            "experiments": {},
            "comparisons": {},
        }

        loaded: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
        for label, folder in EXPERIMENT_RUNS.items():
            run_dir = _latest_run_dir(Path(args.exp_root), folder, split)
            if run_dir is None:
                print(f"[warn] missing run dir for {label} {split}")
                continue
            results = load_experiment_results(run_dir)
            loaded[label] = results
            metrics = recompute_metrics(results, failing_gt)
            metrics["run_dir"] = str(run_dir)
            metrics["label"] = label
            split_out["experiments"][label] = metrics
            _save_json(metrics_dir / split / f"{label}.json", metrics)
            print(
                f"[{split}] {label}: filtered {metrics['num_scenarios_filtered']} "
                f"success={metrics['num_success']} "
                f"rate={metrics['success_rate']:.4f} "
                f"fails={metrics['failure_breakdown']}"
            )

        for cv_label, gt_label, name in [
            ("diffusion_es_cv", "diffusion_es_gt", "diffusion_es"),
            ("fm_es_cv", "fm_es_gt", "fm_es"),
        ]:
            if cv_label not in loaded or gt_label not in loaded:
                continue
            cmp = compare_cv_gt(loaded[cv_label], loaded[gt_label], failing_gt)
            split_out["comparisons"][name] = cmp
            _save_json(metrics_dir / split / f"{name}_cv_vs_gt.json", cmp)

        master["splits"][split] = split_out
        _save_json(metrics_dir / split / "summary.json", split_out)

    _save_json(out_dir / "summary.json", master)

    readme = f"""# GT-filtered ES metrics

Generated: {stamp}

## What this is
1. Evaluate logged ground-truth trajectories with the **same** failure function as
   `simulation/runner.py`:
   - Waymax rollout metrics: overlap, offroad
   - Goal from last valid ego log pose + `check_goal_reaching`
   - `success = goal_reached & ~overlap & ~offroad`
2. Drop scenarios where GT itself is not successful.
3. Recompute Diffusion-ES / FM-ES (cv|gt) metrics on the remaining scenes from
   existing per-scenario JSON result files (no re-simulation).

## Layout
- `gt_filter/{{split}}_gt_filter_summary.json` — how many GT fails and why
- `gt_filter/{{split}}_gt_eval_all.json` — per-scenario GT eval
- `metrics/{{split}}/{{experiment}}.json` — filtered planner metrics
- `metrics/{{split}}/{{planner}}_cv_vs_gt.json` — CV vs GT flips on filtered set
- `summary.json` — everything combined

## Settings
- start_timestep: {args.start_timestep}
- tf_root: {args.tf_root}
- exp_root: {args.exp_root}
"""
    (out_dir / "README.md").write_text(readme)
    print(f"\nDone. Results written to: {out_dir}")


if __name__ == "__main__":
    # Quiet TF / JAX a bit
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
