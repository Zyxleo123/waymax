#!/usr/bin/env python3
"""GT-filter the shared failure-subset ES runs (fail1500) and recompute metrics.

Same success rule as ``simulation/runner.py`` / ``filter_gt_failures_and_reeval.py``:
  success = goal_reached & ~overlap & ~offroad  on the *logged* ego trajectory.

Only evaluates the scenarios listed in the subset ``target_scenarios.json``
(not full shards), then drops GT-failing scenes from Diffusion/FM ES (cv|gt|wm)
result JSONs.

Usage (GPU node recommended for fresh GT eval)::

  python scripts/filter_gt_failures_fail1500.py

Reuse an existing GT filter (skip expensive GT rollout)::

  python scripts/filter_gt_failures_fail1500.py \\
    --gt_filter_summary /path/to/gt_filter_summary.json

Outputs::

  /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/gt_filtered_fail1500_<stamp>/
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
import numpy as np
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

DEFAULT_SUBSET_DIR = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0"
DEFAULT_TF_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training"
DEFAULT_EXP_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp"
DEFAULT_OUT_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp"

EXPERIMENT_RUNS = {
    "diffusion_es_cv": "es_diffusion_cv_fail1500",
    "diffusion_es_gt": "es_diffusion_gt_fail1500",
    "diffusion_es_wm10": "es_diffusion_wm10_fail1500",
    "fm_es_cv": "es_fm_cv_fail1500",
    "fm_es_gt": "es_fm_gt_fail1500",
}


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def evaluate_log_trajectory_like_runner(
    sim_state,
    *,
    start_timestep: int = 10,
    goal_threshold_m: float = 2.0,
    rng_key: jax.Array | None = None,
) -> dict[str, Any]:
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
    per_world = []
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
                "goal_reached": bool(goal_reached[world_idx]),
                "reached_timestep": int(goal_reached_step[world_idx]),
                "min_goal_distance_m": float(goal_eval["min_goal_distance_m"][world_idx]),
                "overlap": bool(overlap[world_idx]),
                "offroad": bool(offroad[world_idx]),
                "success": bool(success[world_idx]),
                "fail_reasons": fail_reasons,
            }
        )
    return {"per_world": per_world}


def build_gt_filter_from_targets(
    *,
    target_map: dict[str, list[int]],
    tf_root: Path,
    start_timestep: int,
    max_num_objects: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    fail_any = Counter()
    fail_exclusive = Counter()
    n_success = 0

    for tf_name, indices in tqdm(sorted(target_map.items()), desc="GT-eval shards"):
        tfrecord_path = tf_root / tf_name
        if not tfrecord_path.exists():
            raise FileNotFoundError(tfrecord_path)
        idxs = [int(i) for i in indices]
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start : start + batch_size]
            cfg = dataclasses.replace(
                waymax_config.WOD_1_3_1_TRAINING,
                path=str(tfrecord_path),
                max_num_objects=int(max_num_objects),
                batch_dims=(len(batch_idxs),),
                shuffle_seed=0,
            )
            try:
                sim_state, scenario_to_batch_idx = load_scenario_state_batch_fast(
                    cfg, batch_idxs
                )
            except Exception as e:
                for scenario_idx in batch_idxs:
                    records.append(
                        {
                            "tfrecord": tf_name,
                            "scenario_idx": int(scenario_idx),
                            "success": False,
                            "overlap": False,
                            "offroad": False,
                            "goal_reached": False,
                            "fail_reasons": ["load_error"],
                            "error": str(e),
                        }
                    )
                    fail_any["load_error"] += 1
                    fail_exclusive["load_error"] += 1
                continue

            rng = jax.random.PRNGKey(int(seed) + start + hash(tf_name) % 100000)
            eval_out = evaluate_log_trajectory_like_runner(
                sim_state, start_timestep=start_timestep, rng_key=rng
            )
            for scenario_idx in batch_idxs:
                local_i = int(scenario_to_batch_idx[int(scenario_idx)])
                row = eval_out["per_world"][local_i]
                entry = {
                    "tfrecord": tf_name,
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
                if row["success"]:
                    n_success += 1
                else:
                    for reason in row["fail_reasons"]:
                        fail_any[reason] += 1
                    reasons = set(row["fail_reasons"])
                    if reasons == {"overlap"}:
                        fail_exclusive["overlap_only"] += 1
                    elif reasons == {"offroad"}:
                        fail_exclusive["offroad_only"] += 1
                    elif reasons == {"goal"}:
                        fail_exclusive["goal_only"] += 1
                    elif reasons == {"overlap", "offroad"}:
                        fail_exclusive["overlap+offroad"] += 1
                    else:
                        fail_exclusive["other_combo"] += 1

    failing_keys = [
        {
            "tfrecord": r["tfrecord"],
            "scenario_idx": r["scenario_idx"],
            "fail_reasons": r["fail_reasons"],
        }
        for r in records
        if not r["success"]
    ]
    return {
        "start_timestep": int(start_timestep),
        "success_definition": "goal_reached & ~overlap & ~offroad (same as simulation/runner.py)",
        "num_scenarios": len(records),
        "num_gt_success": n_success,
        "num_gt_fail": len(records) - n_success,
        "fail_reason_counts_any": dict(fail_any),
        "fail_reason_counts_exclusive": dict(fail_exclusive),
        "failing_keys": failing_keys,
        "records": records,
        "_failing_set": {(r["tfrecord"], int(r["scenario_idx"])) for r in failing_keys},
    }


def _latest_run_dir(exp_root: Path, run_folder: str) -> Path | None:
    root = exp_root / run_folder
    if not root.exists():
        return None
    cands = sorted(
        root.glob("simulation_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def load_experiment_results(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for p in run_dir.glob("*.json"):
        if p.name == "summary.json" or "instruction" in p.name:
            continue
        data = json.loads(p.read_text())
        if "scenario_idx" not in data:
            continue
        tfrecord_name = Path(str(data.get("tfrecord", p.name))).name
        if ".scenario_" in tfrecord_name:
            tfrecord_name = p.name.split(".scenario_")[0]
        out[(tfrecord_name, int(data["scenario_idx"]))] = data
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
    fail_counts = Counter(classify_planner_failure(row) for row in kept.values())
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
    p.add_argument("--subset_dir", type=str, default=DEFAULT_SUBSET_DIR)
    p.add_argument("--tf_root", type=str, default=DEFAULT_TF_ROOT)
    p.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--start_timestep", type=int, default=10)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gt_filter_summary",
        type=str,
        default=None,
        help="Reuse an existing gt_filter_summary.json and skip GT rollouts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    target_path = subset_dir / "target_scenarios.json"
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    target_map = json.loads(target_path.read_text())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"gt_filtered_fail1500_{stamp}"
    filter_dir = out_dir / "gt_filter"
    metrics_dir = out_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gt_filter_summary:
        src = Path(args.gt_filter_summary)
        if not src.exists():
            raise FileNotFoundError(src)
        summary_public = json.loads(src.read_text())
        failing_set = {
            (str(x["tfrecord"]), int(x["scenario_idx"]))
            for x in summary_public.get("failing_keys", [])
        }
        _save_json(filter_dir / "gt_filter_summary.json", summary_public)
        print(
            f"Reusing GT filter from {src}: total={summary_public.get('num_scenarios')} "
            f"ok={summary_public.get('num_gt_success')} fail={summary_public.get('num_gt_fail')} "
            f"exclusive={summary_public.get('fail_reason_counts_exclusive')}"
        )
    else:
        print(f"Evaluating GT on {sum(len(v) for v in target_map.values())} target scenarios...")
        filt = build_gt_filter_from_targets(
            target_map=target_map,
            tf_root=Path(args.tf_root),
            start_timestep=args.start_timestep,
            max_num_objects=args.max_num_objects,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        records = filt.pop("records")
        failing_set = filt["_failing_set"]
        summary_public = {k: v for k, v in filt.items() if k != "_failing_set"}
        _save_json(filter_dir / "gt_filter_summary.json", summary_public)
        _save_json(filter_dir / "gt_eval_all.json", records)
        print(
            f"GT filter: total={summary_public['num_scenarios']} "
            f"ok={summary_public['num_gt_success']} fail={summary_public['num_gt_fail']} "
            f"exclusive={summary_public['fail_reason_counts_exclusive']}"
        )

    loaded: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    experiments: dict[str, Any] = {}
    for label, folder in EXPERIMENT_RUNS.items():
        run_dir = _latest_run_dir(Path(args.exp_root), folder)
        if run_dir is None:
            print(f"[warn] missing {folder}")
            continue
        results = load_experiment_results(run_dir)
        loaded[label] = results
        metrics = recompute_metrics(results, failing_set)
        metrics["run_dir"] = str(run_dir)
        metrics["label"] = label
        experiments[label] = metrics
        _save_json(metrics_dir / f"{label}.json", metrics)
        print(
            f"{label}: filtered N={metrics['num_scenarios_filtered']} "
            f"(dropped {metrics['num_dropped_gt_fail']}) "
            f"success={metrics['num_success']} rate={metrics['success_rate']:.4f} "
            f"fails={metrics['failure_breakdown']}"
        )

    comparisons: dict[str, Any] = {}
    for cv_label, gt_label, name in [
        ("diffusion_es_cv", "diffusion_es_gt", "diffusion_es"),
        ("fm_es_cv", "fm_es_gt", "fm_es"),
    ]:
        if cv_label not in loaded or gt_label not in loaded:
            continue
        cmp = compare_cv_gt(loaded[cv_label], loaded[gt_label], failing_set)
        comparisons[name] = cmp
        _save_json(metrics_dir / f"{name}_cv_vs_gt.json", cmp)
        print(
            f"{name} CV vs GT: shared={cmp['shared_filtered']} "
            f"net={cmp['net_gt_minus_cv']} "
            f"GT fixes={cmp['gt_only_wins_fix_cv_failure']}"
        )

    master = {
        "output_dir": str(out_dir),
        "subset_dir": str(subset_dir),
        "start_timestep": args.start_timestep,
        "success_definition": "goal_reached & ~overlap & ~offroad",
        "gt_filter": summary_public,
        "raw_rates_note": {
            "diffusion_es_gt": 0.5227,
            "diffusion_es_cv": 0.4620,
            "fm_es_gt": 0.3940,
            "fm_es_cv": 0.2267,
        },
        "experiments": experiments,
        "comparisons": comparisons,
    }
    _save_json(out_dir / "summary.json", master)
    (out_dir / "README.md").write_text(
        f"""# GT-filtered fail1500 ES metrics

Generated: {stamp}

Drops scenarios where **logged GT** fails the same closed-loop success rule, then
recomputes Diffusion-ES / FM-ES (cv|gt|wm10) on the shared 1500 FM-failure subset.

See `gt_filter/` and `metrics/` and `summary.json`.
"""
    )
    print(f"\nDone. Results: {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
