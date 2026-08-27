"""Join the closed-loop eval JSONs into one label table.

Policy labels come from a diffusion sim run (``simulation/runner.py`` writes one
JSON per scenario -- for *every* scenario, not just failures, so the successes
are in there too).  Expert labels come from a log-replay run over a subset of
the same shards.

The expert label is a *validity gate*, not a feature: on the 10k scenes where
both runs overlap, the policy fails 93.1% of the time when log replay itself
fails, versus 3.28% when it does not.  A scene the logged expert cannot drive
cleanly is dirty ground truth, not a hard case, and upweighting it would teach
the policy to leave the road.  Anything fit against the raw labels rediscovers
that 28x lift as its top feature, so it has to be masked out up front.

Writes ``labels.npz`` (one row per evaluated scenario) plus ``labels.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

SHARD_RE = re.compile(r"(.*tfrecord-(\d{5})-of-\d{5})\.scenario_(\d+)\.json$")


def _scan(eval_dir: str) -> dict[tuple[str, int], dict]:
    """Read every per-scenario JSON under ``eval_dir`` (recursively)."""
    out: dict[tuple[str, int], dict] = {}
    for path in glob.glob(os.path.join(eval_dir, "**", "*.json"), recursive=True):
        if os.path.basename(path) == "summary.json":
            continue
        match = SHARD_RE.match(os.path.basename(path))
        if match is None:
            continue
        try:
            with open(path) as handle:
                row = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        shard = os.path.basename(row.get("tfrecord", match.group(1)))
        out[(shard, int(row["scenario_idx"]))] = row
    return out


def _shard_index(shard: str) -> int:
    match = re.search(r"tfrecord-(\d{5})-of-\d{5}", shard)
    return int(match.group(1)) if match else -1


def build(policy_dir: str, expert_dir: str | None) -> dict[str, np.ndarray]:
    policy = _scan(policy_dir)
    expert = _scan(expert_dir) if expert_dir else {}
    if not policy:
        raise SystemExit(f"no per-scenario JSONs found under {policy_dir}")

    keys = sorted(policy)
    cols: dict[str, list] = defaultdict(list)
    for shard, scenario_idx in keys:
        row = policy[(shard, scenario_idx)]
        cols["shard"].append(shard)
        cols["shard_idx"].append(_shard_index(shard))
        cols["scenario_idx"].append(scenario_idx)
        cols["ego_idx"].append(int(row.get("ego_idx", -1)))
        goal = row.get("goal_xy", [np.nan, np.nan])
        cols["goal_x"].append(float(goal[0]))
        cols["goal_y"].append(float(goal[1]))
        cols["pol_success"].append(bool(row.get("success", False)))
        cols["pol_offroad"].append(bool(row.get("offroad", False)))
        cols["pol_overlap"].append(bool(row.get("overlap", False)))
        cols["pol_goal_reached"].append(bool(row.get("goal_reached", False)))
        cols["pol_tl_violation"].append(bool(row.get("tl_violation", False)))
        cols["pol_reached_t"].append(int(row.get("reached_timestep", -1)))
        cols["pol_min_goal_dist"].append(float(row.get("min_goal_distance_m", np.nan)))
        cols["pol_final_goal_dist"].append(float(row.get("final_goal_distance_m", np.nan)))

        exp_row = expert.get((shard, scenario_idx))
        cols["exp_labeled"].append(exp_row is not None)
        cols["exp_success"].append(bool(exp_row.get("success", False)) if exp_row else False)
        cols["exp_offroad"].append(bool(exp_row.get("offroad", False)) if exp_row else False)
        cols["exp_overlap"].append(bool(exp_row.get("overlap", False)) if exp_row else False)

    table = {"shard": np.asarray(cols.pop("shard"), dtype=np.str_)}
    for name, values in cols.items():
        arr = np.asarray(values)
        table[name] = arr.astype(np.bool_) if arr.dtype == bool else arr
    return table


def summarize(table: dict[str, np.ndarray]) -> dict:
    n = int(table["scenario_idx"].size)
    fail = ~table["pol_success"]
    labeled = table["exp_labeled"]
    clean = labeled & table["exp_success"]
    dirty = labeled & ~table["exp_success"]

    def rate(mask: np.ndarray) -> float:
        return float(fail[mask].mean()) if mask.any() else float("nan")

    return {
        "n_scenarios": n,
        "n_shards": int(np.unique(table["shard"]).size),
        "n_policy_failures": int(fail.sum()),
        "policy_failure_rate": float(fail.mean()),
        "failure_modes": {
            "offroad": int((fail & table["pol_offroad"]).sum()),
            "overlap": int((fail & table["pol_overlap"]).sum()),
            "no_goal": int((fail & ~table["pol_goal_reached"]).sum()),
            "reached_goal_but_violated": int((fail & table["pol_goal_reached"]).sum()),
        },
        "expert_gate": {
            "n_expert_labeled": int(labeled.sum()),
            "n_expert_dirty": int(dirty.sum()),
            "expert_dirty_rate": float(dirty.sum() / labeled.sum()) if labeled.any() else None,
            "policy_failure_rate_expert_clean": rate(clean),
            "policy_failure_rate_expert_dirty": rate(dirty),
            "n_failures_on_dirty": int((fail & dirty).sum()),
            "frac_failures_that_are_dirty": (
                float((fail & dirty).sum() / (fail & labeled).sum()) if (fail & labeled).any() else None
            ),
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--expert_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)

    table = build(args.policy_dir, args.expert_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "labels.npz"), **table)

    summary = summarize(table)
    with open(os.path.join(args.out_dir, "labels.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
