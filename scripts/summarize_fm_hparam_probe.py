#!/usr/bin/env python3
"""Summarize FM hyperparam probe runs (default: n=250 under Flow_matching_sim)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_FM_ROOT = Path("/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim")
DEFAULT_SUBSET = DEFAULT_FM_ROOT / "fm_hparam_probe_n250_seed0"
DEFAULT_RUNS = DEFAULT_FM_ROOT / "fm_hparam_probe_runs"


def _latest_sim(run_folder: Path) -> Path | None:
    sims = sorted(run_folder.glob("simulation_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sims[0] if sims else None


def load_results(sim_dir: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for p in sim_dir.glob("*.json"):
        if p.name == "summary.json" or "instruction" in p.name:
            continue
        row = json.loads(p.read_text())
        if "scenario_idx" not in row:
            continue
        tf = Path(str(row.get("tfrecord", p.name))).name
        out[(tf, int(row["scenario_idx"]))] = row
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset_dir", type=str, default=str(DEFAULT_SUBSET))
    p.add_argument("--runs_dir", type=str, default=str(DEFAULT_RUNS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subset = Path(args.subset_dir)
    runs = Path(args.runs_dir)
    manifest = json.loads((subset / "subset_manifest.json").read_text())
    cases = {(c["tfrecord"], int(c["scenario_idx"])): c for c in manifest["cases"]}
    buckets = {k: c.get("bucket", "unknown") for k, c in cases.items()}

    print(f"subset n={manifest['n_final']} source={manifest.get('source')} buckets={manifest.get('bucket_counts')}")
    print(f"{'tag':55} {'N':>3} {'ok':>3} {'rate':>7}  bucket breakdown")
    print("-" * 100)

    summary = {"subset": str(subset), "runs": {}}
    for run_dir in sorted(runs.glob("*")):
        if not run_dir.is_dir():
            continue
        sim = _latest_sim(run_dir)
        if sim is None:
            print(f"{run_dir.name:55}  (no simulation_* yet)")
            continue
        res = load_results(sim)
        keys = [k for k in cases if k in res]
        if not keys:
            print(f"{run_dir.name:55}  (0 overlapping results)")
            continue
        succ = [k for k in keys if res[k].get("success")]
        by_bucket_n: dict[str, int] = defaultdict(int)
        by_bucket_ok: dict[str, int] = defaultdict(int)
        for k in keys:
            b = buckets[k]
            by_bucket_n[b] += 1
            if res[k].get("success"):
                by_bucket_ok[b] += 1
        rate = len(succ) / len(keys)
        parts = [
            f"{b}={by_bucket_ok[b]}/{by_bucket_n[b]}"
            for b in sorted(by_bucket_n)
        ]
        print(
            f"{run_dir.name:55} {len(keys):3d} {len(succ):3d} {rate:7.1%}  "
            + " ".join(parts)
        )
        summary["runs"][run_dir.name] = {
            "n": len(keys),
            "success": len(succ),
            "rate": rate,
            "by_bucket": {
                b: {"n": by_bucket_n[b], "success": by_bucket_ok[b]} for b in sorted(by_bucket_n)
            },
            "sim_dir": str(sim),
        }

    out = runs / "probe_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out}")
    print(
        "\nHow to read: compare FM tags against diff_gt_baseline_* on the same "
        "n=250 subset. Higher success rate wins for Phase-1 t_start selection."
    )


if __name__ == "__main__":
    main()
