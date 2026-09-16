#!/usr/bin/env python3
"""Summarize FM hyperparam probe runs on the mixed Diff-OK / Diff-fail GT subset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SUBSET = Path("/zfsauton/scratch/sbellad/immediate_exp/fm_hparam_probe_n20_seed0")
DEFAULT_RUNS = Path("/zfsauton/scratch/sbellad/immediate_exp/fm_hparam_probe_runs")


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
    buckets = {k: c["bucket"] for k, c in cases.items()}

    print(f"subset n={manifest['n_final']} buckets={manifest['bucket_counts']}")
    print(f"{'tag':40} {'N':>3} {'ok':>3} {'rate':>6}  {'diffOK_fmF':>10} {'diffFail':>8}")
    print("-" * 80)

    summary = {"subset": str(subset), "runs": {}}
    for run_dir in sorted(runs.glob("*")):
        if not run_dir.is_dir():
            continue
        sim = _latest_sim(run_dir)
        if sim is None:
            print(f"{run_dir.name:40}  (no simulation_* yet)")
            continue
        res = load_results(sim)
        # Restrict to probe keys if present.
        keys = [k for k in cases if k in res]
        if not keys:
            print(f"{run_dir.name:40}  (0 overlapping results)")
            continue
        succ = [k for k in keys if res[k].get("success")]
        by_bucket_n = defaultdict(int)
        by_bucket_ok = defaultdict(int)
        for k in keys:
            b = buckets[k]
            by_bucket_n[b] += 1
            if res[k].get("success"):
                by_bucket_ok[b] += 1
        rate = len(succ) / len(keys)
        a_n, a_ok = by_bucket_n["diff_ok_fm_fail"], by_bucket_ok["diff_ok_fm_fail"]
        b_n, b_ok = by_bucket_n["diff_fail"], by_bucket_ok["diff_fail"]
        a_s = f"{a_ok}/{a_n}" if a_n else "-"
        b_s = f"{b_ok}/{b_n}" if b_n else "-"
        print(f"{run_dir.name:40} {len(keys):3d} {len(succ):3d} {rate:6.1%}  {a_s:>10} {b_s:>8}")
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
        "\nHow to read: on diff_ok_fm_fail, higher recovery => better FM settings. "
        "On diff_fail, gains are bonus; Diff baseline rate is the soft ceiling."
    )


if __name__ == "__main__":
    main()
