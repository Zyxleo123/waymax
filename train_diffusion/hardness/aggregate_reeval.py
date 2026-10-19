"""Aggregate the re-eval per-scenario JSONs into one comparison table.

The alpha-sweep arms were re-scored offline (reeval_arm.sbatch) with the fixed
eval. ``simulation.runner.run`` drops one ``*.scenario_NNN.json`` per scenario
plus a ``summary.json`` under a timestamped subdir; this reads the per-scenario
files (they carry the full breakdown -- offroad/overlap/tl_violation, not just
success), recomputes the V-Max-parity metric set per (arm, set), and writes both
a machine-readable ``comparison.json`` and a printed table.

Reads from disk only, so the table redraws without touching the GPU. When an arm
has been re-evaluated more than once, the newest timestamped subdir wins.

    python -m train_diffusion.hardness.aggregate_reeval \
        --reeval_dir /data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/hardness_reeval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The sweep arms, in the order that reads as a sweep: control, the clean-data
# ablation, then tempered reweighting alpha=0.3/0.5/1.0.
ARM_ORDER = ["control", "cleanonly", "alpha03", "alpha05", "alpha10"]
ARM_LABEL = {
    "control": "alpha=0  keep dirty (control)",
    "cleanonly": "alpha=0  exclude dirty",
    "alpha03": "alpha=0.3 exclude dirty",
    "alpha05": "alpha=0.5 exclude dirty",
    "alpha10": "alpha=1.0 exclude dirty",
}
SETS = ["failures", "womd"]


def _latest_run_dir(set_dir: Path) -> Path | None:
    """Newest timestamped subdir under <arm>/<set>/ that holds scenario JSONs."""
    if not set_dir.is_dir():
        return None
    runs = [d for d in set_dir.iterdir()
            if d.is_dir() and any(d.glob("*.scenario_*.json"))]
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def _metrics(run_dir: Path) -> dict:
    rows = []
    for f in run_dir.glob("*.scenario_*.json"):
        with open(f) as handle:
            rows.append(json.load(handle))
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def rate(key: str) -> float:
        return sum(1 for r in rows if bool(r.get(key, False))) / n

    reached_but_violated = sum(
        1 for r in rows
        if bool(r.get("goal_reached", False)) and not bool(r.get("success", False))
    ) / n
    return {
        "n": n,
        "safe_goal_reach_rate": rate("success"),
        "goal_reach_rate": rate("goal_reached"),
        "offroad_rate": rate("offroad"),
        "overlap_rate": rate("overlap"),
        "tl_violation_rate": rate("tl_violation"),
        "reached_but_violated_rate": reached_but_violated,
        "run_dir": run_dir.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reeval_dir",
        default="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/hardness_reeval",
    )
    parser.add_argument("--epoch", default="epoch_0010",
                        help="label only, recorded in comparison.json")
    args = parser.parse_args()

    root = Path(args.reeval_dir)
    table: dict[str, dict[str, dict]] = {}
    for arm in ARM_ORDER:
        table[arm] = {}
        for s in SETS:
            run_dir = _latest_run_dir(root / arm / s)
            table[arm][s] = _metrics(run_dir) if run_dir else {"n": 0}

    out = {"epoch": args.epoch, "reeval_dir": root.as_posix(), "arms": table}
    out_path = root / "comparison.json"
    root.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=2)

    # Printed table: safe-goal-reach is the headline; the violation columns say
    # WHY a reached goal still failed, which is the whole point of the parity
    # metric.
    for s in SETS:
        print(f"\n===== {s} =====")
        hdr = (f"{'arm':<28} {'n':>4} {'safe':>7} {'goal':>7} "
               f"{'offrd':>7} {'ovlp':>7} {'tl':>7} {'reach&viol':>11}")
        print(hdr)
        print("-" * len(hdr))
        for arm in ARM_ORDER:
            m = table[arm][s]
            if m.get("n", 0) == 0:
                print(f"{ARM_LABEL[arm]:<28} {'--':>4} {'(no data)':>7}")
                continue
            print(f"{ARM_LABEL[arm]:<28} {m['n']:>4} "
                  f"{m['safe_goal_reach_rate']:>7.3f} {m['goal_reach_rate']:>7.3f} "
                  f"{m['offroad_rate']:>7.3f} {m['overlap_rate']:>7.3f} "
                  f"{m['tl_violation_rate']:>7.3f} {m['reached_but_violated_rate']:>11.3f}")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
