#!/usr/bin/env python3
"""Build a fixed shared failure subset for 1:1 ES comparisons.

Reads FM collect-failure JSONs (keeps success==false), samples N with a fixed seed,
truncates each shard to a multiple of num_worlds, writes:

  - target_scenarios.json
  - subset_manifest.json
  - tfrecords/training/  (symlinks; use with --split training)

Example::

  python scripts/build_failure_subset_for_es.py --n 1500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def classify(row: dict) -> str:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--failure_exp_dir",
        type=str,
        default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_fm_training_collect_failure_20260903_030456",
    )
    p.add_argument(
        "--tf_root",
        type=str,
        default="/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training",
    )
    p.add_argument(
        "--lane_graph_dir",
        type=str,
        default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0",
    )
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_worlds", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fail_dir = Path(args.failure_exp_dir)
    tf_root = Path(args.tf_root)
    lg_dir = Path(args.lane_graph_dir)
    out_dir = Path(args.out_dir)
    # run_simulation.py joins tfrecord_dir / split
    tf_link_dir = out_dir / "tfrecords" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    tf_link_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict] = []
    for p in sorted(fail_dir.glob("*.scenario_*.json")):
        row = json.loads(p.read_text())
        if bool(row.get("success")):
            continue
        tf = Path(row["tfrecord"]).name
        failures.append(
            {
                "tfrecord": tf,
                "scenario_idx": int(row["scenario_idx"]),
                "fail_type": classify(row),
                "overlap": bool(row.get("overlap")),
                "offroad": bool(row.get("offroad")),
                "goal_reached": bool(row.get("goal_reached")),
                "source_json": p.name,
            }
        )
    print(f"Loaded {len(failures)} failures from {fail_dir}")

    nw = int(args.num_worlds)
    target_n = (int(args.n) // nw) * nw
    if target_n <= 0:
        raise SystemExit("--n too small for num_worlds")

    rng = random.Random(int(args.seed))
    pool = list(failures)
    rng.shuffle(pool)

    # Greedy: take shuffled failures, keep per-shard counts divisible by nw.
    by_tf: dict[str, list[dict]] = defaultdict(list)
    accepted = 0
    for row in pool:
        if accepted >= target_n:
            break
        by_tf[row["tfrecord"]].append(row)
        accepted += 1

    final: list[dict] = []
    leftovers: list[dict] = []
    for tf, rows in sorted(by_tf.items()):
        rows = sorted(rows, key=lambda r: r["scenario_idx"])
        n_keep = (len(rows) // nw) * nw
        final.extend(rows[:n_keep])
        leftovers.extend(rows[n_keep:])

    # Top-up from leftovers + remaining pool, only adding full nw-sized blocks
    # from the same shard when possible; otherwise accumulate per shard.
    remaining = leftovers + [r for r in pool if r not in final and r not in leftovers]
    # de-dup
    final_keys = {(r["tfrecord"], r["scenario_idx"]) for r in final}
    remaining_unique = []
    seen = set(final_keys)
    for r in remaining:
        k = (r["tfrecord"], r["scenario_idx"])
        if k in seen:
            continue
        seen.add(k)
        remaining_unique.append(r)
    rng.shuffle(remaining_unique)

    pending: dict[str, list[dict]] = defaultdict(list)
    for r in remaining_unique:
        if len(final) >= target_n:
            break
        pending[r["tfrecord"]].append(r)
        if len(pending[r["tfrecord"]]) >= nw:
            block = pending[r["tfrecord"]][:nw]
            pending[r["tfrecord"]] = pending[r["tfrecord"]][nw:]
            final.extend(sorted(block, key=lambda x: x["scenario_idx"]))

    if len(final) > target_n:
        final = final[:target_n]
    if len(final) < target_n:
        print(
            f"WARNING: only collected {len(final)} (requested {target_n}); "
            "shard remainders could not fill more full batches."
        )

    target_map: dict[str, list[int]] = defaultdict(list)
    for r in final:
        target_map[r["tfrecord"]].append(int(r["scenario_idx"]))
    target_map = {tf: sorted(idxs) for tf, idxs in sorted(target_map.items())}
    for tf, idxs in target_map.items():
        assert len(idxs) % nw == 0, (tf, len(idxs))

    missing = []
    for tf in target_map:
        src = tf_root / tf
        lg = lg_dir / f"{tf}.lanegraph.zip"
        if not src.exists():
            missing.append(f"tfrecord:{src}")
        if not lg.exists():
            missing.append(f"lanegraph:{lg}")
        link = tf_link_dir / tf
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src, link)
    if missing:
        raise SystemExit("Missing assets:\n" + "\n".join(missing[:30]))

    fail_mix = Counter(r["fail_type"] for r in final)
    manifest = {
        "source_failure_exp_dir": str(fail_dir),
        "n_requested": int(args.n),
        "n_final": len(final),
        "seed": int(args.seed),
        "num_worlds": nw,
        "num_shards": len(target_map),
        "failure_type_counts": dict(fail_mix),
        "tfrecord_dir_for_run_simulation": str(out_dir / "tfrecords"),
        "target_scenario_path": str(out_dir / "target_scenarios.json"),
        "scenarios": final,
    }
    (out_dir / "target_scenarios.json").write_text(json.dumps(target_map, indent=2) + "\n")
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(final)} scenarios across {len(target_map)} shards")
    print(f"  failure mix: {dict(fail_mix)}")
    print(f"  target: {out_dir / 'target_scenarios.json'}")
    print(f"  tfrecord_dir: {out_dir / 'tfrecords'}  (use --split training)")
    print(f"  manifest: {out_dir / 'subset_manifest.json'}")


if __name__ == "__main__":
    main()
