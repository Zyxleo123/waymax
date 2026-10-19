"""Rebuild the fail1500 subset from a finished fail1500 run, split into shard groups.

The original subset dir (/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0)
is gone, but every fail1500 run wrote one JSON per scene named
``<tfrecord basename>.scenario_<idx>.json`` with ``tfrecord`` / ``scenario_idx``
inside, so the scene list is recoverable from any of them.

Writes, under ``--out``:
  target_scenarios.json                  all scenes, {tfrecord basename: [idx, ...]}
  group_<g>/tfrecords/training/<shard>   symlinks into --womd_dir, this group's shards only
  group_<g>/target_scenarios.json        this group's scenes
  manifest.json                          counts, source run, groups

Groups are whole tfrecords (the runner walks every file in its tfrecord dir), balanced
by scene count, so each group is one resumable job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(r"^(?P<tf>.+)\.scenario_(?P<idx>\d+)\.json$")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="a fail1500 simulation_* dir with per-scene JSONs")
    p.add_argument("--womd_dir", default="/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training")
    p.add_argument("--out", required=True)
    p.add_argument("--groups", type=int, default=6)
    p.add_argument("--expect", type=int, default=1500)
    a = p.parse_args()

    scenes = defaultdict(list)
    for f in sorted(Path(a.run_dir).glob("*.json")):
        m = NAME_RE.match(f.name)
        if m is None:
            continue
        r = json.loads(f.read_text())
        tf = os.path.basename(str(r["tfrecord"]))
        if tf != m["tf"] or int(r["scenario_idx"]) != int(m["idx"]):
            raise SystemExit(f"{f.name}: name and contents disagree ({tf}, {r['scenario_idx']})")
        scenes[tf].append(int(r["scenario_idx"]))
    n = sum(len(v) for v in scenes.values())
    if n != a.expect:
        raise SystemExit(f"found {n} scenes in {a.run_dir}, expected {a.expect}")
    missing = [tf for tf in scenes if not (Path(a.womd_dir) / tf).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shards missing under {a.womd_dir}, e.g. {missing[:3]}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = {tf: sorted(v) for tf, v in sorted(scenes.items())}
    (out / "target_scenarios.json").write_text(json.dumps(targets, indent=1) + "\n")

    # Greedy balance by scene count: largest shard to the lightest group.
    groups = [dict() for _ in range(a.groups)]
    load = [0] * a.groups
    for tf, idx in sorted(targets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        g = load.index(min(load))
        groups[g][tf] = idx
        load[g] += len(idx)
    for g, tg in enumerate(groups):
        d = out / f"group_{g}" / "tfrecords" / "training"
        d.mkdir(parents=True, exist_ok=True)
        for tf in tg:
            link = d / tf
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(Path(a.womd_dir) / tf, link)
        (out / f"group_{g}" / "target_scenarios.json").write_text(json.dumps(dict(sorted(tg.items())), indent=1) + "\n")
    manifest = {"source_run": str(a.run_dir), "womd_dir": a.womd_dir, "n_scenes": n,
                "n_shards": len(targets), "groups": [{"group": g, "n_shards": len(tg), "n_scenes": load[g]}
                                                     for g, tg in enumerate(groups)]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
