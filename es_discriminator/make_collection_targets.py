"""Target-scenario JSON for ES candidate collection on whole WOMD training shards.

Lists every record of the requested shards except scenes already collected (any
`target_scenarios.json` passed via --exclude) and records without a lane graph. The output
has the runner's `--target_scenario_path` format: {tfrecord basename: [record indices]}.

    python -m es_discriminator.make_collection_targets --first 2 --last 11 \
        --exclude /zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0/target_scenarios.json \
        --out .../target_scenarios.json
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile

LANE_GRAPH_DIR = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"


def shard_name(i: int) -> str:
    return f"training_tfexample.tfrecord-{i:05d}-of-01000"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--first", type=int, required=True, help="first training shard index")
    p.add_argument("--last", type=int, required=True, help="last training shard index (inclusive)")
    p.add_argument("--exclude", nargs="*", default=[], help="target_scenarios.json files already collected")
    p.add_argument("--lane_graph_dir", default=LANE_GRAPH_DIR)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    excluded: dict[str, set[int]] = {}
    for path in a.exclude:
        for name, idx in json.load(open(path)).items():
            excluded.setdefault(name, set()).update(int(i) for i in idx)

    targets, report = {}, {}
    for i in range(a.first, a.last + 1):
        name = shard_name(i)
        with zipfile.ZipFile(os.path.join(a.lane_graph_dir, f"{name}.lanegraph.zip")) as z:
            manifest = json.loads(z.read("manifest.json"))
        n = int(manifest["num_records"])
        missing = set(int(m) for m in manifest.get("missing_indices") or [])
        skip = excluded.get(name, set())
        targets[name] = [r for r in range(n) if r not in skip and r not in missing]
        report[name] = dict(records=n, already_collected=len(skip & set(range(n))),
                            no_lane_graph=len(missing), targets=len(targets[name]))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(targets, f)
    os.replace(tmp, a.out)
    total = {k: sum(r[k] for r in report.values()) for k in ("records", "already_collected", "no_lane_graph", "targets")}
    print(json.dumps({"shards": report, "total": total}, indent=1))


if __name__ == "__main__":
    main()
