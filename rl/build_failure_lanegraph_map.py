#!/usr/bin/env python3
"""Map each ``failures.tfrecord`` scene back to its source WOMD shard + index.

There is no ``failures.tfrecord.lanegraph.zip``: the failure set is a byte-copy
of ScenarioMax records, and lane graphs were only ever extracted for the WOMD
``training_tfexample.*`` shards. Any eval that needs a lane graph -- traffic
light violation is part of the success criterion -- therefore dies with
"lane graph shard zip does not exist" the moment the scenes actually load.

Rather than re-extract road graphs for a set that is a copy of scenes we already
have graphs for, this writes a sidecar that redirects each failure position to
the shard entry that already exists:

    {"entries": [["training_tfexample.tfrecord-00001-of-01000", 251], ...]}

Position ``i`` in ``entries`` is scenario ``i`` of failures.tfrecord.
``lane_graph.lane_graph_utils.LaneGraphLoader`` picks the file up automatically.

The ordering is not free choice -- it has to match how the tfrecord was written.
``rl/build_failure_scenariomax.py`` writes ``for idx in sorted(wanted)``, i.e.
ascending ``scenariomax_record_index`` over the ``scenariomax_hit`` records, so
that is the order reproduced here. ``--verify`` re-checks the correspondence by
comparing logged SDC trajectories scene by scene, which is the check that
matters: a silently wrong map would attach the wrong traffic lights to a scene
and quietly corrupt the metric instead of failing.

Usage::

    python -m rl.build_failure_lanegraph_map --verify 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_TRACE = "/zfsauton/scratch/yixiz/failure_womd_trace.json"
DEFAULT_TFRECORD = "/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord"


def build_entries(trace_path: str) -> list[list]:
    with open(trace_path) as handle:
        trace = json.load(handle)

    records = [
        r for r in trace["records"] if r.get("scenariomax_hit") and r.get("error") is None
    ]
    if not records:
        raise RuntimeError(f"no scenariomax-mapped failures in {trace_path}")

    sources = {r["scenariomax_tfrecord_path"] for r in records}
    if len(sources) != 1:
        # build_failure_scenariomax.py iterates by_source.items() and sorts only
        # *within* each source, so multiple sources make the write order depend
        # on dict insertion order. Refuse rather than guess.
        raise RuntimeError(
            f"trace spans {len(sources)} scenariomax sources; write order is ambiguous."
        )

    records.sort(key=lambda r: int(r["scenariomax_record_index"]))
    return [
        [os.path.basename(r["tfrecord"]), int(r["scenario_idx"])] for r in records
    ]


def verify(tfrecord_path: str, entries: list[list], source_dir: str, num_probes: int) -> bool:
    """Compare logged SDC trajectories between each side of the mapping."""
    import jax

    jax.config.update("jax_platforms", "cpu")
    from types import SimpleNamespace

    from data.scenario_loader import load_scenario_state_batch_fast
    from simulation.runner import _dataset_config_for

    one = SimpleNamespace(max_num_objects=128, num_worlds=1)

    def sdc_xy(state):
        is_sdc = np.asarray(state.object_metadata.is_sdc)[0]
        k = int(np.argmax(is_sdc))
        return np.stack(
            [np.asarray(state.log_trajectory.x)[0, k], np.asarray(state.log_trajectory.y)[0, k]],
            axis=-1,
        )

    n = len(entries)
    probes = sorted({0, n - 1, *np.linspace(0, n - 1, num=max(num_probes, 2), dtype=int).tolist()})
    ok = True
    for i in probes:
        shard, scenario_idx = entries[i]
        source_path = os.path.join(source_dir, shard)
        fail_state, _ = load_scenario_state_batch_fast(
            _dataset_config_for(tfrecord_path, one), [i]
        )
        womd_state, _ = load_scenario_state_batch_fast(
            _dataset_config_for(source_path, one), [scenario_idx]
        )
        a, b = sdc_xy(fail_state), sdc_xy(womd_state)
        m = min(len(a), len(b))
        err = float(np.max(np.abs(a[:m] - b[:m])))
        status = "ok" if err == 0.0 else "MISMATCH"
        print(f"  pos {i:3d} -> {shard} scenario {scenario_idx:3d}  max|dxy|={err:g}  {status}")
        ok = ok and err == 0.0
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", default=DEFAULT_TRACE)
    parser.add_argument("--tfrecord", default=DEFAULT_TFRECORD)
    parser.add_argument(
        "--source-dir",
        default="/zfsauton/scratch/eshau/womd/tf_example/training",
        help="directory holding the WOMD shards the failures were traced back to",
    )
    parser.add_argument("--output", default=None, help="default: <tfrecord>.lanegraph_map.json")
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="cross-check N positions (plus first and last) by comparing SDC trajectories",
    )
    args = parser.parse_args()

    entries = build_entries(args.trace_json)
    print(f"mapped {len(entries)} failure scenes over "
          f"{len({e[0] for e in entries})} source shards")

    if args.verify:
        print("verifying mapping:")
        if not verify(args.tfrecord, entries, args.source_dir, args.verify):
            print("FAILED: mapping does not line up; refusing to write the sidecar.")
            return 1

    output = args.output or f"{args.tfrecord}.lanegraph_map.json"
    with open(output, "w") as handle:
        json.dump({"source_dir": args.source_dir, "entries": entries}, handle, indent=1)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
