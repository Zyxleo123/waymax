"""Converts WOMD tf.Example shards into Diffusion Planner training samples (their npz layout + data list json).

  python diffusion_planner_womd/convert_womd.py --tfrecord .../training_tfexample.tfrecord-00000-of-01000 \
      --lane_graph_dir /zfsauton/scratch/mineuih/waymax_rs/lane_graphs --out_dir <data_root>

writes <data_root>/<shard>/<scenario>_t<step>.npz and <data_root>/lists/<shard>.json. Merge the lists with
``merge_lists.py`` and pass <data_root> / the merged list to their train_predictor.py.

Anchors: like the waymax_rs diffusion policy, a uniformly random step with ego observed over the whole 5 s
horizon; their dataset is fixed on disk, so ``--anchors_per_scenario`` anchors are drawn per scenario (2 gives
~1M samples from WOMD train, the size of their nuPlan training set).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf
from waymax import config as waymax_config
from waymax import dataloader
from waymax.dataloader import womd_factories

from diffusion_planner_womd.features import ScenarioMap, WorldLog, build_sample, valid_anchors
from scores.lane_graph_utils import LaneGraphShardStore

BATCH = 16


def _states(path: str, max_num_objects: int):
    """(scenario index, SimulatorState batch of one) over the shard, in file order."""
    cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=path, max_num_objects=max_num_objects)
    buf, idx = [], []
    for i, serialized in enumerate(tf.data.TFRecordDataset([path])):
        buf.append(serialized)
        idx.append(i)
        if len(buf) == BATCH:
            yield idx, womd_factories.simulator_state_from_womd_dict(dataloader.preprocess_serialized_womd_data(tf.stack(buf), cfg))
            buf, idx = [], []
    if buf:
        yield idx, womd_factories.simulator_state_from_womd_dict(dataloader.preprocess_serialized_womd_data(tf.stack(buf), cfg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", required=True)
    parser.add_argument("--lane_graph_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--anchors_per_scenario", type=int, default=2)
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many scenarios (debugging).")
    args = parser.parse_args()

    shard = os.path.basename(args.tfrecord)
    out = Path(args.out_dir) / shard
    out.mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "lists").mkdir(parents=True, exist_ok=True)
    store = LaneGraphShardStore(str(Path(args.lane_graph_dir) / f"{shard}.lanegraph.zip"))
    rng = np.random.default_rng([args.seed, zlib.crc32(shard.encode())])

    files, skipped = [], 0
    for indices, state in _states(args.tfrecord, args.max_num_objects):
        for w, i in enumerate(indices):
            if args.limit is not None and i >= args.limit:
                break
            graph = store.get_by_record_index(i)
            log = WorldLog.from_state(state, w)
            anchors = valid_anchors(log)
            if graph is None or not len(anchors):
                skipped += 1
                continue
            smap = ScenarioMap(graph, log)
            for t in rng.choice(anchors, size=min(args.anchors_per_scenario, len(anchors)), replace=False):
                name = f"{shard}/{i:05d}_t{int(t):02d}.npz"
                np.savez_compressed(Path(args.out_dir) / name, **build_sample(log, smap, int(t)))
                files.append(name)
        if args.limit is not None and indices[-1] + 1 >= args.limit:
            break
    with open(Path(args.out_dir) / "lists" / f"{shard}.json", "w") as f:
        json.dump(files, f)
    print(f"{shard}: {len(files)} samples, {skipped} scenarios skipped (no lane graph or no valid anchor)")


if __name__ == "__main__":
    main()
