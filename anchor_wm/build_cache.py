"""WOMD tf_example TFRecords -> sim_state cache npz shards (one per TFRecord), in a single pass per file.

Same output as scripts/build_cache.py (`<tfrecord name>.sim_state_cache.npz`, same fields and packing), but
each record is parsed once while streaming the file; scripts/build_cache.py re-reads the file from the
start for every scenario (skip(i).take(1)), which is quadratic in the scenarios per file.

Shards are split across tasks (`--task_id` / `--num_tasks`, e.g. a slurm array), and existing outputs
are skipped, so a re-submitted array only fills what is missing.

  python -m anchor_wm.build_cache --tfrecord_dir .../tf_example/training --output_dir .../cache \
      --task_id 0 --num_tasks 50
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
from glob import glob
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from waymax import config as waymax_config  # noqa: E402
from waymax import dataloader  # noqa: E402
from waymax.dataloader import womd_factories  # noqa: E402

from scripts.build_cache import SIM_STATE_FIELD_EXTRACTORS, _pack_per_scenario_arrays  # noqa: E402


def cache_tfrecord(path: Path, out_dir: Path, max_num_objects: int = 128) -> tuple[Path, int, float]:
    out = out_dir / f"{path.name}.sim_state_cache.npz"
    if out.exists():
        return out, -1, 0.0
    t0 = time.time()
    cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=str(path), max_num_objects=int(max_num_objects),
                              batch_dims=(1,), shuffle_seed=0)
    per_field = {k: [] for k in SIM_STATE_FIELD_EXTRACTORS}
    indices = []
    for i, serialized in enumerate(tf.data.TFRecordDataset([str(path)])):
        processed = dataloader.preprocess_serialized_womd_data(tf.expand_dims(serialized, axis=0), cfg)
        state = womd_factories.simulator_state_from_womd_dict(processed, include_sdc_paths=cfg.include_sdc_paths)
        for k, fn in SIM_STATE_FIELD_EXTRACTORS.items():
            per_field[k].append(fn(state, 0))
        indices.append(i)
    packed = {
        "tfrecord_path": np.asarray(str(path), dtype=np.str_),
        "tfrecord_name": np.asarray(path.name, dtype=np.str_),
        "world_idx": np.asarray(0, dtype=np.int32),
        "num_scenarios": np.asarray(len(indices), dtype=np.int32),
        "scenario_indices": np.asarray(indices, dtype=np.int32),
        "cached_field_names": np.asarray(sorted(SIM_STATE_FIELD_EXTRACTORS), dtype=np.str_),
    }
    packed.update({k: _pack_per_scenario_arrays(v) for k, v in per_field.items()})
    # Write to a temporary name first, so an interrupted task never leaves a truncated shard behind.
    tmp = out_dir / f".{out.name}.tmp.npz"
    np.savez_compressed(tmp, **packed)
    os.replace(tmp, out)
    return out, len(indices), time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_tfrecords", type=int, default=0, help="first N files only (0 = all)")
    p.add_argument("--task_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num_tasks", type=int, default=1)
    p.add_argument("--max_num_objects", type=int, default=128)
    args = p.parse_args()
    paths = [Path(f) for f in sorted(glob(os.path.join(args.tfrecord_dir, "*tfrecord*"))) if os.path.isfile(f)]
    if args.max_tfrecords:
        paths = paths[: args.max_tfrecords]
    mine = paths[args.task_id::args.num_tasks]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"task {args.task_id}/{args.num_tasks}: {len(mine)} of {len(paths)} files", flush=True)
    for path in mine:
        out, n, sec = cache_tfrecord(path, out_dir, args.max_num_objects)
        print(f"{out.name}: " + ("exists" if n < 0 else f"{n} scenarios in {sec:.0f} s"), flush=True)


if __name__ == "__main__":
    main()
