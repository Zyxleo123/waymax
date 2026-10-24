#!/usr/bin/env python3
"""Build byte-offset indexes for a slice of a shard set (one SLURM array task).

Resumable by construction: a shard whose index already exists is skipped, so a
task killed by the time limit is fixed by resubmitting the same array.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.diffusion_ft.scene_sampler import index_path_for  # noqa: E402
from rl.tfrecord_fast import build_offset_index  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard_glob", required=True)
    p.add_argument("--index_dir", default="")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--n_tasks", type=int, default=1)
    a = p.parse_args()

    shards = sorted(glob.glob(a.shard_glob))
    if not shards:
        print(f"FATAL: no shards matched {a.shard_glob!r}", file=sys.stderr)
        return 1
    mine = shards[a.task_id :: a.n_tasks]
    print(f"{len(shards)} shards total; task {a.task_id}/{a.n_tasks} owns {len(mine)}",
          flush=True)

    built = skipped = 0
    t0 = time.time()
    for i, shard in enumerate(mine):
        out = index_path_for(shard, a.index_dir or None)
        if os.path.isfile(out):
            skipped += 1
            continue
        build_offset_index(shard, out, progress_every=0)
        built += 1
        if built % 5 == 0:
            print(f"  {i+1}/{len(mine)} done, "
                  f"{(time.time()-t0)/built:.1f}s/shard", flush=True)
    print(f"DONE built={built} skipped={skipped} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
