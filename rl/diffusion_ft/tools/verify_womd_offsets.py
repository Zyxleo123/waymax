#!/usr/bin/env python3
"""Check that a WOMD tf_example copy is byte-compatible with the cached offset indexes.

Scene ids in the split JSONs resolve through ``<index_dir>/<shard basename>.offsets.npy``,
keyed by shard *basename* only, so a different copy (another WOMD version, a truncated
download) is read silently wrong. Per shard this checks that the index exists, the first
offset is 0, sampled records' headers land exactly on the next offset, and the last record
ends exactly at the file size; with ``--sizes`` the file size must also match the manifest
recorded on the source cluster (``env/womd_sizes.tsv``).

    python -m rl.diffusion_ft.tools.verify_womd_offsets \\
        --glob '/new/womd/tf_example/training/training_tfexample.tfrecord-*' \\
        --index_dir /new/scratch/waymax_rs/womd_offsets --sizes env/womd_sizes.tsv --expect 1000

Exit status 0 only if every matched shard passes (and ``--expect`` shards matched).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from rl.tfrecord_fast import _RECORD_HEADER, _RECORD_META_SIZE


def _record_end(f, offset: int) -> int:
    f.seek(offset)
    header = f.read(_RECORD_HEADER.size)
    if len(header) < _RECORD_HEADER.size:
        return -1
    (length,) = _RECORD_HEADER.unpack(header)
    return offset + _RECORD_META_SIZE + length


def check_shard(path: str, index_dir: str, sizes: dict[str, int], n_samples: int) -> str | None:
    """None if the shard matches its index, else the reason it does not."""
    name = os.path.basename(path)
    size = os.path.getsize(path)
    if sizes and name in sizes and sizes[name] != size:
        return f"size {size} != manifest {sizes[name]}"
    idx = os.path.join(index_dir, name + ".offsets.npy")
    if not os.path.isfile(idx):
        return f"no index {idx}"
    offs = np.load(idx, mmap_mode="r")
    if len(offs) == 0 or int(offs[0]) != 0:
        return "first offset is not 0"
    picks = sorted({int(i) for i in np.linspace(0, len(offs) - 2, n_samples)} if len(offs) > 1 else set())
    with open(path, "rb") as f:
        for i in picks:
            if _record_end(f, int(offs[i])) != int(offs[i + 1]):
                return f"record {i} does not end at offset {i + 1}"
        if _record_end(f, int(offs[-1])) != size:
            return f"last record ({len(offs) - 1}) does not end at file size {size}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", required=True, help="shard glob of the copy to check")
    p.add_argument("--index_dir", required=True, help="dir holding <basename>.offsets.npy")
    p.add_argument("--sizes", default="", help="basename<TAB>bytes manifest (env/womd_sizes.tsv)")
    p.add_argument("--expect", type=int, default=0, help="fail unless exactly this many shards match")
    p.add_argument("--samples", type=int, default=8, help="interior records checked per shard")
    args = p.parse_args()

    sizes: dict[str, int] = {}
    if args.sizes:
        with open(args.sizes) as fh:
            for line in fh:
                if line.strip() and not line.startswith("#"):
                    name, n = line.split("\t")
                    sizes[name] = int(n)
    paths = sorted(glob.glob(args.glob))
    print(f"{len(paths)} shards match {args.glob}; index_dir={args.index_dir}; manifest={len(sizes)} entries")
    bad = 0
    for path in paths:
        why = check_shard(path, args.index_dir, sizes, args.samples)
        if why:
            bad += 1
            print(f"FAIL {os.path.basename(path)}: {why}")
    unlisted = sum(os.path.basename(p) not in sizes for p in paths) if sizes else 0
    if unlisted:
        print(f"note: {unlisted} matched shards are not in the manifest (size not compared)")
    count_ok = not args.expect or len(paths) == args.expect
    if not count_ok:
        print(f"FAIL: expected {args.expect} shards, matched {len(paths)}")
    ok = bad == 0 and count_ok and bool(paths)
    print(f"{'PASS' if ok else 'FAIL'}: {len(paths) - bad}/{len(paths)} shards consistent")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
