#!/usr/bin/env python3
"""Index-preserving compact copy of selected WOMD scenes.

For every source shard holding a target scene, writes a shard with the SAME basename and
the SAME record count. Each target record is the original payload, byte for byte; every
other record is an empty placeholder (a valid 16-byte TFRecord frame). Anything that
addresses scenes by (shard basename, record index) sees the same scene at the same index,
at a fraction of the bytes. That covers the ES runner's ``load_scenario_state_batch_fast``
(it enumerates raw records and parses only the requested ones), ``target_scenarios.json``,
``--start_from`` result names, and the offset-index readers.

Placeholders are not scenes. Never point a reader that parses EVERY record of a shard (e.g. a
splits JSON over "all scenes in the shard") at the output directory.

    python -m rl.diffusion_ft.tools.build_compact_shards build \\
        --targets /zfsauton/scratch/mineuih/waymax_rs/exp/validation_0_150_failure_scenarios.json \\
        --src_dir /zfsauton/datasets/womd/tf_example/validation \\
        --name_fmt 'validation_tfexample.tfrecord-{:05d}-of-00150' \\
        --out_root /zfsauton/scratch/yixiz/waymax_rs/womd_compact/valfail1568 --split validation
    python -m rl.diffusion_ft.tools.build_compact_shards verify \\
        --out_root /zfsauton/scratch/yixiz/waymax_rs/womd_compact/valfail1568

Layout: ``<out_root>/<split>/<shard basenames>`` (shards only, so a ``*`` glob there is
safe), ``<out_root>/target_scenarios.json`` (basename -> indices, the ES runner's format),
``<out_root>/manifest.json``, and ``<out_root>/parts/`` (per-shard records; ``build``
skips shards already done, so a resubmit resumes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from multiprocessing import Pool
from pathlib import Path

import google_crc32c

_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")


def _masked_crc(data: bytes) -> int:
    c = google_crc32c.value(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _frame(payload: bytes) -> bytes:
    header = _U64.pack(len(payload))
    return header + _U32.pack(_masked_crc(header)) + payload + _U32.pack(_masked_crc(payload))


def load_targets(path: str, name_fmt: str) -> dict[str, list[int]]:
    """basename -> sorted scene indices, from a [{tfrecord_index, scenario_index}] list or a
    target_scenarios-style {basename: [indices]} dict."""
    raw = json.load(open(path))
    out: dict[str, set[int]] = {}
    if isinstance(raw, dict):
        for name, idx in raw.items():
            out.setdefault(name, set()).update(int(i) for i in idx)
    else:
        for row in raw:
            out.setdefault(name_fmt.format(int(row["tfrecord_index"])), set()).add(int(row["scenario_index"]))
    return {k: sorted(v) for k, v in sorted(out.items()) if v}


def _build_one(job: tuple[str, str, list[int], str]) -> dict:
    src, dst, targets, part = job
    want = set(targets)
    size = os.path.getsize(src)
    n = 0
    sha: dict[str, str] = {}
    tmp = f"{dst}.tmp.{os.getpid()}"
    with open(src, "rb") as f, open(tmp, "wb") as out:
        while f.tell() < size:
            header = f.read(8)
            (length,) = _U64.unpack(header)
            if _U32.unpack(f.read(4))[0] != _masked_crc(header):
                raise IOError(f"{src}: bad length crc at record {n}")
            if n in want:
                data = f.read(length)
                if len(data) != length or _U32.unpack(f.read(4))[0] != _masked_crc(data):
                    raise IOError(f"{src}: bad data crc at record {n}")
                sha[str(n)] = hashlib.sha256(data).hexdigest()
                out.write(_frame(data))
            else:
                f.seek(length + 4, os.SEEK_CUR)
                out.write(_frame(b""))
            n += 1
    missing = sorted(want - {int(k) for k in sha})
    if missing:
        os.remove(tmp)
        raise ValueError(f"{src}: target indices {missing} beyond its {n} records")
    os.replace(tmp, dst)
    rec = {"basename": os.path.basename(src), "src_size": size, "n_records": n, "targets": targets,
           "sha256": sha, "out_size": os.path.getsize(dst)}
    with open(f"{part}.tmp", "w") as fh:
        json.dump(rec, fh)
    os.replace(f"{part}.tmp", part)
    return rec


def build(args: argparse.Namespace) -> int:
    targets = load_targets(args.targets, args.name_fmt)
    root = Path(args.out_root)
    shard_dir, parts = root / args.split, root / "parts"
    shard_dir.mkdir(parents=True, exist_ok=True)
    parts.mkdir(exist_ok=True)
    jobs, done = [], 0
    for name, idx in targets.items():
        part = parts / f"{name}.json"
        if part.exists() and (shard_dir / name).exists() and json.load(open(part))["targets"] == idx:
            done += 1
            continue
        jobs.append((str(Path(args.src_dir) / name), str(shard_dir / name), idx, str(part)))
    print(f"{len(targets)} shards / {sum(map(len, targets.values()))} scenes; {done} already built, "
          f"{len(jobs)} to build with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_build_one, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {rec['basename']}: {len(rec['targets'])} scenes, "
                      f"{rec['n_records']} records, {rec['out_size'] / 1e6:.1f} MB", flush=True)
    recs = [json.load(open(parts / f"{name}.json")) for name in targets]
    manifest = {"targets_file": os.path.abspath(args.targets), "src_dir": os.path.abspath(args.src_dir),
                "split": args.split, "n_shards": len(recs), "n_scenes": sum(len(r["targets"]) for r in recs),
                "src_bytes": sum(r["src_size"] for r in recs), "out_bytes": sum(r["out_size"] for r in recs),
                "shards": recs}
    json.dump(manifest, open(root / "manifest.json", "w"), indent=1)
    json.dump({r["basename"]: r["targets"] for r in recs}, open(root / "target_scenarios.json", "w"), indent=1)
    print(f"wrote {manifest['n_scenes']} scenes in {manifest['n_shards']} shards: "
          f"{manifest['out_bytes'] / 1e9:.2f} GB (source shards {manifest['src_bytes'] / 1e9:.1f} GB)")
    return 0


def verify(args: argparse.Namespace) -> int:
    """Re-read every compact shard with TensorFlow (independent framing + CRC check)."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    root = Path(args.out_root)
    manifest = json.load(open(root / "manifest.json"))
    shard_dir = root / manifest["split"]
    bad = 0
    for rec in manifest["shards"]:
        path = shard_dir / rec["basename"]
        want = {int(k): v for k, v in rec["sha256"].items()}
        why, n = None, 0
        try:
            for n, s in enumerate(tf.data.TFRecordDataset([str(path)]).as_numpy_iterator()):
                if n in want:
                    if hashlib.sha256(s).hexdigest() != want[n]:
                        why = f"record {n} sha256 differs"
                        break
                    if not tf.train.Example.FromString(s).features.feature["scenario/id"].bytes_list.value:
                        why = f"record {n} has no scenario/id"
                        break
                elif s:
                    why = f"record {n} should be an empty placeholder"
                    break
            else:
                if n + 1 != rec["n_records"]:
                    why = f"{n + 1} records, expected {rec['n_records']}"
        except Exception as e:  # truncated file, bad CRC
            why = f"{type(e).__name__}: {e}"
        if why:
            bad += 1
            print(f"FAIL {rec['basename']}: {why}")
    extra = sorted(set(os.listdir(shard_dir)) - {r["basename"] for r in manifest["shards"]})
    if extra:
        bad += 1
        print(f"FAIL: {len(extra)} files in {shard_dir} not in the manifest, e.g. {extra[:3]}")
    ok = bad == 0
    print(f"{'PASS' if ok else 'FAIL'}: {manifest['n_shards'] - bad}/{manifest['n_shards']} shards, "
          f"{manifest['n_scenes']} scenes")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--targets", required=True)
    b.add_argument("--src_dir", required=True)
    b.add_argument("--name_fmt", default="validation_tfexample.tfrecord-{:05d}-of-00150")
    b.add_argument("--out_root", required=True)
    b.add_argument("--split", default="validation")
    b.add_argument("--workers", type=int, default=8)
    v = sub.add_parser("verify")
    v.add_argument("--out_root", required=True)
    args = p.parse_args()
    return build(args) if args.cmd == "build" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
