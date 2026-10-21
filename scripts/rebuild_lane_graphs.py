"""Rebuild the ES runner's lane-graph shard zips on Babel (the autonlab originals were not copied).

The originals (mineuih/waymax_rs/lane_graphs/<tf_example shard>.lanegraph.zip) were written by
scripts/extract_roadgraph.py (mini-hong/waymax_rs@main; its graph code is unchanged since April):
one LaneGraph per scene, built from the WOMD v1.3.1 *Scenario* protos and matched to tf_example
records by scenario id. This script reuses that file's functions unchanged and only changes how
much is read:

  targets  which records need a graph: every record of validation shards 0-9 (val10), the fail1500
           targets (training shards, read at their byte offsets) and the valfail1568 targets
           (validation, from the compact copy) -> <out>/targets.json {shard: {record index: id}}
  scan     array task k of K reads every K-th Scenario file, gets each record's scenario_id from
           the top-level field walk (no full parse), and builds a graph only for wanted ids
           -> <out>/parts/<scenario file>.zip ({id}.npz) + .done marker (requeue-safe)
  merge    one <tf_example shard>.lanegraph.zip per target shard, same layout as the originals
           (manifest.json + {index:06d}_{id}.npz); fails if any wanted id was not found

Differences from the originals: a zip holds only the targeted records (manifest "partial": true,
the rest listed in missing_indices), which is all the runner reads (by record index).

Needs the Waymo protos on PYTHONPATH (compiled from waymo-open-dataset v1.5.2's .proto files; the
fields used here are unchanged since v1.0): /data/user_data/yixiz/tools/wod_protos/gen.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import zipfile
from pathlib import Path

import matplotlib  # before tensorflow: TF loads the system libstdc++, which matplotlib's extension rejects

matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402,F401  (extract_roadgraph imports it)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_roadgraph as er  # noqa: E402
from waymo_open_dataset.protos import scenario_pb2  # noqa: E402

NEWROOT = "/data/user_data/yixiz/autonlab"
YX = f"{NEWROOT}/zfsauton/scratch/yixiz/waymax_rs"
TFEX = f"{NEWROOT}/zfsauton/datasets/womd/tf_example"
SCEN = "/data/datasets/waymo/waymo-open-dataset-v1.3.1/scenario"


# ---------------------------------------------------------------- TFRecord framing
def iter_records(path: str):
    """(index, bytes) for every record (length-prefixed framing; CRCs not checked)."""
    with open(path, "rb") as f:
        i = 0
        while True:
            head = f.read(12)
            if not head:
                return
            (n,) = struct.unpack("<Q", head[:8])
            data = f.read(n)
            f.read(4)
            if len(data) != n:
                raise IOError(f"truncated record {i} in {path}")
            yield i, data
            i += 1


def read_record_at(f, offset: int) -> bytes:
    f.seek(int(offset))
    (n,) = struct.unpack("<Q", f.read(8))
    f.read(4)
    return f.read(n)


# ---------------------------------------------------------------- Scenario id without a full parse
def _varint(buf: bytes, i: int):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, i
        shift += 7


def scenario_id_fast(buf: bytes) -> str:
    """Scenario.scenario_id (field 5) by walking the top-level fields only."""
    i, n = 0, len(buf)
    while i < n:
        tag, i = _varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            _, i = _varint(buf, i)
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        elif wire == 2:
            ln, i = _varint(buf, i)
            if field == 5:
                return buf[i:i + ln].decode("utf-8")
            i += ln
        else:
            raise ValueError(f"unexpected wire type {wire}")
    raise KeyError("no scenario_id")


# ---------------------------------------------------------------- modes
def cmd_targets(a) -> None:
    targets: dict[str, dict[int, str]] = {}
    nrec: dict[str, int] = {}

    def add(shard: str, idx: int, sid: str) -> None:
        old = targets.setdefault(shard, {}).get(idx)
        if old is not None and old != sid:
            raise RuntimeError(f"{shard}[{idx}]: id {old} vs {sid}")
        targets[shard][idx] = sid

    for k in range(10):  # val10: every record
        name = f"validation_tfexample.tfrecord-{k:05d}-of-00150"
        n = 0
        for i, raw in iter_records(f"{TFEX}/validation/{name}"):
            add(name, i, er.get_scenario_id_from_tfexample_bytes(raw))
            n += 1
        nrec[name] = n
        print(f"[targets] {name}: {n} records (all)", flush=True)

    fail = json.load(open(f"{YX}/es_posthoc/fail1500_subset_g2/target_scenarios.json"))
    for name, idxs in sorted(fail.items()):
        offs = __import__("numpy").load(f"{YX}/womd_offsets/{name}.offsets.npy")
        nrec[name] = len(offs)
        with open(f"{TFEX}/training/{name}", "rb") as f:
            for i in idxs:
                add(name, int(i), er.get_scenario_id_from_tfexample_bytes(read_record_at(f, offs[int(i)])))
    print(f"[targets] fail1500: {sum(map(len, fail.values()))} records in {len(fail)} shards", flush=True)

    comp = f"{YX}/womd_compact/valfail1568"
    vf = json.load(open(f"{comp}/target_scenarios.json"))
    for name, idxs in sorted(vf.items()):
        want = {int(i) for i in idxs}
        n = 0
        for i, raw in iter_records(f"{comp}/validation/{name}"):
            n += 1
            if i in want:
                add(name, i, er.get_scenario_id_from_tfexample_bytes(raw))
        if name in nrec and nrec[name] != n:
            raise RuntimeError(f"{name}: {nrec[name]} records vs compact {n}")
        nrec[name] = n
        got = {i for i in want if i in targets.get(name, {})}
        if got != want:
            raise RuntimeError(f"{name}: compact copy lacks {sorted(want - got)}")
    print(f"[targets] valfail1568: {sum(map(len, vf.values()))} records in {len(vf)} shards", flush=True)

    out = {"num_records": nrec, "targets": {s: {str(i): sid for i, sid in sorted(d.items())} for s, d in sorted(targets.items())}}
    ids = [sid for d in targets.values() for sid in d.values()]
    print(f"[targets] {len(targets)} shards, {len(ids)} records, {len(set(ids))} unique ids", flush=True)
    os.makedirs(a.out, exist_ok=True)
    tmp = f"{a.out}/targets.json.tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, f"{a.out}/targets.json")


def scenario_files() -> list[str]:
    return sorted(glob.glob(f"{SCEN}/training/training.tfrecord-*")) + sorted(glob.glob(f"{SCEN}/validation/validation.tfrecord-*"))


def cmd_scan(a) -> None:
    tg = json.load(open(f"{a.out}/targets.json"))["targets"]
    want = {sid for d in tg.values() for sid in d.values()}
    files = scenario_files()
    if len(files) != 1150:
        raise RuntimeError(f"expected 1150 scenario files, found {len(files)}")
    mine = files[a.part::a.nparts]
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    print(f"[scan] part {a.part}/{a.nparts}: {len(mine)} files, {len(want)} wanted ids", flush=True)
    for path in mine:
        base = os.path.basename(path)
        zp, done = f"{a.out}/parts/{base}.zip", f"{a.out}/parts/{base}.done"
        if os.path.exists(done):
            continue
        n = hit = 0
        with zipfile.ZipFile(zp + ".tmp", "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for _, raw in iter_records(path):
                n += 1
                sid = scenario_id_fast(raw)
                if sid not in want:
                    continue
                sc = scenario_pb2.Scenario()
                sc.ParseFromString(raw)
                if sc.scenario_id != sid:
                    raise RuntimeError(f"fast id {sid} != parsed {sc.scenario_id} in {base}")
                zf.writestr(f"{sid}.npz", er.lane_graph_to_npz_bytes(er.build_lane_graph_from_scenario(sc)))
                hit += 1
        os.replace(zp + ".tmp", zp)
        with open(done, "w") as f:
            json.dump({"records": n, "matched": hit}, f)
        print(f"[scan] {base}: {n} records, {hit} matched", flush=True)


def cmd_selftest(a) -> None:
    """scenario_id_fast agrees with the full parse on the first records of one file of each split."""
    for path in (scenario_files()[0], scenario_files()[-1]):
        for i, raw in iter_records(path):
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(raw)
            assert scenario_id_fast(raw) == sc.scenario_id, (path, i)
            if i == 49:
                break
    print("[selftest] scenario_id_fast == full parse on 2 x 50 records")


def cmd_merge(a) -> None:
    meta = json.load(open(f"{a.out}/targets.json"))
    tg, nrec = meta["targets"], meta["num_records"]
    dones = glob.glob(f"{a.out}/parts/*.done")
    if len(dones) != 1150:
        raise RuntimeError(f"only {len(dones)}/1150 scenario files scanned")
    where: dict[str, str] = {}
    for zp in sorted(glob.glob(f"{a.out}/parts/*.zip")):
        for name in zipfile.ZipFile(zp).namelist():
            sid = name[:-4]
            if sid in where:
                raise RuntimeError(f"id {sid} in {where[sid]} and {zp}")
            where[sid] = zp
    want = {sid for d in tg.values() for sid in d.values()}
    missing = sorted(want - set(where))
    print(f"[merge] {len(want)} wanted ids, {len(where)} built, {len(missing)} unresolved", flush=True)
    if missing:
        raise RuntimeError(f"unresolved ids (first 10): {missing[:10]}")
    os.makedirs(a.dest, exist_ok=True)
    handles: dict[str, zipfile.ZipFile] = {}
    for shard, d in sorted(tg.items()):
        n = int(nrec[shard])
        ids = [None] * n
        for i, sid in d.items():
            ids[int(i)] = sid
        manifest = {
            "schema_version": "lanegraph_shard_v1",
            "tfexample_file": shard,
            "num_records": n,
            "scenario_ids": ids,
            "missing_indices": [i for i in range(n) if ids[i] is None],
            "partial": True,
            "built_by": "scripts/rebuild_lane_graphs.py on Babel 2026-10-19 (extract_roadgraph.py graph code, "
                        "WOMD v1.3.1 scenario protos); only targeted records (val10 all, fail1500, valfail1568)",
        }
        out = f"{a.dest}/{shard}.lanegraph.zip"
        with zipfile.ZipFile(out + ".tmp", "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for i, sid in sorted(((int(i), s) for i, s in d.items())):
                src = where[sid]
                if src not in handles:
                    handles[src] = zipfile.ZipFile(src)
                zf.writestr(f"{i:06d}_{sid}.npz", handles[src].read(f"{sid}.npz"))
        os.replace(out + ".tmp", out)
    print(f"[merge] wrote {len(tg)} shard zips to {a.dest}", flush=True)


def cmd_check(a) -> None:
    """Every targeted record loads through the runner's own reader (lane_graph_utils.LaneGraphShardStore)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lane_graph.lane_graph_utils import LaneGraphShardStore

    tg = json.load(open(f"{a.out}/targets.json"))["targets"]
    n = 0
    for shard, d in sorted(tg.items()):
        store = LaneGraphShardStore(f"{a.dest}/{shard}.lanegraph.zip")
        for i in d:
            g = store.get_by_record_index(int(i))
            if g is None or len(g.nodes_xyz) == 0 or not g.lane_id_to_node_range:
                raise RuntimeError(f"{shard}[{i}]: empty or missing lane graph")
            n += 1
    print(f"[check] {n} records in {len(tg)} shards load via LaneGraphShardStore, all non-empty", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["targets", "scan", "merge", "check", "selftest"])
    p.add_argument("--out", default=f"{YX}/lane_graphs_babel_work")
    p.add_argument("--dest", default=f"{YX}/lane_graphs_babel")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--nparts", type=int, default=1)
    a = p.parse_args()
    {"targets": cmd_targets, "scan": cmd_scan, "merge": cmd_merge, "check": cmd_check, "selftest": cmd_selftest}[a.mode](a)


if __name__ == "__main__":
    main()
