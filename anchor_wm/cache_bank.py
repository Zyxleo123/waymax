"""Cache AnchorWM inputs + per-candidate geometry for a recorded Diffusion-ES bank (e.g. the v3
discriminator's fail1500_plus_womd_train_s02_11 set), for scorer fine-tuning.

For each `<batch>.npz` (5 scenes x 8 replans = 40 groups x 320 candidates) this rebuilds the scenes once
from their tfrecord (es_discriminator.analyze_es_decisions.load_batch_state, which checks scenario ids)
and writes `<out_dir>/<batch>.anchor_cache.npz`, row-aligned with the batch's groups:

  A  model inputs at each replan (data.build_arrays on the replan state = original log with the executed
     ego, as wm_cache / eval_bank): agent_history, map, lights, current poses, types, ego size / velocity,
     road-edge label points + scorer tokens; goal_xy from scen_goal_world. Floats stored as float16.
  B  per candidate and world step t+1 .. t+S (0.1 s, S = 50; log-replay objects of the original log,
     2-D, candidate boxes from cand_traj x / y / yaw):
       b_gap      [G,K,S] float16  smallest separating-axis gap to any valid object (m; <= 0 = overlap)
       b_type     [G,K,S] int8     type of that closest object (Waymax ObjectTypeIds)
       b_offroad  [G,K,S] float16  largest signed road-edge distance of the 4 corners (m; > 0 = offroad)
     Steps past the group's label horizon are filled with +50 / 0 / -50.
  C  lane centerline points (lane types 1-3) in a forward box: lane_xy, lane_dir, lane_id, lane_valid.
  cand_rows [G,K] int32  rows of the batch file's cand_* arrays (and of its label files) per group.

Labels themselves (Waymax first overlap / offroad steps, goal gain, progress, ...) stay in the original
label files; `bank_data.BankGroups` joins them with this cache. Resumable (skips finished files) and
shardable (--shard i --num_shards n over the sorted batch files).

  python -m anchor_wm.cache_bank --bank_dir .../training_sets/<set>/es_candidates \
      --tfrecord_dir /zfsauton/datasets/womd/tf_example/training --out_dir .../anchor_cache/<set> \
      --shard 0 --num_shards 22 [--check]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import struct
import time

import numpy as np
import torch

from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import build_arrays, future_boxes, payload_from_sim_state, rotate_xy_np
from anchor_wm.geometry import box_corners, edge_signed_distance, sat_gap

STEPS = 50
LANE_TYPES = (1, 2, 3)
LANE_BOX = (-30.0, 150.0, 60.0)
NUM_LANE_POINTS = 1024
A_KEYS = ("agent_history", "agent_history_valid", "agent_valid", "map_features", "map_valid",
          "traffic_light_features", "traffic_light_valid", "current_local_xy", "current_yaw_local",
          "object_types", "is_sdc", "ego_idx", "ego_lw", "ego_vel",
          "edge_xy", "edge_dir", "edge_prev_dir", "edge_same_prev", "edge_valid",
          "tok_edge_xy", "tok_edge_dir", "tok_edge_valid")
FRAME_TOL_M, FRAME_TOL_RAD = 1e-2, 1e-3


def batch_files(bank_dir: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(bank_dir, "*.npz")) if re.search(r"\.scen_[\d_]+\.npz$", f))


def tfrecord_of(batch_file: str) -> str:
    """'<tfrecord>.scen_000_001_....npz' -> '<tfrecord>' (every batch holds scenes of one tfrecord)."""
    return re.sub(r"\.scen_[\d_]+\.npz$", "", os.path.basename(batch_file))


def plan_units(files: list[str], done: set[str], files_per_unit: int) -> list[list[str]]:
    """Remaining batch files grouped by source tfrecord and split into units of <= files_per_unit
    (deterministic order), so each unit needs one tfrecord pass."""
    groups: dict[str, list[str]] = {}
    for f in sorted(files):
        if os.path.basename(f) not in done:
            groups.setdefault(tfrecord_of(f), []).append(f)
    units = []
    for tf in sorted(groups):
        g = groups[tf]
        n = max(1, math.ceil(len(g) / files_per_unit))
        size = math.ceil(len(g) / n)
        units += [g[i:i + size] for i in range(0, len(g), size)]
    return units


def read_records(path: str, needed: set[int]) -> dict[int, bytes]:
    """Raw serialized records at the given indices of an uncompressed TFRecord file, reading only each
    record's 12-byte header and seeking past the data of records that are not needed
    (record = uint64 length, uint32 crc, data, uint32 crc). CRCs are not checked; the parsed scenario
    ids are checked by the caller."""
    out: dict[int, bytes] = {}
    with open(path, "rb") as fh:
        idx = 0
        while len(out) < len(needed):
            hdr = fh.read(12)
            if len(hdr) < 12:
                break
            (length,) = struct.unpack("<Q", hdr[:8])
            if idx in needed:
                data = fh.read(length)
                if len(data) != length:
                    raise ValueError(f"{path}: truncated record {idx}")
                out[idx] = data
                fh.seek(4, 1)
            else:
                fh.seek(length + 4, 1)
            idx += 1
    return out


def load_states_one_pass(batches: list[dict], tfrecord_dir: str) -> list:
    """Simulator states for several batch files of one tfrecord from a single scan of it. Same Waymax
    config, parsing and scenario-id check as es_discriminator.analyze_es_decisions.load_batch_state."""
    import dataclasses

    import tensorflow as tf
    from waymax import config as waymax_config
    from waymax import dataloader
    from waymax.dataloader import womd_factories

    path = os.path.join(tfrecord_dir, str(batches[0]["scen_tfrecord"][0]))
    needed = {int(i) for d in batches for i in d["scen_index"]}
    records = read_records(path, needed)          # header reads + seeks instead of a full TFRecordDataset scan
    if len(records) != len(needed):
        raise ValueError(f"{path}: scenario indices {sorted(needed - set(records))} not found")
    states = []
    for d in batches:
        idx = [int(i) for i in d["scen_index"]]
        if str(d["scen_tfrecord"][0]) != os.path.basename(path) or idx != sorted(set(idx)):
            raise ValueError(f"batch file scenes not from {path} in ascending order: {idx}")
        cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=path, max_num_objects=128,
                                  batch_dims=(len(idx),), shuffle_seed=0)
        serialized = tf.constant([records[i] for i in idx], dtype=tf.string)
        processed = dataloader.preprocess_serialized_womd_data(serialized, cfg)
        st = womd_factories.simulator_state_from_womd_dict(processed, include_sdc_paths=cfg.include_sdc_paths)
        ids = [tf.train.Example.FromString(records[i]).features.feature["scenario/id"].bytes_list.value[0].decode()
               for i in idx]
        if ids != [str(s) for s in d["scen_id"]]:
            raise RuntimeError(f"scenario id mismatch: {ids} vs {list(d['scen_id'])}")
        states.append(st)
    return states


def lane_points(payload: dict, w: int, origin: np.ndarray, heading: float) -> dict:
    keep = (np.asarray(payload["roadgraph_points_valid"][w]).astype(bool)
            & np.isin(payload["roadgraph_points_types"][w], LANE_TYPES))
    idx = np.flatnonzero(keep)
    out = {"lane_xy": np.zeros((NUM_LANE_POINTS, 2), np.float16), "lane_dir": np.zeros((NUM_LANE_POINTS, 2), np.float16),
           "lane_id": np.full((NUM_LANE_POINTS,), -1, np.int32), "lane_valid": np.zeros((NUM_LANE_POINTS,), bool)}
    if len(idx) == 0:
        return out
    xy = np.stack([payload["roadgraph_points_x"][w][idx], payload["roadgraph_points_y"][w][idx]], -1).astype(np.float64)
    local = rotate_xy_np(xy - origin, heading)
    back, ahead, half = LANE_BOX
    sel = np.flatnonzero((local[:, 0] >= back) & (local[:, 0] <= ahead) & (np.abs(local[:, 1]) <= half))
    if len(sel) > NUM_LANE_POINTS:
        sel = sel[::math.ceil(len(sel) / NUM_LANE_POINTS)][:NUM_LANE_POINTS]
    k = len(sel)
    dirs = np.stack([payload["roadgraph_points_dir_x"][w][idx], payload["roadgraph_points_dir_y"][w][idx]], -1)
    d = rotate_xy_np(dirs[sel], heading)
    out["lane_xy"][:k] = local[sel]
    out["lane_dir"][:k] = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
    out["lane_id"][:k] = np.asarray(payload["roadgraph_points_ids"][w])[idx][sel]
    out["lane_valid"][:k] = True
    return out


OBJ_MARGIN_M = 20.0     # objects never within this distance of the candidates' extent are skipped
EDGE_MARGIN_M = 30.0    # road-edge points farther than this from the candidates' extent are skipped


@torch.no_grad()
def step_geometry(cand: np.ndarray, ego_lw: np.ndarray, fut: dict, edges: dict, horizon: int,
                  obj_chunk: int = 32, prefilter: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cand [K,S,3] (x, y, yaw; ego frame), objects fut [1,N,S,...] -> gap, closest type, offroad [K,S].

    With `prefilter`, only objects valid within the horizon that come within OBJ_MARGIN_M of the
    candidates' bounding box (over the horizon), and road-edge points within EDGE_MARGIN_M of it, are
    tested. Overlap flags and any gap below about OBJ_MARGIN_M - (half box diagonals) are unchanged
    (a skipped object is at least that far from every candidate); the nearest road edge is unchanged
    wherever one lies within EDGE_MARGIN_M."""
    k, s, _ = cand.shape
    h = max(1, min(int(horizon), s))
    if prefilter:
        pts = cand[:, :h, :2].reshape(-1, 2)
        lo, hi = pts.min(0), pts.max(0)
        o_pos, o_val = fut["xy"][0][:, :h], fut["valid"][0][:, :h]
        inside = ((o_pos >= lo - OBJ_MARGIN_M) & (o_pos <= hi + OBJ_MARGIN_M)).all(-1) & o_val
        keep = inside.any(1)
        fut = {key: (v[:, keep] if key != "type" else v[keep]) for key, v in fut.items()}
        exy = edges["edge_xy"].astype(np.float32)
        near = ((exy >= lo - EDGE_MARGIN_M) & (exy <= hi + EDGE_MARGIN_M)).all(-1) & edges["edge_valid"].astype(bool)
        if near.any():                     # no edge nearby: keep them all, so the nearest one is still found
            edges = {key: v[near] for key, v in edges.items()}
    xy = torch.from_numpy(cand[..., :2].astype(np.float32))
    yaw = torch.from_numpy(cand[..., 2].astype(np.float32))
    ec = box_corners(xy, yaw, torch.from_numpy(np.asarray(ego_lw, np.float32)).expand(k, s, 2))      # [K,S,4,2]
    o_xy = torch.from_numpy(fut["xy"][0])                                                             # [N,S,2]
    o_yaw = torch.from_numpy(fut["yaw"][0])
    o_lw = torch.from_numpy(fut["lw"][0])
    o_ok = torch.from_numpy(fut["valid"][0])
    o_type = torch.from_numpy(np.asarray(fut["type"]).astype(np.int64))                              # [N]
    best = torch.full((k, s), float("inf"))
    best_type = torch.zeros((k, s), dtype=torch.long)
    for n0 in range(0, o_xy.shape[0], obj_chunk):
        sl = slice(n0, n0 + obj_chunk)
        oc = box_corners(o_xy[sl], o_yaw[sl], o_lw[sl])                                               # [n,S,4,2]
        gap = sat_gap(ec[:, None], yaw[:, None], oc[None], o_yaw[None, sl])                          # [K,n,S]
        gap = gap.masked_fill(~o_ok[None, sl], float("inf"))
        g_min, g_arg = gap.min(1)
        better = g_min < best
        best = torch.where(better, g_min, best)
        best_type = torch.where(better, o_type[sl][g_arg], best_type)
    e = {key: torch.from_numpy(np.ascontiguousarray(v[None])) for key, v in edges.items()}
    e = {key: (v.float() if v.is_floating_point() else v) for key, v in e.items()}
    sd = edge_signed_distance(ec.reshape(1, -1, 2), e).reshape(k, s, 4).amax(-1)
    in_h = torch.arange(1, s + 1) <= horizon
    best = torch.where(torch.isfinite(best), best, torch.full_like(best, 50.0)).clamp(max=50.0)
    best = best.masked_fill(~in_h[None], 50.0)
    best_type = best_type.masked_fill(~in_h[None] | (best >= 50.0), 0)
    sd = sd.masked_fill(~in_h[None], -50.0).clamp(-50.0, 50.0)
    return best.numpy().astype(np.float16), best_type.numpy().astype(np.int8), sd.numpy().astype(np.float16)


def process_file(path: str, tfrecord_dir: str, cfg: AnchorWMConfig, check: bool, state=None,
                 d: dict | None = None, prefilter: bool = True) -> dict:
    """`state` (+ the loaded batch `d`): a simulator state already built by load_states_one_pass."""
    import jax
    from es_discriminator.analyze_es_decisions import load_batch_state
    from es_discriminator.real_es.check_alignment import replaced_state

    d = d if d is not None else dict(np.load(path))
    lab = dict(np.load(path[:-4] + ".labels_waymax.npz"))
    st = jax.device_get(state if state is not None else load_batch_state(d, tfrecord_dir))
    log_payload = payload_from_sim_state(st)                              # other agents: log replay
    model_payload = payload_from_sim_state(replaced_state(st, d))        # executed ego history
    g_n = len(d["group_t"])
    order = np.argsort(d["cand_group"], kind="stable")
    counts = np.bincount(d["cand_group"], minlength=g_n)
    k = int(counts[0])
    if not np.all(counts == k):
        raise ValueError(f"{path}: groups with != {k} candidates")
    cand_rows = order.reshape(g_n, k).astype(np.int32)
    out: dict[str, list] = {}
    frame = {"origin_err_m": 0.0, "heading_err_rad": 0.0}
    agree = {"overlap_tp": 0, "overlap_fp": 0, "overlap_fn": 0, "overlap_tn": 0}
    n_obj = min(cfg.max_agents, log_payload["log_trajectory_x"].shape[1])
    by_t: dict[int, np.ndarray] = {}
    for t in np.unique(d["group_t"]):
        by_t[int(t)] = build_arrays(model_payload, anchor_step=int(t), cfg=cfg, with_targets=False)
    for g in range(g_n):
        t, w = int(d["group_t"][g]), int(d["group_scen"][g])
        arr = by_t[t]
        origin, heading = d["group_origin"][g].astype(np.float64), float(d["group_heading"][g])
        frame["origin_err_m"] = max(frame["origin_err_m"], float(np.abs(arr["origin_xy"][w] - origin).max()))
        dh = (arr["origin_yaw"][w] - heading + np.pi) % (2 * np.pi) - np.pi
        frame["heading_err_rad"] = max(frame["heading_err_rad"], float(abs(dh)))
        rec = {key: arr[key][w] for key in A_KEYS}
        goal = rotate_xy_np(d["scen_goal_world"][w].astype(np.float64) - origin, heading)
        rec["goal_xy"] = goal.astype(np.float32)
        rec.update(lane_points(log_payload, w, origin, heading))
        # B: geometry against the original log (log replay), in the group frame.
        pw = {key: v[w:w + 1] for key, v in log_payload.items()}
        fut = future_boxes(pw, t, origin[None], np.array([heading]), STEPS, 1, n_obj)
        is_sdc = pw["object_metadata_is_sdc"][0, :n_obj].astype(bool)
        fut["valid"] = fut["valid"] & ~is_sdc[None, :, None]
        fut["type"] = pw["object_metadata_object_types"][0, :n_obj]
        # Road edges were built in the replan frame, which equals the group frame (checked below).
        edges = {key: arr[key][w] for key in ("edge_xy", "edge_dir", "edge_prev_dir", "edge_same_prev", "edge_valid")}
        cand =d["cand_traj"][cand_rows[g]][:, :STEPS].astype(np.float32)                       # world steps t+1..t+S
        h = int(lab["label_horizon"][g])
        gap, typ, off = step_geometry(np.nan_to_num(cand), d["group_ego_state"][g, 5:7], fut, edges, h,
                                      prefilter=prefilter)
        rec.update(b_gap=gap, b_type=typ, b_offroad=off)
        if check:
            ours = (gap.astype(np.float32) <= 0).any(1)
            ref = lab["gt_overlap_step"][cand_rows[g]] > 0
            agree["overlap_tp"] += int((ours & ref).sum())
            agree["overlap_fp"] += int((ours & ~ref).sum())
            agree["overlap_fn"] += int((~ours & ref).sum())
            agree["overlap_tn"] += int((~ours & ~ref).sum())
        for key, v in rec.items():
            out.setdefault(key, []).append(v)
    if frame["origin_err_m"] > FRAME_TOL_M or frame["heading_err_rad"] > FRAME_TOL_RAD:
        raise RuntimeError(f"{path}: replan frame != recorded group frame {frame}")
    arrays = {key: np.stack(v) for key, v in out.items()}
    for key, v in arrays.items():
        if v.dtype == np.float32 or v.dtype == np.float64:
            arrays[key] = v.astype(np.float16)
    arrays["goal_xy"] = arrays["goal_xy"].astype(np.float32)
    arrays["cand_rows"] = cand_rows
    arrays["group_t"] = d["group_t"].astype(np.int16)
    arrays["group_scen"] = d["group_scen"].astype(np.int16)
    return {"arrays": arrays, "frame": frame, "agree": agree if check else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank_dir", required=True)
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--check", action="store_true", help="report overlap agreement with labels_waymax")
    p.add_argument("--by_tfrecord", action="store_true",
                   help="group the remaining files by source tfrecord; one tfrecord pass per unit of files")
    p.add_argument("--files_per_unit", type=int, default=24)
    p.add_argument("--max_units", type=int, default=0)
    p.add_argument("--only_files", default="", help="comma-separated batch basenames (validation runs)")
    p.add_argument("--base_shards", default="",
                   help="restrict to the files of these shards of an earlier split, e.g. '12,13' with --base_num_shards 22")
    p.add_argument("--base_num_shards", type=int, default=0)
    p.add_argument("--no_prefilter", action="store_true", help="test every object / road-edge point")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = AnchorWMConfig()
    os.makedirs(args.out_dir, exist_ok=True)
    all_files = batch_files(args.bank_dir)
    if args.only_files:
        keep = set(args.only_files.split(","))
        all_files = [f for f in all_files if os.path.basename(f) in keep]
    if args.base_shards:
        if not args.base_num_shards:
            raise SystemExit("--base_shards needs --base_num_shards")
        own = {int(x) for x in args.base_shards.split(",")}
        all_files = [f for i, f in enumerate(all_files) if i % args.base_num_shards in own]
    target_of = lambda f: os.path.join(args.out_dir, os.path.basename(f)[:-4] + ".anchor_cache.npz")  # noqa: E731
    done = skipped = 0
    t0 = time.time()
    agree_tot: dict[str, int] = {}

    def finish(f: str, res: dict, seconds: float, extra: dict | None = None) -> None:
        nonlocal done
        target = target_of(f)
        tmp = target + ".tmp.npz"
        np.savez_compressed(tmp, **res["arrays"])
        os.replace(tmp, target)
        done += 1
        if res["agree"]:
            for key, v in res["agree"].items():
                agree_tot[key] = agree_tot.get(key, 0) + v
        print(json.dumps({"file": os.path.basename(f), "seconds": round(seconds, 1),
                          "mb": round(os.path.getsize(target) / 2 ** 20, 2), **res["frame"], **(extra or {}),
                          **({"agree": res["agree"]} if res["agree"] else {})}), flush=True)

    if args.by_tfrecord:
        finished = {os.path.basename(f) for f in all_files if os.path.exists(target_of(f))}
        units = plan_units(all_files, finished, args.files_per_unit)[args.shard::args.num_shards]
        if args.max_units:
            units = units[:args.max_units]
        files = [f for u in units for f in u]
        skipped = len(finished)
        for unit in units:
            ds = [dict(np.load(f)) for f in unit]
            t1 = time.time()
            states = load_states_one_pass(ds, args.tfrecord_dir)
            scan = time.time() - t1
            print(json.dumps({"unit": tfrecord_of(unit[0]), "files": len(unit), "scan_seconds": round(scan, 1)}), flush=True)
            for f, d, st in zip(unit, ds, states):
                if os.path.exists(target_of(f)):                                  # finished elsewhere meanwhile
                    continue
                t2 = time.time()
                finish(f, process_file(f, args.tfrecord_dir, cfg, args.check, state=st, d=d,
                                    prefilter=not args.no_prefilter), time.time() - t2)
    else:
        files = all_files[args.shard::args.num_shards]
        if args.max_files:
            files = files[:args.max_files]
        for f in files:
            if os.path.exists(target_of(f)):
                skipped += 1
                continue
            t1 = time.time()
            finish(f, process_file(f, args.tfrecord_dir, cfg, args.check, prefilter=not args.no_prefilter),
                   time.time() - t1)
    summary = {"shard": args.shard, "num_shards": args.num_shards, "files": len(files), "done": done,
               "skipped": skipped, "minutes": round((time.time() - t0) / 60, 1), "by_tfrecord": args.by_tfrecord}
    if agree_tot:
        n = sum(agree_tot.values())
        summary["overlap_agreement"] = (agree_tot["overlap_tp"] + agree_tot["overlap_tn"]) / max(n, 1)
        summary["overlap_recall"] = agree_tot["overlap_tp"] / max(agree_tot["overlap_tp"] + agree_tot["overlap_fn"], 1)
        summary["overlap_counts"] = agree_tot
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
