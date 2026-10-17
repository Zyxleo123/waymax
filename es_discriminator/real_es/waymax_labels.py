"""Waymax-exact safety labels for recorded Diffusion-ES candidates and the logged expert.

For every `<batch>.npz` written by `es_candidate_recorder` (with `.labels.npz` and
`.labels_v2.npz` already present) writes `<batch>.labels_waymax.npz`, row-aligned
with the `cand_*` arrays:

  gt_overlap_step      first future step (1..T, 0.1 s) at which the candidate's box
                       overlaps a valid logged object, by `waymax.utils.geometry.has_overlap`
                       (the predicate behind Waymax's OverlapMetric); 0 = none within
                       the group's label horizon
  gt_offroad_step      Waymax-exact offroad (copied from labels_v2)
  label_horizon        per group, valid future steps (copied from labels)
  anchor_overlap / anchor_offroad      ego already in violation at the replan pose
  expert_overlap_step / expert_offroad_step
                       the same checks on the logged ego future over the label horizon
                       (0 = the logged human is clean there)

Other agents follow the original log, as in log replay. Resumable and shardable.

  python -m es_discriminator.real_es.waymax_labels --data_dir .../es_candidates \
      --tfrecord_dir .../tfrecords/training --summary_out .../waymax_labels_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from waymax.utils import geometry

from es_discriminator.analyze_es_decisions import (
    EPISODE_STEPS, batch_files, first_step, load_batch_state, offroad_poses, road_edges,
)
from es_discriminator.label_bank import _to_world

SUFFIX = ".labels_waymax.npz"
K_PAD = 64


@jax.jit
def overlap_steps(ego_kt5: jax.Array, obj_ot5: jax.Array, valid_ot: jax.Array) -> jax.Array:
    """[K,T] True where box k at step t overlaps any valid object; boxes are (x, y, length, width, yaw)."""
    k, t = ego_kt5.shape[:2]
    o = obj_ot5.shape[0]
    a = jnp.broadcast_to(ego_kt5[:, None], (k, o, t, 5))
    b = jnp.broadcast_to(obj_ot5[None], (k, o, t, 5))
    return jnp.any(geometry.has_overlap(a, b) & valid_ot[None], axis=1)


def _boxes(xyyaw: np.ndarray, length, width) -> np.ndarray:
    xyyaw = np.asarray(xyyaw, dtype=np.float64)
    shape = xyyaw.shape[:-1]
    return np.stack([xyyaw[..., 0], xyyaw[..., 1], np.broadcast_to(length, shape),
                     np.broadcast_to(width, shape), xyyaw[..., 2]], -1).astype(np.float32)


def _padded_overlap(ego_kt5: np.ndarray, obj_ot5: np.ndarray, valid_ot: np.ndarray) -> np.ndarray:
    k = len(ego_kt5)
    kp = max(K_PAD, int(np.ceil(k / K_PAD) * K_PAD))
    ego = np.zeros((kp,) + ego_kt5.shape[1:], np.float32)
    ego[:k] = ego_kt5
    ego[k:, :, 2:4] = 1e-3  # degenerate far-away boxes for padding rows
    ego[k:, :, :2] = 1e7
    return np.asarray(overlap_steps(jnp.asarray(ego), jnp.asarray(obj_ot5), jnp.asarray(valid_ot)))[:k]


def label_file(path: str, tfrecord_dir: str) -> str | None:
    out = path[:-4] + SUFFIX
    if os.path.exists(out):
        return None
    d = dict(np.load(path))
    v1 = dict(np.load(path[:-4] + ".labels.npz"))
    v2 = dict(np.load(path[:-4] + ".labels_v2.npz"))
    st = load_batch_state(d, tfrecord_dir)
    log = st.log_trajectory
    lx, ly, lyaw = np.asarray(log.x), np.asarray(log.y), np.asarray(log.yaw)
    llen, lwid, lz = np.asarray(log.length), np.asarray(log.width), np.asarray(log.z)
    lvalid = np.asarray(log.valid).astype(bool)
    is_sdc = np.asarray(st.object_metadata.is_sdc).astype(bool)

    G = len(d["group_t"])
    T = d["cand_traj"].shape[1]
    n = len(d["cand_id"])
    ov_step = np.zeros((n,), np.int16)
    anchor_ov = np.zeros((G,), bool)
    exp_ov = np.zeros((G,), np.int16)
    exp_off = np.zeros((G,), np.int16)
    edges = {}
    for g in range(G):
        rows = np.flatnonzero(d["cand_group"] == g)
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        h = int(v1["label_horizon"][g])
        e = int(np.argmax(is_sdc[w]))
        origin, heading = d["group_origin"][g], float(d["group_heading"][g])
        L, W = d["group_ego_state"][g, 5], d["group_ego_state"][g, 6]

        steps = np.arange(t + 1, t + 1 + T)
        steps_c = np.minimum(steps, EPISODE_STEPS - 1)
        in_h = np.arange(1, T + 1) <= h
        others = np.ones(lx.shape[1], bool)
        others[e] = False
        obj = _boxes(np.stack([lx[w][:, steps_c], ly[w][:, steps_c], lyaw[w][:, steps_c]], -1),
                     llen[w][:, steps_c], lwid[w][:, steps_c])  # [O,T,5]
        obj_valid = lvalid[w][:, steps_c] & in_h[None] & others[:, None]

        world = _to_world(np.nan_to_num(d["cand_traj"][rows].astype(np.float64)), origin, heading)
        hits = _padded_overlap(_boxes(world, L, W), obj, obj_valid)
        ov_step[rows] = first_step(hits & in_h[None])

        now = _boxes(np.array([[[origin[0], origin[1], heading]]]), L, W)  # [1,1,5]
        o_now = _boxes(np.stack([lx[w][:, t], ly[w][:, t], lyaw[w][:, t]], -1)[:, None],
                       llen[w][:, t, None], lwid[w][:, t, None])
        anchor_ov[g] = bool(_padded_overlap(now, o_now, (lvalid[w][:, t] & others)[:, None])[0, 0])

        e_valid = lvalid[w, e, steps_c] & in_h
        exp_xyyaw = np.stack([lx[w, e, steps_c], ly[w, e, steps_c], lyaw[w, e, steps_c]], -1)
        exp_box = _boxes(exp_xyyaw, llen[w, e, steps_c], lwid[w, e, steps_c])[None]
        e_hits = _padded_overlap(exp_box, obj, obj_valid)[0] & e_valid
        exp_ov[g] = first_step(e_hits[None])[0]
        if w not in edges:
            edges[w] = road_edges(st, w)
        e_off = np.zeros((T,), bool)
        idx = np.flatnonzero(e_valid)
        if len(idx):
            poses = np.stack([exp_xyyaw[idx, 0], exp_xyyaw[idx, 1], exp_xyyaw[idx, 2],
                              llen[w, e, steps_c[idx]], lwid[w, e, steps_c[idx]]], -1)
            e_off[idx] = offroad_poses(poses, lz[w, e, steps_c[idx]], edges[w])
        exp_off[g] = first_step(e_off[None])[0]

    v1_unsafe = v1["gt_collision_step"] > 0
    wm_unsafe = ov_step > 0
    agree = np.array([n, (v1_unsafe & wm_unsafe).sum(), (v1_unsafe & ~wm_unsafe).sum(),
                      (~v1_unsafe & wm_unsafe).sum()], np.int64)
    tmp = out + ".tmp.npz"
    np.savez_compressed(
        tmp, gt_overlap_step=ov_step, gt_offroad_step=v2["gt_offroad_step"], label_horizon=v1["label_horizon"],
        anchor_overlap=anchor_ov, anchor_offroad=v2["anchor_offroad"],
        expert_overlap_step=exp_ov, expert_offroad_step=exp_off,
        overlap_vs_v1_collision=agree, label_version=np.asarray(3))
    os.replace(tmp, out)
    return out


def summarize(data_dir: str) -> dict:
    tot = np.zeros(4, np.int64)
    G = anchor_ov = anchor_off = exp_ov = exp_off = exp_any = 0
    cand_ov = cand_off = cand_any = 0
    files = [f for f in batch_files(data_dir) if os.path.exists(f[:-4] + SUFFIX)]
    for f in files:
        z = np.load(f[:-4] + SUFFIX)
        tot += z["overlap_vs_v1_collision"]
        G += len(z["label_horizon"])
        anchor_ov += int(z["anchor_overlap"].sum())
        anchor_off += int(z["anchor_offroad"].sum())
        exp_ov += int((z["expert_overlap_step"] > 0).sum())
        exp_off += int((z["expert_offroad_step"] > 0).sum())
        exp_any += int(((z["expert_overlap_step"] > 0) | (z["expert_offroad_step"] > 0)).sum())
        cand_ov += int((z["gt_overlap_step"] > 0).sum())
        cand_off += int((z["gt_offroad_step"] > 0).sum())
        cand_any += int(((z["gt_overlap_step"] > 0) | (z["gt_offroad_step"] > 0)).sum())
    n = max(int(tot[0]), 1)
    return {
        "files": len(files), "groups": G, "candidates": int(tot[0]),
        "candidate_rates": {"overlap": cand_ov / n, "offroad": cand_off / n, "unsafe": cand_any / n},
        "overlap_vs_v1_polygon_collision": {
            "both": int(tot[1]), "v1_only": int(tot[2]), "waymax_only": int(tot[3]),
            "agreement": 1.0 - float(tot[2] + tot[3]) / n},
        "groups_anchor_overlap": anchor_ov, "groups_anchor_offroad": anchor_off,
        "groups_expert_overlap": exp_ov, "groups_expert_offroad": exp_off, "groups_expert_unsafe": exp_any,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_files", type=int, default=0, help="only the first N batch files (smoke tests)")
    p.add_argument("--summary_out", default="")
    a = p.parse_args()
    files = batch_files(a.data_dir)
    if a.max_files:
        files = files[:a.max_files]
    files = files[a.shard::a.num_shards]
    t0 = time.time()
    done = 0
    for i, f in enumerate(files):
        if label_file(f, a.tfrecord_dir):
            done += 1
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"waymax labels {i + 1}/{len(files)} ({done} new) {(time.time() - t0) / (i + 1):.1f}s/batch",
                  flush=True)
    if a.summary_out:
        s = summarize(a.data_dir)
        os.makedirs(os.path.dirname(os.path.abspath(a.summary_out)), exist_ok=True)
        with open(a.summary_out, "w") as f:
            json.dump(s, f, indent=2)
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
