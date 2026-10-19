"""Agreement of AnchorWM's online labels with the Waymax-exact ES-bank labels.

For recorded candidates of a few batch files, recompute overlap / offroad with
`candidates.label_candidates` (log-replay boxes and road edges from the original log, in the
recorded group frame, over the recorded label horizon) and compare with `<batch>.labels_waymax.npz`
(gt_overlap_step / gt_offroad_step > 0). Done on the recorded 0.1 s grid (exactness of the
geometry) and on the 0.2 s training grid. Known differences: no road height (Waymax uses z), and
the 0.2 s grid can miss overlaps that last a single 0.1 s step.

  python -m anchor_wm.label_check --bank_dir .../es_candidates --tfrecord_dir ... --max_files 3 --out ...json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from anchor_wm.candidates import label_candidates
from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import build_edges, future_boxes, payload_from_sim_state


def _confusion(pred: np.ndarray, ref: np.ndarray) -> dict:
    tp, fp = int((pred & ref).sum()), int((pred & ~ref).sum())
    fn, tn = int((~pred & ref).sum()), int((~pred & ~ref).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _summary(c: dict) -> dict:
    n = sum(c.values())
    return {**c, "agreement": (c["tp"] + c["tn"]) / max(n, 1), "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
            "precision": c["tp"] / max(c["tp"] + c["fp"], 1)}


def check_file(path: str, tfrecord_dir: str, cfg: AnchorWMConfig, device: torch.device) -> dict:
    import jax
    from es_discriminator.analyze_es_decisions import load_batch_state

    d = dict(np.load(path))
    lab = dict(np.load(path[:-4] + ".labels_waymax.npz"))
    payload = payload_from_sim_state(jax.device_get(load_batch_state(d, tfrecord_dir)))
    t_raw = d["cand_traj"].shape[1]
    res = {grid: {"overlap": dict(tp=0, fp=0, fn=0, tn=0), "offroad": dict(tp=0, fp=0, fn=0, tn=0)}
           for grid in ("0.1s", "0.2s")}
    for g in np.flatnonzero(d["group_valid"].astype(bool)):
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        rows = np.flatnonzero(d["cand_group"] == g)
        h = int(lab["label_horizon"][g])
        if h <= 1 or len(rows) == 0:
            continue
        pw = {k: v[w:w + 1] for k, v in payload.items()}
        origin = d["group_origin"][g][None].astype(np.float64)
        heading = np.array([float(d["group_heading"][g])])
        fut = future_boxes(pw, t, origin, heading, t_raw, 1, min(cfg.max_agents, pw["log_trajectory_x"].shape[1]))
        is_sdc = pw["object_metadata_is_sdc"][:, :fut["xy"].shape[1]].astype(bool)
        f = {"fut_xy": fut["xy"], "fut_yaw": fut["yaw"], "fut_lw": fut["lw"],
             "fut_valid": fut["valid"] & ~is_sdc[:, :, None],
             "ego_lw": d["group_ego_state"][g, 5:7][None].astype(np.float32),
             "ego_future_xy": np.zeros((1, t_raw, 2), np.float32), "ego_future_valid": np.zeros((1, t_raw), bool),
             "goal_xy": np.zeros((1, 2), np.float32), "ego_vel": np.zeros((1, 2), np.float32)}
        f.update(build_edges(pw, origin, heading, cfg))
        f = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in f.items()}
        traj = torch.from_numpy(np.nan_to_num(d["cand_traj"][rows][..., :3]).astype(np.float32)).to(device)[None]
        ref_ov = lab["gt_overlap_step"][rows] > 0
        ref_off = lab["gt_offroad_step"][rows] > 0
        for grid, sl, hh, dt in (("0.1s", slice(None), h, 0.1), ("0.2s", slice(1, None, 2), h // 2, 0.2)):
            fg = dict(f)
            for k in ("fut_xy", "fut_yaw", "fut_lw", "fut_valid"):
                fg[k] = f[k][:, :, sl]
            x = traj[:, :, sl]
            out = label_candidates(fg, x[..., :2], x[..., 2], dt, horizon=torch.tensor([hh], device=device))
            for name, ref in (("overlap", ref_ov), ("offroad", ref_off)):
                c = _confusion(out[name][0].cpu().numpy(), ref)
                for k in c:
                    res[grid][name][k] += c[k]
    return res


def diag_offroad(path: str, tfrecord_dir: str, cfg: AnchorWMConfig, device: torch.device, max_groups: int) -> dict:
    """Where offroad disagreements come from, per candidate (0.1 s grid, no midpoints):
    ref        Waymax with the logged ego height, as es_discriminator's relabel (should equal the bank label)
    waymax_2d  Waymax with all heights zeroed (pure 2-D nearest edge)
    ours       label_candidates with the training edge set (box + subsampling)
    ours_full  label_candidates with every road-edge point"""
    import dataclasses
    import jax
    from es_discriminator.analyze_es_decisions import EPISODE_STEPS, load_batch_state, offroad_poses, road_edges
    from es_discriminator.label_bank import _to_world

    d = dict(np.load(path))
    lab = dict(np.load(path[:-4] + ".labels_waymax.npz"))
    st = jax.device_get(load_batch_state(d, tfrecord_dir))
    payload = payload_from_sim_state(st)
    log_z = np.asarray(st.log_trajectory.z)
    is_sdc = np.asarray(st.object_metadata.is_sdc).astype(bool)
    full_cfg = dataclasses.replace(cfg, num_label_edges=200000, edge_box=(-1e5, 1e5, 1e5))
    labels = {k: [] for k in ("bank", "ref", "waymax_2d", "ours", "ours_full")}
    for g in np.flatnonzero(d["group_valid"].astype(bool))[:max_groups]:
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        rows = np.flatnonzero(d["cand_group"] == g)
        h = int(lab["label_horizon"][g])
        if h <= 1:
            continue
        e = int(np.argmax(is_sdc[w]))
        origin, heading = d["group_origin"][g].astype(np.float64), float(d["group_heading"][g])
        length, width = d["group_ego_state"][g, 5], d["group_ego_state"][g, 6]
        local = np.nan_to_num(d["cand_traj"][rows][:, :h].astype(np.float64))
        world = _to_world(local, origin, heading)
        poses = np.concatenate([world, np.broadcast_to([length, width], world.shape[:2] + (2,))], -1)
        z = log_z[w, e, np.minimum(np.arange(t + 1, t + 1 + h), EPISODE_STEPS - 1)]
        rg = road_edges(st, w)
        labels["ref"].append(offroad_poses(poses, z[None, :], rg).any(1))
        rg0 = dataclasses.replace(rg, z=np.zeros_like(rg.z))
        labels["waymax_2d"].append(offroad_poses(poses, np.zeros_like(z)[None, :], rg0).any(1))
        labels["bank"].append(lab["gt_offroad_step"][rows] > 0)
        pw = {k: v[w:w + 1] for k, v in payload.items()}
        traj = torch.from_numpy(local.astype(np.float32)).to(device)[None]
        for name, c in (("ours", cfg), ("ours_full", full_cfg)):
            f = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device)
                 for k, v in build_edges(pw, origin[None], np.array([heading]), c).items()}
            n0 = 1
            f.update(fut_xy=torch.zeros(1, n0, h, 2, device=device), fut_yaw=torch.zeros(1, n0, h, device=device),
                     fut_lw=torch.ones(1, n0, h, 2, device=device), fut_valid=torch.zeros(1, n0, h, dtype=torch.bool, device=device),
                     ego_lw=torch.tensor([[length, width]], dtype=torch.float32, device=device),
                     ego_future_xy=torch.zeros(1, h, 2, device=device), ego_future_valid=torch.zeros(1, h, dtype=torch.bool, device=device),
                     goal_xy=torch.zeros(1, 2, device=device), ego_vel=torch.zeros(1, 2, device=device))
            out = label_candidates(f, traj[..., :2], traj[..., 2], 0.1, midpoints=False)
            labels[name].append(out["offroad"][0].cpu().numpy())
    lab_all = {k: np.concatenate(v) for k, v in labels.items()}
    rep = {"candidates": int(len(lab_all["ref"])), "positives": {k: int(v.sum()) for k, v in lab_all.items()}}
    for a, b in (("ref", "bank"), ("waymax_2d", "ref"), ("ours", "ref"), ("ours", "waymax_2d"),
                 ("ours_full", "waymax_2d"), ("ours_full", "ours")):
        rep[f"{a}_vs_{b}"] = _summary(_confusion(lab_all[a], lab_all[b]))
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank_dir", required=True)
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--max_files", type=int, default=3)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--diag_offroad", type=int, default=0, help="only run the offroad diagnostic on N groups per file")
    args = p.parse_args()

    from es_discriminator.analyze_es_decisions import batch_files
    import os

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = AnchorWMConfig()
    files = [f for f in batch_files(args.bank_dir) if os.path.exists(f[:-4] + ".labels_waymax.npz")][:args.max_files]
    if args.diag_offroad:
        reps = {os.path.basename(f): diag_offroad(f, args.tfrecord_dir, cfg, device, args.diag_offroad) for f in files}
        print(json.dumps(reps, indent=1))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(reps, fh, indent=1)
        return
    total = None
    for f in files:
        r = check_file(f, args.tfrecord_dir, cfg, device)
        if total is None:
            total = r
        else:
            for grid in r:
                for name in r[grid]:
                    for k in r[grid][name]:
                        total[grid][name][k] += r[grid][name][k]
        print(os.path.basename(f), json.dumps({g: {n: _summary(v) for n, v in r[g].items()} for g in r}), flush=True)
    report = {"files": len(files), **{g: {n: _summary(v) for n, v in total[g].items()} for g in total}}
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
