"""Fit Cartesian trajectory anchors (k-means) per agent group from WOMD cache shards.

Futures are taken in each agent's frame (position and heading at the anchor step), fully valid
over the horizon. Clustering uses 5 waypoints (1 s apart, MultiPath style); each anchor is the
mean full 25-step trajectory of its cluster. Near-stationary futures (< 0.5 m travelled) are
kept at 10% so k-means spends its clusters on motion. Writes

  <out>.npz   anchors [3,K,T,2] (m, agent frame; vehicle / pedestrian / cyclist),
              counts [3,K], fitted_samples [3]
  <out>.json  oracle best-anchor ADE / FDE / miss rate (FDE > 2 m) on validation shards

  python -m anchor_wm.anchors --cache_dir .../sim_state_cache_npz --out .../anchors_k64 --num_anchors 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from anchor_wm.config import ANCHOR_GROUPS, AnchorWMConfig
from anchor_wm.data import load_npz, resolve_cache_paths, rotate_xy_np, split_cache_paths

WAYPOINTS = (4, 9, 14, 19, 24)
STATIONARY_M = 0.5
STATIONARY_KEEP = 0.1


def agent_frame_futures(payload: dict, anchor: int, cfg: AnchorWMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Fully valid futures [M,T,2] in each agent's own frame, and their groups [M] (0/1/2)."""
    x, y = payload["log_trajectory_x"], payload["log_trajectory_y"]
    yaw, valid = payload["log_trajectory_yaw"], payload["log_trajectory_valid"].astype(bool)
    types = payload["object_metadata_object_types"].astype(np.int32)
    is_sdc = payload["object_metadata_is_sdc"].astype(bool)
    n_steps = x.shape[2]
    raw = anchor + np.arange(1, cfg.future_steps + 1) * cfg.future_stride
    if raw[-1] >= n_steps:
        return np.zeros((0, cfg.future_steps, 2), np.float32), np.zeros(0, np.int64)
    ok = valid[:, :, anchor] & valid[:, :, raw].all(-1) & (np.isin(types, (1, 2, 3)) | is_sdc)
    s_idx, n_idx = np.nonzero(ok)
    cur = np.stack([x[s_idx, n_idx, anchor], y[s_idx, n_idx, anchor]], -1).astype(np.float64)
    fut = np.stack([x[s_idx, n_idx][:, raw], y[s_idx, n_idx][:, raw]], -1).astype(np.float64)
    h = yaw[s_idx, n_idx, anchor].astype(np.float64)
    local = rotate_xy_np(fut - cur[:, None], h[:, None])
    t = types[s_idx, n_idx]
    group = np.where(is_sdc[s_idx, n_idx], 0, np.where(t == 2, 1, np.where(t == 3, 2, 0)))
    return local.astype(np.float32), group.astype(np.int64)


def collect(paths, cfg, anchor_steps, max_per_group: int, seed: int):
    rng = np.random.default_rng(seed)
    bank = {g: [] for g in range(3)}
    for i, p in enumerate(paths):
        payload = load_npz(p)
        for a in anchor_steps:
            fut, grp = agent_frame_futures(payload, a, cfg)
            for g in range(3):
                bank[g].append(fut[grp == g])
        if (i + 1) % 10 == 0:
            print(f"read {i + 1}/{len(paths)} shards", flush=True)
    out = {}
    for g in range(3):
        arr = np.concatenate(bank[g]) if bank[g] else np.zeros((0, cfg.future_steps, 2), np.float32)
        moving = np.linalg.norm(arr[:, -1], axis=-1) >= STATIONARY_M
        keep = moving | (rng.random(len(arr)) < STATIONARY_KEEP)
        arr = arr[keep]
        if len(arr) > max_per_group:
            arr = arr[rng.choice(len(arr), max_per_group, replace=False)]
        out[g] = arr
    return out


def kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    """k-means++ init + Lloyd. x [M,D] -> assignment [M]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = x.shape[0]
    centers = [x[torch.randint(m, (1,), generator=g).item()]]
    d2 = (x - centers[0]).square().sum(-1)
    for _ in range(1, k):
        probs = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(x[idx])
        d2 = torch.minimum(d2, (x - x[idx]).square().sum(-1))
    c = torch.stack(centers)
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(-1)
        new = torch.zeros_like(c).index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=k).float()
        empty = cnt == 0
        c = torch.where(empty[:, None], c, new / cnt.clamp_min(1)[:, None])
        if empty.any():   # re-seed empty clusters at the worst-fit points
            far = torch.cdist(x, c).amin(-1).topk(int(empty.sum())).indices
            c[empty] = x[far]
    return torch.cdist(x, c).argmin(-1)


def fit(futures: np.ndarray, k: int, iters: int, seed: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    t = futures.shape[1]
    if len(futures) < k:
        raise ValueError(f"only {len(futures)} samples for {k} anchors")
    x = torch.from_numpy(futures[:, WAYPOINTS].reshape(len(futures), -1)).to(device)
    assign = kmeans(x, k, iters, seed)
    full = torch.from_numpy(futures.reshape(len(futures), -1)).to(device)
    sums = torch.zeros((k, full.shape[1]), device=device).index_add_(0, assign, full)
    cnt = torch.bincount(assign, minlength=k).float()
    anchors = (sums / cnt.clamp_min(1)[:, None]).reshape(k, t, 2)
    order = torch.argsort(anchors[:, -1, 0])          # sort by forward endpoint for readability
    return anchors[order].cpu().numpy(), cnt[order].cpu().numpy().astype(np.int64)


def oracle(anchors: np.ndarray, futures: np.ndarray, device: str) -> dict:
    if len(futures) == 0:
        return {"samples": 0}
    a = torch.from_numpy(anchors).to(device)
    f = torch.from_numpy(futures).to(device)
    ade, fde = [], []
    for s in range(0, len(f), 20000):
        err = (f[s:s + 20000, None] - a[None]).norm(dim=-1)                                  # [m,K,T]
        e_ade = err.mean(-1)
        best = e_ade.argmin(-1)
        ade.append(e_ade.gather(1, best[:, None]).squeeze(1))
        fde.append(err[:, :, -1].amin(-1))
    ade, fde = torch.cat(ade), torch.cat(fde)
    return {"samples": int(len(f)), "oracle_ade_m": float(ade.mean()), "oracle_fde_m": float(fde.mean()),
            "oracle_miss_rate_2m": float((fde > 2.0).float().mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--num_anchors", type=int, default=64)
    p.add_argument("--max_train_shards", type=int, default=40)
    p.add_argument("--max_val_shards", type=int, default=5)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--anchor_steps", type=int, nargs="+", default=[20, 30, 40])
    p.add_argument("--max_per_group", type=int, default=300000)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = AnchorWMConfig(num_anchors=args.num_anchors)
    train_paths, val_paths = split_cache_paths(resolve_cache_paths(args.cache_dir), args.val_fraction)
    rng = np.random.default_rng(args.seed)
    train_paths = [train_paths[i] for i in sorted(rng.choice(len(train_paths), min(args.max_train_shards, len(train_paths)), replace=False))]
    t0 = time.time()
    train = collect(train_paths, cfg, args.anchor_steps, args.max_per_group, args.seed)
    val = collect(val_paths[:args.max_val_shards], cfg, args.anchor_steps, args.max_per_group, args.seed + 1)
    anchors = np.zeros((3, args.num_anchors, cfg.future_steps, 2), np.float32)
    counts = np.zeros((3, args.num_anchors), np.int64)
    report = {"train_shards": len(train_paths), "val_shards": min(args.max_val_shards, len(val_paths)),
              "num_anchors": args.num_anchors, "groups": {}}
    for g, name in enumerate(ANCHOR_GROUPS):
        fallback = len(train[g]) < 20 * args.num_anchors and g > 0
        if fallback:   # too few samples: reuse the vehicle anchors for this group
            anchors[g], counts[g] = anchors[0], 0
        else:
            anchors[g], counts[g] = fit(train[g], args.num_anchors, args.iters, args.seed + g, args.device)
        report["groups"][name] = {"fitted_samples": int(len(train[g])), "vehicle_fallback": bool(fallback),
                                  "val": oracle(anchors[g], val[g], args.device),
                                  "empty_clusters": int((counts[g] == 0).sum())}
        print(name, json.dumps(report["groups"][name]), flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out) + ".npz", anchors=anchors, counts=counts,
             fitted_samples=np.array([len(train[g]) for g in range(3)]))
    with open(str(out) + ".json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
