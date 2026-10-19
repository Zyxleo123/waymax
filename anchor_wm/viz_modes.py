"""Side-by-side K=6 mode plots, MF-Gaussian vs AnchorWM, on WOMD validation scenes with many moving agents.

Scenes are chosen from the logged futures only (never from model outputs): the first scene of each
validation shard with at least --min_moving moving agents (non-SDC dynamic, within --radius_m of the
ego, logged displacement over the horizon >= --move_m). Modes are drawn for the moving agents.
Also prints the mean 6-mode endpoint spread / minFDE over moving agents of the first --stat_scenes
scenes that qualify (one per shard).

  python -m anchor_wm.viz_modes --cache_dir ... --mf_ckpt .../epoch_0450.pt --ckpt .../final.pt --out modes.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from anchor_wm.config import DYNAMIC_TYPES, ROAD_EDGE_TYPES  # noqa: E402
from anchor_wm.data import (LANE_TYPES, build_arrays, load_npz, resolve_cache_paths, rotate_xy_np,  # noqa: E402
                            split_cache_paths)
from anchor_wm.eval_bank import load_model  # noqa: E402
from anchor_wm.eval_wm import load_mf  # noqa: E402
from anchor_wm.model import select_modes  # noqa: E402

MODE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def moving_mask(a: dict, b: int, radius_m: float, move_m: float) -> np.ndarray:
    tv = a["target_valid"][b]                                                     # [N,T]
    has = tv.any(-1) & a["agent_valid"][b] & np.isin(a["object_types"][b], DYNAMIC_TYPES)
    last = np.where(has, tv.shape[1] - 1 - np.argmax(tv[:, ::-1], axis=1), 0)
    disp = np.linalg.norm(a["target_delta_xy"][b, np.arange(tv.shape[0]), last], axis=-1) * 50.0
    near = np.linalg.norm(a["current_local_xy"][b], axis=-1) <= radius_m
    return has & near & (disp >= move_m)


@torch.no_grad()
def predict(models: dict, a: dict, b: int, device: torch.device) -> dict:
    """{name: (xy [N,6,T,2] metres ego frame (absolute), probs [N,6])} for scene b."""
    f = {k: torch.from_numpy(np.ascontiguousarray(v[b:b + 1])).to(device) for k, v in a.items()}
    cur = a["current_local_xy"][b][:, None, None]
    out = {}
    for name, (kind, model, rng, nms) in models.items():
        if kind == "mf":
            o = model(f)
            pred, probs = o["pred_delta_xy_norm_k"].float(), o["mode_probs"].float()
        else:
            _, dec = model(f)
            sel = select_modes(dec, 6, nms, rng)
            pred = dec["pred_xy_k"].float().gather(2, sel[..., None, None].expand(-1, -1, -1, *dec["pred_xy_k"].shape[3:]))
            probs = dec["probs"].float().gather(-1, sel)
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
        out[name] = (pred[0].cpu().numpy() * rng + cur, probs[0].cpu().numpy())
    return out


def scene_stats(xy: np.ndarray, gt: np.ndarray, tv: np.ndarray, mov: np.ndarray) -> dict:
    end = xy[mov, :, -1]                                                          # [M,6,2]
    pd = np.linalg.norm(end[:, :, None] - end[:, None], axis=-1)
    spread = pd.sum((1, 2)) / 30.0
    last = tv.shape[1] - 1 - np.argmax(tv[mov, ::-1], axis=1)
    gt_end = gt[mov, last]
    fde = np.linalg.norm(xy[mov, :, :, :][np.arange(mov.sum()), :, last] - gt_end[:, None], axis=-1).min(-1)
    return {"spread": spread, "minfde": fde}


def draw_box(ax, xy, yaw, size, **kw):
    c, s = np.cos(yaw), np.sin(yaw)
    l, w = size[0] / 2, size[1] / 2
    pts = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) @ np.array([[c, s], [-s, c]]) + xy
    ax.add_patch(Polygon(pts, closed=True, **kw))


def draw_panel(ax, payload, a, b, xy, probs, mov, title, window):
    s_idx = b
    ox, oy = a["origin_xy"][b]
    hd = a["origin_yaw"][b]
    rv = np.asarray(payload["roadgraph_points_valid"][s_idx]).astype(bool)
    types = np.asarray(payload["roadgraph_points_types"][s_idx])
    pts = np.stack([payload["roadgraph_points_x"][s_idx], payload["roadgraph_points_y"][s_idx]], -1).astype(np.float64)
    local = rotate_xy_np(pts - np.array([ox, oy]), hd)
    for tset, col, sz in ((LANE_TYPES, "#c8c8c8", 0.6), (ROAD_EDGE_TYPES, "#555555", 0.8)):
        m = rv & np.isin(types, tset)
        ax.scatter(local[m, 0], local[m, 1], s=sz, c=col, linewidths=0, zorder=1)
    n = a["current_local_xy"].shape[1]
    lw_all = a["fut_lw"][b][:, 0]
    cur, cyaw = a["current_local_xy"][b], a["current_yaw_local"][b]
    agent_ok = a["agent_valid"][b] & np.isin(a["object_types"][b], DYNAMIC_TYPES)
    ego = int(a["ego_idx"][b])
    for i in range(n):
        if not agent_ok[i] or i == ego:
            continue
        draw_box(ax, cur[i], cyaw[i], lw_all[i], fc="#4a90d9" if mov[i] else "#bbbbbb", ec="k", lw=0.4, zorder=3)
    draw_box(ax, np.zeros(2), 0.0, a["ego_lw"][b], fc="#d62728", ec="k", lw=0.6, zorder=4)
    gt = a["fut_xy"][b]
    tv = a["target_valid"][b]
    for i in np.flatnonzero(mov):
        pm = probs[i].max()
        for k in range(xy.shape[1]):
            alpha = 0.3 + 0.7 * probs[i, k] / max(pm, 1e-9)
            ax.plot(xy[i, k, :, 0], xy[i, k, :, 1], color=MODE_COLORS[k], lw=1.2, alpha=alpha, zorder=5)
            ax.scatter(xy[i, k, -1, 0], xy[i, k, -1, 1], s=8, color=MODE_COLORS[k], alpha=alpha, zorder=6)
        g = gt[i][tv[i]]
        ax.plot(np.r_[cur[i, 0], g[:, 0]], np.r_[cur[i, 1], g[:, 1]], "k--", lw=1.0, zorder=7)
    (x0, x1), (y0, y1) = window
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--mf_ckpt", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_scenes", type=int, default=5)
    p.add_argument("--stat_scenes", type=int, default=60)
    p.add_argument("--anchor_step", type=int, default=20)
    p.add_argument("--min_moving", type=int, default=8)
    p.add_argument("--move_m", type=float, default=3.0)
    p.add_argument("--radius_m", type=float, default=40.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    awm = load_model(args.ckpt, device)
    cfg = awm.config
    rng = float(cfg.max_range)
    models = {"MF-Gaussian": ("mf", load_mf(args.mf_ckpt, device), rng, 0.0),
              "AnchorWM": ("anchor", awm, rng, float(cfg.nms_dist_m))}
    _, val_paths = split_cache_paths(resolve_cache_paths(args.cache_dir), 0.1)

    picked, stats = [], {name: {"spread": [], "minfde": []} for name in models}
    n_stat = 0
    for path in val_paths:
        if n_stat >= args.stat_scenes:
            break
        payload = load_npz(path)
        a = build_arrays(payload, anchor_step=args.anchor_step, cfg=cfg)
        for b in range(a["agent_history"].shape[0]):
            mov = moving_mask(a, b, args.radius_m, args.move_m)
            if mov.sum() < args.min_moving:
                continue
            preds = predict(models, a, b, device)
            for name, (xy, _) in preds.items():
                st = scene_stats(xy, a["fut_xy"][b], a["target_valid"][b], mov)
                for k in st:
                    stats[name][k].append(st[k])
            n_stat += 1
            if len(picked) < args.num_scenes:
                picked.append((path, payload, a, b, mov, preds))
            break

    summary = {name: {"scenes": n_stat, "moving_agents": int(sum(len(x) for x in s["spread"])),
                      "endpoint_spread_m": round(float(np.concatenate(s["spread"]).mean()), 3),
                      "min_fde_6_m": round(float(np.concatenate(s["minfde"]).mean()), 3)}
               for name, s in stats.items()}
    print(json.dumps(summary, indent=1))

    # Tight window around the moving agents, their logged futures and both models' modes; long thin
    # scenes (straight roads) get a short row instead of a mostly empty square.
    windows = []
    for path, payload, a, b, mov, preds in picked:
        allpts = [np.zeros((1, 2)), a["current_local_xy"][b][mov], a["fut_xy"][b][mov][a["target_valid"][b][mov]]]
        allpts += [xy[mov].reshape(-1, 2) for xy, _ in preds.values()]
        allpts = np.concatenate(allpts)
        lo, hi = np.percentile(allpts, 0.5, axis=0) - 5, np.percentile(allpts, 99.5, axis=0) + 5
        c, size = (lo + hi) / 2, np.maximum(hi - lo, 30.0)
        size[1] = max(size[1], 0.3 * size[0])
        size[0] = max(size[0], 0.6 * size[1])
        windows.append(((c[0] - size[0] / 2, c[0] + size[0] / 2), (c[1] - size[1] / 2, c[1] + size[1] / 2)))
    ratios = [(w[1][1] - w[1][0]) / (w[0][1] - w[0][0]) for w in windows]
    panel_w = 7.0
    fig, axes = plt.subplots(len(picked), 2, figsize=(2 * panel_w, sum(panel_w * q + 0.8 for q in ratios) + 1.2),
                             gridspec_kw={"height_ratios": [q + 0.8 / panel_w for q in ratios]})
    axes = np.atleast_2d(axes)
    for r, ((path, payload, a, b, mov, preds), window) in enumerate(zip(picked, windows)):
        shard = path.rsplit("/", 1)[-1].split("-of-")[0][-5:]
        for col, (name, (xy, probs)) in enumerate(preds.items()):
            st = scene_stats(xy, a["fut_xy"][b], a["target_valid"][b], mov)
            title = (f"{name} | shard {shard} scene {b} | {int(mov.sum())} moving\n"
                     f"spread {st['spread'].mean():.1f} m, minFDE@6 {st['minfde'].mean():.2f} m")
            draw_panel(axes[r, col], payload, a, b, xy, probs, mov, title, window)
    fig.suptitle("K=6 modes for moving agents (dashed black = logged future, red = ego, "
                 "blue = moving agents, opacity ~ mode probability)\n"
                 f"Aggregate over {n_stat} val scenes: " + ", ".join(
                     f"{n} spread {s['endpoint_spread_m']:.2f} m / minFDE@6 {s['min_fde_6_m']:.2f} m"
                     for n, s in summary.items()), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
