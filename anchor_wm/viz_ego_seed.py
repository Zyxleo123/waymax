"""Schematics of Diffusion-ES replans with / without AnchorWM ego-mode seeding (dumps of anchor_wm.es_debug).

One PNG per scene (batch of a --num_worlds 1 run): rows = runs (e.g. no seeds | 6 seeds), columns = the executed
closed-loop path, then the chosen replan steps in the ego frame at that step. Each replan panel shows the road
edges and lanes, the agents now, the diffusion initial pool, the AnchorWM's own ego modes (width ~ probability),
the final ES pool's top plans and the chosen plan (green when it descends from an ego seed).

  python -m anchor_wm.viz_ego_seed --run "no seeds=.../v1ft_seed0" --run "6 AnchorWM seeds=.../v1ft_seed6" \
      --out_dir ... --steps 10 30 50 70
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from anchor_wm.config import DYNAMIC_TYPES, ROAD_EDGE_TYPES  # noqa: E402
from anchor_wm.data import LANE_TYPES  # noqa: E402

MODE_COLORS = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]
POOL, TOP, CHOSEN_DIFF, CHOSEN_SEED = "#9ecae1", "#3182bd", "#08306b", "#1a9850"


def load_run(d: str) -> tuple[dict, dict]:
    dumps = {}
    for f in sorted(glob.glob(f"{d}/dumps/b*_t*.npz")):
        b, t = os.path.basename(f)[1:-4].split("_t")
        dumps.setdefault(int(b), {})[int(t)] = dict(np.load(f))
    results = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.scenario_*.json"))]
    return dumps, results


def match_result(scene: dict, results: list[dict]) -> dict | None:
    goal = np.asarray(scene[min(scene)]["goal"]).reshape(-1)[:2]
    best = min(results, key=lambda r: np.linalg.norm(np.asarray(r["goal_xy"]) - goal), default=None)
    return best if best is not None and np.linalg.norm(np.asarray(best["goal_xy"]) - goal) < 0.5 else None


def to_frame(xy, origin, yaw):
    d = np.asarray(xy, np.float64) - origin
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)


def box(ax, xy, yaw, size, **kw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * (np.asarray(size) / 2)
    ax.add_patch(Polygon(pts @ np.array([[c, s], [-s, c]]) + xy, closed=True, **kw))


def draw_road(ax, road: dict, origin, yaw):
    v = road["roadgraph_points_valid"][0].astype(bool)
    types = road["roadgraph_points_types"][0]
    pts = to_frame(np.stack([road["roadgraph_points_x"][0], road["roadgraph_points_y"][0]], -1), origin, yaw)
    for tset, col, sz in ((LANE_TYPES, "#d0d0d0", 0.5), (ROAD_EDGE_TYPES, "#303030", 1.2)):
        m = v & np.isin(types, tset)
        ax.scatter(pts[m, 0], pts[m, 1], s=sz, c=col, linewidths=0, zorder=1)


def ego_pose(z: dict) -> tuple[np.ndarray, float, np.ndarray, int]:
    e = int(np.argmax(z["object_metadata_is_sdc"][0]))
    xy = np.array([z["log_trajectory_x"][0, e], z["log_trajectory_y"][0, e]], np.float64)
    lw = np.array([z["log_trajectory_length"][0, e], z["log_trajectory_width"][0, e]])
    return xy, float(z["log_trajectory_yaw"][0, e]), lw, e


def draw_replan(ax, z: dict, road: dict, window: float):
    origin, yaw, ego_lw, e = ego_pose(z)
    draw_road(ax, road, origin, yaw)
    ok = z["log_trajectory_valid"][0].astype(bool) & np.isin(z["object_metadata_object_types"][0], DYNAMIC_TYPES)
    axy = to_frame(np.stack([z["log_trajectory_x"][0], z["log_trajectory_y"][0]], -1), origin, yaw)
    for i in np.flatnonzero(ok):
        if i != e:
            box(ax, axy[i], z["log_trajectory_yaw"][0, i] - yaw,
                (z["log_trajectory_length"][0, i], z["log_trajectory_width"][0, i]), fc="#bdbdbd", ec="k", lw=0.4, zorder=3)
    box(ax, np.zeros(2), 0.0, ego_lw, fc="#e41a1c", ec="k", lw=0.6, zorder=8)
    init, origin_k = z["init_world"][0], z["init_origin"][0]
    for k in np.flatnonzero(origin_k == 0):
        p = to_frame(init[k, :, :2], origin, yaw)
        ax.plot(p[:, 0], p[:, 1], color=POOL, lw=0.5, alpha=0.6, zorder=2)
    fin = z["final_world"][0]
    for k in range(min(8, fin.shape[0])):
        p = to_frame(fin[k, :, :2], origin, yaw)
        ax.plot(p[:, 0], p[:, 1], color=TOP, lw=0.9, alpha=0.8, zorder=4)
    probs = z["awm_ego_probs"][0]
    for m in range(probs.shape[0]):
        p = to_frame(z["awm_ego_xy"][0, m], origin, yaw)
        p = np.concatenate([np.zeros((1, 2)), p])
        ax.plot(p[:, 0], p[:, 1], color=MODE_COLORS[m % 6], lw=0.8 + 4.0 * probs[m], ls="--", alpha=0.9, zorder=5,
                label=f"AWM ego mode p={probs[m]:.2f}")
    best = to_frame(z["best_world"][0, :, :2], origin, yaw)
    from_seed = bool(z["best_origin"][0])
    ax.plot(best[:, 0], best[:, 1], color=CHOSEN_SEED if from_seed else CHOSEN_DIFF, lw=2.6, zorder=7)
    ax.plot(best[:11, 0], best[:11, 1], color="#ffd92f", lw=1.2, zorder=7.5)              # executed until next replan
    g = to_frame(np.asarray(z["goal"]).reshape(-1)[:2], origin, yaw)
    if np.all(np.abs(g) < window * 1.5):
        ax.scatter([g[0]], [g[1]], marker="*", s=120, c="gold", ec="k", zorder=9)
    ax.set_xlim(-0.35 * window, 1.15 * window)
    ax.set_ylim(-0.75 * window / 1.0, 0.75 * window / 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    n_seed = int(z["num_seeds"])
    ranks = np.argsort(np.argsort(-z["init_scores"][0]))[init.shape[0] - n_seed:] if n_seed else []
    ax.set_title(f"t={int(z['timestep'])}  chosen from {'SEED' if from_seed else 'diffusion'}  "
                 f"score {float(z['final_scores'][0, 0]):.2f}"
                 + (f"\nseed ranks in init pool {sorted(int(r) for r in ranks)} of {init.shape[0]}" if n_seed else ""),
                 fontsize=8.5)


def draw_overview(ax, scene: dict, road: dict, label: str, result: dict | None):
    steps = sorted(scene)
    z0 = scene[steps[0]]
    origin, yaw, _, _ = ego_pose(z0)
    pts = []
    for t in steps:
        b = to_frame(scene[t]["best_world"][0, :11, :2], origin, yaw)
        ax.plot(b[:, 0], b[:, 1], color=CHOSEN_SEED if scene[t]["best_origin"][0] else CHOSEN_DIFF, lw=2.2, zorder=5)
        ax.text(b[0, 0], b[0, 1], str(t), fontsize=6, zorder=6)
        pts.append(b)
    last = to_frame(scene[steps[-1]]["best_world"][0, :, :2], origin, yaw)
    ax.plot(last[:, 0], last[:, 1], color="#555555", lw=1.0, ls=":", zorder=4)
    g = to_frame(np.asarray(z0["goal"]).reshape(-1)[:2], origin, yaw)
    ax.scatter([g[0]], [g[1]], marker="*", s=140, c="gold", ec="k", zorder=9)
    allp = np.concatenate(pts + [last, g[None]])
    lo, hi = allp.min(0) - 12, allp.max(0) + 12
    c, size = (lo + hi) / 2, max((hi - lo).max(), 40)
    draw_road(ax, road, origin, yaw)
    ax.set_xlim(c[0] - size / 2, c[0] + size / 2)
    ax.set_ylim(c[1] - size / 2, c[1] + size / 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    out = "?" if result is None else ("SUCCESS" if result["success"] else
                                      "+".join(k for k in ("overlap", "offroad") if result[k]) or "goal miss")
    ax.set_title(f"{label}: executed path ({out})\ngreen = from ego seed, blue = from diffusion", fontsize=8.5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="label=run_dir (a run with --es_debug_dir run_dir/dumps)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[10, 30, 50, 70])
    p.add_argument("--window_m", type=float, default=45.0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = [(lab, *load_run(d)) for lab, d in (r.split("=", 1) for r in args.run)]
    for b in sorted(runs[0][1]):
        fig, axes = plt.subplots(len(runs), 1 + len(args.steps), figsize=(4.2 * (1 + len(args.steps)), 3.9 * len(runs)),
                                 squeeze=False)
        name = ""
        for r, (lab, dumps, results) in enumerate(runs):
            scene = dumps[b]
            road = scene[min(scene)]
            res = match_result(scene, results)
            if res is not None:
                name = f"{os.path.basename(res['tfrecord']).split('-of-')[0][-5:]}_s{int(res['scenario_idx']):03d}"
            draw_overview(axes[r, 0], scene, road, lab, res)
            for c, t in enumerate(args.steps):
                if t in scene:
                    draw_replan(axes[r, c + 1], scene[t], road, args.window_m)
                axes[r, c + 1].set_ylabel(lab if c == 0 else "")
        h, l = axes[-1, 1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.5, frameon=False)
        fig.suptitle(f"{name}: Diffusion-ES with the v1 fine-tuned AnchorWM. Light blue = diffusion initial pool, "
                     "dashed = AnchorWM ego modes (width ~ prob), mid blue = final top-8, thick = chosen "
                     "(green from seed), yellow = executed 1 s, red box = ego, star = goal", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        path = os.path.join(args.out_dir, f"ego_seed_{name or f'b{b:03d}'}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
