"""Why is a scene flagged offroad? Per-step offroad geometry of executed paths, plus a zoomed plot.

For each run's executed path (and the logged human drive) the metric's offroad check is broken down at
every step: the worst corner, its signed distance, and the road-edge point it was measured against
(edge id, whether it is a polyline end, its height vs the frozen box height). The plot overlays the
paths, their boxes at the flagged steps and the road-edge polylines near them.

    python scripts/offroad_debug_scene.py --tfrecord training_tfexample.tfrecord-00031-of-01000 --scenario 49 \
        --run ft=<run dir> --run a2margin=<run dir> --out scene049.png
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from scores.scorer_lane_graph import polygon_corners, road_edge_points

SUBSET_DIR = "/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
START_T, GOAL_T = 10, 90


def breakdown(boxes, edges, z0):
    """Per step: worst corner signed distance and the edge point behind it (same rule as the metric)."""
    exyz, edir, eids = (np.asarray(e) for e in edges)
    corners = np.asarray(polygon_corners(jnp.asarray(boxes[None])))[0]               # [T, 4, 2]
    out = []
    for t in range(len(boxes)):
        best = None
        for c in range(4):
            p = np.array([*corners[t, c], z0], np.float32)
            dd = ((exyz - p) * np.array([1, 1, 2], np.float32)) ** 2
            i = int(np.argmin(dd.sum(-1)))
            j = max(i - 1, 0)
            v = p[:2] - exyz[i, :2]
            cross = v[0] * edir[i, 1] - v[1] * edir[i, 0]
            crossp = v[0] * edir[j, 1] - v[1] * edir[j, 0]
            same = eids[i] == eids[j]
            sgn = np.sign(crossp if (same and crossp < cross) else cross)
            d = float(np.linalg.norm(v) * sgn)
            nxt_same = i + 1 < len(eids) and eids[i + 1] == eids[i]
            end = (not same) or (not nxt_same) or i == j
            if best is None or d > best["d"]:
                best = {"d": d, "corner": c, "edge_idx": i, "edge_id": int(eids[i]), "edge_xyz": exyz[i].tolist(),
                        "polyline_end": bool(end), "edge_dz": float(exyz[i, 2] - z0), "corner_xy": corners[t, c].tolist()}
        out.append(best)
    return out, corners


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", required=True)
    p.add_argument("--scenario", type=int, required=True)
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING,
                              path=os.path.join(SUBSET_DIR, "tfrecords", "training", a.tfrecord),
                              max_num_objects=128, batch_dims=(1,), shuffle_seed=0)
    st, to_batch = load_scenario_state_batch_fast(cfg, [a.scenario])
    w = int(to_batch[a.scenario])
    log = st.log_trajectory
    e = int(np.argmax(np.asarray(st.object_metadata.is_sdc[w])))
    g = lambda arr: np.asarray(arr[w, e])  # noqa: E731
    lw = np.array([g(log.length)[START_T], g(log.width)[START_T]], np.float32)
    z0 = float(g(log.z)[START_T])
    edges = road_edge_points(st.roadgraph_points, w)
    paths = {"log": np.stack([g(log.x), g(log.y), g(log.yaw)], -1)}
    for r in a.run:
        name, rd = r.split("=", 1)
        f = glob.glob(os.path.join(rd, "chunk_*", f"{a.tfrecord}.scenario_{a.scenario:03d}.trajectory.npz"))
        paths[name] = np.load(f[0])["predicted_trajectory"][:, :3]
    win = slice(START_T, GOAL_T + 1)
    fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
    exyz, _, eids = (np.asarray(x) for x in edges)
    colors = {"log": "black", **{n: c for n, c in zip([r.split("=", 1)[0] for r in a.run], ["tab:red", "tab:blue", "tab:green"])}}
    focus = []
    for name, pth in paths.items():
        boxes = np.concatenate([pth[win], np.broadcast_to(lw, (len(pth[win]), 2))], -1).astype(np.float32)
        info, corners = breakdown(boxes, edges, z0)
        d = np.array([x["d"] for x in info])
        flagged = np.flatnonzero(d > 0)
        print(f"== {name}: offroad steps {len(flagged)}; worst {d.max():+.2f} m at t={int(np.argmax(d)) + START_T}")
        for k in flagged[:6].tolist() + ([int(np.argmax(d))] if len(flagged) else []):
            x = info[k]
            print(f"   t={k + START_T:3d} d={x['d']:+.2f} corner={x['corner']} edge_id={x['edge_id']} idx={x['edge_idx']} "
                  f"polyline_end={x['polyline_end']} edge_dz={x['edge_dz']:+.2f} edge_xy=({x['edge_xyz'][0]:.1f},{x['edge_xyz'][1]:.1f}) "
                  f"corner_xy=({x['corner_xy'][0]:.1f},{x['corner_xy'][1]:.1f})")
        ax.plot(pth[win, 0], pth[win, 1], "-", color=colors[name], lw=1.5, label=f"{name} ({len(flagged)} offroad steps)")
        for k in flagged[:: max(1, len(flagged) // 6)]:
            poly = np.vstack([corners[k], corners[k][:1]])
            ax.plot(poly[:, 0], poly[:, 1], "-", color=colors[name], lw=0.8)
            x = info[k]
            ax.plot([x["corner_xy"][0], x["edge_xyz"][0]], [x["corner_xy"][1], x["edge_xyz"][1]], ":", color=colors[name])
            focus.append(x["corner_xy"])
    # road edges near the paths
    allp = np.concatenate([pth[win, :2] for pth in paths.values()])
    lo, hi = allp.min(0) - 15, allp.max(0) + 15
    near = (exyz[:, 0] > lo[0]) & (exyz[:, 0] < hi[0]) & (exyz[:, 1] > lo[1]) & (exyz[:, 1] < hi[1])
    for i in np.unique(eids[near]):
        m = near & (eids == i)
        ax.plot(exyz[m, 0], exyz[m, 1], ".-", color="gray", ms=2, lw=0.8)
        ax.annotate(str(int(i)), exyz[m][0, :2], fontsize=6, color="gray")
    if focus:
        f = np.array(focus)
        c = f.mean(0)
        r = max(12.0, float(np.abs(f - c).max()) + 8)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{a.tfrecord} #{a.scenario}: paths, boxes at flagged steps, road edges (gray, ids)")
    fig.savefig(a.out, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
