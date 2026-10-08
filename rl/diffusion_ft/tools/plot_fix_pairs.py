"""Static side-by-side figures, base vs a repair arm, on scenes the arm FIXED
(``--mode fixed``, the default) or on episodes the arm still FAILS (``--mode failing``).

One PNG per scene, 2 x 2: columns are the base policy and the arm on the SAME
paired draw (scene, repeat); the top row is the whole episode on the map, the
bottom row zooms on the moment the base violates -- its first box overlap with
another car, or its closest approach to a road edge -- with every car's box drawn
at that instant. In the arm's column, dashed segments are the base proposal at
each replan wherever the executed plan departs from it, i.e. what the repair
replaced.

CPU only, no model, no env: the static geometry (road edges, lanes, other cars,
goal) comes from an existing ``dump_reach_from_select`` bundle -- it depends on
the scene, not the arm -- and the executed paths and outcomes from the select
run's own ``feat/*.npz`` and ``rows.jsonl``.

Caveats, drawn into the caption: the bundle does not carry the ego's size, so the
ego box is nominal (``--ego_lw``); and the off-road marker is the closest UNSIGNED
approach of that box to an edge point, so it locates the excursion, it does not
re-derive waymax's offroad flag (the outcome labels come from the run's rows).

    python -m rl.diffusion_ft.tools.plot_fix_pairs --scenes_json <bundle.json> \
      --select_dir <run>/select --arm o_lpA3 --ids 33174,122120 --out_dir <dir>

``--mode failing``: one PNG per scene where the arm leaves some repeat unsolved
(every such scene in the bundle, or ``--ids``). The repeat drawn is the failing one
where the arm executed the most repairs (its strongest attempt), and the zoom is on
the ARM's own violation -- its first box overlap, its closest approach to a road
edge, or, for a clean episode that never arrives, its last pose. Files are
``fail_<sid>_r<rep>.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

BASE_C, ARM_C, EDGE_C, LANE_C, AGENT_C, HIT_C = (
    "#767b84", "#2a78d6", "#1b1b1b", "#c9c7c1", "#9a9893", "#d64550")


def corners(x, y, yaw, l, w):
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([l, l, -l, -l]) / 2
    dy = np.array([w, -w, -w, w]) / 2
    return np.stack([x + c * dx - s * dy, y + s * dx + c * dy], -1)       # [4, 2]


def sat_overlap(a, b):
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def agent_at(ag, t):
    """(x, y, yaw) of a logged agent at absolute step t, or None outside its log."""
    ts = np.asarray(ag["t"], float)
    if t < ts[0] or t > ts[-1]:
        return None
    xy = np.asarray(ag["xy"])
    yaw = np.asarray(ag["yaw"])
    return (np.interp(t, ts, xy[:, 0]), np.interp(t, ts, xy[:, 1]),
            yaw[int(np.argmin(np.abs(ts - t)))])


def polylines(pts, gap=1.5):
    """Split an ordered, cropped point list into polylines at jumps > gap metres."""
    p = np.asarray(pts, float)
    if len(p) < 2:
        return []
    cut = np.nonzero(np.linalg.norm(np.diff(p, axis=0), axis=1) > gap)[0] + 1
    return [s for s in np.split(p, cut) if len(s) > 1]


def episode(npz, key, col):
    """[replans, steps, 5] -> concatenated [T, 5] executed path, NaN-trimmed."""
    tr = npz[key][col].reshape(-1, 5)
    return tr[~np.isnan(tr[:, 0])]


def critical_step(path, start, sc, lw, cause):
    """Index into ``path`` where it violates (see module docstring); ``nogoal`` = its
    last pose."""
    l, w = lw
    if cause == "nogoal":
        return len(path) - 1
    if cause == "overlap":
        best, best_d = None, np.inf
        for i, (x, y, yaw) in enumerate(path[:, :3]):
            ego = corners(x, y, yaw, l, w)
            for ag in sc["agents"]:
                p = agent_at(ag, start + i)
                if p is None:
                    continue
                if sat_overlap(ego, corners(*p, ag["l"], ag["w"])):
                    return i
                d = np.hypot(p[0] - x, p[1] - y)
                if d < best_d:
                    best, best_d = i, d
        return best
    edges = np.asarray(sc["edges"], float)
    d = [np.min(np.linalg.norm(corners(x, y, yaw, l, w)[:, None] - edges[None], axis=-1))
         for x, y, yaw in path[:, :3]]
    return int(np.argmin(d))


def label(r):
    if r is None:
        return "?"
    if r["solved"]:
        return "solved"
    return " + ".join(n for n, hit in (("collision", r["overlap"]), ("off-road", r["offroad"]),
                                       ("no goal", not r["reached"])) if hit)


def draw_map(ax, sc, box):
    for pl in polylines(sc["lanes"], gap=4.0):
        ax.plot(pl[:, 0], pl[:, 1], color=LANE_C, lw=0.6, zorder=1)
    for pl in polylines(sc["edges"]):
        ax.plot(pl[:, 0], pl[:, 1], color=EDGE_C, lw=1.1, zorder=2)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def draw_panel(ax, sc, path, color, start, t_star, lw, box, zoom, cf=None, hit_agent=None):
    draw_map(ax, sc, box)
    if not zoom:
        for ag in sc["agents"]:
            xy = np.asarray(ag["xy"])
            ax.plot(xy[:, 0], xy[:, 1], color=AGENT_C, lw=0.6, alpha=0.5, zorder=2)
        g = sc["goal"]
        ax.plot(g[0], g[1], marker="*", ms=14, color="#e8a317", mec="k", mew=0.6, zorder=6)
    if cf is not None:
        for seg in cf:
            ax.plot(seg[:, 0], seg[:, 1], color=BASE_C, lw=1.4, ls="--", zorder=3)
    ax.plot(path[:, 0], path[:, 1], color=color, lw=2.2, zorder=4)
    T = start + t_star
    for ag in sc["agents"]:
        p = agent_at(ag, T)
        if p is None or not (box[0] - 5 < p[0] < box[1] + 5 and box[2] - 5 < p[1] < box[3] + 5):
            continue
        hit = ag is hit_agent
        ax.add_patch(Polygon(corners(*p, ag["l"], ag["w"]), closed=True, zorder=5,
                             fc=HIT_C if hit else "#e6e5e1", ec="k", lw=0.7, alpha=0.9))
    steps = [t_star] if not zoom else [max(t_star - 10, 0), max(t_star - 5, 0), t_star]
    for k, i in enumerate(steps):
        i = min(i, len(path) - 1)
        last = k == len(steps) - 1
        ax.add_patch(Polygon(corners(*path[i, :3], *lw), closed=True, zorder=6,
                             fc=color, ec="k", lw=0.8 if last else 0.4,
                             alpha=0.95 if last else 0.3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes_json", required=True)
    p.add_argument("--select_dir", type=Path, required=True)
    p.add_argument("--arm", default="o_lpA3")
    p.add_argument("--base", default="base")
    p.add_argument("--ids", default="", help="comma scene ids (required in --mode fixed)")
    p.add_argument("--rep", type=int, default=0, help="--mode fixed: the repeat drawn")
    p.add_argument("--mode", choices=("fixed", "failing"), default="fixed")
    p.add_argument("--reps", type=int, default=4, help="paired repeats in the run")
    p.add_argument("--ego_lw", default="4.8,2.1", help="nominal ego length,width (m)")
    p.add_argument("--zoom_m", type=float, default=12.0)
    p.add_argument("--out_dir", type=Path, required=True)
    a = p.parse_args()
    lw = tuple(float(v) for v in a.ego_lw.split(","))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {s["scene_id"]: s for s in json.load(open(a.scenes_json))["scenes"]}
    pidx = {}
    for f in sorted(glob.glob(str(a.select_dir / "tasks/*/feat/*.npz"))):
        for j, sid in enumerate(np.load(f)["scene_ids"]):
            pidx.setdefault(int(sid), (f, j))
    rows = {}
    for f in glob.glob(str(a.select_dir / "tasks/*/rows.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["arm"] in (a.base, a.arm):
                # termination-aware flags where the run has them
                for k in ("solved", "offroad", "overlap", "reached"):
                    r[k] = r.get(k + "_term", r[k])
                rows[(int(r["scene_id"]), r["arm"], int(r["repeat"]))] = r

    failing = a.mode == "failing"
    ids = [int(x) for x in a.ids.split(",") if x.strip()]
    if not ids:
        if not failing:
            raise SystemExit("--mode fixed needs --ids")
        ids = sorted(bundle)
    jobs = []
    for sid in ids:
        if not failing:
            jobs.append((sid, a.rep))
            continue
        bad = [rp for rp in range(a.reps)
               if (sid, a.arm, rp) in rows and not rows[(sid, a.arm, rp)]["solved"]]
        if bad:
            jobs.append((sid, max(bad, key=lambda rp: (rows[(sid, a.arm, rp)].get("n_switch", 0), -rp))))
    if failing:
        print(f"[fix_pairs] failing mode: {len(jobs)} scenes with an unsolved {a.arm} repeat",
              flush=True)

    for sid, rep in jobs:
        if sid not in bundle or sid not in pidx:
            print(f"[fix_pairs] scene {sid}: not in bundle or run, skipped", flush=True)
            continue
        sc, (f, col) = bundle[sid], pidx[sid]
        npz = np.load(f)
        pb = episode(npz, f"{a.base}/r{rep}/traj", col)
        pa = episode(npz, f"{a.arm}/r{rep}/traj", col)
        # base proposal at each replan of the arm's episode, kept where it differs
        ex = npz[f"{a.arm}/r{rep}/traj"][col]
        cfb = npz[f"{a.arm}/r{rep}/traj_base"][col]
        cf = []
        for e, c in zip(ex, cfb):
            ok = ~np.isnan(e[:, 0]) & ~np.isnan(c[:, 0])
            if ok.sum() > 1 and np.max(np.linalg.norm(e[ok, :2] - c[ok, :2], axis=1)) > 0.05:
                cf.append(c[ok, :2])
        rb, ra = rows.get((sid, a.base, rep)), rows.get((sid, a.arm, rep))
        # The violation the zoom is about: the base's (fixed) or the arm's (failing).
        rf, pf = (ra, pa) if failing else (rb, pb)
        cause = ("overlap" if rf and rf["overlap"] else "offroad" if rf and rf["offroad"]
                 else "nogoal" if failing else "offroad")
        start = int(sc["start_step"])
        ts = critical_step(pf, start, sc, lw, cause)
        if ts is None:
            ts = len(pf) - 1
        hit = None
        if cause == "overlap":
            ego = corners(*pf[ts, :3], *lw)
            for ag in sc["agents"]:
                q = agent_at(ag, start + ts)
                if q is not None and sat_overlap(ego, corners(*q, ag["l"], ag["w"])):
                    hit = ag
                    break
        allxy = np.concatenate([pb[:, :2], pa[:, :2], np.asarray(sc["goal"])[None]])
        pad = 8.0
        full = (allxy[:, 0].min() - pad, allxy[:, 0].max() + pad,
                allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
        cx, cy = pf[ts, 0], pf[ts, 1]
        z = (cx - a.zoom_m, cx + a.zoom_m, cy - a.zoom_m, cy + a.zoom_m)

        fig, axs = plt.subplots(2, 2, figsize=(11, 10.5), facecolor="white",
                                gridspec_kw={"height_ratios": [1, 1]})
        for row, (box, zoom) in enumerate(((full, False), (z, True))):
            draw_panel(axs[row, 0], sc, pb, BASE_C, start, ts, lw, box, zoom,
                       hit_agent=None if failing else hit)
            draw_panel(axs[row, 1], sc, pa, ARM_C, start, ts, lw, box, zoom, cf=cf,
                       hit_agent=hit if failing else None)
            if not zoom:
                for ax in axs[row]:
                    ax.add_patch(plt.Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2],
                                               fill=False, ec="k", lw=0.8, ls=":", zorder=7))
        what = "first box overlap" if cause == "overlap" and hit is not None else (
            "closest approach to another car" if cause == "overlap" else
            "last pose, goal not reached" if cause == "nogoal" else
            "closest approach to a road edge")
        if failing:
            what = f"{a.arm}'s {what}"
        axs[0, 0].set_title(f"base policy: {label(rb)}", fontsize=13, color=HIT_C
                            if rb and not rb["solved"] else "k")
        axs[0, 1].set_title(f"{a.arm}: {label(ra)}  ({ra['n_switch'] if ra else '?'} repairs "
                            f"executed)", fontsize=13,
                            color=HIT_C if ra and not ra["solved"] else "k")
        axs[1, 0].set_title(f"zoom at t = {ts / 10:.1f} s ({what})", fontsize=10)
        axs[1, 1].set_title(f"same instant; dashed = base proposal the repair replaced",
                            fontsize=10)
        fig.suptitle(f"scene {sid} ({sc['split'].replace('_heldout', '')} split), "
                     f"paired draw: repeat {rep}", fontsize=14, y=0.995)
        fig.text(0.5, 0.005, f"black = road edge, grey = lane centres and other cars' paths, "
                 f"star = goal; boxes at the zoom instant (red = car hit). Ego box nominal "
                 f"{lw[0]}x{lw[1]} m; outcomes from the run's rows.", ha="center",
                 fontsize=8.5, color="#52514e")
        fig.tight_layout(rect=(0, 0.015, 1, 0.98))
        out = a.out_dir / f"{'fail' if failing else 'fix'}_{sid}_r{rep}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[fix_pairs] scene {sid}: base {label(rb)} / {a.arm} {label(ra)}, "
              f"t*={ts} ({what}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
