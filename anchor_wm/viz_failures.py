"""Where Diffusion-ES + AnchorWM fails although the logged expert does not (notebooks/anchor_wm_failures.ipynb).

  scene_table(runs, expert_run)   one row per scene: each closed-loop run's and the expert's (log replay) outcome
  run_checkpoints(run)            the AnchorWM forecaster / scorer checkpoints a closed-loop run used
  write_targets(keys, path)       targets json for anchor_wm/slurm/babel/es_dumps.sh (re-run with per-replan dumps)
  load_scene(out_dir)             one re-run scene: its dumps (anchor_wm.es_debug) and result
  replan_table(scene)             per replan: chosen plan vs the expert (score, log-replay labels, goal distance)
                                  and whether the final ES pool held a safe plan
  animate(scene)                  executed ego and chosen plans vs the logged expert, 0.1 s per frame
  replan_figure(scene, t)         ES pools coloured by score, scores against labels, the decoded ego anchors

Labels are log replay over a plan's 5.1 s (other agents follow the log): `overlap` / `offroad` = the first
violation of a plan, `safe` = neither within the plan's horizon (or before the episode ends).
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, colors
from matplotlib.patches import Polygon

from anchor_wm.config import DYNAMIC_TYPES, ROAD_EDGE_TYPES
from anchor_wm.data import LANE_TYPES

SURFACE, EDGE, LANE, AGENT, INK2 = "#fcfcfb", "#3a3a38", "#d6d5d0", "#b9b8b3", "#52514e"
CHOSEN, EXEC, EGO, EXPERT, FAIL = "#eb6834", "#0b0b0b", "#2a78d6", "#7a7974", "#c62828"
POP_CMAP = colors.LinearSegmentedColormap.from_list("pop", ["#cfe0f5", "#0d3b75"])   # one hue, light = low score
LABEL_STYLE = {"safe": ("#2a78d6", "o"), "overlap": ("#c62828", "X"), "offroad": ("#d99a00", "s")}
STEP_S = 0.1


# ---- closed-loop results -----------------------------------------------------------------------------------

def outcome(r: dict | None) -> str | None:
    if r is None:
        return None
    return "success" if r["success"] else "overlap" if r["overlap"] else "offroad" if r["offroad"] else "goal_miss"


def fail_time_s(r: dict | None) -> float | None:
    if r is None:
        return None
    steps = [t for t in (r.get("overlap_timestep", -1), r.get("offroad_timestep", -1)) if t is not None and t >= 0]
    return round(min(steps) * STEP_S, 1) if steps else None


def load_results(run: str) -> dict[str, dict]:
    """<tfrecord>.scenario_<idx>.json name -> result, over a run's chunks (or a flat run dir)."""
    files = glob.glob(f"{run}/chunk_*/*.scenario_*.json") + glob.glob(f"{run}/*.scenario_*.json")
    return {os.path.basename(f): json.load(open(f)) for f in files}


def scene_name(key: str) -> str:
    tf, idx = key.split(".scenario_")
    return f"{tf.split('-of-')[0][-5:]}_s{int(idx.split('.')[0]):03d}"


def scene_table(runs: dict[str, str], expert_run: str) -> pd.DataFrame:
    """Scenes every run (and the expert) finished: `<run>` outcome and `<run>_t` failure time (s) per run."""
    res = {name: load_results(d) for name, d in runs.items()}
    expert = load_results(expert_run)
    keys = sorted(set(expert).intersection(*map(set, res.values())))
    rows = []
    for k in keys:
        row = {"scene": scene_name(k), "key": k, "expert": outcome(expert[k]), "expert_t": fail_time_s(expert[k])}
        for name, r in res.items():
            row[name], row[f"{name}_t"] = outcome(r[k]), fail_time_s(r[k])
        rows.append(row)
    return pd.DataFrame(rows).set_index("scene")


def run_checkpoints(run: str) -> tuple[str, str]:
    """(forecaster, scorer) checkpoints from a closed-loop run's anchor_scorer_config.json."""
    cfg = json.load(open(sorted(glob.glob(f"{run}/chunk_*/anchor_scorer_config.json") +
                                glob.glob(f"{run}/anchor_scorer_config.json"))[0]))
    return cfg.get("planner_world_model_ckpt") or cfg["checkpoint"], cfg["checkpoint"]


def write_targets(keys: list[str], path: str) -> None:
    targets: dict[str, list[int]] = {}
    for k in keys:
        tf, idx = k.split(".scenario_")
        targets.setdefault(tf, []).append(int(idx.split(".")[0]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(targets, open(path, "w"), indent=1)


# ---- one re-run scene with dumps ---------------------------------------------------------------------------

def load_scene(out_dir: str) -> dict:
    """A one-scene es_dumps.sh run: per-replan dumps (batch 0), result, executed path, expert log."""
    steps = {int(os.path.basename(f)[6:-4]): dict(np.load(f)) for f in sorted(glob.glob(f"{out_dir}/dumps/b000_t*.npz"))}
    if not steps:
        raise FileNotFoundError(f"no dumps in {out_dir}/dumps")
    results = list(load_results(out_dir).values())
    first = steps[min(steps)]
    ego = int(np.argmax(first["object_metadata_is_sdc"][0]))
    full = {k[len("full_log_trajectory_"):]: first[k][0] for k in first if k.startswith("full_log_trajectory_")}
    scene = {"steps": steps, "first": first, "result": results[0] if results else None, "ego": ego, "full": full,
             "goal": np.asarray(first["goal"]).reshape(-1)[:2],
             "agents": np.isin(first["object_metadata_object_types"][0], DYNAMIC_TYPES)}
    scene["exe"], scene["exe_t"] = executed_path(steps)
    scene["failure"] = failure(scene["result"])
    return scene


def executed_path(steps: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ego world [x, y, yaw] at every step from the first replan on: each chosen plan until the next replan."""
    ts = sorted(steps)
    xs, t_of = [], []
    for j, t in enumerate(ts):
        plan = steps[t]["best_world"][0]                                       # [52, 5] t .. t+5.1 s
        n = min((ts[j + 1] - t) if j + 1 < len(ts) else plan.shape[0], plan.shape[0])
        xs.append(plan[:n, :3])
        t_of.append(t + np.arange(n))
    return np.concatenate(xs), np.concatenate(t_of)


def failure(res: dict | None) -> tuple[str, int]:
    if res is None:
        return "unknown", -1
    steps = [(res.get("overlap_timestep", -1), "overlap"), (res.get("offroad_timestep", -1), "offroad")]
    steps = [(t, k) for t, k in steps if t is not None and t >= 0]
    if steps:
        t, k = min(steps)
        return k, int(t)
    return ("goal_miss" if not res["goal_reached"] else "success"), -1


def labels(z: dict, name: str) -> np.ndarray:
    """'safe' / 'overlap' / 'offroad' per trajectory of a labelled set (init, final, best, expert_aligned)."""
    ov, off = z[f"{name}_overlap_step"][0], z[f"{name}_offroad_step"][0]
    first_ov = np.where(ov > 0, ov, 10 ** 4)
    first_off = np.where(off > 0, off, 10 ** 4)
    return np.where((ov == 0) & (off == 0), "safe", np.where(first_ov <= first_off, "overlap", "offroad"))


def replan_table(scene: dict) -> pd.DataFrame:
    """Per replan. `verdict`: chosen plan safe -> ok; else a safe plan in the final pool -> scorer (it ranked an
    unsafe plan first); else one in the initial pool only -> ES dropped it; else no safe proposal."""
    rows = []
    for t, z in sorted(scene["steps"].items()):
        sc = z["final_scores"][0]
        lab_f, lab_i, chosen = labels(z, "final"), labels(z, "init"), labels(z, "best")[0]
        safe_f = lab_f == "safe"
        row = {"t_s": t * STEP_S, "chosen": chosen, "chosen_score": float(sc.max()),
               "best_safe_score": float(sc[safe_f].max()) if safe_f.any() else np.nan,
               "safe_final": int(safe_f.sum()), "safe_init": int((lab_i == "safe").sum()), "pool": len(sc),
               "expert": labels(z, "expert_aligned")[0], "expert_offset_m": float(z["expert_offset_m"][0])}
        if "expert_score" in z:
            es = float(z["expert_score"][0])
            row.update(expert_score=es, expert_rank=int(1 + (sc > es).sum()))
        end = lambda p: float(np.linalg.norm(p[-1, :2] - scene["goal"]))   # noqa: E731
        row.update(chosen_goal_m=end(z["best_world"][0]), expert_goal_m=end(z["expert_aligned"][0]),
                   top_anchor_p=float(z["awm_ego_probs"][0].max()))
        row["verdict"] = ("ok" if chosen == "safe" else "scorer" if safe_f.any()
                          else "ES dropped safe" if row["safe_init"] else "no safe proposal")
        rows.append(row)
    return pd.DataFrame(rows).set_index("t_s").round(2)


# ---- drawing ------------------------------------------------------------------------------------------------

def to_frame(xy, origin, yaw):
    d = np.asarray(xy, np.float64) - origin
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)


def box(ax, xy, yaw, size, **kw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * (np.asarray(size, np.float64) / 2)
    ax.add_patch(Polygon(pts @ np.array([[c, s], [-s, c]]) + xy, closed=True, **kw))


def draw_road(ax, first, origin, yaw, scale=1.0):
    v = first["roadgraph_points_valid"][0].astype(bool)
    types = first["roadgraph_points_types"][0]
    pts = to_frame(np.stack([first["roadgraph_points_x"][0], first["roadgraph_points_y"][0]], -1), origin, yaw)
    for tset, col, sz in ((LANE_TYPES, LANE, 0.5), (ROAD_EDGE_TYPES, EDGE, 1.3)):
        m = v & np.isin(types, tset)
        ax.scatter(pts[m, 0], pts[m, 1], s=sz * scale, c=col, linewidths=0, zorder=1)


def draw_agents(ax, scene, t, origin, yaw, future_s=0.0):
    full = scene["full"]
    for i in np.flatnonzero(scene["agents"]):
        if i == scene["ego"] or not full["valid"][i, t]:
            continue
        xy = to_frame(np.array([full["x"][i, t], full["y"][i, t]]), origin, yaw)
        box(ax, xy, full["yaw"][i, t] - yaw, (full["length"][i, t], full["width"][i, t]), fc=AGENT, ec="#6f6e69",
            lw=0.4, zorder=3)
        if future_s > 0:
            t1 = min(full["x"].shape[1], t + int(future_s / STEP_S) + 1)
            m = full["valid"][i, t:t1].astype(bool)
            fut = to_frame(np.stack([full["x"][i, t:t1][m], full["y"][i, t:t1][m]], -1), origin, yaw)
            ax.plot(fut[:, 0], fut[:, 1], color="#8f8e89", lw=0.6, alpha=0.7, zorder=2)


def expert_xy(scene, t0, t1):
    """The logged ego (world xy, yaw) over [t0, t1)."""
    f, e = scene["full"], scene["ego"]
    m = f["valid"][e, t0:t1].astype(bool)
    return np.stack([f["x"][e, t0:t1][m], f["y"][e, t0:t1][m]], -1), f["yaw"][e, t0:t1][m]


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, color=INK2)


def animate(scene: dict, fps: int = 10, view_m: float = 70.0) -> animation.FuncAnimation:
    """Map frame following the ego. Blue box + black path = executed; grey outline + dashed = the logged expert;
    orange = the plan chosen at the latest replan; red X = the failure."""
    steps, first, full, e = scene["steps"], scene["first"], scene["full"], scene["ego"]
    exe, exe_t = scene["exe"], scene["exe_t"]
    kind, t_fail = scene["failure"]
    ts = sorted(steps)
    size = (full["length"][e, ts[0]], full["width"][e, ts[0]])
    frames = list(range(ts[0], min(int(exe_t.max()), full["x"].shape[1] - 1) + 1))
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(SURFACE)
    zero = np.zeros(2)

    def draw(t):
        ax.clear()
        k = int(np.argmin(np.abs(exe_t - t)))
        c = exe[k, :2]
        draw_road(ax, first, zero, 0.0, scale=0.8)
        draw_agents(ax, scene, t, zero, 0.0)
        exp, _ = expert_xy(scene, ts[0], full["x"].shape[1])
        ax.plot(exp[:, 0], exp[:, 1], color=EXPERT, lw=1.2, ls=(0, (4, 3)), zorder=5)
        if full["valid"][e, t]:
            box(ax, (full["x"][e, t], full["y"][e, t]), full["yaw"][e, t], size, fc="none", ec=EXPERT, lw=1.4,
                ls="--", zorder=8)
        r = max(x for x in ts if x <= t)
        plan = steps[r]["best_world"][0]
        ax.plot(plan[:, 0], plan[:, 1], color=CHOSEN, lw=2.2, zorder=6)
        done = exe[exe_t <= t]
        ax.plot(done[:, 0], done[:, 1], color=EXEC, lw=2.4, zorder=7)
        failed = 0 <= t_fail <= t
        box(ax, exe[k, :2], float(exe[k, 2]), size, fc=FAIL if failed else EGO, ec="k", lw=0.7, zorder=9)
        if failed:
            kf = int(np.argmin(np.abs(exe_t - t_fail)))
            ax.scatter([exe[kf, 0]], [exe[kf, 1]], marker="X", s=170, c=FAIL, ec="white", lw=0.8, zorder=10)
        ax.scatter(*scene["goal"], marker="*", s=150, c="#f2c400", ec="k", lw=0.6, zorder=9)
        ax.set_xlim(c[0] - view_m / 2, c[0] + view_m / 2)
        ax.set_ylim(c[1] - view_m / 2, c[1] + view_m / 2)
        status = f"{kind} at {t_fail * STEP_S:.1f} s" if t_fail >= 0 else kind
        _style(ax, f"t = {t * STEP_S:.1f} s (plan from the {r * STEP_S:.0f} s replan)   {status}\n"
                   "blue: executed ego · dashed grey: logged expert · orange: chosen plan")

    ani = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    plt.close(fig)
    return ani


def replan_figure(scene: dict, t: int, view=(-18, 55, -27, 27)):
    """Three panels at replan step t, ego frame: (1) final ES pool coloured by learned score, chosen plan, the
    expert path moved onto the ego, the executed second; (2) every initial / final candidate's score with its
    log-replay label, the chosen plan ringed, the expert's score dashed; (3) the AnchorWM decoded ego anchors
    (all, alpha ~ probability; the NMS top 6 with their probabilities)."""
    z, first = scene["steps"][t], scene["first"]
    k = int(np.argmin(np.abs(scene["exe_t"] - t)))
    origin, yaw = scene["exe"][k, :2].astype(np.float64), float(scene["exe"][k, 2])
    e = scene["ego"]
    size = (scene["full"]["length"][e, t], scene["full"]["width"][e, t])
    kind, t_fail = scene["failure"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.35, 1, 1.35]})
    fig.patch.set_facecolor(SURFACE)

    def scene_layer(ax, future_s):
        draw_road(ax, first, origin, yaw)
        draw_agents(ax, scene, t, origin, yaw, future_s=future_s)
        box(ax, np.zeros(2), 0.0, size, fc=EGO, ec="k", lw=0.6, zorder=8)
        g = to_frame(scene["goal"], origin, yaw)
        if np.all(np.abs(g) < 80):
            ax.scatter(*g, marker="*", s=130, c="#f2c400", ec="k", lw=0.6, zorder=9)
        ax.set_xlim(*view[:2])
        ax.set_ylim(*view[2:])

    def path(ax, p, **kw):
        q = to_frame(p[:, :2], origin, yaw)
        ax.plot(q[:, 0], q[:, 1], **kw)

    expert = z["expert_aligned"][0]
    # (1) population by score
    ax = axes[0]
    scene_layer(ax, 5.0)
    pop, sc = z["final_world"][0], z["final_scores"][0]
    norm = colors.Normalize(vmin=float(sc.min()), vmax=float(sc.max()))
    for j in np.argsort(sc):
        path(ax, pop[j], color=POP_CMAP(norm(sc[j])), lw=0.8, zorder=4)
    path(ax, expert, color=EXPERT, lw=1.6, ls=(0, (4, 3)), zorder=5)
    path(ax, z["best_world"][0], color=CHOSEN, lw=2.4, zorder=6, solid_capstyle="round")
    seg = scene["exe"][(scene["exe_t"] >= t) & (scene["exe_t"] <= t + 10)]
    path(ax, seg, color=EXEC, lw=3.2, zorder=7, solid_capstyle="round")
    lead = f"   {kind} in {(t_fail - t) * STEP_S:.1f} s" if t_fail >= t else ""
    _style(ax, f"t = {t * STEP_S:.0f} s: final pool by score (dark = high){lead}\n"
               "orange: chosen · black: executed 1 s · dashed: expert moved onto the ego")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=POP_CMAP), ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=7)

    # (2) scores vs labels
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for row, name in ((1, "init"), (0, "final")):
        scores = z[f"{name}_scores"][0]
        lab = labels(z, name)
        for lname, (col, mk) in LABEL_STYLE.items():
            m = lab == lname
            ax.scatter(scores[m], row + rng.uniform(-0.18, 0.18, m.sum()), c=col, marker=mk, s=26, lw=0,
                       alpha=0.85, label=f"{lname}" if row == 0 else None, zorder=3)
    best = float(sc.max())
    ax.scatter([best], [0], s=230, facecolors="none", edgecolors=CHOSEN, lw=2, zorder=4, label="chosen")
    if "expert_score" in z:
        ax.axvline(float(z["expert_score"][0]), color=EXPERT, ls="--", lw=1.4,
                   label=f"expert ({labels(z, 'expert_aligned')[0]})")
    ax.set_yticks([0, 1], ["final pool", "initial pool"])
    ax.set_xlabel("learned score")
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=5)
    ax.set_title("scores vs log-replay labels (5 s)", fontsize=9.5, color=INK2)

    # (3) decoded ego anchors
    ax = axes[2]
    scene_layer(ax, 0.0)
    all_xy, all_p = z["awm_ego_all_xy"][0], z["awm_ego_all_probs"][0]
    alpha = 0.08 + 0.9 * all_p / max(float(all_p.max()), 1e-9)
    for j in np.argsort(all_p):
        q = to_frame(all_xy[j], origin, yaw)
        ax.plot(q[:, 0], q[:, 1], color="#0d3b75", lw=0.7, alpha=float(alpha[j]), zorder=4)
    for xy, p in zip(z["awm_ego_xy"][0], z["awm_ego_probs"][0]):
        q = to_frame(xy, origin, yaw)
        ax.plot(q[:, 0], q[:, 1], color="#0d3b75", lw=1.8, zorder=5)
        if view[0] < q[-1, 0] < view[1] and view[2] < q[-1, 1] < view[3]:
            ax.text(q[-1, 0], q[-1, 1], f"{p:.2f}", fontsize=7.5, color=INK2, zorder=9)
    path(ax, expert, color=EXPERT, lw=1.6, ls=(0, (4, 3)), zorder=6)
    path(ax, z["best_world"][0], color=CHOSEN, lw=1.8, zorder=6)
    _style(ax, "AnchorWM decoded ego anchors (alpha ~ probability; bold = NMS top 6 with p)\n"
               "orange: chosen · dashed: expert")
    fig.tight_layout()
    return fig
