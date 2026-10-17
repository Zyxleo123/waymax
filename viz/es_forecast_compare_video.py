#!/usr/bin/env python3
"""Side-by-side videos of Diffusion-ES with GT / CV / WM other-agent forecasts.

Scenes: validation failures (shards 00/01) of es_diffusion_{gt,cv,wm10}, after dropping
scenes where the logged GT trajectory itself fails. ``--scene_set union`` (default)
takes every scene that fails for at least one forecast; ``cv`` / ``gt`` / ``wm`` take
only that run's failures.

Phase ``run`` (GPU) re-simulates each scene's original 5-scene batch three times, once
per forecast, with the same seed and RNG path as simulation/runner.py, recording at each
replan the ES pool (collision / offroad verdict of every candidate), the elites, the
executed plan, and the other-agent forecast the scorer used.

Phase ``render`` (CPU) writes per scene one mp4 with three panels (GT | CV | WM) that
play the closed loop together, a snapshot png, and an ``index.html``.

Example::

  python viz/es_forecast_compare_video.py --phase all --max_batches 1
  sbatch viz/run_es_forecast_compare_video.sh
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Running as `python viz/<this file>` puts viz/ on sys.path, where viz/viz.py would shadow the viz package.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRATCH = "/zfsauton/scratch/sbellad"
ES_ROOT = f"{SCRATCH}/Diffusion_ES_sim"
DEFAULT_VAL_RESULT_DIRS = {
    "gt": f"{ES_ROOT}/es_diffusion_gt/simulation_es_validation_es_diffusion_gt_VAL_tfrecord00_01_20260907_203053",
    "cv": f"{ES_ROOT}/es_diffusion_cv/simulation_es_validation_es_diffusion_cv_VAL_tfrecord00_01_20260907_171131",
    "wm": f"{ES_ROOT}/es_diffusion_wm10/simulation_es_validation_es_diffusion_wm10_VAL_tfrecord00_01_20260916_164314",
}
DEFAULT_GT_EVAL = f"{ES_ROOT}/gt_filtered_es_metrics_20260908_114522/gt_filter/validation_gt_eval_all.json"
DEFAULT_WOMD_ROOT = "/zfsauton/datasets/womd/tf_example"
DEFAULT_LANE_GRAPH_DIR = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DEFAULT_DIFFUSION_CKPT = f"{SCRATCH}/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
DEFAULT_WM_CKPT = f"{SCRATCH}/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt"
DEFAULT_OUT_ROOT = f"{SCRATCH}/viz"

MODES = ("gt", "cv", "wm")
MODE_LABEL = {"gt": "GT forecast", "cv": "CV forecast", "wm": "WM forecast (det., refresh 10)"}

C_GT_FUT = "#1b9e77"
C_FORECAST = {"gt": C_GT_FUT, "cv": "#2c6fdb", "wm": "#ff7f00"}
C_EGO = "#e7298a"
C_FEAS = "#b3b3e6"
C_COLL = "#e41a1c"
C_OFF = "#a65628"
C_BOTH = "#7f0000"
C_ELITE = "#54278f"
C_AGENT = "#9a9a9a"
C_NEIGHBOR = "#5a5a5a"
DYNAMIC_TYPES = (1, 2, 3)

_LANE_TYPES = (1, 2, 3)
_WHITE_LINES = (5, 6, 7, 8)
_YELLOW_LINES = (9, 10, 11, 12, 13)
_EDGE_TYPES = (14, 15, 16)
_CROSSWALK = (18,)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["run", "render", "all"], default="all")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--scene_set", choices=["union", "gt", "cv", "wm"], default="union")
    p.add_argument("--gt_result_dir", type=str, default=DEFAULT_VAL_RESULT_DIRS["gt"])
    p.add_argument("--cv_result_dir", type=str, default=DEFAULT_VAL_RESULT_DIRS["cv"])
    p.add_argument("--wm_result_dir", type=str, default=DEFAULT_VAL_RESULT_DIRS["wm"])
    p.add_argument("--gt_eval_json", type=str, default=DEFAULT_GT_EVAL)
    p.add_argument("--womd_root", type=str, default=DEFAULT_WOMD_ROOT)
    p.add_argument("--lane_graph_dir", type=str, default=DEFAULT_LANE_GRAPH_DIR)
    p.add_argument("--diffusion_ckpt_path", type=str, default=DEFAULT_DIFFUSION_CKPT)
    p.add_argument("--world_model_ckpt", type=str, default=DEFAULT_WM_CKPT)
    p.add_argument("--world_model_device", type=str, default="auto")
    p.add_argument("--wm_refresh_steps", type=int, default=10)
    p.add_argument("--num_worlds", type=int, default=5, help="Must match the original runs for exact RNG.")
    p.add_argument("--start_timestep", type=int, default=10)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--population_size", type=int, default=64)
    p.add_argument("--elite_size", type=int, default=8)
    p.add_argument("--num_es_iterations", type=int, default=4)
    p.add_argument("--resample_timesteps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0, help="0 = all batches.")
    p.add_argument("--view_half_m", type=float, default=32.0, help="Half-width of each panel (m).")
    p.add_argument("--neighbor_radius_m", type=float, default=35.0, help="Forecasts drawn for dynamic agents within this radius.")
    p.add_argument("--fps", type=int, default=5, help="Sim runs at 10 Hz; 5 fps = half speed.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Scene selection
# ----------------------------------------------------------------------------


def _key(tfrecord: str, idx: int) -> tuple[str, int]:
    return Path(tfrecord).name, int(idx)


def load_scenes(args: argparse.Namespace) -> list[dict[str, Any]]:
    gt_eval = {_key(r["tfrecord"], r["scenario_idx"]): r for r in json.loads(Path(args.gt_eval_json).read_text())}
    gt_fail = {k for k, r in gt_eval.items() if not r["success"]}
    dirs = {"gt": args.gt_result_dir, "cv": args.cv_result_dir, "wm": args.wm_result_dir}
    orig: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for mode, d in dirs.items():
        for f in Path(d).glob("*.scenario_*.json"):
            r = json.loads(f.read_text())
            orig[_key(r["tfrecord"], r["scenario_idx"])][mode] = {
                k: r[k]
                for k in ("success", "overlap", "offroad", "goal_reached", "reached_timestep", "min_goal_distance_m")
            }
    scenes = []
    for k, per_mode in sorted(orig.items()):
        if k in gt_fail or len(per_mode) < len(MODES):
            continue
        fails = [m for m in MODES if not per_mode[m]["success"]]
        if not fails or (args.scene_set != "union" and args.scene_set not in fails):
            continue
        tf_name, idx = k
        scenes.append(
            {
                "tfrecord": tf_name,
                "scenario_idx": idx,
                "stem": f"val_{tf_name.split('tfrecord-')[1][:5]}_sc{idx:03d}",
                "orig": per_mode,
                "fails_in": fails,
            }
        )
    return scenes


def group_into_batches(scenes: list[dict[str, Any]], num_worlds: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for s in scenes:
        groups[(s["tfrecord"], (s["scenario_idx"] // num_worlds) * num_worlds)].append(s)
    return [{"tfrecord": k[0], "batch_start": k[1], "targets": v} for k, v in sorted(groups.items())]


# ----------------------------------------------------------------------------
# Phase 1: re-simulation (GPU)
# ----------------------------------------------------------------------------


def _outcome(ev: dict[str, Any]) -> str:
    if ev["success"]:
        return f"success (goal at t={ev['reached_timestep']})"
    parts = []
    if ev.get("overlap"):
        t = ev.get("first_overlap_t", -1)
        parts.append("collision" + (f" at t={t}" if t is not None and t >= 0 else ""))
    if ev.get("offroad"):
        t = ev.get("first_offroad_t", -1)
        parts.append("offroad" + (f" at t={t}" if t is not None and t >= 0 else ""))
    if not ev["goal_reached"]:
        parts.append(f"goal not reached ({ev['min_goal_distance_m']:.1f} m short)")
    return "FAIL: " + ", ".join(parts)


def run_batch(args, planner, wm_forecaster, batch: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    from simulation.evaluation_utils import infer_goals_from_sim_state
    from viz.es_recording import attach_lane_graphs, evaluate_batch, load_batch, simulate_batch

    nw = int(args.num_worlds)
    tf_path = Path(args.womd_root) / "validation" / batch["tfrecord"]
    indices = list(range(int(batch["batch_start"]), int(batch["batch_start"]) + nw))
    sim_state, scenario_ids = load_batch(str(tf_path), indices, max_num_objects=int(args.max_num_objects))
    attach_lane_graphs(planner, sim_state, str(tf_path), indices, args.lane_graph_dir)
    goal_xy_b2, _, ego_idx_b = infer_goals_from_sim_state(sim_state)
    targets = {int(s["scenario_idx"]): s for s in batch["targets"]}
    start = int(args.start_timestep)

    for w, idx in enumerate(indices):
        if idx not in targets:
            continue
        rg = sim_state.roadgraph_points
        lt = sim_state.log_trajectory
        rg_valid = np.asarray(rg.valid[w]).astype(bool)
        np.savez_compressed(
            data_dir / f"{targets[idx]['stem']}_scene.npz",
            rg_xy=np.asarray(rg.xy[w], dtype=np.float32)[rg_valid],
            rg_types=np.asarray(rg.types[w], dtype=np.int32)[rg_valid],
            rg_ids=np.asarray(rg.ids[w], dtype=np.int32)[rg_valid],
            obj_xy=np.asarray(lt.xy[w], dtype=np.float32),
            obj_yaw=np.asarray(lt.yaw[w], dtype=np.float32),
            obj_length=np.asarray(lt.length[w], dtype=np.float32),
            obj_width=np.asarray(lt.width[w], dtype=np.float32),
            obj_valid=np.asarray(lt.valid[w]).astype(bool),
            obj_types=np.asarray(sim_state.object_metadata.object_types[w], dtype=np.int32),
            ego_idx=np.int32(ego_idx_b[w]),
            goal_xy=np.asarray(goal_xy_b2[w], dtype=np.float32),
        )

    repro: dict[int, dict[str, Any]] = defaultdict(dict)
    for mode in MODES:
        wm = wm_forecaster if mode == "wm" else None
        planner.set_forecast(mode, wm)
        if wm is not None:
            wm.clear_cache()
            if hasattr(wm, "set_scene_ids"):
                wm.set_scene_ids(scenario_ids)
        replaced, records = simulate_batch(
            planner,
            sim_state,
            goal_xy_b2,
            start_timestep=start,
            replan_interval_steps=int(args.replan_interval_steps),
            seed=int(args.seed),
            record_forecasts=True,
        )
        evals = evaluate_batch(replaced, goal_xy_b2, start_timestep=start, seed=int(args.seed))
        rt = replaced.log_trajectory
        for w, idx in enumerate(indices):
            if idx not in targets:
                continue
            ego = int(ego_idx_b[w])
            ev = evals[w]
            payload: dict[str, Any] = {
                "exec_xy": np.asarray(rt.xy[w, ego], dtype=np.float32),
                "exec_yaw": np.asarray(rt.yaw[w, ego], dtype=np.float32),
                "overlap_ts": ev["overlap_ts"],
                "offroad_ts": ev["offroad_ts"],
                "replan_steps": np.asarray(sorted(records), dtype=np.int32),
            }
            for step, rec in records.items():
                pre = f"t{step:03d}"
                final = rec["stages"][-1]
                payload[f"{pre}_pool"] = final["traj"][w]
                payload[f"{pre}_coll"] = final["coll"][w]
                payload[f"{pre}_off"] = final["off"][w]
                payload[f"{pre}_elite"] = final["elite_idx"][w]
                payload[f"{pre}_best"] = np.int32(rec["best_idx"][w])
                payload[f"{pre}_chosen"] = rec["chosen"][w]
                payload[f"{pre}_horizon"] = np.int32(rec["horizon_scored"][w])
                payload[f"{pre}_feasible_per_iter"] = np.asarray(
                    [int((~st["coll"][w] & ~st["off"][w]).sum()) for st in rec["stages"]], dtype=np.int32
                )
                payload[f"{pre}_fc_xyyaw"] = rec["forecast_traj"][w][..., :3]
                payload[f"{pre}_fc_mask"] = rec["forecast_mask"][w]
            np.savez_compressed(data_dir / f"{targets[idx]['stem']}_{mode}.npz", **payload)
            repro[idx][mode] = {k: v for k, v in ev.items() if not k.endswith("_ts")}
            print(f"  sc{idx:03d} {mode}: {_outcome(repro[idx][mode])}", flush=True)

    rows = []
    for idx, s in targets.items():
        same = {
            m: all(
                bool(repro[idx][m][k]) == bool(s["orig"][m][k]) for k in ("overlap", "offroad", "goal_reached")
            )
            for m in MODES
        }
        rows.append({**s, "repro": repro[idx], "reproduced": same, "data_dir": str(data_dir)})
    return rows


def phase_run(args: argparse.Namespace, out_dir: Path) -> None:
    import tensorflow as tf

    # TF only parses TFRecords; left on the GPU it reserves nearly all memory and starves JAX.
    tf.config.set_visible_devices([], "GPU")

    from viz.es_recording import build_recording_planner, load_deterministic_world_model

    scenes = load_scenes(args)
    batches = group_into_batches(scenes, int(args.num_worlds))
    if int(args.max_batches) > 0:
        batches = batches[: int(args.max_batches)]
    print(f"[run] {len(scenes)} scenes ({args.scene_set}) in {len(batches)} batches -> {out_dir}", flush=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    wm = load_deterministic_world_model(
        args.world_model_ckpt, device=args.world_model_device, refresh_steps=int(args.wm_refresh_steps)
    )
    planner = build_recording_planner(
        diffusion_ckpt_path=args.diffusion_ckpt_path,
        seed=int(args.seed),
        population_size=int(args.population_size),
        elite_size=int(args.elite_size),
        num_iterations=int(args.num_es_iterations),
        resample_timesteps=int(args.resample_timesteps),
        num_worlds=int(args.num_worlds),
        other_agent_forecast="gt",
    )
    jsonl = out_dir / "per_scene.jsonl"
    done = set()
    if jsonl.exists():
        done = {json.loads(l)["stem"] for l in jsonl.read_text().splitlines() if l.strip() and "error" not in l}
    with jsonl.open("a") as f:
        for i, batch in enumerate(batches, start=1):
            if all(s["stem"] in done for s in batch["targets"]):
                continue
            print(
                f"[{i}/{len(batches)}] {batch['tfrecord']} batch {batch['batch_start']}.."
                f"{batch['batch_start'] + int(args.num_worlds) - 1} targets={[s['scenario_idx'] for s in batch['targets']]}",
                flush=True,
            )
            try:
                rows = run_batch(args, planner, wm, batch, data_dir)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                rows = [{**s, "error": f"{type(exc).__name__}: {exc}"} for s in batch["targets"]]
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()


# ----------------------------------------------------------------------------
# Phase 2: rendering (CPU)
# ----------------------------------------------------------------------------


def _box_corners(x, y, yaw, length, width) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([0.5, 0.5, -0.5, -0.5]) * length
    dy = np.array([0.5, -0.5, -0.5, 0.5]) * width
    return np.stack([x + dx * c - dy * s, y + dx * s + dy * c], axis=-1)


def _draw_box(ax, x, y, yaw, length, width, *, face, edge=None, alpha=0.9, lw=0.8, ls="-", z=10):
    from matplotlib.patches import Polygon

    corners = _box_corners(x, y, yaw, length, width)
    ax.add_patch(
        Polygon(corners, closed=True, facecolor=face, edgecolor=edge or face, alpha=alpha, lw=lw, ls=ls, zorder=z)
    )
    if face != "none":
        nose = 0.5 * (corners[0] + corners[1])
        ax.plot([x, nose[0]], [y, nose[1]], color="k", lw=0.6, alpha=alpha, zorder=z + 1)


def _draw_map(ax, sc: dict[str, np.ndarray], center: np.ndarray, half: float) -> None:
    from matplotlib.collections import LineCollection

    xy, types, ids = sc["rg_xy"], sc["rg_types"], sc["rg_ids"]
    lo, hi = center - half - 5.0, center + half + 5.0
    inside = np.all((xy >= lo) & (xy <= hi), axis=-1)
    a, b = xy[:-1], xy[1:]
    keep = (ids[:-1] == ids[1:]) & (inside[:-1] | inside[1:]) & (np.linalg.norm(b - a, axis=-1) < 5.0)
    seg_types = types[:-1]
    for type_set, color, lw, ls in (
        (_LANE_TYPES, "#d0d0d0", 0.6, (0, (4, 4))),
        (_WHITE_LINES, "#b0b0b0", 0.6, "-"),
        (_YELLOW_LINES, "#d9b44a", 0.7, "-"),
        (_CROSSWALK, "#9cc3eb", 0.8, "-"),
        (_EDGE_TYPES, "#202020", 1.3, "-"),
    ):
        m = keep & np.isin(seg_types, type_set)
        if m.any():
            ax.add_collection(
                LineCollection(np.stack([a[m], b[m]], axis=1), colors=[color], linewidths=lw, linestyles=ls, zorder=1)
            )


def _status_color(coll: bool, off: bool) -> str:
    if coll and off:
        return C_BOTH
    if coll:
        return C_COLL
    if off:
        return C_OFF
    return C_FEAS


def _latest_replan(steps: np.ndarray, t: int) -> int:
    past = steps[steps <= t]
    return int(past[-1]) if past.size else int(steps[0])


def _neighbors(sc, ego_xy: np.ndarray, t: int, radius: float) -> list[int]:
    ego = int(sc["ego_idx"])
    out = []
    for i in range(sc["obj_xy"].shape[0]):
        if i == ego or not sc["obj_valid"][i, t] or int(sc["obj_types"][i]) not in DYNAMIC_TYPES:
            continue
        if np.linalg.norm(sc["obj_xy"][i, t] - ego_xy) <= radius:
            out.append(i)
    return out


def draw_panel(ax, sc, md, mode: str, t: int, ev: dict[str, Any], args) -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    steps = md["replan_steps"]
    start = int(args.start_timestep)
    n_steps = md["exec_xy"].shape[0]
    t = int(np.clip(t, start, n_steps - 1))
    r = _latest_replan(steps, t)
    pre = f"t{r:03d}"
    h = int(md[f"{pre}_horizon"])
    ego_xy = md["exec_xy"][t]
    half = float(args.view_half_m)

    ax.cla()
    ax.set_facecolor("white")
    _draw_map(ax, sc, ego_xy, half)

    # Other agents (logged = GT motion) now.
    ego = int(sc["ego_idx"])
    neigh = set(_neighbors(sc, md["exec_xy"][r], r, float(args.neighbor_radius_m)))
    for i in range(sc["obj_xy"].shape[0]):
        if i == ego or not sc["obj_valid"][i, t]:
            continue
        x, y = sc["obj_xy"][i, t]
        if abs(x - ego_xy[0]) > half + 8 or abs(y - ego_xy[1]) > half + 8:
            continue
        _draw_box(
            ax, x, y, sc["obj_yaw"][i, t], sc["obj_length"][i, t], sc["obj_width"][i, t],
            face=C_NEIGHBOR if i in neigh else C_AGENT, alpha=0.75 if i in neigh else 0.4, z=6,
        )

    # Forecasts made at the latest replan for neighbouring dynamic agents.
    fc_xy, fc_mask = md[f"{pre}_fc_xyyaw"], md[f"{pre}_fc_mask"]
    hf = min(h, fc_xy.shape[1])
    k_now = t - r
    for i in neigh:
        sl = slice(r, min(n_steps, r + hf))
        v = sc["obj_valid"][i, sl]
        gt_pts = sc["obj_xy"][i, sl][v]
        if len(gt_pts) > 1:
            ax.plot(gt_pts[:, 0], gt_pts[:, 1], color=C_GT_FUT, lw=1.6, alpha=0.9, zorder=7)
        if mode == "gt":
            continue
        m = fc_mask[i, :hf]
        pts = fc_xy[i, :hf][m]
        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], color=C_FORECAST[mode], lw=1.6, ls=(0, (3, 2)), alpha=0.95, zorder=8)
        if k_now < hf and m[k_now] and sc["obj_valid"][i, t]:
            px, py, pyaw = fc_xy[i, k_now]
            _draw_box(
                ax, px, py, pyaw, sc["obj_length"][i, r], sc["obj_width"][i, r],
                face="none", edge=C_FORECAST[mode], alpha=1.0, lw=1.4, ls="--", z=9,
            )
            ax.plot([px, sc["obj_xy"][i, t, 0]], [py, sc["obj_xy"][i, t, 1]], color=C_FORECAST[mode], lw=0.8,
                    ls=":", zorder=9)

    # ES pool of the latest replan.
    pool = md[f"{pre}_pool"][:, :h, :2]
    coll, off = md[f"{pre}_coll"].astype(bool), md[f"{pre}_off"].astype(bool)
    elites = set(int(e) for e in md[f"{pre}_elite"])
    by_color: dict[str, list] = defaultdict(list)
    for k in range(len(pool)):
        if k not in elites:
            by_color[_status_color(bool(coll[k]), bool(off[k]))].append(pool[k])
    for col, segs in by_color.items():
        ax.add_collection(LineCollection(segs, colors=[col], linewidths=0.7, alpha=0.55, zorder=10))
    for k in elites:
        ax.plot(pool[k, :, 0], pool[k, :, 1], color=C_ELITE, lw=1.8, alpha=0.85, zorder=11)
        if coll[k] or off[k]:
            ax.plot(pool[k, :, 0], pool[k, :, 1], color=_status_color(bool(coll[k]), bool(off[k])), lw=0.8, zorder=12)
    chosen = md[f"{pre}_chosen"][:h]
    ax.plot(chosen[:, 0], chosen[:, 1], color="k", lw=2.0, zorder=13)

    # Ego.
    trail = md["exec_xy"][start : t + 1]
    ax.plot(trail[:, 0], trail[:, 1], color=C_EGO, lw=2.6, alpha=0.9, zorder=14, solid_capstyle="round")
    _draw_box(
        ax, ego_xy[0], ego_xy[1], md["exec_yaw"][t], sc["obj_length"][ego, t], sc["obj_width"][ego, t],
        face=C_EGO, alpha=0.95, z=15,
    )
    goal = sc["goal_xy"]
    ax.plot(*goal, marker="*", color="gold", mec="k", ms=15, zorder=16)
    ax.add_patch(Circle(goal, 2.0, fill=False, ls=":", color="k", zorder=16))

    ax.set_xlim(ego_xy[0] - half, ego_xy[0] + half)
    ax.set_ylim(ego_xy[1] - half, ego_xy[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    n_feas = int((~coll & ~off).sum())
    best = int(md[f"{pre}_best"])
    chosen_ok = not (coll[best] or off[best])
    per_iter = " → ".join(str(int(x)) for x in md[f"{pre}_feasible_per_iter"])
    dist = float(np.linalg.norm(ego_xy - goal))
    info = (
        f"t = {t}   (plan from replan t={r})\n"
        f"feasible per ES iter: {per_iter}  (of {len(pool) - len(elites)}, then {len(pool)})\n"
        f"final pool: {n_feas}/{len(pool)} feasible, {int((coll & ~off).sum())} coll, "
        f"{int((off & ~coll).sum())} offroad, {int((coll & off).sum())} both\n"
        f"executed plan: {'feasible' if chosen_ok else 'INFEASIBLE (no feasible candidate)'}\n"
        f"distance to goal: {dist:.1f} m"
    )
    ax.text(0.01, 0.99, info, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"), zorder=20)

    badges = []
    if md["overlap_ts"][t] > 0:
        badges.append(("COLLISION", C_COLL))
    if md["offroad_ts"][t] > 0:
        badges.append(("OFFROAD", C_OFF))
    if ev.get("goal_reached") and 0 <= ev.get("reached_timestep", -1) <= t:
        badges.append(("GOAL REACHED", "#1a7f37"))
    for j, (txt, col) in enumerate(badges):
        ax.text(0.99, 0.99 - 0.07 * j, txt, transform=ax.transAxes, va="top", ha="right", fontsize=11,
                color="white", weight="bold", bbox=dict(facecolor=col, edgecolor="none", alpha=0.95), zorder=21)

    title_color = "#1a7f37" if ev.get("success") else "#b00000"
    ax.set_title(f"{MODE_LABEL[mode]}\n{_outcome(ev)}", fontsize=10, color=title_color)


def _legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=C_EGO, label="ego (trail = executed path)"),
        Line2D([0], [0], color="k", lw=2, label="ES chosen plan (latest replan)"),
        Line2D([0], [0], color=C_ELITE, lw=1.8, label="ES elites (top-8)"),
        Line2D([0], [0], color=C_FEAS, lw=1.2, label="candidate: feasible"),
        Line2D([0], [0], color=C_COLL, lw=1.2, label="candidate: collision"),
        Line2D([0], [0], color=C_OFF, lw=1.2, label="candidate: offroad"),
        Line2D([0], [0], color=C_BOTH, lw=1.2, label="candidate: both"),
        Patch(facecolor=C_NEIGHBOR, alpha=0.75, label="neighbour agent now (actual)"),
        Line2D([0], [0], color=C_GT_FUT, lw=1.6, label="neighbour GT future"),
        Line2D([0], [0], color=C_FORECAST["cv"], lw=1.6, ls=(0, (3, 2)), label="CV forecast (box = predicted now)"),
        Line2D([0], [0], color=C_FORECAST["wm"], lw=1.6, ls=(0, (3, 2)), label="WM forecast (box = predicted now)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="k", ms=12, label="goal (2 m)"),
    ]


def _suptitle(row: dict[str, Any]) -> str:
    orig = " | ".join(
        f"{m.upper()}: {'ok' if row['orig'][m]['success'] else 'FAIL'}" for m in MODES
    )
    return f"{row['stem']}  ({row['tfrecord']} scenario {row['scenario_idx']})     original runs: {orig}"


def render_scene(row: dict[str, Any], args, out_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    data_dir = Path(row["data_dir"])
    sc = dict(np.load(data_dir / f"{row['stem']}_scene.npz"))
    mds = {m: dict(np.load(data_dir / f"{row['stem']}_{m}.npz")) for m in MODES}
    start = int(args.start_timestep)
    n_steps = mds["gt"]["exec_xy"].shape[0]
    frames = list(range(start, n_steps)) + [n_steps - 1] * max(1, int(args.fps))

    fig, axes = plt.subplots(1, 3, figsize=(21, 8.4), dpi=90)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=6, fontsize=8.5, frameon=False)
    fig.suptitle(_suptitle(row), fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.11, wspace=0.03)

    vid_dir = out_dir / "videos"
    snap_dir = out_dir / "snapshots"
    vid_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    mp4 = vid_dir / f"{row['stem']}.mp4"
    writer = FFMpegWriter(fps=int(args.fps), codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(mp4), dpi=90):
        for t in frames:
            for ax, m in zip(axes, MODES):
                draw_panel(ax, sc, mds[m], m, t, row["repro"][m], args)
            writer.grab_frame()

    events = [
        x
        for m in MODES
        for x in (row["repro"][m]["first_overlap_t"], row["repro"][m]["first_offroad_t"])
        if x is not None and x >= 0
    ]
    t_snap = min(events) if events else (start + n_steps) // 2
    for ax, m in zip(axes, MODES):
        draw_panel(ax, sc, mds[m], m, t_snap, row["repro"][m], args)
    png = snap_dir / f"{row['stem']}_t{t_snap:03d}.png"
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return {"video": f"videos/{mp4.name}", "snapshot": f"snapshots/{png.name}", "t_snap": t_snap}


EXPLAINER = """
<div style='background:#f5f5f5;padding:10px 14px;border-radius:6px;font-size:13px;max-width:1200px'>
Each video plays the same scene three times side by side: Diffusion-ES scored with <b>GT</b>, <b>CV</b> and
<b>WM</b> forecasts of the other agents. Everything else is identical (diffusion model, seed, ES budget:
64 samples, top-8 elites, 4 iterations, replan every 10 steps). Other agents always move as in the log;
only the ego is planned. The camera follows each panel's own ego.<br>
<b>Forecasts.</b> For dynamic agents within 35 m, the green line is their true (logged) future from the
latest replan. In the CV / WM panels the dashed blue / orange line is what that forecast predicted, and
the dashed box is where it predicted the agent to be <i>now</i>; compare it with the grey box (where the
agent actually is). The planner only sees the dashed prediction.<br>
<b>ES.</b> Thin lines are the final ES pool of the latest replan: lavender = feasible (score 1), red =
predicted collision, brown = offroad (score 0). Purple = the 8 elites, black = executed plan. The score is
binary (collision &times; offroad, no progress term), so among feasible plans ES picks the lowest index, and
if none is feasible it executes an arbitrary infeasible one. Badges show the simulator's events at that timestep.
</div>
"""


def phase_render(args: argparse.Namespace, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg

    rows = {}
    for line in (out_dir / "per_scene.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["stem"]] = r
    cards = []
    for i, row in enumerate(sorted(rows.values(), key=lambda r: r["stem"]), start=1):
        if "error" in row:
            cards.append(f"<h3>{html.escape(row['stem'])}</h3><p style='color:#c00'>ERROR: {html.escape(row['error'])}</p>")
            continue
        print(f"[render {i}/{len(rows)}] {row['stem']}", flush=True)
        out = render_scene(row, args, out_dir)
        trs = "".join(
            f"<tr><td>{m.upper()}</td><td>{'success' if row['orig'][m]['success'] else 'FAIL'}"
            f" (overlap={row['orig'][m]['overlap']}, offroad={row['orig'][m]['offroad']}, "
            f"goal={row['orig'][m]['goal_reached']})</td><td>{html.escape(_outcome(row['repro'][m]))}</td>"
            f"<td>{'yes' if row['reproduced'][m] else '<b style=color:#c00>no</b>'}</td></tr>"
            for m in MODES
        )
        cards.append(
            f"<div style='margin:26px 0;padding-top:8px;border-top:2px solid #444'>"
            f"<h3 id='{row['stem']}'>{html.escape(row['stem'])} &mdash; {html.escape(row['tfrecord'])} scenario "
            f"{row['scenario_idx']} &mdash; fails with: {', '.join(m.upper() for m in row['fails_in'])}</h3>"
            "<table border=1 cellspacing=0 cellpadding=3 style='font-size:12px;border-collapse:collapse'>"
            "<tr><th>forecast</th><th>original run</th><th>this re-run</th><th>same outcome</th></tr>"
            f"{trs}</table>"
            f"<video controls loop muted preload='metadata' style='width:100%;max-width:1900px;margin-top:8px' "
            f"src='{out['video']}'></video>"
            f"<div style='font-size:12px;color:#555'>snapshot at t={out['t_snap']} (first collision/offroad):</div>"
            f"<a href='{out['snapshot']}'><img src='{out['snapshot']}' style='max-width:100%;border:1px solid #ccc'/></a>"
            "</div>"
        )
    toc = ", ".join(f"<a href='#{s}'>{s}</a>" for s in sorted(rows))
    page = (
        "<html><head><meta charset='utf-8'><title>Diffusion-ES: GT vs CV vs WM</title></head>"
        "<body style='font-family:sans-serif;margin:20px'>"
        f"<h1>Diffusion-ES with GT / CV / WM forecasts &mdash; {len(rows)} validation failures</h1>"
        f"{EXPLAINER}<p>{toc}</p>{''.join(cards)}</body></html>"
    )
    (out_dir / "index.html").write_text(page)
    print(f"[render] wrote {out_dir / 'index.html'}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT_ROOT) / (
        f"es_gt_cv_wm_val_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("run", "all"):
        phase_run(args, out_dir)
    if args.phase in ("render", "all"):
        phase_render(args, out_dir)


if __name__ == "__main__":
    main()
