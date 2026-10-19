#!/usr/bin/env python3
"""Re-run Diffusion-ES (GT forecast) on its train/val failures and show every ES iteration.

Scenes: every ``success == false`` scenario of the es_diffusion_gt trainval run
(training + validation shards 00/01). Scenes where the logged GT trajectory itself
fails (gt_filtered_es_metrics) are tagged ``GT itself fails``.

Phase ``run`` (GPU) re-simulates each failing scene inside its original 5-scene batch
(same seed, same RNG splits as ``simulation/runner.py``) with a recording ES planner
that stores, at every replan step and every ES iteration, the whole population, each
candidate's collision / offroad verdict from the scorer, the elites that were kept and
the plan that was executed.

Phase ``render`` (CPU) writes per scene:
  *_overview.png      executed ego path vs GT ego path, goal, first overlap/offroad
  *_es_critical.png   ES iterations at the replan that produced the failure
  *_timeline.png      feasible/infeasible counts per replan + sim events over time
  *_es_all.png        grid: every replan step x every ES iteration
and an ``index.html`` grouped by failure type.

Example::

  python scripts/viz_es_gt_failures.py --phase all --max_batches 1
  sbatch slurm/sim/viz_es_gt_failures.sh
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRATCH = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad"
ES_GT_DIR = f"{SCRATCH}/Diffusion_ES_sim/es_diffusion_gt"
DEFAULT_RESULT_DIRS = {
    "training": f"{ES_GT_DIR}/simulation_es_training_es_diffusion_gt_TRAIN_tfrecord00_01_20260907_102456",
    "validation": f"{ES_GT_DIR}/simulation_es_validation_es_diffusion_gt_VAL_tfrecord00_01_20260907_203053",
}
DEFAULT_GT_EVAL_DIR = f"{SCRATCH}/Diffusion_ES_sim/gt_filtered_es_metrics_20260908_114522/gt_filter"
DEFAULT_WOMD_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example"
DEFAULT_LANE_GRAPH_DIR = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DEFAULT_DIFFUSION_CKPT = (
    f"{SCRATCH}/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
)
DEFAULT_OUT_ROOT = f"{SCRATCH}/Diffusion_ES_sim"

COLOR_FEAS = (0.10, 0.62, 0.25)
COLOR_COLL = (0.85, 0.10, 0.10)
COLOR_OFF = (1.00, 0.55, 0.00)
COLOR_BOTH = (0.55, 0.10, 0.65)
COLOR_ELITE = (0.10, 0.30, 0.95)
COLOR_EGO = (0.55, 0.10, 0.75)
COLOR_GT_EGO = (0.05, 0.55, 0.10)
COLOR_AGENT = (0.45, 0.45, 0.45)

CATEGORY_ORDER = [
    ("gt_fail", "GT itself fails (dataset / label issue)"),
    ("offroad", "Offroad (planner)"),
    ("no_goal", "Goal not reached (planner)"),
    ("overlap", "Collision (planner)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["run", "render", "all"], default="all")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--train_result_dir", type=str, default=DEFAULT_RESULT_DIRS["training"])
    p.add_argument("--val_result_dir", type=str, default=DEFAULT_RESULT_DIRS["validation"])
    p.add_argument("--gt_eval_dir", type=str, default=DEFAULT_GT_EVAL_DIR)
    p.add_argument("--womd_root", type=str, default=DEFAULT_WOMD_ROOT)
    p.add_argument("--lane_graph_dir", type=str, default=DEFAULT_LANE_GRAPH_DIR)
    p.add_argument("--diffusion_ckpt_path", type=str, default=DEFAULT_DIFFUSION_CKPT)
    p.add_argument("--other_agent_forecast", type=str, default="gt")
    p.add_argument("--num_worlds", type=int, default=5, help="Must match the original run for exact RNG.")
    p.add_argument("--start_timestep", type=int, default=10)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--population_size", type=int, default=64)
    p.add_argument("--elite_size", type=int, default=8)
    p.add_argument("--num_es_iterations", type=int, default=4)
    p.add_argument("--resample_timesteps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0, help="0 = all batches.")
    p.add_argument("--no_grid", action="store_true", help="Skip the all-replans grid figure.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Scene selection
# ----------------------------------------------------------------------------


def load_failures(args: argparse.Namespace) -> list[dict[str, Any]]:
    result_dirs = {"training": args.train_result_dir, "validation": args.val_result_dir}
    scenes: list[dict[str, Any]] = []
    for split, rdir in result_dirs.items():
        gt_path = Path(args.gt_eval_dir) / f"{split}_gt_eval_all.json"
        gt = {
            (Path(r["tfrecord"]).name, int(r["scenario_idx"])): r
            for r in json.loads(gt_path.read_text())
        }
        for f in sorted(Path(rdir).glob("*.scenario_*.json")):
            r = json.loads(f.read_text())
            if r["success"]:
                continue
            tf_name = Path(r["tfrecord"]).name
            idx = int(r["scenario_idx"])
            g = gt.get((tf_name, idx))
            gt_ok = g is None or bool(g["success"])
            if not gt_ok:
                category = "gt_fail"
            elif r["overlap"]:
                category = "overlap"
            elif r["offroad"]:
                category = "offroad"
            else:
                category = "no_goal"
            scenes.append(
                {
                    "split": split,
                    "tfrecord": tf_name,
                    "scenario_idx": idx,
                    "category": category,
                    "gt_status": "ok" if gt_ok else "+".join(g["fail_reasons"]),
                    "orig": {
                        k: r[k]
                        for k in (
                            "ego_idx",
                            "overlap",
                            "offroad",
                            "goal_reached",
                            "reached_timestep",
                            "min_goal_distance_m",
                            "final_goal_distance_m",
                        )
                    },
                }
            )
    return scenes


def group_into_batches(scenes: list[dict[str, Any]], num_worlds: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for s in scenes:
        start = (int(s["scenario_idx"]) // num_worlds) * num_worlds
        groups[(s["split"], s["tfrecord"], start)].append(s)
    return [
        {"split": k[0], "tfrecord": k[1], "batch_start": k[2], "targets": v}
        for k, v in sorted(groups.items())
    ]


def scene_stem(s: dict[str, Any]) -> str:
    split = "train" if s["split"] == "training" else "val"
    shard = s["tfrecord"].split("tfrecord-")[1][:5]
    return f"{split}_{shard}_sc{int(s['scenario_idx']):03d}"


# ----------------------------------------------------------------------------
# Phase 1: re-simulation with a recording ES planner
# ----------------------------------------------------------------------------


def _build_recording_planner(args: argparse.Namespace):
    from viz.es_recording import build_recording_planner

    return build_recording_planner(
        diffusion_ckpt_path=args.diffusion_ckpt_path,
        seed=int(args.seed),
        population_size=int(args.population_size),
        elite_size=int(args.elite_size),
        num_iterations=int(args.num_es_iterations),
        resample_timesteps=int(args.resample_timesteps),
        num_worlds=int(args.num_worlds),
        other_agent_forecast=str(args.other_agent_forecast),
    )


def run_batch(args: argparse.Namespace, planner, batch: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    from simulation.evaluation_utils import infer_goals_from_sim_state
    from viz.es_recording import attach_lane_graphs, evaluate_batch, load_batch, simulate_batch

    nw = int(args.num_worlds)
    tf_path = Path(args.womd_root) / batch["split"] / batch["tfrecord"]
    indices = list(range(int(batch["batch_start"]), int(batch["batch_start"]) + nw))
    sim_state, _ = load_batch(str(tf_path), indices, max_num_objects=int(args.max_num_objects))
    attach_lane_graphs(planner, sim_state, str(tf_path), indices, args.lane_graph_dir)
    goal_xy_b2, _, ego_idx_b = infer_goals_from_sim_state(sim_state)
    start = int(args.start_timestep)
    replaced, records = simulate_batch(
        planner,
        sim_state,
        goal_xy_b2,
        start_timestep=start,
        replan_interval_steps=int(args.replan_interval_steps),
        seed=int(args.seed),
    )
    evals = evaluate_batch(replaced, goal_xy_b2, start_timestep=start, seed=int(args.seed))

    rows = []
    targets = {int(s["scenario_idx"]): s for s in batch["targets"]}
    for w, idx in enumerate(indices):
        if idx not in targets:
            continue
        scene = targets[idx]
        ev = evals[w]
        ego = int(ego_idx_b[w])
        lt = sim_state.log_trajectory
        rt = replaced.log_trajectory
        rg = sim_state.roadgraph_points
        rg_valid = np.asarray(rg.valid[w]).astype(bool)
        payload: dict[str, Any] = {
            "rg_xy": np.asarray(rg.xy[w], dtype=np.float32)[rg_valid],
            "rg_types": np.asarray(rg.types[w], dtype=np.int32)[rg_valid],
            "rg_ids": np.asarray(rg.ids[w], dtype=np.int32)[rg_valid],
            "obj_xy": np.asarray(lt.xy[w], dtype=np.float32),
            "obj_yaw": np.asarray(lt.yaw[w], dtype=np.float32),
            "obj_length": np.asarray(lt.length[w], dtype=np.float32),
            "obj_width": np.asarray(lt.width[w], dtype=np.float32),
            "obj_valid": np.asarray(lt.valid[w]).astype(bool),
            "obj_types": np.asarray(sim_state.object_metadata.object_types[w], dtype=np.int32),
            "ego_idx": np.int32(ego),
            "exec_xy": np.asarray(rt.xy[w, ego], dtype=np.float32),
            "exec_yaw": np.asarray(rt.yaw[w, ego], dtype=np.float32),
            "goal_xy": np.asarray(goal_xy_b2[w], dtype=np.float32),
            "overlap_ts": ev["overlap_ts"],
            "offroad_ts": ev["offroad_ts"],
            "replan_steps": np.asarray(sorted(records), dtype=np.int32),
        }
        for step, rec in records.items():
            pre = f"t{step:03d}"
            payload[f"{pre}_best"] = np.int32(rec["best_idx"][w])
            payload[f"{pre}_chosen"] = rec["chosen"][w]
            payload[f"{pre}_chosen_offroad_steps"] = rec["chosen_offroad_steps"][w]
            payload[f"{pre}_horizon_scored"] = np.int32(rec["horizon_scored"][w])
            for s_i, st in enumerate(rec["stages"]):
                sp = f"{pre}_s{s_i}"
                payload[f"{sp}_traj"] = st["traj"][w]
                payload[f"{sp}_score"] = st["score"][w]
                payload[f"{sp}_coll"] = st["coll"][w]
                payload[f"{sp}_off"] = st["off"][w]
                payload[f"{sp}_elite"] = st["elite_idx"][w]
                payload[f"{sp}_ncarried"] = np.int32(st["n_carried"])
        npz_path = data_dir / f"{scene_stem(scene)}.npz"
        np.savez_compressed(npz_path, **payload)

        orig = scene["orig"]
        repro = {k: v for k, v in ev.items() if not k.endswith("_ts")}
        same = (
            repro["overlap"] == bool(orig["overlap"])
            and repro["offroad"] == bool(orig["offroad"])
            and repro["goal_reached"] == bool(orig["goal_reached"])
        )
        rows.append(
            {
                **{k: scene[k] for k in ("split", "tfrecord", "scenario_idx", "category", "gt_status", "orig")},
                "stem": scene_stem(scene),
                "npz": str(npz_path),
                "ego_idx": ego,
                "ego_idx_matches": ego == int(orig["ego_idx"]),
                "repro": repro,
                "reproduced_same_failure": bool(same),
            }
        )
    return rows


def phase_run(args: argparse.Namespace, out_dir: Path) -> None:
    import tensorflow as tf

    # TF only parses TFRecords; left on the GPU it reserves nearly all memory and starves JAX.
    tf.config.set_visible_devices([], "GPU")

    scenes = load_failures(args)
    batches = group_into_batches(scenes, int(args.num_worlds))
    if int(args.max_batches) > 0:
        batches = batches[: int(args.max_batches)]
    print(
        f"[run] {len(scenes)} failures in {len(batches)} batches "
        f"({sum(len(b['targets']) for b in batches)} scenes this run) -> {out_dir}",
        flush=True,
    )
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    planner = _build_recording_planner(args)
    jsonl = out_dir / "per_scene.jsonl"
    done = set()
    if jsonl.exists():
        done = {json.loads(l)["stem"] for l in jsonl.read_text().splitlines() if l.strip()}
    with jsonl.open("a") as f:
        for i, batch in enumerate(batches, start=1):
            if all(scene_stem(s) in done for s in batch["targets"]):
                continue
            print(
                f"[{i}/{len(batches)}] {batch['split']} {batch['tfrecord']} "
                f"batch {batch['batch_start']}..{batch['batch_start'] + int(args.num_worlds) - 1} "
                f"targets={[s['scenario_idx'] for s in batch['targets']]}",
                flush=True,
            )
            try:
                rows = run_batch(args, planner, batch, data_dir)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                for s in batch["targets"]:
                    row = {
                        **{k: s[k] for k in ("split", "tfrecord", "scenario_idx", "category", "gt_status", "orig")},
                        "stem": scene_stem(s),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    f.write(json.dumps(row) + "\n")
                f.flush()
                continue
            for row in rows:
                r = row["repro"]
                print(
                    f"  sc{row['scenario_idx']:03d} [{row['category']}] reproduced={row['reproduced_same_failure']} "
                    f"overlap={r['overlap']}@{r['first_overlap_t']} offroad={r['offroad']}@{r['first_offroad_t']} "
                    f"goal={r['goal_reached']}@{r['reached_timestep']}",
                    flush=True,
                )
                f.write(json.dumps(row) + "\n")
            f.flush()


# ----------------------------------------------------------------------------
# Phase 2: rendering
# ----------------------------------------------------------------------------

_LANE_TYPES = {1, 2, 3}
_WHITE_LINES = {5, 6, 7, 8}
_YELLOW_LINES = {9, 10, 11, 12, 13}
_EDGE_TYPES = {14, 15, 16}
_CROSSWALK = {18}


def _status_color(coll: bool, off: bool):
    if coll and off:
        return COLOR_BOTH
    if coll:
        return COLOR_COLL
    if off:
        return COLOR_OFF
    return COLOR_FEAS


def _box_corners(x, y, yaw, length, width) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([0.5, 0.5, -0.5, -0.5]) * length
    dy = np.array([0.5, -0.5, -0.5, 0.5]) * width
    return np.stack([x + dx * c - dy * s, y + dx * s + dy * c], axis=-1)


def _draw_box(ax, x, y, yaw, length, width, *, color, alpha=0.9, fill=True, lw=1.0, z=10):
    from matplotlib.patches import Polygon

    corners = _box_corners(x, y, yaw, length, width)
    ax.add_patch(
        Polygon(
            corners,
            closed=True,
            facecolor=color if fill else "none",
            edgecolor=color,
            alpha=alpha,
            lw=lw,
            zorder=z,
        )
    )
    nose = 0.5 * (corners[0] + corners[1])
    ax.plot([x, nose[0]], [y, nose[1]], color="k" if fill else color, lw=0.6, alpha=alpha, zorder=z + 1)


def _draw_map(ax, d: dict[str, np.ndarray], center: np.ndarray, half: float) -> None:
    from matplotlib.collections import LineCollection

    xy, types, ids = d["rg_xy"], d["rg_types"], d["rg_ids"]
    lo, hi = center - half - 5.0, center + half + 5.0
    inside = np.all((xy >= lo) & (xy <= hi), axis=-1)
    a, b = xy[:-1], xy[1:]
    keep = (ids[:-1] == ids[1:]) & (inside[:-1] | inside[1:])
    keep &= np.linalg.norm(b - a, axis=-1) < 5.0
    seg_types = types[:-1]
    for type_set, color, lw, ls in (
        (_LANE_TYPES, (0.80, 0.80, 0.80), 0.6, (0, (4, 4))),
        (_WHITE_LINES, (0.65, 0.65, 0.65), 0.6, "-"),
        (_YELLOW_LINES, (0.85, 0.70, 0.20), 0.7, "-"),
        (_CROSSWALK, (0.55, 0.75, 0.95), 0.8, "-"),
        (_EDGE_TYPES, (0.10, 0.10, 0.10), 1.3, "-"),
    ):
        m = keep & np.isin(seg_types, list(type_set))
        if m.any():
            segs = np.stack([a[m], b[m]], axis=1)
            ax.add_collection(LineCollection(segs, colors=[color], linewidths=lw, linestyles=ls, zorder=1))


def _draw_agents(ax, d, t: int, center, half, *, future: int = 0, highlight: set[int] | None = None) -> None:
    ego = int(d["ego_idx"])
    n_steps = d["obj_xy"].shape[1]
    t = int(np.clip(t, 0, n_steps - 1))
    for i in range(d["obj_xy"].shape[0]):
        if i == ego or not d["obj_valid"][i, t]:
            continue
        x, y = d["obj_xy"][i, t]
        if abs(x - center[0]) > half + 10 or abs(y - center[1]) > half + 10:
            continue
        hl = highlight is not None and i in highlight
        color = (0.10, 0.10, 0.10) if hl else COLOR_AGENT
        _draw_box(
            ax, x, y, d["obj_yaw"][i, t], d["obj_length"][i, t], d["obj_width"][i, t],
            color=color, alpha=0.75 if hl else 0.45, z=6,
        )
        if future > 0:
            sl = slice(t, min(n_steps, t + future))
            v = d["obj_valid"][i, sl]
            pts = d["obj_xy"][i, sl][v]
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.8, alpha=0.5, zorder=5)
                ax.plot(pts[-1, 0], pts[-1, 1], marker=".", color=color, ms=3, alpha=0.6, zorder=5)


def _ego_box_at(d, t: int):
    ego = int(d["ego_idx"])
    t = int(np.clip(t, 0, d["exec_xy"].shape[0] - 1))
    x, y = d["exec_xy"][t]
    return x, y, d["exec_yaw"][t], d["obj_length"][ego, t], d["obj_width"][ego, t]


def _setup_ax(ax, center, half, title: str, fontsize: int = 9) -> None:
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)


def _stage(d, t: int, s: int) -> dict[str, np.ndarray]:
    p = f"t{t:03d}_s{s}"
    return {
        "traj": d[f"{p}_traj"],
        "score": d[f"{p}_score"],
        "coll": d[f"{p}_coll"].astype(bool),
        "off": d[f"{p}_off"].astype(bool),
        "elite": d[f"{p}_elite"],
        "n_carried": int(d[f"{p}_ncarried"]),
    }


def _num_stages(d, t: int) -> int:
    s = 0
    while f"t{t:03d}_s{s}_traj" in d:
        s += 1
    return s


def _counts(st) -> dict[str, int]:
    c, o = st["coll"], st["off"]
    return {
        "n": int(len(c)),
        "feasible": int((~c & ~o).sum()),
        "coll": int((c & ~o).sum()),
        "off": int((~c & o).sum()),
        "both": int((c & o).sum()),
    }


def _draw_stage(ax, d, t: int, s: int, center, half, *, is_last: bool, interval: int, compact: bool) -> None:
    from matplotlib.collections import LineCollection

    st = _stage(d, t, s)
    h = int(d[f"t{t:03d}_horizon_scored"])
    traj = st["traj"][:, :h, :2]
    _draw_map(ax, d, center, half)
    _draw_agents(ax, d, t, center, half, future=h)
    ex, ey, eyaw, el, ew = _ego_box_at(d, t)
    _draw_box(ax, ex, ey, eyaw, el, ew, color=COLOR_EGO, alpha=0.9, z=12)

    elite_set = set(int(i) for i in st["elite"])
    carried = set(range(st["n_carried"]))
    order = sorted(range(len(traj)), key=lambda k: (not (st["coll"][k] or st["off"][k]), k))
    segs_by_style: dict[tuple, list] = defaultdict(list)
    for k in order:
        if k in elite_set:
            continue
        col = _status_color(bool(st["coll"][k]), bool(st["off"][k]))
        segs_by_style[(col, "--" if k in carried else "-")].append(traj[k])
    for (col, ls), segs in segs_by_style.items():
        ax.add_collection(LineCollection(segs, colors=[col], linewidths=0.7, alpha=0.45, linestyles=ls, zorder=7))
    for k in st["elite"]:
        k = int(k)
        col = _status_color(bool(st["coll"][k]), bool(st["off"][k]))
        ax.plot(traj[k, :, 0], traj[k, :, 1], color=COLOR_ELITE, lw=3.0, alpha=0.55, zorder=8, solid_capstyle="round")
        ax.plot(traj[k, :, 0], traj[k, :, 1], color=col, lw=1.1, alpha=1.0, zorder=9)

    if is_last:
        chosen = d[f"t{t:03d}_chosen"][:h]
        ax.plot(chosen[:, 0], chosen[:, 1], color="k", lw=2.2, zorder=14)
        ex_seg = chosen[: interval + 1]
        ax.plot(ex_seg[:, 0], ex_seg[:, 1], color="k", lw=4.0, zorder=14, solid_capstyle="round")
        off_steps = d[f"t{t:03d}_chosen_offroad_steps"][:h].astype(bool)
        if off_steps.any():
            ax.scatter(chosen[off_steps, 0], chosen[off_steps, 1], s=10, color=COLOR_OFF, edgecolors="k",
                       linewidths=0.3, zorder=15)

    c = _counts(st)
    elite_feas = sum(1 for k in st["elite"] if not (st["coll"][int(k)] or st["off"][int(k)]))
    if s == 0:
        head = f"t={t}  initial sample ({c['n']})"
    else:
        head = f"t={t}  iter {s}: {st['n_carried']} elites + {c['n'] - st['n_carried']} resampled"
    if compact:
        title = f"{head}\nok {c['feasible']} | coll {c['coll']} | off {c['off']} | both {c['both']}"
    else:
        title = (
            f"{head}\nfeasible {c['feasible']}/{c['n']}   collision {c['coll']}   offroad {c['off']}   both {c['both']}\n"
            f"elites kept: {elite_feas}/{len(st['elite'])} feasible"
        )
        if is_last:
            best = int(d[f"t{t:03d}_best"])
            bc, bo = bool(st["coll"][best]), bool(st["off"][best])
            verdict = "feasible" if not (bc or bo) else ("collision" if bc and not bo else "offroad" if bo and not bc else "coll+offroad")
            title += f"\nCHOSEN = pool[{best}] ({verdict} per scorer)"
    _setup_ax(ax, center, half, title, fontsize=7 if compact else 8)


def _stage_window(d, t: int) -> tuple[np.ndarray, float]:
    center = np.asarray(_ego_box_at(d, t)[:2], dtype=np.float32)
    h = int(d[f"t{t:03d}_horizon_scored"])
    pts = [d[f"t{t:03d}_chosen"][:h, :2]]
    n = _num_stages(d, t)
    for s in range(n):
        st = _stage(d, t, s)
        pts.append(st["traj"][st["elite"], :h, :2].reshape(-1, 2))
    pts = np.concatenate(pts, axis=0)
    half = float(np.clip(np.abs(pts - center).max() + 6.0, 12.0, 70.0))
    return center, half


def _legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        Line2D([0], [0], color=COLOR_FEAS, lw=1.5, label="feasible (score 1)"),
        Line2D([0], [0], color=COLOR_COLL, lw=1.5, label="collision (score 0)"),
        Line2D([0], [0], color=COLOR_OFF, lw=1.5, label="offroad (score 0)"),
        Line2D([0], [0], color=COLOR_BOTH, lw=1.5, label="collision + offroad"),
        Line2D([0], [0], color="0.4", lw=1.0, ls="--", label="elite carried from previous iter"),
        Line2D([0], [0], color=COLOR_ELITE, lw=3.0, alpha=0.55, label="selected as elite (top-8)"),
        Line2D([0], [0], color="k", lw=2.2, label="chosen plan (thick = executed 10 steps)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OFF, markeredgecolor="k", ms=5,
               label="chosen plan offroad per scorer"),
        Patch(facecolor=COLOR_EGO, alpha=0.9, label="ego now"),
        Patch(facecolor=COLOR_AGENT, alpha=0.45, label="other agents now (+ GT future)"),
    ]


def critical_replan(row: dict[str, Any], d, interval: int, start: int) -> tuple[int, str]:
    steps = [int(t) for t in d["replan_steps"]]
    r = row.get("repro", {})
    fails = [t for t in (r.get("first_overlap_t", -1), r.get("first_offroad_t", -1)) if t is not None and t >= 0]
    if fails:
        t_fail = min(fails)
        t_crit = start + ((t_fail - start) // interval) * interval
        t_crit = max(s for s in steps if s <= max(t_crit, steps[0]))
        kind = "overlap" if t_fail == r.get("first_overlap_t") else "offroad"
        return t_crit, f"first {kind} at t={t_fail}; executed segment planned at t={t_crit}"
    feas = []
    for t in steps:
        st = _stage(d, t, _num_stages(d, t) - 1)
        feas.append(_counts(st)["feasible"])
    t_crit = steps[int(np.argmin(feas))]
    return t_crit, f"no overlap/offroad; replan with fewest feasible candidates (t={t_crit})"


def render_es_row(d, t: int, out_png: Path, title: str, interval: int) -> None:
    import matplotlib.pyplot as plt

    n = _num_stages(d, t)
    center, half = _stage_window(d, t)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.6), dpi=110)
    for s in range(n):
        _draw_stage(axes[s], d, t, s, center, half, is_last=(s == n - 1), interval=interval, compact=False)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out_png)
    plt.close(fig)


def render_es_grid(d, out_png: Path, title: str, interval: int) -> None:
    import matplotlib.pyplot as plt

    steps = [int(t) for t in d["replan_steps"]]
    n = max(_num_stages(d, t) for t in steps)
    fig, axes = plt.subplots(len(steps), n, figsize=(3.3 * n, 3.6 * len(steps)), dpi=75, squeeze=False)
    for r_i, t in enumerate(steps):
        center, half = _stage_window(d, t)
        for s in range(n):
            ax = axes[r_i, s]
            if s >= _num_stages(d, t):
                ax.axis("off")
                continue
            _draw_stage(ax, d, t, s, center, half, is_last=(s == n - 1), interval=interval, compact=True)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png)
    plt.close(fig)


def render_overview(row, d, out_png: Path, title: str, start: int, interval: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ego = int(d["ego_idx"])
    r = row["repro"]
    exec_xy = d["exec_xy"]
    gt_xy = d["obj_xy"][ego]
    gt_valid = d["obj_valid"][ego]
    goal = d["goal_xy"]
    n_steps = exec_xy.shape[0]
    fails = [t for t in (r["first_overlap_t"], r["first_offroad_t"]) if t >= 0]
    t_focus = min(fails) if fails else int(np.argmin(np.linalg.norm(exec_xy[start:] - goal, axis=-1))) + start

    pts = np.concatenate([exec_xy[start:], gt_xy[gt_valid], goal[None]], axis=0)
    c_all = 0.5 * (pts.min(0) + pts.max(0))
    half_all = float(np.clip(0.5 * (pts.max(0) - pts.min(0)).max() + 12.0, 25.0, 120.0))
    c_zoom = exec_xy[t_focus]
    half_zoom = 14.0

    near = set()
    for i in range(d["obj_xy"].shape[0]):
        if i != ego and d["obj_valid"][i, t_focus] and np.linalg.norm(d["obj_xy"][i, t_focus] - c_zoom) < 6.0:
            near.add(i)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), dpi=110)
    for ax, center, half, zoom in ((axes[0], c_all, half_all, False), (axes[1], c_zoom, half_zoom, True)):
        _draw_map(ax, d, center, half)
        _draw_agents(ax, d, t_focus, center, half, future=0, highlight=near)
        if not zoom:
            for i in range(d["obj_xy"].shape[0]):
                if i == ego:
                    continue
                v = d["obj_valid"][i, start:]
                p = d["obj_xy"][i, start:][v]
                if len(p) > 1 and np.abs(p - center).min(axis=0).max() < half:
                    ax.plot(p[:, 0], p[:, 1], color=COLOR_AGENT, lw=0.5, alpha=0.3, zorder=3)
        ax.plot(gt_xy[gt_valid, 0], gt_xy[gt_valid, 1], color=COLOR_GT_EGO, lw=2.0, ls="--", zorder=9,
                label="GT (logged) ego path")
        seg_pts = exec_xy[start:]
        segs = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", linewidths=2.6, zorder=10)
        lc.set_array(np.arange(start, n_steps - 1))
        lc.set_clim(start, n_steps - 1)
        ax.add_collection(lc)
        for t in d["replan_steps"]:
            t = int(t)
            ax.plot(*exec_xy[t], "o", color="k", ms=3.5, zorder=11)
            if not zoom:
                ax.annotate(f"{t}", exec_xy[t], fontsize=6, xytext=(3, 3), textcoords="offset points", zorder=11)
        off_t = np.flatnonzero(d["offroad_ts"][: n_steps] > 0)
        off_t = off_t[off_t >= start]
        ov_t = np.flatnonzero(d["overlap_ts"][: n_steps] > 0)
        ov_t = ov_t[ov_t >= start]
        if len(off_t):
            ax.scatter(exec_xy[off_t, 0], exec_xy[off_t, 1], s=22, marker="x", color=COLOR_OFF, zorder=12,
                       label="offroad timesteps")
        if len(ov_t):
            ax.scatter(exec_xy[ov_t, 0], exec_xy[ov_t, 1], s=22, marker="x", color=COLOR_COLL, zorder=12,
                       label="overlap timesteps")
        ax.plot(*goal, marker="*", color="gold", mec="k", ms=16, zorder=13, label="goal")
        ax.add_patch(plt.Circle(goal, 2.0, fill=False, ls=":", color="k", zorder=13))
        ex, ey, eyaw, el, ew = _ego_box_at(d, t_focus)
        _draw_box(ax, ex, ey, eyaw, el, ew, color=COLOR_EGO, alpha=0.9, z=14)
        sub = (
            f"zoom at t={t_focus} (ego box; nearby agents dark)"
            if zoom
            else "whole episode: executed path colored by time, replan points labeled"
        )
        _setup_ax(ax, center, half, sub)
    cb = fig.colorbar(lc, ax=axes[1], fraction=0.04, pad=0.01)
    cb.set_label("timestep")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png)
    plt.close(fig)


def render_timeline(row, d, out_png: Path, title: str, start: int, interval: int) -> None:
    import matplotlib.pyplot as plt

    steps = [int(t) for t in d["replan_steps"]]
    r = row["repro"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), dpi=110, sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1.0]})
    w = interval * 0.38
    for t in steps:
        n = _num_stages(d, t)
        for j, s in enumerate((0, n - 1)):
            c = _counts(_stage(d, t, s))
            x = t + (j - 0.5) * w * 1.05 + interval * 0.0
            bottom = 0
            for key, col in (("feasible", COLOR_FEAS), ("coll", COLOR_COLL), ("off", COLOR_OFF), ("both", COLOR_BOTH)):
                frac = c[key] / max(c["n"], 1)
                ax0.bar(x, frac, width=w, bottom=bottom, color=col, edgecolor="white", lw=0.4)
                bottom += frac
            ax0.text(x, 1.01, "init" if j == 0 else "final", ha="center", fontsize=6)
        st = _stage(d, t, n - 1)
        best = int(d[f"t{t:03d}_best"])
        ok = not (st["coll"][best] or st["off"][best])
        ax0.plot(t, 1.12, marker="o" if ok else "X", color=COLOR_FEAS if ok else COLOR_COLL, ms=9, mec="k")
    ax0.set_ylim(0, 1.2)
    ax0.set_ylabel("fraction of candidates")
    ax0.set_title("ES pool per replan: left bar = initial 64 samples, right bar = final pool; "
                  "marker = chosen plan feasible (o) / infeasible (X) per scorer", fontsize=9)

    n_steps = d["exec_xy"].shape[0]
    ts = np.arange(n_steps)
    dist = np.linalg.norm(d["exec_xy"] - d["goal_xy"][None], axis=-1)
    ax1.plot(ts[start:], dist[start:], color="k", lw=1.5, label="ego distance to goal (m)")
    ax1.axhline(2.0, color="k", ls=":", lw=0.8)
    ax1b = ax1.twinx()
    ax1b.fill_between(ts, 0, d["offroad_ts"][:n_steps], step="mid", color=COLOR_OFF, alpha=0.5, label="offroad")
    ax1b.fill_between(ts, 0, d["overlap_ts"][:n_steps] * 0.6, step="mid", color=COLOR_COLL, alpha=0.6,
                      label="overlap")
    ax1b.set_ylim(0, 3)
    ax1b.set_yticks([])
    for t in steps:
        ax0.axvline(t - interval * 0.5, color="0.85", lw=0.5, zorder=0)
        ax1.axvline(t, color="0.85", lw=0.5, zorder=0)
    for t, col in ((r["first_offroad_t"], COLOR_OFF), (r["first_overlap_t"], COLOR_COLL)):
        if t >= 0:
            ax0.axvline(t, color=col, lw=1.5, ls="--")
            ax1.axvline(t, color=col, lw=1.5, ls="--")
    if r["goal_reached"] and r["reached_timestep"] >= 0:
        ax1.axvline(r["reached_timestep"], color=COLOR_GT_EGO, lw=1.5, ls="--", label="goal reached")
    ax1.set_xlabel("timestep (replan every 10; dashed = first offroad / overlap / goal)")
    ax1.set_ylabel("distance to goal (m)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.set_xlim(start - interval, n_steps)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png)
    plt.close(fig)


def replan_table(d, interval: int) -> list[dict[str, Any]]:
    rows = []
    goal = d["goal_xy"]
    n_steps = d["exec_xy"].shape[0]
    for t in (int(x) for x in d["replan_steps"]):
        n = _num_stages(d, t)
        c0 = _counts(_stage(d, t, 0))
        st = _stage(d, t, n - 1)
        cf = _counts(st)
        best = int(d[f"t{t:03d}_best"])
        h = int(d[f"t{t:03d}_horizon_scored"])
        chosen = d[f"t{t:03d}_chosen"]
        off_steps = np.flatnonzero(d[f"t{t:03d}_chosen_offroad_steps"][:h])
        seg = slice(t, min(n_steps, t + interval))
        per_iter = []
        for s in range(n):
            c = _counts(_stage(d, t, s))
            per_iter.append(f"{c['feasible']}/{c['n']}")
        rows.append(
            {
                "t": t,
                "init": c0,
                "final": cf,
                "per_iter_feasible": per_iter,
                "elite_feasible": sum(1 for k in st["elite"] if not (st["coll"][int(k)] or st["off"][int(k)])),
                "chosen_idx": best,
                "chosen_coll": bool(st["coll"][best]),
                "chosen_off": bool(st["off"][best]),
                "chosen_first_offroad_step": int(off_steps[0]) if off_steps.size else -1,
                "chosen_end_goal_dist": float(np.linalg.norm(chosen[h - 1, :2] - goal)),
                "sim_offroad_in_segment": bool(d["offroad_ts"][seg].sum() > 0),
                "sim_overlap_in_segment": bool(d["overlap_ts"][seg].sum() > 0),
            }
        )
    return rows


def _fmt_t(t: int) -> str:
    return "-" if t is None or t < 0 else str(t)


def _scene_card(row: dict[str, Any], table: list[dict[str, Any]], crit_note: str, imgs: dict[str, str]) -> str:
    o, r = row["orig"], row["repro"]
    repro_badge = (
        "<span style='color:#1a7f37'>reproduced same failure</span>"
        if row["reproduced_same_failure"]
        else "<span style='color:#c00'>rerun differs from original run</span>"
    )
    head = (
        f"<h3 id='{row['stem']}'>{html.escape(row['stem'])} &mdash; {html.escape(row['tfrecord'])} "
        f"scenario {row['scenario_idx']}</h3>"
        f"<p><b>original:</b> overlap={o['overlap']} offroad={o['offroad']} goal={o['goal_reached']} "
        f"(reached t={o['reached_timestep']}, min goal dist {o['min_goal_distance_m']:.2f} m) &nbsp; "
        f"<b>GT replay:</b> {html.escape(row['gt_status'])}<br>"
        f"<b>rerun:</b> overlap={r['overlap']} (first t={_fmt_t(r['first_overlap_t'])}) "
        f"offroad={r['offroad']} (first t={_fmt_t(r['first_offroad_t'])}) goal={r['goal_reached']} "
        f"(t={_fmt_t(r['reached_timestep'])}, min dist {r['min_goal_distance_m']:.2f} m, final "
        f"{r['final_goal_distance_m']:.2f} m) &nbsp; {repro_badge}<br>"
        f"<b>critical replan:</b> {html.escape(crit_note)}</p>"
    )
    trs = []
    for x in table:
        chosen_v = "feasible" if not (x["chosen_coll"] or x["chosen_off"]) else (
            "coll+off" if x["chosen_coll"] and x["chosen_off"] else ("collision" if x["chosen_coll"] else "offroad")
        )
        style = " style='background:#fde2e2'" if (x["sim_offroad_in_segment"] or x["sim_overlap_in_segment"]) else ""
        events = ", ".join(
            e for e, f in (("offroad", x["sim_offroad_in_segment"]), ("overlap", x["sim_overlap_in_segment"])) if f
        ) or "-"
        i0, f = x["init"], x["final"]
        trs.append(
            f"<tr{style}><td>{x['t']}</td>"
            f"<td>{i0['feasible']}/{i0['n']} (c{i0['coll']} o{i0['off']} b{i0['both']})</td>"
            f"<td>{' &rarr; '.join(x['per_iter_feasible'])}</td>"
            f"<td>{f['feasible']}/{f['n']} (c{f['coll']} o{f['off']} b{f['both']})</td>"
            f"<td>{x['elite_feasible']}/8</td>"
            f"<td>pool[{x['chosen_idx']}] {chosen_v}</td>"
            f"<td>{_fmt_t(x['chosen_first_offroad_step'])}</td>"
            f"<td>{x['chosen_end_goal_dist']:.1f}</td>"
            f"<td>{events}</td></tr>"
        )
    table_html = (
        "<table border=1 cellspacing=0 cellpadding=3 style='font-size:12px;border-collapse:collapse'>"
        "<tr><th>replan t</th><th>initial 64: feasible (c=coll o=offroad b=both)</th>"
        "<th>feasible per ES iteration</th><th>final pool</th><th>elites feasible</th>"
        "<th>chosen plan (scorer)</th><th>chosen plan first offroad step (+k)</th>"
        "<th>chosen plan end dist to goal (m)</th><th>sim events in next 10 steps</th></tr>"
        + "".join(trs)
        + "</table>"
    )
    img_html = "".join(
        f"<div style='margin:8px 0'><div style='font-size:12px;color:#555'>{html.escape(cap)}</div>"
        f"<a href='{src}'><img src='{src}' style='max-width:100%;border:1px solid #ccc'/></a></div>"
        for cap, src in imgs.items()
    )
    return f"<div style='margin:28px 0;padding-top:8px;border-top:2px solid #444'>{head}{table_html}{img_html}</div>"


EXPLAINER = """
<div style='background:#f5f5f5;padding:10px 14px;border-radius:6px;font-size:13px;max-width:1100px'>
<b>How Diffusion-ES picks a plan here.</b> At each replan (every 10 steps from t=10) the diffusion policy
samples 64 ego trajectories. Each is scored by the scorer with metrics <code>[collision, offroad]</code>,
both multiplicative and binary, so a candidate scores <b>1</b> (no collision with the GT-forecast agents and no
offroad) or <b>0</b>. There is no progress / goal / comfort term. Each ES iteration keeps the top-8 (elites),
re-noises them for 5 diffusion steps into 64 new samples, scores those, and pools elites + new (72). After
4 iterations the top-1 of the pool is executed for 10 steps. Because scores are 0/1, ties are broken by
index (<code>jax.lax.top_k</code>): among all feasible candidates the one with the lowest pool index wins, and if
<i>no</i> candidate is feasible the planner executes pool[0], an arbitrary infeasible plan.<br>
<b>Colors:</b> green feasible, red collision, orange offroad, purple both; blue halo = selected elite;
dashed = elite carried over; black = chosen plan (thick part = the 10 executed steps); orange dots on the
chosen plan = timesteps the scorer itself flags as offroad. Grey boxes/lines = other agents now and their
GT future (which is what the GT-forecast scorer sees).
</div>
"""


def phase_render(args: argparse.Namespace, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    jsonl = out_dir / "per_scene.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    by_stem = {r["stem"]: r for r in rows}
    rows = list(by_stem.values())
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    start, interval = int(args.start_timestep), int(args.replan_interval_steps)

    cards: dict[str, list[str]] = defaultdict(list)
    summary_rows = []
    for i, row in enumerate(sorted(rows, key=lambda r: (r["category"], r["split"], r["tfrecord"], r["scenario_idx"])), 1):
        stem = row["stem"]
        if "error" in row:
            cards[row["category"]].append(
                f"<div><h3>{html.escape(stem)}</h3><p style='color:#c00'>ERROR: {html.escape(row['error'])}</p></div>"
            )
            continue
        print(f"[render {i}/{len(rows)}] {stem}", flush=True)
        d = dict(np.load(row["npz"]))
        t_crit, crit_note = critical_replan(row, d, interval, start)
        cat_label = dict(CATEGORY_ORDER)[row["category"]]
        title = (
            f"{stem}  ({row['tfrecord']} scen {row['scenario_idx']})  [{cat_label}]  GT replay: {row['gt_status']}\n"
            f"rerun: overlap={row['repro']['overlap']}@{_fmt_t(row['repro']['first_overlap_t'])}  "
            f"offroad={row['repro']['offroad']}@{_fmt_t(row['repro']['first_offroad_t'])}  "
            f"goal={row['repro']['goal_reached']}@{_fmt_t(row['repro']['reached_timestep'])}  "
            f"min goal dist={row['repro']['min_goal_distance_m']:.2f} m"
        )
        imgs = {}
        p = png_dir / f"{stem}_overview.png"
        render_overview(row, d, p, title, start, interval)
        imgs["Episode overview"] = f"png/{p.name}"
        p = png_dir / f"{stem}_es_critical.png"
        render_es_row(d, t_crit, p, f"{title}\nES iterations at critical replan t={t_crit}: {crit_note}", interval)
        imgs[f"ES iterations at critical replan t={t_crit}"] = f"png/{p.name}"
        p = png_dir / f"{stem}_timeline.png"
        render_timeline(row, d, p, title, start, interval)
        imgs["Replan timeline"] = f"png/{p.name}"
        if not args.no_grid:
            p = png_dir / f"{stem}_es_all.png"
            render_es_grid(d, p, f"{title}\nrows = replan steps, columns = ES iterations", interval)
            imgs["All replans x all ES iterations (click to enlarge)"] = f"png/{p.name}"
        table = replan_table(d, interval)
        cards[row["category"]].append(_scene_card(row, table, crit_note, imgs))
        summary_rows.append(
            {
                "stem": stem,
                "category": row["category"],
                "gt_status": row["gt_status"],
                "reproduced_same_failure": row["reproduced_same_failure"],
                "critical_replan": t_crit,
                "replans_with_no_feasible_candidate": [x["t"] for x in table if x["final"]["feasible"] == 0],
                "replans_where_chosen_plan_infeasible": [
                    x["t"] for x in table if x["chosen_coll"] or x["chosen_off"]
                ],
                "repro": row["repro"],
            }
        )

    toc, body = [], []
    for key, label in CATEGORY_ORDER:
        items = cards.get(key, [])
        if not items:
            continue
        stems = sorted(r["stem"] for r in rows if r["category"] == key)
        toc.append(
            f"<li><a href='#cat_{key}'>{html.escape(label)}</a> ({len(items)}): "
            + ", ".join(f"<a href='#{s}'>{s}</a>" for s in stems)
            + "</li>"
        )
        body.append(f"<h2 id='cat_{key}'>{html.escape(label)} &mdash; {len(items)}</h2>" + "".join(items))
    n_repro = sum(1 for r in rows if r.get("reproduced_same_failure"))
    page = (
        "<html><head><meta charset='utf-8'><title>Diffusion-ES (GT forecast) failures</title></head>"
        "<body style='font-family:sans-serif;margin:20px'>"
        f"<h1>Diffusion-ES with GT forecasts: {len(rows)} train/val failures</h1>"
        f"<p>Re-simulated in the original 5-scene batches (seed {args.seed}); "
        f"{n_repro}/{len(rows)} reproduce the original failure exactly.</p>"
        f"{EXPLAINER}<ul>{''.join(toc)}</ul>{''.join(body)}</body></html>"
    )
    (out_dir / "index.html").write_text(page)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"[render] wrote {out_dir / 'index.html'}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(DEFAULT_OUT_ROOT) / f"viz_es_gt_failures_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("run", "all"):
        phase_run(args, out_dir)
    if args.phase in ("render", "all"):
        phase_render(args, out_dir)


if __name__ == "__main__":
    main()
