#!/usr/bin/env python3
"""Collision-critical CV vs WM forecast errors on shared overlap failures.

Scene filter (original fail1500 compare JSON):
  GT rollout succeeds, WM fails with overlap, CV also fails with overlap.

Re-runs Diffusion-ES + WM (collision-only scoring) to recover the executed
ego plan, then scores that plan against GT / WM / CV other-agent boxes.
Does not change the planner, scorer, or training.

Outputs CSV, summary JSON/tables, a margin-vs-recall sweep, and a few BEV
example PNGs.

Example::

  python scripts/analyze_wm_cv_collider_errors.py --max_scenes 2
  sbatch slurm_analyze_wm_cv_collider.sh
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

import diagnose_wm_overlap_gap as diag  # noqa: E402
import viz_wm_cv_gt_forecast as vizmod  # noqa: E402
from data.scenario_loader import load_scenario_state_batch_fast  # noqa: E402
from lane_graph.lane_graph_utils import lane_graph_zip_path_for_tfrecord  # noqa: E402
from planner.diffusion_es_planner import DiffusionESPlanner  # noqa: E402
from scores import helpers as scorer_helpers  # noqa: E402
from scores.lane_graph_utils import LaneGraphShardStore  # noqa: E402
from simulation.evaluation_utils import (  # noqa: E402
    check_goal_reaching,
    infer_goals_from_sim_state,
    rollout_predicted_trajectories_with_metrics,
)
from simulation.planning_utils import apply_ego_replacements_to_expanded_state  # noqa: E402
from waymax import config as waymax_config  # noqa: E402

DEFAULT_COMPARE_DIR = (
    "/zfsauton/scratch/sbellad/immediate_exp/cv_wm_gt_compare_20260915_175107"
)
DEFAULT_TFRECORD_DIR = (
    "/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0/tfrecords/training"
)
DEFAULT_LANE_GRAPH_DIR = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DEFAULT_DIFFUSION_CKPT = (
    "/zfsauton/scratch/sbellad/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
)
DEFAULT_WM_CKPT = (
    "/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt"
)
DEFAULT_OUT_ROOT = "/zfsauton/scratch/sbellad/immediate_exp"

MARGINS_M = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)

CSV_FIELDS = [
    "tfrecord",
    "scenario_idx",
    "replan_t",
    "hit_t",
    "is_pre_overlap_replan",
    "collider_id",
    "hits_gt",
    "hits_wm",
    "hits_cv",
    "wm_fn",
    "cv_fn",
    "dt_to_hit",
    "wm_center_error_m",
    "cv_center_error_m",
    "wm_err_x_m",
    "cv_err_x_m",
    "wm_err_y_m",
    "cv_err_y_m",
    "wm_yaw_error_rad",
    "cv_yaw_error_rad",
    "wm_yaw_error_deg",
    "cv_yaw_error_deg",
    "wm_box_corner_error_m",
    "cv_box_corner_error_m",
    "wm_collider_ade_m",
    "cv_collider_ade_m",
    "wm_collider_fde_m",
    "cv_collider_fde_m",
    "wm_pred_gt_box_signed_m",
    "cv_pred_gt_box_signed_m",
    "ego_wm_signed_m",
    "ego_cv_signed_m",
    "ego_gt_signed_m",
    "min_ego_wm_signed_horizon_m",
    "min_ego_cv_signed_horizon_m",
    "min_ego_gt_signed_horizon_m",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare_dir", type=str, default=DEFAULT_COMPARE_DIR)
    p.add_argument("--tfrecord_dir", type=str, default=DEFAULT_TFRECORD_DIR)
    p.add_argument("--lane_graph_dir", type=str, default=DEFAULT_LANE_GRAPH_DIR)
    p.add_argument("--diffusion_ckpt_path", type=str, default=DEFAULT_DIFFUSION_CKPT)
    p.add_argument("--world_model_ckpt", type=str, default=DEFAULT_WM_CKPT)
    p.add_argument("--world_model_device", type=str, default="auto")
    p.add_argument("--wm_refresh_steps", type=int, default=10)
    p.add_argument("--start_timestep", type=int, default=10)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--population_size", type=int, default=64)
    p.add_argument("--elite_size", type=int, default=8)
    p.add_argument("--num_es_iterations", type=int, default=4)
    p.add_argument("--resample_timesteps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nearby_radius_m", type=float, default=20.0)
    p.add_argument("--max_scenes", type=int, default=0, help="0 = all matching scenes.")
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


def load_gt_ok_wm_cv_overlap_fail_scenes(compare_dir: Path) -> list[dict[str, Any]]:
    wm_gt = json.loads((compare_dir / "wm_vs_gt.json").read_text())
    cv_gt = json.loads((compare_dir / "cv_vs_gt.json").read_text())
    wm_fail = {
        (str(r["tfrecord"]), int(r["scenario_idx"]))
        for r in wm_gt.get("gt_only_win_scenes", [])
        if str(r.get("wm_failure", "")).lower() == "overlap"
    }
    cv_fail = {
        (str(r["tfrecord"]), int(r["scenario_idx"]))
        for r in cv_gt.get("gt_only_win_scenes", [])
        if str(r.get("cv_failure", "")).lower() in {"overlap", "overlap+offroad"}
    }
    keys = sorted(wm_fail & cv_fail)
    if not keys:
        raise ValueError(f"No GT-ok / WM-overlap / CV-overlap scenes in {compare_dir}")
    return [{"tfrecord": tf, "scenario_idx": idx} for tf, idx in keys]


def _wrap_pi(rad: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(rad) + np.pi) % (2.0 * np.pi) - np.pi


def _obb_corners(box5: np.ndarray) -> np.ndarray:
    x, y, yaw, length, width = [float(v) for v in np.asarray(box5, dtype=np.float64)[:5]]
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = 0.5 * length * c, 0.5 * length * s
    wx, wy = 0.5 * width * s, 0.5 * width * c
    return np.array(
        [
            [x + dx - wx, y + dy + wy],
            [x + dx + wx, y + dy - wy],
            [x - dx + wx, y - dy - wy],
            [x - dx - wx, y - dy + wy],
        ],
        dtype=np.float64,
    )


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _polygon_separation(pa: np.ndarray, pb: np.ndarray) -> float:
    d = 1e9
    for poly, other in ((pa, pb), (pb, pa)):
        for i in range(4):
            a, b = poly[i], poly[(i + 1) % 4]
            for p in other:
                d = min(d, _point_seg_dist(p, a, b))
    return float(d)


def signed_obb_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Positive = separated, 0 ~ touching, negative = SAT penetration depth."""
    pa = _obb_corners(box_a)
    pb = _obb_corners(box_b)
    edges_a = np.roll(pa, -1, axis=0) - pa
    edges_b = np.roll(pb, -1, axis=0) - pb
    axes = np.concatenate(
        [np.stack([-edges_a[:, 1], edges_a[:, 0]], axis=1), np.stack([-edges_b[:, 1], edges_b[:, 0]], axis=1)],
        axis=0,
    )
    norms = np.linalg.norm(axes, axis=1, keepdims=True).clip(1e-6)
    axes = axes / norms
    min_overlap = 1e9
    separated = False
    for axis in axes:
        a_proj = pa @ axis
        b_proj = pb @ axis
        overlap = float(min(a_proj.max(), b_proj.max()) - max(a_proj.min(), b_proj.min()))
        if overlap < 0.0:
            separated = True
        else:
            min_overlap = min(min_overlap, overlap)
    if separated:
        return _polygon_separation(pa, pb)
    return -float(min_overlap)


def inflate_boxes(boxes: np.ndarray, margin_m: float) -> np.ndarray:
    out = np.array(boxes, copy=True, dtype=np.float32)
    out[..., 3] = out[..., 3] + (2.0 * float(margin_m))
    out[..., 4] = out[..., 4] + (2.0 * float(margin_m))
    return out


def _finite(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _stats(values: list[float | None]) -> dict[str, float | None]:
    xs = [_finite(v) for v in values]
    xs = [x for x in xs if x is not None]
    if not xs:
        return {k: None for k in ("n", "mean", "median", "p75", "p90", "p95", "max", "std")}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _min_signed_over_horizon(ego_t5: np.ndarray, obj_t5: np.ndarray, mask: np.ndarray) -> float | None:
    h = min(int(ego_t5.shape[0]), int(obj_t5.shape[0]), int(mask.shape[0]))
    ds: list[float] = []
    for t in range(h):
        if not bool(mask[t]):
            continue
        ds.append(signed_obb_distance(ego_t5[t], obj_t5[t]))
    if not ds:
        return None
    return float(min(ds))


def collider_errors_at(
    forecasts: dict[str, Any],
    *,
    collider_idx: int,
    dt: int,
    ego_t5: np.ndarray,
) -> dict[str, Any]:
    i = int(collider_idx)
    h = int(forecasts["h"])
    if not (0 <= dt < h):
        return {}
    gt = np.asarray(forecasts["gt"][i, dt], dtype=np.float64)
    wm = np.asarray(forecasts["wm"][i, dt], dtype=np.float64)
    cv = np.asarray(forecasts["cv"][i, dt], dtype=np.float64)
    gt_ok = bool(forecasts["gt_mask"][i, dt])
    if not gt_ok:
        return {}

    def _vs(pred: np.ndarray) -> dict[str, float]:
        dxy = pred[:2] - gt[:2]
        yaw = float(_wrap_pi(pred[2] - gt[2]))
        corners_p = _obb_corners(pred)
        corners_g = _obb_corners(gt)
        corner = float(np.linalg.norm(corners_p - corners_g, axis=1).mean())
        return {
            "center": float(np.linalg.norm(dxy)),
            "x": float(dxy[0]),
            "y": float(dxy[1]),
            "yaw_rad": yaw,
            "yaw_deg": float(abs(np.rad2deg(yaw))),
            "corner": corner,
            "pred_gt_signed": signed_obb_distance(pred, gt),
        }

    wm_e = _vs(wm)
    cv_e = _vs(cv)
    ego_box = np.asarray(ego_t5[min(dt, int(ego_t5.shape[0]) - 1)], dtype=np.float64)
    wm_ade, wm_fde, _ = diag._ade_fde(
        forecasts["wm"][i, :h, :2],
        forecasts["gt"][i, :h, :2],
        forecasts["gt_mask"][i, :h],
    )
    cv_ade, cv_fde, _ = diag._ade_fde(
        forecasts["cv"][i, :h, :2],
        forecasts["gt"][i, :h, :2],
        forecasts["gt_mask"][i, :h],
    )
    return {
        "wm": wm_e,
        "cv": cv_e,
        "ego_wm_signed": signed_obb_distance(ego_box, wm),
        "ego_cv_signed": signed_obb_distance(ego_box, cv),
        "ego_gt_signed": signed_obb_distance(ego_box, gt),
        "min_ego_wm": _min_signed_over_horizon(ego_t5, forecasts["wm"][i], forecasts["wm_mask"][i]),
        "min_ego_cv": _min_signed_over_horizon(ego_t5, forecasts["cv"][i], forecasts["cv_mask"][i]),
        "min_ego_gt": _min_signed_over_horizon(ego_t5, forecasts["gt"][i], forecasts["gt_mask"][i]),
        "wm_ade": wm_ade,
        "wm_fde": wm_fde,
        "cv_ade": cv_ade,
        "cv_fde": cv_fde,
    }


def margin_hits(ego_boxes: np.ndarray, obj: np.ndarray, mask: np.ndarray, margin_m: float) -> bool:
    return diag.plan_collides(ego_boxes, inflate_boxes(obj, margin_m), mask)


def run_scene(args: argparse.Namespace, planner: DiffusionESPlanner, scene: dict[str, Any]) -> dict[str, Any]:
    tf_name = str(scene["tfrecord"])
    scenario_idx = int(scene["scenario_idx"])
    tf_path = Path(args.tfrecord_dir) / tf_name
    if not tf_path.exists():
        raise FileNotFoundError(tf_path)

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tf_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(1,),
        shuffle_seed=0,
    )
    sim_state, _ = load_scenario_state_batch_fast(ds_cfg, [scenario_idx])
    zip_path = lane_graph_zip_path_for_tfrecord(str(tf_path), args.lane_graph_dir)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    store = LaneGraphShardStore(zip_path.as_posix())
    planner.scorers[0].reset()
    planner.scorers[0].set_lane_graph(store.get_by_record_index(scenario_idx))
    scorer_helpers.update_lane_points(planner.scorers[0], sim_state, 0)
    if planner.world_model_forecaster is not None:
        planner.world_model_forecaster.clear_cache()

    goal_xy, _, _ = infer_goals_from_sim_state(sim_state)
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t = int(args.start_timestep)
    interval = int(args.replan_interval_steps)

    rng = jax.random.PRNGKey(int(args.seed))
    replaced = sim_state
    stored: dict[int, dict[str, Any]] = {}
    last_plan_t = start_t
    for step in range(start_t, episode_num_steps - 1, interval):
        rng, key_plan = jax.random.split(rng)
        plan_result = planner.plan_trajectory(
            replaced,
            goal_xy,
            rng=key_plan,
            timestep=step,
            mask_goal=False,
            use_subgoal=False,
        )
        plan_t5 = np.asarray(plan_result.trajectory_world_bt5[0], dtype=np.float32)
        forecasts = diag.forecast_bundle(
            replaced,
            timestep=step,
            horizon=int(plan_t5.shape[0]),
            wm_forecaster=planner.world_model_forecaster,
        )
        h = int(forecasts["h"])
        ego_boxes = diag.ego_plan_boxes(plan_t5, replaced, timestep=step, horizon=h)
        stored[int(step)] = {
            "plan_t5": plan_t5,
            "ego_boxes": ego_boxes,
            "forecasts": forecasts,
            "hits": {
                "gt": diag.plan_collides(ego_boxes, forecasts["gt"], forecasts["gt_mask"]),
                "wm": diag.plan_collides(ego_boxes, forecasts["wm_col"], forecasts["wm_col_mask"]),
                "cv": diag.plan_collides(ego_boxes, forecasts["cv_col"], forecasts["cv_col_mask"]),
            },
        }
        replaced = apply_ego_replacements_to_expanded_state(
            replaced,
            start_t_b=step,
            traj_bt5=plan_t5[None, ...],
        )
        last_plan_t = int(step)

    rollout = rollout_predicted_trajectories_with_metrics(
        replaced,
        rng_key=jax.random.PRNGKey(0),
        metric_names=("overlap",),
    )
    t_hit = diag.first_overlap_step(
        np.asarray(rollout["metric_timeseries"]["overlap"]),
        start_timestep=start_t,
        world_idx=0,
    )
    goal_eval = check_goal_reaching(replaced, goal_xy, start_timestep=start_t)
    collider_idx, collider_dist = (None, None)
    if t_hit is not None:
        collider_idx, collider_dist = diag.colliding_agent_at(replaced, timestep=t_hit, world_idx=0)
    t_pre = last_plan_t
    if t_hit is not None:
        t_pre = diag.last_replan_at_or_before(
            t_hit, start_timestep=start_t, interval=interval, last_replan=last_plan_t
        )

    rows: list[dict[str, Any]] = []
    for step, blob in stored.items():
        hits = blob["hits"]
        forecasts = blob["forecasts"]
        h = int(forecasts["h"])
        dt = None if t_hit is None else int(t_hit) - int(step)
        err = {}
        if collider_idx is not None and dt is not None and 0 <= dt < h:
            err = collider_errors_at(
                forecasts,
                collider_idx=int(collider_idx),
                dt=dt,
                ego_t5=blob["ego_boxes"],
            )
        wm_e = err.get("wm") or {}
        cv_e = err.get("cv") or {}
        rows.append(
            {
                "tfrecord": tf_name,
                "scenario_idx": scenario_idx,
                "replan_t": int(step),
                "hit_t": t_hit,
                "is_pre_overlap_replan": bool(t_hit is not None and int(step) == int(t_pre)),
                "collider_id": collider_idx,
                "hits_gt": bool(hits["gt"]),
                "hits_wm": bool(hits["wm"]),
                "hits_cv": bool(hits["cv"]),
                "wm_fn": bool((not hits["wm"]) and hits["gt"]),
                "cv_fn": bool((not hits["cv"]) and hits["gt"]),
                "dt_to_hit": dt,
                "wm_center_error_m": wm_e.get("center"),
                "cv_center_error_m": cv_e.get("center"),
                "wm_err_x_m": wm_e.get("x"),
                "cv_err_x_m": cv_e.get("x"),
                "wm_err_y_m": wm_e.get("y"),
                "cv_err_y_m": cv_e.get("y"),
                "wm_yaw_error_rad": wm_e.get("yaw_rad"),
                "cv_yaw_error_rad": cv_e.get("yaw_rad"),
                "wm_yaw_error_deg": wm_e.get("yaw_deg"),
                "cv_yaw_error_deg": cv_e.get("yaw_deg"),
                "wm_box_corner_error_m": wm_e.get("corner"),
                "cv_box_corner_error_m": cv_e.get("corner"),
                "wm_collider_ade_m": err.get("wm_ade"),
                "cv_collider_ade_m": err.get("cv_ade"),
                "wm_collider_fde_m": err.get("wm_fde"),
                "cv_collider_fde_m": err.get("cv_fde"),
                "wm_pred_gt_box_signed_m": wm_e.get("pred_gt_signed"),
                "cv_pred_gt_box_signed_m": cv_e.get("pred_gt_signed"),
                "ego_wm_signed_m": err.get("ego_wm_signed"),
                "ego_cv_signed_m": err.get("ego_cv_signed"),
                "ego_gt_signed_m": err.get("ego_gt_signed"),
                "min_ego_wm_signed_horizon_m": err.get("min_ego_wm"),
                "min_ego_cv_signed_horizon_m": err.get("min_ego_cv"),
                "min_ego_gt_signed_horizon_m": err.get("min_ego_gt"),
            }
        )

    margin_rows = []
    for step, blob in stored.items():
        ego_boxes = blob["ego_boxes"]
        fc = blob["forecasts"]
        gt_hit = bool(blob["hits"]["gt"])
        rec = {
            "replan_t": int(step),
            "is_pre_overlap_replan": bool(t_hit is not None and int(step) == int(t_pre)),
            "hits_gt": gt_hit,
        }
        for m in MARGINS_M:
            rec[f"wm_m{m:.2f}"] = margin_hits(ego_boxes, fc["wm_col"], fc["wm_col_mask"], m)
            rec[f"cv_m{m:.2f}"] = margin_hits(ego_boxes, fc["cv_col"], fc["cv_col_mask"], m)
        margin_rows.append(rec)

    pre_blob = stored.get(int(t_pre))
    return {
        "tfrecord": tf_name,
        "scenario_idx": scenario_idx,
        "sim_state": sim_state,
        "reproduced_overlap": t_hit is not None,
        "goal_reached": bool(goal_eval["reached"][0]),
        "t_hit": t_hit,
        "t_replan": int(t_pre),
        "collider_idx": collider_idx,
        "collider_dist_m": collider_dist,
        "rows": rows,
        "margin_rows": margin_rows,
        "pre_blob": pre_blob,
        "pre_hits": None if pre_blob is None else pre_blob["hits"],
    }


def _fmt_table(stats_wm: dict, stats_cv: dict) -> str:
    keys = ("mean", "median", "p75", "p90", "p95", "max", "std")
    lines = [f"{'metric':<28} {'wm':>10} {'cv':>10}"]
    for k in keys:
        wv, cv = stats_wm.get(k), stats_cv.get(k)
        ws = "nan" if wv is None else f"{wv:10.4f}"
        cs = "nan" if cv is None else f"{cv:10.4f}"
        lines.append(f"{k:<28} {ws} {cs}")
    return "\n".join(lines)


def _pick_examples(scene_payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep one scene payload per requested visualization category."""
    best: dict[str, tuple[float, dict[str, Any]]] = {}

    def consider(name: str, score: float, payload: dict[str, Any]) -> None:
        cur = best.get(name)
        if cur is None or score > cur[0]:
            best[name] = (score, payload)

    for p in scene_payloads:
        pre = next((r for r in p["rows"] if r["is_pre_overlap_replan"]), None)
        if pre is None or pre.get("wm_center_error_m") is None:
            continue
        wm_c = float(pre["wm_center_error_m"])
        cv_c = float(pre["cv_center_error_m"])
        if pre["wm_fn"]:
            consider("wm_fn_about_10cm", -abs(wm_c - 0.10), p)
            if wm_c <= 0.03:
                consider("near_zero_center_but_fn", -wm_c, p)
        consider("cv_much_worse_than_wm", cv_c - wm_c, p)
        consider("cv_better_than_wm", wm_c - cv_c, p)
        consider("both_fail_badly", min(wm_c, cv_c) + 0.25 * (wm_c + cv_c), p)
        if int(p["scenario_idx"]) == 207:
            consider("scene_207", 1.0, p)
    return {k: v[1] for k, v in best.items()}


def _margin_summary(margin_rows: list[dict[str, Any]], *, pre_only: bool) -> list[dict[str, Any]]:
    rows = [r for r in margin_rows if (r["is_pre_overlap_replan"] if pre_only else True)]
    pos = [r for r in rows if r["hits_gt"]]
    neg = [r for r in rows if not r["hits_gt"]]
    out = []
    for m in MARGINS_M:
        wm_tp = sum(1 for r in pos if r[f"wm_m{m:.2f}"])
        cv_tp = sum(1 for r in pos if r[f"cv_m{m:.2f}"])
        wm_fp = sum(1 for r in neg if r[f"wm_m{m:.2f}"])
        cv_fp = sum(1 for r in neg if r[f"cv_m{m:.2f}"])
        n_pos, n_neg = len(pos), len(neg)
        out.append(
            {
                "margin_m": m,
                "n_gt_positive": n_pos,
                "wm_tp": wm_tp,
                "cv_tp": cv_tp,
                "wm_fn": n_pos - wm_tp,
                "cv_fn": n_pos - cv_tp,
                "wm_recall": (wm_tp / n_pos) if n_pos else None,
                "cv_recall": (cv_tp / n_pos) if n_pos else None,
                "n_gt_negative": n_neg,
                "wm_fp": wm_fp,
                "cv_fp": cv_fp,
                "wm_fp_rate": (wm_fp / n_neg) if n_neg else None,
                "cv_fp_rate": (cv_fp / n_neg) if n_neg else None,
            }
        )
    return out


def _write_margin_txt(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [title, f"{'margin':>8} {'WM rec':>10} {'CV rec':>10} {'WM FN':>8} {'CV FN':>8} {'WM FP':>8} {'CV FP':>8}"]
    for r in rows:
        wr = "nan" if r["wm_recall"] is None else f"{r['wm_recall']:.3f}"
        cr = "nan" if r["cv_recall"] is None else f"{r['cv_recall']:.3f}"
        lines.append(
            f"{r['margin_m']:8.2f} {wr:>10} {cr:>10} {r['wm_fn']:8d} {r['cv_fn']:8d} "
            f"{r['wm_fp']:8d} {r['cv_fp']:8d}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    scenes = load_gt_ok_wm_cv_overlap_fail_scenes(Path(args.compare_dir))
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT_ROOT) / f"wm_cv_collider_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] {len(scenes)} GT-ok / WM-overlap / CV-overlap scenes -> {out_dir}", flush=True)

    planner = vizmod.build_planner(args)
    csv_path = out_dir / "collider_replan_rows.csv"
    all_rows: list[dict[str, Any]] = []
    all_margin: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    n_err = 0
    n_overlap = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i, scene in enumerate(scenes, start=1):
            print(f"[{i}/{len(scenes)}] {scene['tfrecord']} scen {scene['scenario_idx']}", flush=True)
            try:
                payload = run_scene(args, planner, scene)
            except Exception as exc:
                n_err += 1
                print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
                continue
            n_overlap += int(bool(payload["reproduced_overlap"]))
            for row in payload["rows"]:
                writer.writerow({k: row.get(k) for k in CSV_FIELDS})
                all_rows.append(row)
            all_margin.extend(payload["margin_rows"])
            payloads.append(payload)
            pre = next((r for r in payload["rows"] if r["is_pre_overlap_replan"]), None)
            if pre:
                print(
                    f"  overlap={payload['reproduced_overlap']} t_hit={payload['t_hit']} "
                    f"hits gt/wm/cv={pre['hits_gt']}/{pre['hits_wm']}/{pre['hits_cv']} "
                    f"wm@hit={pre['wm_center_error_m']} cv@hit={pre['cv_center_error_m']}",
                    flush=True,
                )

    pre_rows = [r for r in all_rows if r.get("is_pre_overlap_replan")]
    gt_pos_all = [r for r in all_rows if r.get("hits_gt")]
    label_all = {
        "n_replans": len(all_rows),
        "wm_fn": sum(1 for r in all_rows if r["wm_fn"]),
        "cv_fn": sum(1 for r in all_rows if r["cv_fn"]),
        "wm_tp": sum(1 for r in all_rows if r["hits_wm"] and r["hits_gt"]),
        "cv_tp": sum(1 for r in all_rows if r["hits_cv"] and r["hits_gt"]),
        "n_gt_positive": len(gt_pos_all),
        "wm_recall": (
            sum(1 for r in gt_pos_all if r["hits_wm"]) / len(gt_pos_all) if gt_pos_all else None
        ),
        "cv_recall": (
            sum(1 for r in gt_pos_all if r["hits_cv"]) / len(gt_pos_all) if gt_pos_all else None
        ),
    }
    label_pre = {
        "n": len(pre_rows),
        "wm_fn": sum(1 for r in pre_rows if r["wm_fn"]),
        "cv_fn": sum(1 for r in pre_rows if r["cv_fn"]),
        "wm_recall": (
            sum(1 for r in pre_rows if r["hits_wm"] and r["hits_gt"])
            / max(1, sum(1 for r in pre_rows if r["hits_gt"]))
        ),
        "cv_recall": (
            sum(1 for r in pre_rows if r["hits_cv"] and r["hits_gt"])
            / max(1, sum(1 for r in pre_rows if r["hits_gt"]))
        ),
        "n_gt_positive": sum(1 for r in pre_rows if r["hits_gt"]),
    }

    crit_stats = {
        "pre_overlap_center_l2_m": {
            "wm": _stats([r["wm_center_error_m"] for r in pre_rows]),
            "cv": _stats([r["cv_center_error_m"] for r in pre_rows]),
        },
        "pre_overlap_yaw_deg": {
            "wm": _stats([r["wm_yaw_error_deg"] for r in pre_rows]),
            "cv": _stats([r["cv_yaw_error_deg"] for r in pre_rows]),
        },
        "pre_overlap_box_corner_m": {
            "wm": _stats([r["wm_box_corner_error_m"] for r in pre_rows]),
            "cv": _stats([r["cv_box_corner_error_m"] for r in pre_rows]),
        },
        "fn_only_center_l2_m": {
            "wm": _stats([r["wm_center_error_m"] for r in pre_rows if r["wm_fn"]]),
            "cv": _stats([r["cv_center_error_m"] for r in pre_rows if r["cv_fn"]]),
        },
    }
    margin_pre = _margin_summary(all_margin, pre_only=True)
    margin_all = _margin_summary(all_margin, pre_only=False)

    summary = {
        "scene_filter": "gt_success & wm_overlap_fail & cv_overlap_fail",
        "n_scenes_queued": len(scenes),
        "n_ok": len(payloads),
        "n_errors": n_err,
        "n_reproduced_overlap": n_overlap,
        "collision_labels_all_replans": label_all,
        "collision_labels_pre_overlap": label_pre,
        "critical_error_stats": crit_stats,
        "margin_sweep_pre_overlap": margin_pre,
        "margin_sweep_all_replans": margin_all,
        "csv": str(csv_path),
        "out_dir": str(out_dir),
        "note": (
            "Executed ego is WM-ES (the overlapping closed-loop). CV numbers are "
            "CV forecasts of others against that same ego plan, not a CV-ES re-rollout."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    txt = []
    txt.append("WM vs CV collision-critical errors (pre-overlap replan, collider @ hit)\n")
    txt.append("center L2 (m)")
    txt.append(_fmt_table(crit_stats["pre_overlap_center_l2_m"]["wm"], crit_stats["pre_overlap_center_l2_m"]["cv"]))
    txt.append("\nyaw |deg|")
    txt.append(_fmt_table(crit_stats["pre_overlap_yaw_deg"]["wm"], crit_stats["pre_overlap_yaw_deg"]["cv"]))
    txt.append("\nbox corner mean L2 (m)")
    txt.append(_fmt_table(crit_stats["pre_overlap_box_corner_m"]["wm"], crit_stats["pre_overlap_box_corner_m"]["cv"]))
    txt.append("\nlabels pre-overlap: " + json.dumps(label_pre))
    txt.append("labels all replans: " + json.dumps(label_all))
    (out_dir / "error_stats.txt").write_text("\n".join(txt) + "\n")
    _write_margin_txt(out_dir / "margin_sweep_pre_overlap.txt", margin_pre, "Pre-overlap replan (hits_gt positives)")
    _write_margin_txt(out_dir / "margin_sweep_all_replans.txt", margin_all, "All replans (hits_gt positives)")

    examples_dir = out_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    picks = _pick_examples(payloads)
    for name, payload in picks.items():
        blob = payload.get("pre_blob")
        if blob is None:
            continue
        pre = next((r for r in payload["rows"] if r["is_pre_overlap_replan"]), None)
        png = examples_dir / f"{name}_{Path(payload['tfrecord']).name}_scen{payload['scenario_idx']}.png"
        vizmod.render_scene_png(
            sim_state=payload["sim_state"],
            plan_t5=blob["plan_t5"],
            forecasts=blob["forecasts"],
            t_replan=int(payload["t_replan"]),
            t_hit=int(payload["t_hit"] if payload["t_hit"] is not None else payload["t_replan"]),
            collider_idx=payload["collider_idx"],
            hits=blob["hits"],
            collider_ade={
                "wm_ade_m": None if pre is None else pre.get("wm_collider_ade_m"),
                "cv_ade_m": None if pre is None else pre.get("cv_collider_ade_m"),
            },
            err_at_hit=(
                None
                if pre is None or pre.get("wm_center_error_m") is None
                else {
                    "wm_err_m": pre["wm_center_error_m"],
                    "cv_err_m": pre["cv_center_error_m"],
                }
            ),
            nearby_radius_m=float(args.nearby_radius_m),
            title=f"{name}  {payload['tfrecord']} scen {payload['scenario_idx']}",
            out_png=png,
        )
        print(f"  example {name} -> {png.name}", flush=True)

    print(json.dumps({k: summary[k] for k in ("n_ok", "n_errors", "n_reproduced_overlap", "out_dir")}, indent=2), flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"margin table: {out_dir / 'margin_sweep_pre_overlap.txt'}", flush=True)


if __name__ == "__main__":
    main()
