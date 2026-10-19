#!/usr/bin/env python3
"""BEV overlay of GT vs CV vs WM other-agent forecasts.

Default scene set: CV and GT both succeeded, WM failed with overlap
(intersection of cv_vs_wm cv-only overlap wins and wm_vs_gt gt-only overlap wins).

Re-runs Diffusion-ES + WM (collision-only scoring, to avoid the offroad OOM)
so the ego polyline is the plan that actually collided. At the replan before
the first overlap, draws:

  left: 30 m context, centerlines of nearby agents
  right: zoom on the collider, oriented boxes at the hit timestep

Example::

  python scripts/viz_wm_cv_gt_forecast.py --max_scenes 3
  sbatch slurm_viz_wm_cv_gt_forecast.sh
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
from flax import nnx  # noqa: E402
from waymax import config as waymax_config  # noqa: E402

import diagnose_wm_overlap_gap as diag  # noqa: E402
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
from simulation.run_simulation import _load_world_model_forecaster  # noqa: E402
from train_diffusion.utils.infer import load_model_for_inference  # noqa: E402
from viz.utils import plot_numpy_bounding_boxes  # noqa: E402
from viz.viz import _index_pytree, plot_roadgraph_points  # noqa: E402

DEFAULT_COMPARE_DIR = (
    "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/cv_wm_gt_compare_20260915_175107"
)
DEFAULT_TFRECORD_DIR = (
    "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0/tfrecords/training"
)
DEFAULT_LANE_GRAPH_DIR = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DEFAULT_DIFFUSION_CKPT = (
    "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
)
DEFAULT_WM_CKPT = (
    "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt"
)
DEFAULT_OUT_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp"

COLOR_GT = np.array([0.12, 0.55, 0.18], dtype=np.float32)
COLOR_CV = np.array([0.12, 0.38, 0.88], dtype=np.float32)
COLOR_WM = np.array([0.88, 0.22, 0.10], dtype=np.float32)
COLOR_EGO = np.array([0.62, 0.12, 0.78], dtype=np.float32)
COLOR_OTHER = np.array([0.45, 0.45, 0.45], dtype=np.float32)


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


def load_cv_gt_ok_wm_overlap_scenes(compare_dir: Path) -> list[dict[str, Any]]:
    """Scenes where CV and GT both succeed and WM fails with overlap."""
    gt_payload = json.loads((compare_dir / "wm_vs_gt.json").read_text())
    cv_payload = json.loads((compare_dir / "cv_vs_wm.json").read_text())
    gt_keys = {
        (str(r["tfrecord"]), int(r["scenario_idx"]))
        for r in gt_payload.get("gt_only_win_scenes", [])
        if str(r.get("wm_failure", "")).lower() == "overlap"
    }
    cv_keys = {
        (str(r["tfrecord"]), int(r["scenario_idx"]))
        for r in cv_payload.get("cv_only_win_scenes", [])
        if str(r.get("wm_failure", "")).lower() == "overlap"
    }
    keys = sorted(gt_keys & cv_keys)
    if not keys:
        raise ValueError(
            f"No CV&GT-ok / WM-overlap-fail scenes in {compare_dir} "
            f"(gt_only overlap={len(gt_keys)}, cv_only overlap={len(cv_keys)})"
        )
    return [{"tfrecord": tf, "scenario_idx": idx} for tf, idx in keys]


def build_planner(args: argparse.Namespace) -> DiffusionESPlanner:
    inference = load_model_for_inference(
        checkpoint_path=args.diffusion_ckpt_path,
        use_ema=True,
        metadata_path=None,
        seed=int(args.seed),
        skip_checkpoint_load=False,
    )
    policy = nnx.merge(inference.graphdef, inference.params_state, inference.nonparam_state)
    wm_args = argparse.Namespace(
        other_agent_forecast="wm",
        world_model_ckpt=args.world_model_ckpt,
        world_model_device=args.world_model_device,
        wm_refresh_steps=int(args.wm_refresh_steps),
    )
    # Collision only: offroad v2 OOMs on 24 GB for dense roadgraphs.
    return DiffusionESPlanner(
        policy=policy,
        preprocess_cfg=inference.preprocess_cfg,
        population_size=int(args.population_size),
        resample_timesteps=int(args.resample_timesteps),
        elite_size=int(args.elite_size),
        num_iterations=int(args.num_es_iterations),
        num_worlds=1,
        metrics=["collision"],
        other_agent_forecast="wm",
        world_model_forecaster=_load_world_model_forecaster(wm_args),
    )


def _box_plot5(box_xyyawlw: np.ndarray) -> np.ndarray:
    """(x, y, yaw, length, width) -> (x, y, length, width, yaw) for viz helper."""
    arr = np.asarray(box_xyyawlw, dtype=np.float32)
    return np.stack([arr[..., 0], arr[..., 1], arr[..., 3], arr[..., 4], arr[..., 2]], axis=-1)


def _nearby_mask(sim_state, *, timestep: int, radius_m: float) -> np.ndarray:
    ego = int(np.argmax(np.asarray(sim_state.object_metadata.is_sdc[0]).astype(np.int32)))
    t = int(np.clip(timestep, 0, int(sim_state.log_trajectory.x.shape[-1]) - 1))
    xy = np.asarray(sim_state.log_trajectory.xy[0, :, t], dtype=np.float32)
    valid = np.asarray(sim_state.log_trajectory.valid[0, :, t]).astype(bool)
    types = np.asarray(sim_state.object_metadata.object_types[0], dtype=np.int32)
    is_sdc = np.asarray(sim_state.object_metadata.is_sdc[0]).astype(bool)
    dist = np.linalg.norm(xy - xy[ego], axis=-1)
    dynamic = np.isin(types, list(diag._DYNAMIC_TYPES))
    return valid & (~is_sdc) & dynamic & (dist <= float(radius_m))


def _draw_polyline(ax, xy: np.ndarray, *, color, lw: float, ls: str, alpha: float, z: int) -> None:
    pts = np.asarray(xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        lw=lw,
        ls=ls,
        alpha=alpha,
        zorder=z,
        solid_capstyle="round",
    )


def render_scene_png(
    *,
    sim_state,
    plan_t5: np.ndarray,
    forecasts: dict[str, Any],
    t_replan: int,
    t_hit: int,
    collider_idx: int | None,
    hits: dict[str, bool],
    collider_ade: dict[str, Any] | None,
    err_at_hit: dict[str, Any] | None,
    nearby_radius_m: float,
    title: str,
    out_png: Path,
) -> None:
    state0 = _index_pytree(sim_state, 0)
    ego = int(np.argmax(np.asarray(state0.object_metadata.is_sdc).astype(np.int32)))
    h = int(forecasts["h"])
    dt = int(np.clip(int(t_hit) - int(t_replan), 0, h - 1))
    nearby = _nearby_mask(sim_state, timestep=t_replan, radius_m=nearby_radius_m)

    ego_now = np.asarray(plan_t5[0, :2], dtype=np.float32)
    if collider_idx is not None:
        col_now = np.asarray(forecasts["gt"][int(collider_idx), 0, :2], dtype=np.float32)
        origin = 0.5 * (ego_now + col_now)
        zoom_origin = 0.5 * (
            np.asarray(plan_t5[min(dt, len(plan_t5) - 1), :2], dtype=np.float32)
            + np.asarray(forecasts["gt"][int(collider_idx), dt, :2], dtype=np.float32)
        )
    else:
        origin = ego_now
        zoom_origin = ego_now

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), dpi=120)
    ctx_m, zoom_m = 28.0, 10.0

    for ax, half, is_zoom in ((axes[0], ctx_m, False), (axes[1], zoom_m, True)):
        plot_roadgraph_points(ax, state0.roadgraph_points, verbose=False)
        center = zoom_origin if is_zoom else origin
        ax.set_xlim(float(center[0]) - half, float(center[0]) + half)
        ax.set_ylim(float(center[1]) - half, float(center[1]) + half)
        ax.set_aspect("equal")
        ax.set_title("hit-time boxes (zoom)" if is_zoom else "futures at replan")
        ax.grid(False)

        t = int(np.clip(t_replan, 0, int(state0.log_trajectory.x.shape[-1]) - 1))
        xy = np.asarray(state0.log_trajectory.xy[:, t], dtype=np.float32)
        yaw = np.asarray(state0.log_trajectory.yaw[:, t], dtype=np.float32)
        length = np.asarray(state0.log_trajectory.length[:, t], dtype=np.float32)
        width = np.asarray(state0.log_trajectory.width[:, t], dtype=np.float32)
        valid = np.asarray(state0.log_trajectory.valid[:, t]).astype(bool)
        cur = np.stack([xy[:, 0], xy[:, 1], length, width, yaw], axis=-1)
        show = valid & nearby
        if collider_idx is not None:
            show[int(collider_idx)] = False
        if bool(show.any()):
            plot_numpy_bounding_boxes(ax, cur[show], COLOR_OTHER, alpha=0.35)

        n_agents = int(forecasts["gt"].shape[0])
        for i in range(n_agents):
            if not bool(nearby[i]) and i != collider_idx:
                continue
            is_col = collider_idx is not None and i == int(collider_idx)
            alpha = 1.0 if is_col else 0.28
            lw = 2.6 if is_col else 1.0
            z = 18 if is_col else 8
            gt_m = np.asarray(forecasts["gt_mask"][i]).astype(bool)
            wm_m = np.asarray(forecasts["wm_mask"][i]).astype(bool)
            cv_m = np.asarray(forecasts["cv_mask"][i]).astype(bool)
            if not is_zoom:
                if bool(gt_m.any()):
                    _draw_polyline(ax, forecasts["gt"][i, gt_m, :2], color=COLOR_GT, lw=lw, ls="-", alpha=alpha, z=z)
                if bool(cv_m.any()):
                    _draw_polyline(ax, forecasts["cv"][i, cv_m, :2], color=COLOR_CV, lw=lw, ls="--", alpha=alpha, z=z + 1)
                if bool(wm_m.any()):
                    _draw_polyline(ax, forecasts["wm"][i, wm_m, :2], color=COLOR_WM, lw=lw, ls="-.", alpha=alpha, z=z + 2)

        _draw_polyline(
            ax,
            plan_t5[:h, :2],
            color=COLOR_EGO,
            lw=2.8,
            ls="-",
            alpha=1.0,
            z=20,
        )

        ego_hit = np.asarray(plan_t5[min(dt, len(plan_t5) - 1)], dtype=np.float32)
        ego_l = float(state0.log_trajectory.length[ego, t])
        ego_w = float(state0.log_trajectory.width[ego, t])
        ego_box = np.array([[ego_hit[0], ego_hit[1], ego_l, ego_w, ego_hit[2]]], dtype=np.float32)
        plot_numpy_bounding_boxes(ax, ego_box, COLOR_EGO, alpha=0.95)
        if collider_idx is not None:
            i = int(collider_idx)
            for src, color in (("gt", COLOR_GT), ("cv", COLOR_CV), ("wm", COLOR_WM)):
                box = np.asarray(forecasts[src][i, dt], dtype=np.float32)
                plot_numpy_bounding_boxes(ax, _box_plot5(box)[None, ...], color, alpha=0.95)

    ade_txt = ""
    if collider_ade:
        wm_ade = collider_ade.get("wm_ade_m")
        cv_ade = collider_ade.get("cv_ade_m")
        ade_txt = f"  collider ADE  WM={wm_ade}  CV={cv_ade}"
    hit_txt = ""
    if err_at_hit:
        hit_txt = (
            f"  err@hit  WM={err_at_hit.get('wm_err_m'):.3f}m  "
            f"CV={err_at_hit.get('cv_err_m'):.3f}m"
        )
    fig.suptitle(
        f"{title}\n"
        f"replan t={t_replan}  hit t={t_hit}  "
        f"plan hits  GT={hits['gt']}  CV={hits['cv']}  WM={hits['wm']}"
        f"{ade_txt}{hit_txt}",
        fontsize=10,
    )
    legend = [
        Line2D([0], [0], color=COLOR_GT, lw=2, label="GT others"),
        Line2D([0], [0], color=COLOR_CV, lw=2, ls="--", label="CV others"),
        Line2D([0], [0], color=COLOR_WM, lw=2, ls="-.", label="WM others"),
        Line2D([0], [0], color=COLOR_EGO, lw=2.5, label="WM-ES ego plan"),
        Patch(facecolor=COLOR_OTHER, edgecolor=COLOR_OTHER, alpha=0.4, label="nearby now"),
    ]
    axes[0].legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


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
    graph = store.get_by_record_index(scenario_idx)
    planner.scorers[0].reset()
    planner.scorers[0].set_lane_graph(graph)
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
        horizon = int(plan_t5.shape[0])
        forecasts = diag.forecast_bundle(
            replaced,
            timestep=step,
            horizon=horizon,
            wm_forecaster=planner.world_model_forecaster,
        )
        h = int(forecasts["h"])
        ego_boxes = diag.ego_plan_boxes(plan_t5, replaced, timestep=step, horizon=h)
        stored[int(step)] = {
            "plan_t5": plan_t5,
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
    overlap = t_hit is not None
    goal_eval = check_goal_reaching(replaced, goal_xy, start_timestep=start_t)

    plot_t = last_plan_t
    if t_hit is not None:
        plot_t = diag.last_replan_at_or_before(
            t_hit,
            start_timestep=start_t,
            interval=interval,
            last_replan=last_plan_t,
        )
    blob = stored[int(plot_t)]
    collider_idx, collider_dist = (None, None)
    if t_hit is not None:
        collider_idx, collider_dist = diag.colliding_agent_at(replaced, timestep=t_hit, world_idx=0)

    collider_ade = None
    err_at_hit = None
    if collider_idx is not None:
        mets = diag.agent_metrics(
            blob["forecasts"],
            replaced,
            timestep=plot_t,
            nearby_radius_m=float(args.nearby_radius_m),
            collider_idx=collider_idx,
        )
        collider_ade = mets.get("collider")
        dt = int(t_hit) - int(plot_t)
        h = int(blob["forecasts"]["h"])
        if 0 <= dt < h:
            gt_xy = blob["forecasts"]["gt"][int(collider_idx), dt, :2]
            wm_xy = blob["forecasts"]["wm"][int(collider_idx), dt, :2]
            cv_xy = blob["forecasts"]["cv"][int(collider_idx), dt, :2]
            err_at_hit = {
                "dt": dt,
                "wm_err_m": float(np.linalg.norm(wm_xy - gt_xy)),
                "cv_err_m": float(np.linalg.norm(cv_xy - gt_xy)),
            }

    return {
        "tfrecord": tf_name,
        "scenario_idx": scenario_idx,
        "sim_state": sim_state,
        "reproduced_overlap": bool(overlap),
        "goal_reached": bool(goal_eval["reached"][0]),
        "t_hit": t_hit if t_hit is not None else int(plot_t),
        "t_replan": int(plot_t),
        "collider_idx": collider_idx,
        "collider_dist_m": collider_dist,
        "plan_t5": blob["plan_t5"],
        "forecasts": blob["forecasts"],
        "hits": blob["hits"],
        "collider_ade": collider_ade,
        "err_at_hit": err_at_hit,
    }


def _write_index(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    items = []
    for row in rows:
        png = row.get("png")
        if not png:
            continue
        rel = Path(png).name
        err = row.get("error")
        if err:
            cap = f"{row['tfrecord']} scen {row['scenario_idx']} ERROR: {err}"
        else:
            ade = row.get("collider_ade") or {}
            cap = (
                f"{row['tfrecord']} scen {row['scenario_idx']}  "
                f"overlap={row.get('reproduced_overlap')}  "
                f"hits GT/CV/WM={row.get('hits')}  "
                f"cv_ade={ade.get('cv_ade_m')} wm_ade={ade.get('wm_ade_m')}"
            )
        items.append(
            f'<div style="margin:16px 0"><h3>{cap}</h3>'
            f'<img src="{rel}" style="max-width:100%;border:1px solid #ccc"/></div>'
        )
    html = (
        "<html><body style='font-family:sans-serif'>"
        "<h1>WM vs CV vs GT (CV&amp;GT ok, WM overlap fail)</h1>"
        + "\n".join(items)
        + "</body></html>"
    )
    (out_dir / "index.html").write_text(html)


def main() -> None:
    args = parse_args()
    scenes = load_cv_gt_ok_wm_overlap_scenes(Path(args.compare_dir))
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT_ROOT) / f"viz_wm_cv_gt_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] {len(scenes)} CV&GT-ok / WM-overlap-fail scenes -> {out_dir}", flush=True)

    planner = build_planner(args)
    meta_rows: list[dict[str, Any]] = []
    jsonl_path = out_dir / "per_scene.jsonl"
    with jsonl_path.open("w") as f:
        for i, scene in enumerate(scenes, start=1):
            print(
                f"[{i}/{len(scenes)}] {scene['tfrecord']} scenario {scene['scenario_idx']}",
                flush=True,
            )
            try:
                row = run_scene(args, planner, scene)
                stem = f"{i:02d}_{Path(scene['tfrecord']).name}_scen{int(scene['scenario_idx'])}"
                png = out_dir / f"{stem}.png"
                render_scene_png(
                    sim_state=row["sim_state"],
                    plan_t5=row["plan_t5"],
                    forecasts=row["forecasts"],
                    t_replan=int(row["t_replan"]),
                    t_hit=int(row["t_hit"]),
                    collider_idx=row["collider_idx"],
                    hits=row["hits"],
                    collider_ade=row["collider_ade"],
                    err_at_hit=row["err_at_hit"],
                    nearby_radius_m=float(args.nearby_radius_m),
                    title=f"{scene['tfrecord']}  scen {scene['scenario_idx']}",
                    out_png=png,
                )
                slim = {
                    "tfrecord": row["tfrecord"],
                    "scenario_idx": row["scenario_idx"],
                    "reproduced_overlap": row["reproduced_overlap"],
                    "goal_reached": row["goal_reached"],
                    "t_hit": row["t_hit"],
                    "t_replan": row["t_replan"],
                    "collider_idx": row["collider_idx"],
                    "collider_dist_m": row["collider_dist_m"],
                    "hits": row["hits"],
                    "collider_ade": row["collider_ade"],
                    "err_at_hit": row["err_at_hit"],
                    "png": str(png),
                }
                print(
                    f"  overlap={slim['reproduced_overlap']} t_hit={slim['t_hit']} "
                    f"hits={slim['hits']} png={png.name}",
                    flush=True,
                )
            except Exception as exc:
                slim = {
                    "tfrecord": scene["tfrecord"],
                    "scenario_idx": int(scene["scenario_idx"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "png": "",
                }
                print(f"  ERROR: {slim['error']}", flush=True)
            meta_rows.append(slim)
            f.write(json.dumps(slim) + "\n")
            f.flush()

    _write_index(out_dir, meta_rows)
    summary = {
        "n_scenes": len(scenes),
        "n_ok": sum(1 for r in meta_rows if "error" not in r),
        "n_errors": sum(1 for r in meta_rows if "error" in r),
        "n_reproduced_overlap": sum(1 for r in meta_rows if r.get("reproduced_overlap")),
        "out_dir": str(out_dir),
        "index_html": str(out_dir / "index.html"),
        "scene_filter": "cv_success & gt_success & wm_overlap_fail",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
