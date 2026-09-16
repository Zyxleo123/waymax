#!/usr/bin/env python3
"""Select Diff-ES success / FM-ES fail cases, replan both, save scenario videos.

Writes per-case:
  - diff.mp4 / fm.mp4 / side_by_side.mp4  (full closed-loop ego + ES elite overlays)
      * magenta = non-elite population (~64 after ES)
      * orange  = other elites (7)
      * green   = highest-rated / selected elite
      * labels in black
  - trajectories.npz  (includes per-replan full ES population)
  - diagnostics.json / meta.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import jax
import numpy as np
from flax import nnx
from waymax import config as waymax_config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scenario_loader import load_scenario_state_batch_fast
from lane_graph.lane_graph_utils import lane_graph_zip_path_for_tfrecord
from scores.scorer_lane_graph import LaneGraphShardStore
from scores import helpers as scorer_helpers
from simulation.evaluation_utils import infer_goals_from_sim_state
from simulation.planning_utils import (
    apply_ego_replacements_to_expanded_state,
    sim_state_to_ego_trajectory,
)
from simulation.run_simulation import _load_fm_model_for_inference, _parse_args
from train_diffusion.utils.infer import load_model_for_inference
from viz.render import _apply_ego_override_to_state_batch
from viz.utils import VizConfig, img_from_fig


DEFAULT_EXP_ROOT = Path("/zfsauton/scratch/sbellad/immediate_exp")
DEFAULT_OUT = DEFAULT_EXP_ROOT / "diff_ok_fm_fail_viewer_gt"
DEFAULT_DIFF_CKPT = (
    "/zfsauton/scratch/sbellad/diffusion/"
    "diffusion_baseline_baseline_20260810_144559/epoch_0500"
)
DEFAULT_FM_CKPT = (
    "/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/"
    "pretrain_fm_policy_20260828_011255/epoch_0500"
)
DEFAULT_LANE_GRAPH_DIR = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"

# Hyperparams used for the fail1500 ES runs (keep in sync with those jobs).
ES_COMMON = dict(
    population_size=64,
    elite_size=8,
    num_es_iterations=4,
    replan_interval_steps=10,
    start_timestep=10,
    num_worlds=1,
    metrics=["collision", "offroad"],
)
DIFF_HPARAMS = dict(resample_timesteps=5, **ES_COMMON)
FM_HPARAMS = dict(resample_t_start=0.2, resample_num_steps=None, **ES_COMMON)


def _latest_sim(run_folder: Path) -> Path:
    sims = sorted(run_folder.glob("simulation_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sims:
        raise FileNotFoundError(f"No simulation_* under {run_folder}")
    return sims[0]


def load_results(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for p in run_dir.glob("*.json"):
        if p.name == "summary.json" or "instruction" in p.name:
            continue
        row = json.loads(p.read_text())
        if "scenario_idx" not in row:
            continue
        tf = Path(str(row.get("tfrecord", p.name))).name
        out[(tf, int(row["scenario_idx"]))] = row
    return out


def fm_fail_type(row: dict[str, Any]) -> str:
    if row.get("overlap") and row.get("offroad"):
        return "overlap+offroad"
    if row.get("overlap"):
        return "overlap"
    if row.get("offroad"):
        return "offroad"
    if not row.get("goal_reached"):
        return "goal"
    return "other"


def select_cases(
    *,
    mode: str,
    n: int,
    seed: int,
    exp_root: Path,
) -> list[dict[str, Any]]:
    diff = load_results(_latest_sim(exp_root / f"es_diffusion_{mode}_fail1500"))
    fm = load_results(_latest_sim(exp_root / f"es_fm_{mode}_fail1500"))
    keys = sorted(set(diff) & set(fm))
    candidates = [k for k in keys if diff[k]["success"] and not fm[k]["success"]]
    by_type: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for k in candidates:
        by_type[fm_fail_type(fm[k])].append(k)

    rng = random.Random(seed)
    quotas = {}
    total = len(candidates)
    for t, lst in by_type.items():
        quotas[t] = max(1, round(n * len(lst) / total)) if total else 0
    while sum(quotas.values()) > n:
        t = max(quotas, key=lambda x: quotas[x])
        if quotas[t] > 1:
            quotas[t] -= 1
        else:
            break
    while sum(quotas.values()) < n:
        t = max(by_type, key=lambda x: len(by_type[x]) - quotas.get(x, 0))
        quotas[t] = quotas.get(t, 0) + 1

    chosen: list[tuple[str, int]] = []
    for t, q in quotas.items():
        pool = list(by_type[t])
        rng.shuffle(pool)
        chosen.extend(pool[:q])
    if len(chosen) < n:
        leftover = [k for k in candidates if k not in chosen]
        rng.shuffle(leftover)
        chosen.extend(leftover[: n - len(chosen)])
    chosen = chosen[:n]

    cases = []
    for i, (tf, idx) in enumerate(chosen):
        cases.append(
            {
                "case_id": i,
                "tfrecord": tf,
                "scenario_idx": int(idx),
                "mode": mode,
                "diff": {
                    "success": bool(diff[(tf, idx)]["success"]),
                    "overlap": bool(diff[(tf, idx)]["overlap"]),
                    "offroad": bool(diff[(tf, idx)]["offroad"]),
                    "goal_reached": bool(diff[(tf, idx)]["goal_reached"]),
                    "min_goal_distance_m": float(diff[(tf, idx)].get("min_goal_distance_m", -1)),
                    "tfrecord_path": str(diff[(tf, idx)].get("tfrecord", "")),
                },
                "fm": {
                    "success": bool(fm[(tf, idx)]["success"]),
                    "overlap": bool(fm[(tf, idx)]["overlap"]),
                    "offroad": bool(fm[(tf, idx)]["offroad"]),
                    "goal_reached": bool(fm[(tf, idx)]["goal_reached"]),
                    "min_goal_distance_m": float(fm[(tf, idx)].get("min_goal_distance_m", -1)),
                    "fail_type": fm_fail_type(fm[(tf, idx)]),
                    "tfrecord_path": str(fm[(tf, idx)].get("tfrecord", "")),
                },
            }
        )
    return cases


def _build_planners(mode: str, diffusion_ckpt: str, fm_ckpt: str):
    from planner.diffusion_es_planner import DiffusionESPlanner
    from planner.fm_es_planner import FMESPlanner

    inf = load_model_for_inference(
        checkpoint_path=diffusion_ckpt,
        use_ema=True,
        metadata_path=None,
        seed=0,
        skip_checkpoint_load=False,
    )
    diff_policy = nnx.merge(inf.graphdef, inf.params_state, inf.nonparam_state)
    diff_planner = DiffusionESPlanner(
        policy=diff_policy,
        preprocess_cfg=inf.preprocess_cfg,
        population_size=DIFF_HPARAMS["population_size"],
        resample_timesteps=DIFF_HPARAMS["resample_timesteps"],
        elite_size=DIFF_HPARAMS["elite_size"],
        num_iterations=DIFF_HPARAMS["num_es_iterations"],
        num_worlds=DIFF_HPARAMS["num_worlds"],
        metrics=DIFF_HPARAMS["metrics"],
        other_agent_forecast=mode,
    )

    fm_policy, fm_cfg = _load_fm_model_for_inference(
        checkpoint_path=fm_ckpt,
        use_ema=True,
        metadata_path=None,
        seed=0,
        skip_checkpoint_load=False,
    )
    fm_planner = FMESPlanner(
        policy=fm_policy,
        preprocess_cfg=fm_cfg,
        population_size=FM_HPARAMS["population_size"],
        elite_size=FM_HPARAMS["elite_size"],
        num_iterations=FM_HPARAMS["num_es_iterations"],
        num_worlds=FM_HPARAMS["num_worlds"],
        metrics=FM_HPARAMS["metrics"],
        resample_t_start=FM_HPARAMS["resample_t_start"],
        other_agent_forecast=mode,
    )
    return diff_planner, fm_planner


def _attach_lane_graph(planner, *, tf_path: Path, scenario_idx: int, lane_graph_dir: str, sim_state) -> None:
    """Mirror simulation/runner.py: ES scorers need the per-scenario lane graph."""
    if getattr(planner, "scorers", None) is None:
        return
    zip_path = lane_graph_zip_path_for_tfrecord(str(tf_path), lane_graph_dir)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Lane graph zip not found for {tf_path.name}: {zip_path}"
        )
    store = LaneGraphShardStore(zip_path.as_posix())
    graph = store.get_by_record_index(int(scenario_idx))
    for world_idx, scorer in enumerate(planner.scorers):
        scorer.set_lane_graph(graph)
        scorer_helpers.update_lane_points(scorer, sim_state, world_idx)


def plan_one(planner, sim_state, goal_xy, args_ns, rng):
    """Closed-loop replan; records full final ES population at each replan step."""
    start_timestep = ES_COMMON["start_timestep"]
    replan_interval_steps = ES_COMMON["replan_interval_steps"]
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    replaced_state = sim_state
    elite_steps: list[dict[str, Any]] = []
    elite_size = int(getattr(planner, "elite_size", 8))

    for step_offset in range(start_timestep, episode_num_steps - 1, replan_interval_steps):
        rng, key_plan = jax.random.split(rng)
        plan_result = planner.plan_trajectory(
            replaced_state,
            goal_xy,
            rng=key_plan,
            timestep=step_offset,
            mask_goal=False,
            instruction_texts=None,
            use_subgoal=False,
        )
        plan_traj_bt5 = np.asarray(plan_result.trajectory_world_bt5, dtype=np.float32)
        aux = plan_result.aux or {}
        pop = aux.get("es_pop_world_bkt5")
        pop_scores = aux.get("es_pop_scores_bk")
        elite = aux.get("es_elite_world_bet5")
        elite_scores = aux.get("es_elite_scores_be")
        best_score = aux.get("es_best_score_b")
        if pop is not None:
            pop_np = np.asarray(pop[0], dtype=np.float32)  # [K, T, 5] score-sorted desc
            scores_np = (
                np.asarray(pop_scores[0], dtype=np.float32)
                if pop_scores is not None
                else np.zeros((pop_np.shape[0],), dtype=np.float32)
            )
        elif elite is not None:
            pop_np = np.asarray(elite[0], dtype=np.float32)
            scores_np = (
                np.asarray(elite_scores[0], dtype=np.float32)
                if elite_scores is not None
                else np.zeros((pop_np.shape[0],), dtype=np.float32)
            )
            order = np.argsort(-scores_np)
            pop_np = pop_np[order]
            scores_np = scores_np[order]
        else:
            pop_np = None
            scores_np = None

        if pop_np is not None:
            elite_steps.append(
                {
                    "t": int(step_offset),
                    "pop_kt5": pop_np,
                    "scores_k": scores_np,
                    "elite_size": elite_size,
                    "best_score": float(np.asarray(best_score).reshape(-1)[0])
                    if best_score is not None
                    else float(scores_np[0]),
                    "selected_t5": plan_traj_bt5[0],
                    # keep legacy keys for drawing helpers
                    "elite_et5": pop_np[:elite_size],
                    "scores_e": scores_np[:elite_size],
                }
            )
        replaced_state = apply_ego_replacements_to_expanded_state(
            replaced_state,
            start_t_b=step_offset,
            traj_bt5=plan_traj_bt5,
        )

    traj_bt5 = np.asarray(sim_state_to_ego_trajectory(replaced_state)[0], dtype=np.float32)
    return traj_bt5, replaced_state, elite_steps


def _speed_mps(traj_t5: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(traj_t5)[:, 3:5], axis=-1)


def _xy(traj_t5: np.ndarray) -> np.ndarray:
    return np.asarray(traj_t5)[:, :2]


def _path_length_m(xy: np.ndarray) -> float:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    return float(np.nansum(d))


def _sdc_log_xy(sim_state, world_idx: int = 0) -> np.ndarray:
    is_sdc = np.asarray(sim_state.object_metadata.is_sdc[world_idx])
    ego = int(np.argmax(is_sdc.astype(np.int32)))
    return np.asarray(sim_state.log_trajectory.xy[world_idx, ego], dtype=np.float32)


def _put_label_black(img: np.ndarray, text: str, org=(12, 28)) -> np.ndarray:
    """Black text with a light outline so it stays readable on any background."""
    out = np.asarray(img).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    # light halo then black fill
    cv2.putText(out, text, org, font, scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(out, text, org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def _active_elite_step(elite_steps: list[dict[str, Any]], t: int) -> dict[str, Any] | None:
    active = None
    for step in elite_steps:
        if int(step["t"]) <= int(t):
            active = step
        else:
            break
    return active


def _draw_elites_on_ax(ax, elite_step: dict[str, Any] | None) -> None:
    """Draw full post-ES pool with 3 distinct colors.

    - magenta: non-elite population (~64)
    - orange: other elites (elite_size-1)
    - green: best / selected elite
    """
    if elite_step is None:
        return
    pop = np.asarray(elite_step.get("pop_kt5", elite_step.get("elite_et5")), dtype=np.float32)
    if pop.ndim != 3 or pop.shape[0] == 0:
        return
    elite_size = int(elite_step.get("elite_size", min(8, pop.shape[0])))
    n = pop.shape[0]
    n_elite = min(elite_size, n)

    color_pop = (0.75, 0.10, 0.75)          # magenta — ~64 non-elites
    color_other_elite = (1.00, 0.55, 0.00)  # orange — other 7 elites
    color_best = (0.00, 0.80, 0.30)         # green — most elite / selected

    # 1) Non-elite population (underneath).
    for rank in range(n - 1, n_elite - 1, -1):
        xy = pop[rank, :, :2]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color_pop,
            lw=1.1,
            alpha=0.35,
            zorder=12,
            solid_capstyle="round",
        )

    # 2) Other elites (not the best).
    for rank in range(n_elite - 1, 0, -1):
        xy = pop[rank, :, :2]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color_other_elite,
            lw=2.0,
            alpha=0.75,
            zorder=16,
            solid_capstyle="round",
        )

    # 3) Best / selected on top.
    xy = pop[0, :, :2]
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color_best,
        lw=3.2,
        alpha=1.0,
        zorder=20,
        solid_capstyle="round",
    )
    ax.scatter(
        [xy[-1, 0]],
        [xy[-1, 1]],
        c=[color_best],
        s=36,
        zorder=21,
        edgecolors="black",
        linewidths=0.6,
    )


def _render_frame_with_elites(
    state_for_render,
    *,
    batch_idx: int,
    t: int,
    goal_xy: np.ndarray,
    elite_step: dict[str, Any] | None,
    viz_config: VizConfig,
    width: int,
    height: int,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    state_t = dataclasses.replace(
        state_for_render, timestep=np.full((state_for_render.shape[0],), t, dtype=np.int32)
    )
    # Use viz internals: plot base scene then overlay elites before rasterizing.
    # Replicate plot_simulator_state but keep ax open for overlays.
    from viz.utils import init_fig_ax
    from viz.viz import (
        _index_pytree,
        plot_roadgraph_points,
        plot_traffic_light_signals_as_points,
        plot_trajectory,
    )
    from waymax import datatypes
    from waymax import config as _wcfg
    import matplotlib.patches

    state = _index_pytree(state_t, batch_idx)
    fig, ax = init_fig_ax(viz_config)
    traj = state.log_trajectory
    is_controlled = datatypes.get_control_mask(
        state.object_metadata, _wcfg.ObjectType.SDC
    )
    indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None
    plot_trajectory(ax, traj, is_controlled, time_idx=state.timestep, indices=indices)
    plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
    plot_traffic_light_signals_as_points(
        ax, state.log_traffic_light, state.timestep, verbose=False
    )
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    ax.add_patch(
        matplotlib.patches.Circle(
            (float(goal[0]), float(goal[1])),
            radius=2.0,
            fill=False,
            edgecolor="gold",
            linewidth=2.0,
            alpha=0.9,
            zorder=12,
        )
    )
    _draw_elites_on_ax(ax, elite_step)

    current_xy = traj.xy[:, state.timestep, :]
    xy = current_xy[state.object_metadata.is_sdc]
    origin_x, origin_y = xy[0, :2]
    ax.axis(
        (
            origin_x - viz_config.back_x,
            origin_x + viz_config.front_x,
            origin_y - viz_config.back_y,
            origin_y + viz_config.front_y,
        )
    )
    ax.set_axis_off()
    frame = img_from_fig(fig)
    frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return frame


def _render_ego_video(
    *,
    tf_path: Path,
    scenario_idx: int,
    ego_traj: np.ndarray,
    goal_xy: np.ndarray,
    elite_steps: list[dict[str, Any]],
    out_mp4: Path,
    max_num_objects: int,
    label: str,
) -> Path:
    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tf_path),
        max_num_objects=int(max_num_objects),
        batch_dims=(1,),
        shuffle_seed=0,
    )
    state_batch, _ = load_scenario_state_batch_fast(ds_cfg, [int(scenario_idx)])
    state_batch = _apply_ego_override_to_state_batch(
        state_batch,
        batch_idx=0,
        start_t=0,
        ego_traj_l5=np.asarray(ego_traj, dtype=np.float32),
    )

    viz_config = VizConfig(
        show_agent_id=True,
        center_agent_idx=-1,
        front_x=30.0,
        back_x=30.0,
        front_y=30.0,
        back_y=30.0,
        px_per_meter=4.0,
    )
    total_steps = int(state_batch.log_trajectory.x.shape[-1])
    steps = min(91, total_steps)
    frames: list[np.ndarray] = []
    for t in range(steps):
        elite_step = _active_elite_step(elite_steps, t)
        frame = _render_frame_with_elites(
            state_batch,
            batch_idx=0,
            t=t,
            goal_xy=goal_xy,
            elite_step=elite_step,
            viz_config=viz_config,
            width=480,
            height=480,
        )
        score_txt = ""
        if elite_step is not None:
            score_txt = f"  elites@{elite_step['t']} best={elite_step['best_score']:.2f}"
        frame = _put_label_black(frame, f"{label}  t={t}{score_txt}")
        frames.append(frame)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_mp4.as_posix(), frames, fps=10, codec="libx264")
    return out_mp4


def _side_by_side(diff_mp4: Path, fm_mp4: Path, out_mp4: Path) -> Path:
    d_frames = list(imageio.mimread(diff_mp4.as_posix()))
    f_frames = list(imageio.mimread(fm_mp4.as_posix()))
    n = min(len(d_frames), len(f_frames))
    combined = []
    for i in range(n):
        left = np.asarray(d_frames[i])
        right = np.asarray(f_frames[i])
        h = min(left.shape[0], right.shape[0])
        w = min(left.shape[1], right.shape[1])
        left = left[:h, :w]
        right = right[:h, :w]
        sep = np.full((h, 4, left.shape[2]), 255, dtype=left.dtype)
        combined.append(np.concatenate([left, sep, right], axis=1))
    imageio.mimsave(out_mp4.as_posix(), combined, fps=10, codec="libx264")
    return out_mp4


def build_diagnostics(
    *,
    case: dict[str, Any],
    diff_traj: np.ndarray,
    fm_traj: np.ndarray,
    goal_xy: np.ndarray,
    log_xy: np.ndarray,
    start_timestep: int,
) -> dict[str, Any]:
    """Time series + summaries useful for tuning FM resample_t_start / ES knobs."""
    dxy, fxy = _xy(diff_traj), _xy(fm_traj)
    t = np.arange(dxy.shape[0], dtype=np.int32)
    lat = np.linalg.norm(fxy - dxy, axis=-1)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    g_diff = np.linalg.norm(dxy - goal[None, :], axis=-1)
    g_fm = np.linalg.norm(fxy - goal[None, :], axis=-1)
    log = np.asarray(log_xy, dtype=np.float32)
    n = min(len(log), len(dxy))
    err_d_log = np.linalg.norm(dxy[:n] - log[:n], axis=-1)
    err_f_log = np.linalg.norm(fxy[:n] - log[:n], axis=-1)
    spd_d, spd_f = _speed_mps(diff_traj), _speed_mps(fm_traj)

    yaw_d = np.asarray(diff_traj)[:, 2]
    yaw_f = np.asarray(fm_traj)[:, 2]
    dyaw = np.abs(np.arctan2(np.sin(yaw_f - yaw_d), np.cos(yaw_f - yaw_d)))

    post = slice(start_timestep, None)
    note = (
        "FM resample_t_start ∈ (0,1]: larger = more corruption toward Gaussian prior "
        "before ODE integration back to data. Current fail1500 used 0.2 (mild). "
        "If FM looks sticky / under-exploring vs Diff, try 0.4–0.8 (more noise). "
        "Diffusion analogue is resample_timesteps (fail1500 used 5)."
    )
    return {
        "hyperparams": {
            "mode": case.get("mode"),
            "diffusion_es": DIFF_HPARAMS,
            "fm_es": FM_HPARAMS,
            "note_on_noising": note,
        },
        "series": {
            "t": t.tolist(),
            "diff_speed_mps": spd_d.astype(float).tolist(),
            "fm_speed_mps": spd_f.astype(float).tolist(),
            "lateral_err_fm_vs_diff_m": lat.astype(float).tolist(),
            "yaw_err_fm_vs_diff_rad": dyaw.astype(float).tolist(),
            "goal_dist_diff_m": g_diff.astype(float).tolist(),
            "goal_dist_fm_m": g_fm.astype(float).tolist(),
            "diff_vs_log_xy_err_m": err_d_log.astype(float).tolist(),
            "fm_vs_log_xy_err_m": err_f_log.astype(float).tolist(),
        },
        "summary": {
            "start_timestep": int(start_timestep),
            "max_lateral_fm_vs_diff_m": float(np.max(lat[post])),
            "mean_lateral_fm_vs_diff_m": float(np.mean(lat[post])),
            "t_at_max_lateral": int(start_timestep + np.argmax(lat[post])),
            "max_yaw_err_rad": float(np.max(dyaw[post])),
            "mean_speed_diff_mps": float(np.mean(spd_d[post])),
            "mean_speed_fm_mps": float(np.mean(spd_f[post])),
            "path_length_diff_m": _path_length_m(dxy[post]),
            "path_length_fm_m": _path_length_m(fxy[post]),
            "final_goal_dist_diff_m": float(g_diff[-1]),
            "final_goal_dist_fm_m": float(g_fm[-1]),
            "min_goal_dist_diff_m": float(np.min(g_diff[post])),
            "min_goal_dist_fm_m": float(np.min(g_fm[post])),
            "mean_fm_vs_log_err_m": float(np.mean(err_f_log[start_timestep:n])),
            "mean_diff_vs_log_err_m": float(np.mean(err_d_log[start_timestep:n])),
            "fm_fail_type": case["fm"]["fail_type"],
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", type=str, default="gt", choices=["cv", "gt"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_root", type=str, default=str(DEFAULT_EXP_ROOT))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--diffusion_ckpt", type=str, default=DEFAULT_DIFF_CKPT)
    p.add_argument("--fm_ckpt", type=str, default=DEFAULT_FM_CKPT)
    p.add_argument("--tf_root", type=str, default="/zfsauton/scratch/eshau/womd/tf_example/training")
    p.add_argument("--lane_graph_dir", type=str, default=DEFAULT_LANE_GRAPH_DIR)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--select_only", action="store_true", help="Only write cases.json, no videos")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(
        mode=args.mode,
        n=args.n,
        seed=args.seed,
        exp_root=Path(args.exp_root),
    )
    mix = Counter(c["fm"]["fail_type"] for c in cases)
    index = {
        "mode": args.mode,
        "n": len(cases),
        "seed": args.seed,
        "fm_fail_type_mix": dict(mix),
        "hyperparams": {
            "diffusion_es": DIFF_HPARAMS,
            "fm_es": FM_HPARAMS,
            "note_on_noising": (
                "FM resample_t_start larger => more noising toward prior. "
                "fail1500 used 0.2 (mild)."
            ),
        },
        "cases": cases,
    }
    (out_dir / "cases.json").write_text(json.dumps(index, indent=2) + "\n")
    (out_dir / "STATUS.txt").write_text("selected_only\n")
    print(f"Selected {len(cases)} cases; FM fail mix={dict(mix)}")
    print(f"Wrote {out_dir / 'cases.json'}")
    if args.select_only:
        return

    try:
        devices = jax.devices()
        has_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    except Exception as e:
        raise SystemExit(f"JAX device check failed: {e}") from e
    if not has_gpu:
        (out_dir / "STATUS.txt").write_text(
            "ERROR: no GPU visible. Re-run via sbatch slurm_export_diff_ok_fm_fail_viewer.sh\n"
        )
        raise SystemExit(
            "No GPU visible to JAX. Do NOT run the full export on a CPU node.\n"
            "Use: sbatch slurm_export_diff_ok_fm_fail_viewer.sh"
        )
    print(f"JAX devices: {devices}", flush=True)

    sys.argv = ["run_simulation.py"]
    dummy = _parse_args()
    dummy.planner = "es"
    dummy.other_agent_forecast = args.mode
    dummy.population_size = ES_COMMON["population_size"]
    dummy.elite_size = ES_COMMON["elite_size"]
    dummy.num_es_iterations = ES_COMMON["num_es_iterations"]
    dummy.resample_timesteps = DIFF_HPARAMS["resample_timesteps"]
    dummy.resample_t_start = FM_HPARAMS["resample_t_start"]
    dummy.replan_interval_steps = ES_COMMON["replan_interval_steps"]
    dummy.start_timestep = ES_COMMON["start_timestep"]
    dummy.num_worlds = ES_COMMON["num_worlds"]

    print("Loading planners...", flush=True)
    diff_planner, fm_planner = _build_planners(args.mode, args.diffusion_ckpt, args.fm_ckpt)
    (out_dir / "STATUS.txt").write_text("rendering_videos\n")

    for case in cases:
        tf = case["tfrecord"]
        idx = int(case["scenario_idx"])
        tf_path = Path(args.tf_root) / tf
        for src in (case["diff"]["tfrecord_path"], case["fm"]["tfrecord_path"]):
            if src and Path(src).exists():
                tf_path = Path(src)
                break
        print(
            f"[{case['case_id']}] {tf} scenario {idx} (fm fail={case['fm']['fail_type']})",
            flush=True,
        )
        cfg = dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING,
            path=str(tf_path),
            max_num_objects=int(args.max_num_objects),
            batch_dims=(1,),
            shuffle_seed=0,
        )
        sim_state, _ = load_scenario_state_batch_fast(cfg, [idx])
        goal_xy, _, _ = infer_goals_from_sim_state(sim_state)
        goal0 = np.asarray(goal_xy[0], dtype=np.float32)
        log_xy = _sdc_log_xy(sim_state, 0)

        # ES collision/offroad scoring requires lane graphs (same as runner.py).
        _attach_lane_graph(
            diff_planner,
            tf_path=tf_path,
            scenario_idx=idx,
            lane_graph_dir=args.lane_graph_dir,
            sim_state=sim_state,
        )
        _attach_lane_graph(
            fm_planner,
            tf_path=tf_path,
            scenario_idx=idx,
            lane_graph_dir=args.lane_graph_dir,
            sim_state=sim_state,
        )

        rng = jax.random.PRNGKey(args.seed + 17 * case["case_id"])
        rng, r1, r2 = jax.random.split(rng, 3)
        diff_traj, _, diff_elites = plan_one(diff_planner, sim_state, goal_xy, dummy, r1)
        fm_traj, _, fm_elites = plan_one(fm_planner, sim_state, goal_xy, dummy, r2)

        case_dir = out_dir / f"case_{case['case_id']:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        # Pack full ES population (+ elites) for later re-render / analysis.
        diff_elite_t = np.asarray([e["t"] for e in diff_elites], dtype=np.int32)
        fm_elite_t = np.asarray([e["t"] for e in fm_elites], dtype=np.int32)
        diff_pop_kt5 = (
            np.stack([e["pop_kt5"] for e in diff_elites], axis=0)
            if diff_elites
            else np.zeros((0, 0, 0, 5), np.float32)
        )
        fm_pop_kt5 = (
            np.stack([e["pop_kt5"] for e in fm_elites], axis=0)
            if fm_elites
            else np.zeros((0, 0, 0, 5), np.float32)
        )
        diff_pop_scores = (
            np.stack([e["scores_k"] for e in diff_elites], axis=0)
            if diff_elites
            else np.zeros((0, 0), np.float32)
        )
        fm_pop_scores = (
            np.stack([e["scores_k"] for e in fm_elites], axis=0)
            if fm_elites
            else np.zeros((0, 0), np.float32)
        )
        np.savez_compressed(
            case_dir / "trajectories.npz",
            diff_traj=diff_traj,
            fm_traj=fm_traj,
            goal_xy=goal0,
            log_xy=log_xy,
            start_timestep=np.int32(ES_COMMON["start_timestep"]),
            diff_elite_t=diff_elite_t,
            fm_elite_t=fm_elite_t,
            diff_pop_kt5=diff_pop_kt5,
            fm_pop_kt5=fm_pop_kt5,
            diff_pop_scores=diff_pop_scores,
            fm_pop_scores=fm_pop_scores,
            diff_elite_et5=diff_pop_kt5[:, :8] if diff_pop_kt5.size else diff_pop_kt5,
            fm_elite_et5=fm_pop_kt5[:, :8] if fm_pop_kt5.size else fm_pop_kt5,
        )
        diag = build_diagnostics(
            case=case,
            diff_traj=diff_traj,
            fm_traj=fm_traj,
            goal_xy=goal0,
            log_xy=log_xy,
            start_timestep=ES_COMMON["start_timestep"],
        )
        (case_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
        (case_dir / "meta.json").write_text(json.dumps(case, indent=2) + "\n")

        diff_mp4 = case_dir / "diff.mp4"
        fm_mp4 = case_dir / "fm.mp4"
        _render_ego_video(
            tf_path=tf_path,
            scenario_idx=idx,
            ego_traj=diff_traj,
            goal_xy=goal0,
            elite_steps=diff_elites,
            out_mp4=diff_mp4,
            max_num_objects=args.max_num_objects,
            label="Diffusion-ES OK",
        )
        _render_ego_video(
            tf_path=tf_path,
            scenario_idx=idx,
            ego_traj=fm_traj,
            goal_xy=goal0,
            elite_steps=fm_elites,
            out_mp4=fm_mp4,
            max_num_objects=args.max_num_objects,
            label=f"FM-ES FAIL ({case['fm']['fail_type']})",
        )
        _side_by_side(diff_mp4, fm_mp4, case_dir / "side_by_side.mp4")
        d_n = int(diff_elites[0]["pop_kt5"].shape[0]) if diff_elites else 0
        f_n = int(fm_elites[0]["pop_kt5"].shape[0]) if fm_elites else 0
        print(
            f"  wrote {diff_mp4.name}, {fm_mp4.name}, side_by_side.mp4 "
            f"(diff_pop={d_n} fm_pop={f_n} replans={len(diff_elites)}/{len(fm_elites)})",
            flush=True,
        )

    n_vid = len(list(out_dir.glob("case_*/side_by_side.mp4")))
    (out_dir / "STATUS.txt").write_text(f"done videos={n_vid}\n")
    print(f"Done. videos={n_vid} Viewer data in {out_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
