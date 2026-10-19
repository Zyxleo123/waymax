#!/usr/bin/env python3
"""Collider ADE/FDE + collision-label agreement on GT-only overlap wins.

Re-runs Diffusion-ES with WM on the scenes where GT succeeds and WM fails
with overlap. At each replan (and at the replan before the first overlap):

  1. ADE/FDE of WM vs CV vs logged GT, for nearby agents AND the collider.
  2. Collision labels of the WM-chosen ego plan vs:
       - logged other-agent boxes (eval world)
       - WM boxes (what ES scored)
       - CV boxes (CV world model)

Original fail1500 runs did not save trajectories, so this re-plans. RNG may
not reproduce every original overlap; collision-label stats are still valid.

Example::

  python scripts/diagnose_wm_overlap_gap.py
  python scripts/diagnose_wm_overlap_gap.py --max_scenes 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from waymax import config as waymax_config
from waymax.datatypes.object_state import ObjectTypeIds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from data.scenario_loader import load_scenario_state_batch_fast  # noqa: E402
from lane_graph.lane_graph_utils import lane_graph_zip_path_for_tfrecord  # noqa: E402
from planner.diffusion_es_planner import DiffusionESPlanner  # noqa: E402
from scores import helpers as scorer_helpers  # noqa: E402
from scores.lane_graph_utils import LaneGraphShardStore  # noqa: E402
from scores.polygon import pairwise_intersections, polygon_corners  # noqa: E402
from simulation.evaluation_utils import (  # noqa: E402
    check_goal_reaching,
    infer_goals_from_sim_state,
    rollout_predicted_trajectories_with_metrics,
)
from simulation.planning_utils import apply_ego_replacements_to_expanded_state  # noqa: E402
from simulation.run_simulation import _load_world_model_forecaster  # noqa: E402
from train_diffusion.utils.infer import load_model_for_inference  # noqa: E402

DEFAULT_COMPARE_JSON = (
    "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/cv_wm_gt_compare_20260915_175107/wm_vs_gt.json"
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

_DYNAMIC_TYPES = {
    int(ObjectTypeIds.VEHICLE.value),
    int(ObjectTypeIds.PEDESTRIAN.value),
    int(ObjectTypeIds.CYCLIST.value),
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "forecast_xy"}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare_json", type=str, default=DEFAULT_COMPARE_JSON)
    p.add_argument("--tfrecord_dir", type=str, default=DEFAULT_TFRECORD_DIR)
    p.add_argument("--lane_graph_dir", type=str, default=DEFAULT_LANE_GRAPH_DIR)
    p.add_argument("--diffusion_ckpt_path", type=str, default=DEFAULT_DIFFUSION_CKPT)
    p.add_argument("--world_model_ckpt", type=str, default=DEFAULT_WM_CKPT)
    p.add_argument("--world_model_device", type=str, default="auto")
    p.add_argument("--wm_refresh_steps", type=int, default=10)
    p.add_argument("--other_agent_forecast", type=str, default="wm")
    p.add_argument("--start_timestep", type=int, default=10)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--population_size", type=int, default=64)
    p.add_argument("--elite_size", type=int, default=8)
    p.add_argument("--num_es_iterations", type=int, default=4)
    p.add_argument("--resample_timesteps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nearby_radius_m", type=float, default=20.0)
    p.add_argument("--max_scenes", type=int, default=0, help="0 = all overlap GT-only wins.")
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


def load_overlap_scenes(compare_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(compare_json.read_text())
    scenes = [
        row
        for row in payload.get("gt_only_win_scenes", [])
        if str(row.get("wm_failure", "")).lower() == "overlap"
    ]
    if not scenes:
        raise ValueError(f"No GT-only overlap wins in {compare_json}")
    return scenes


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def _ade_fde(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    valid: np.ndarray,
) -> tuple[float | None, float | None, int]:
    if pred_xy.shape != gt_xy.shape:
        raise ValueError(f"pred/gt shape mismatch {pred_xy.shape} vs {gt_xy.shape}")
    err = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    mask = np.asarray(valid).astype(bool)
    if mask.shape != err.shape:
        raise ValueError(f"valid shape mismatch {mask.shape} vs {err.shape}")
    n = int(mask.sum())
    if n == 0:
        return None, None, 0
    ade = float(err[mask].mean())
    last = np.flatnonzero(mask)[-1]
    fde = float(err[last])
    return ade, fde, n


def _mean_ignore_none(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(np.mean(xs))


def boxes_from_log(sim_state, *, timestep: int, horizon: int, world_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    traj = sim_state.log_trajectory
    t = int(np.clip(timestep, 0, int(traj.x.shape[-1]) - 1))
    h = min(int(horizon), int(traj.x.shape[-1]) - t)
    xy = _to_numpy(traj.xy[world_idx, :, t : t + h]).astype(np.float32)
    yaw = _to_numpy(traj.yaw[world_idx, :, t : t + h]).astype(np.float32)
    length = _to_numpy(traj.length[world_idx, :, t : t + h]).astype(np.float32)
    width = _to_numpy(traj.width[world_idx, :, t : t + h]).astype(np.float32)
    valid = _to_numpy(traj.valid[world_idx, :, t : t + h]).astype(bool)
    is_sdc = _to_numpy(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
    boxes = np.stack([xy[..., 0], xy[..., 1], yaw, length, width], axis=-1)
    masks = valid & (~is_sdc[:, None])
    return boxes, masks


def ego_plan_boxes(plan_t5: np.ndarray, sim_state, *, timestep: int, horizon: int, world_idx: int = 0) -> np.ndarray:
    ego = int(np.argmax(_to_numpy(sim_state.object_metadata.is_sdc[world_idx]).astype(np.int32)))
    t = int(np.clip(timestep, 0, int(sim_state.log_trajectory.length.shape[-1]) - 1))
    length = float(sim_state.log_trajectory.length[world_idx, ego, t])
    width = float(sim_state.log_trajectory.width[world_idx, ego, t])
    h = min(int(horizon), int(plan_t5.shape[0]))
    out = np.zeros((h, 5), dtype=np.float32)
    out[:, 0] = plan_t5[:h, 0]
    out[:, 1] = plan_t5[:h, 1]
    out[:, 2] = plan_t5[:h, 2]
    out[:, 3] = length
    out[:, 4] = width
    return out


def plan_collides(ego_t5: np.ndarray, obj_nt5: np.ndarray, obj_nt: np.ndarray) -> bool:
    h = min(int(ego_t5.shape[0]), int(obj_nt5.shape[1]), int(obj_nt.shape[1]))
    if h <= 0:
        return False
    ego = jnp.asarray(ego_t5[None, :h], dtype=jnp.float32)
    obj = jnp.asarray(obj_nt5[:, :h], dtype=jnp.float32)
    mask = jnp.asarray(obj_nt[:, :h], dtype=bool)
    ego_poly = polygon_corners(ego)
    obj_poly = polygon_corners(obj)

    def _time_pairwise(ego_t: jax.Array, object_t: jax.Array) -> jax.Array:
        return pairwise_intersections(ego_t, object_t)

    hits = jax.vmap(_time_pairwise, in_axes=(1, 1), out_axes=0)(ego_poly, obj_poly)
    hits = hits & jnp.swapaxes(mask, 0, 1)[:, None, :]
    return bool(np.asarray(jnp.any(hits)))


def colliding_agent_at(
    sim_state,
    *,
    timestep: int,
    world_idx: int = 0,
) -> tuple[int | None, float | None]:
    ego = int(np.argmax(_to_numpy(sim_state.object_metadata.is_sdc[world_idx]).astype(np.int32)))
    t = int(np.clip(timestep, 0, int(sim_state.log_trajectory.x.shape[-1]) - 1))
    xy = _to_numpy(sim_state.log_trajectory.xy[world_idx, :, t]).astype(np.float32)
    yaw = _to_numpy(sim_state.log_trajectory.yaw[world_idx, :, t]).astype(np.float32)
    length = _to_numpy(sim_state.log_trajectory.length[world_idx, :, t]).astype(np.float32)
    width = _to_numpy(sim_state.log_trajectory.width[world_idx, :, t]).astype(np.float32)
    valid = _to_numpy(sim_state.log_trajectory.valid[world_idx, :, t]).astype(bool)
    is_sdc = _to_numpy(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
    boxes = np.stack([xy[:, 0], xy[:, 1], yaw, length, width], axis=-1)
    polys = np.asarray(polygon_corners(jnp.asarray(boxes[:, None, :], dtype=jnp.float32)))[:, 0]
    hits = np.asarray(pairwise_intersections(jnp.asarray(polys[ego : ego + 1]), jnp.asarray(polys)))[0]
    hits = hits & valid & (~is_sdc)
    hits[ego] = False
    dist = np.linalg.norm(xy - xy[ego], axis=-1)
    if not bool(hits.any()):
        dist_masked = np.where(valid & (~is_sdc), dist, np.inf)
        idx = int(np.argmin(dist_masked))
        if not np.isfinite(dist_masked[idx]):
            return None, None
        return idx, float(dist_masked[idx])
    idxs = np.flatnonzero(hits)
    pick = int(idxs[int(np.argmin(dist[idxs]))])
    return pick, float(dist[pick])


def forecast_bundle(sim_state, *, timestep: int, horizon: int, wm_forecaster) -> dict[str, Any]:
    cv_traj, cv_mask, cv_col_traj, cv_col_mask = scorer_helpers.forecast_other_agents(
        sim_state,
        timestep=int(timestep),
        traj_length=int(horizon),
        world_idx=0,
        mode="cv",
    )
    gt_traj, gt_mask, _, _ = scorer_helpers.forecast_other_agents(
        sim_state,
        timestep=int(timestep),
        traj_length=int(horizon),
        world_idx=0,
        mode="gt",
    )
    wm_traj, wm_mask, wm_col_traj, wm_col_mask = wm_forecaster.forecast(
        sim_state,
        timestep=int(timestep),
        traj_length=int(horizon),
        world_idx=0,
    )
    h = min(int(cv_traj.shape[1]), int(gt_traj.shape[1]), int(wm_traj.shape[1]), int(horizon))
    return {
        "h": h,
        "cv": np.asarray(cv_traj[:, :h]),
        "cv_mask": np.asarray(cv_mask[:, :h]).astype(bool),
        "cv_col": np.asarray(cv_col_traj[:, :h]),
        "cv_col_mask": np.asarray(cv_col_mask[:, :h]).astype(bool),
        "gt": np.asarray(gt_traj[:, :h]),
        "gt_mask": np.asarray(gt_mask[:, :h]).astype(bool),
        "wm": np.asarray(wm_traj[:, :h]),
        "wm_mask": np.asarray(wm_mask[:, :h]).astype(bool),
        "wm_col": np.asarray(wm_col_traj[:, :h]),
        "wm_col_mask": np.asarray(wm_col_mask[:, :h]).astype(bool),
    }


def agent_metrics(
    forecasts: dict[str, Any],
    sim_state,
    *,
    timestep: int,
    nearby_radius_m: float,
    collider_idx: int | None = None,
) -> dict[str, Any]:
    ego = int(np.argmax(_to_numpy(sim_state.object_metadata.is_sdc[0]).astype(np.int32)))
    t = int(np.clip(timestep, 0, int(sim_state.log_trajectory.x.shape[-1]) - 1))
    xy_now = _to_numpy(sim_state.log_trajectory.xy[0, :, t]).astype(np.float32)
    valid_now = _to_numpy(sim_state.log_trajectory.valid[0, :, t]).astype(bool)
    types = _to_numpy(sim_state.object_metadata.object_types[0]).astype(np.int32)
    is_sdc = _to_numpy(sim_state.object_metadata.is_sdc[0]).astype(bool)
    dist = np.linalg.norm(xy_now - xy_now[ego], axis=-1)
    dynamic = np.isin(types, list(_DYNAMIC_TYPES))
    nearby = valid_now & (~is_sdc) & dynamic & (dist <= float(nearby_radius_m))

    h = int(forecasts["h"])
    gt_xy = forecasts["gt"][:, :h, :2]
    wm_xy = forecasts["wm"][:, :h, :2]
    cv_xy = forecasts["cv"][:, :h, :2]
    gt_mask = forecasts["gt_mask"][:, :h]
    n_agents = int(gt_xy.shape[0])

    def _pool(mask: np.ndarray) -> dict[str, float | None]:
        wm_ades, wm_fdes, cv_ades, cv_fdes = [], [], [], []
        for i in range(n_agents):
            if not bool(mask[i]):
                continue
            valid = gt_mask[i] & np.isfinite(gt_xy[i]).all(axis=-1)
            wm_ade, wm_fde, _ = _ade_fde(wm_xy[i], gt_xy[i], valid)
            cv_ade, cv_fde, _ = _ade_fde(cv_xy[i], gt_xy[i], valid)
            wm_ades.append(wm_ade)
            wm_fdes.append(wm_fde)
            cv_ades.append(cv_ade)
            cv_fdes.append(cv_fde)
        return {
            "n": int(mask.sum()),
            "wm_ade_m": _mean_ignore_none(wm_ades),
            "wm_fde_m": _mean_ignore_none(wm_fdes),
            "cv_ade_m": _mean_ignore_none(cv_ades),
            "cv_fde_m": _mean_ignore_none(cv_fdes),
        }

    out = {
        "all_dynamic": _pool(valid_now & (~is_sdc) & dynamic),
        "nearby": _pool(nearby),
        "collider": None,
    }
    if collider_idx is not None and 0 <= int(collider_idx) < n_agents:
        i = int(collider_idx)
        valid = gt_mask[i] & np.isfinite(gt_xy[i]).all(axis=-1)
        wm_ade, wm_fde, n = _ade_fde(wm_xy[i], gt_xy[i], valid)
        cv_ade, cv_fde, _ = _ade_fde(cv_xy[i], gt_xy[i], valid)
        out["collider"] = {
            "idx": i,
            "n_valid": n,
            "dist_now_m": float(dist[i]) if np.isfinite(dist[i]) else None,
            "wm_ade_m": wm_ade,
            "wm_fde_m": wm_fde,
            "cv_ade_m": cv_ade,
            "cv_fde_m": cv_fde,
        }
    return out


def first_overlap_step(timeseries: np.ndarray, *, start_timestep: int, world_idx: int = 0) -> int | None:
    arr = np.asarray(timeseries)
    if arr.ndim != 2:
        raise ValueError(f"expected overlap timeseries rank-2, got {arr.shape}")
    # Prefer [T, B] as produced by evaluation_utils transpose; fall back to [B, T].
    if arr.shape[0] < arr.shape[1]:
        series = arr[int(world_idx)]
    else:
        series = arr[:, int(world_idx)]
    sl = np.asarray(series[int(start_timestep) :]).astype(bool)
    hits = np.flatnonzero(sl)
    if hits.size == 0:
        return None
    return int(start_timestep + hits[0])


def last_replan_at_or_before(t_hit: int, *, start_timestep: int, interval: int, last_replan: int) -> int:
    rel = int(t_hit) - int(start_timestep)
    t = int(start_timestep) + (rel // int(interval)) * int(interval)
    t = min(t, int(last_replan))
    return max(int(start_timestep), t)


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
    return DiffusionESPlanner(
        policy=policy,
        preprocess_cfg=inference.preprocess_cfg,
        population_size=int(args.population_size),
        resample_timesteps=int(args.resample_timesteps),
        elite_size=int(args.elite_size),
        num_iterations=int(args.num_es_iterations),
        num_worlds=1,
        metrics=["collision", "offroad"],
        other_agent_forecast="wm",
        world_model_forecaster=_load_world_model_forecaster(wm_args),
    )


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

    goal_xy, _, ego_idx_b = infer_goals_from_sim_state(sim_state)
    ego_idx = int(ego_idx_b[0])
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t = int(args.start_timestep)
    interval = int(args.replan_interval_steps)

    rng = jax.random.PRNGKey(int(args.seed))
    replaced = sim_state
    replans: list[dict[str, Any]] = []
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
        forecasts = forecast_bundle(
            replaced,
            timestep=step,
            horizon=horizon,
            wm_forecaster=planner.world_model_forecaster,
        )
        h = int(forecasts["h"])
        ego_boxes = ego_plan_boxes(plan_t5, replaced, timestep=step, horizon=h)
        hits_gt = plan_collides(ego_boxes, forecasts["gt"], forecasts["gt_mask"])
        hits_wm = plan_collides(ego_boxes, forecasts["wm_col"], forecasts["wm_col_mask"])
        hits_cv = plan_collides(ego_boxes, forecasts["cv_col"], forecasts["cv_col_mask"])
        ade = agent_metrics(
            forecasts,
            replaced,
            timestep=step,
            nearby_radius_m=float(args.nearby_radius_m),
        )
        replans.append(
            {
                "t": int(step),
                "horizon": h,
                "plan_hits_gt": bool(hits_gt),
                "plan_hits_wm": bool(hits_wm),
                "plan_hits_cv": bool(hits_cv),
                "ade": ade,
                "forecast_xy": {
                    "gt": forecasts["gt"][..., :2].astype(np.float32),
                    "wm": forecasts["wm"][..., :2].astype(np.float32),
                    "cv": forecasts["cv"][..., :2].astype(np.float32),
                    "gt_mask": forecasts["gt_mask"],
                    "wm_mask": forecasts["wm_mask"],
                    "cv_mask": forecasts["cv_mask"],
                },
            }
        )
        replaced = apply_ego_replacements_to_expanded_state(
            replaced,
            start_t_b=step,
            traj_bt5=plan_t5[None, ...],
        )
        last_plan_t = int(step)

    rollout = rollout_predicted_trajectories_with_metrics(replaced, rng_key=jax.random.PRNGKey(0))
    goal_eval = check_goal_reaching(replaced, goal_xy, start_timestep=start_t)
    overlap_ts = np.asarray(rollout["metric_timeseries"]["overlap"])
    offroad_ts = np.asarray(rollout["metric_timeseries"]["offroad"])
    t_hit = first_overlap_step(overlap_ts, start_timestep=start_t, world_idx=0)
    overlap = t_hit is not None
    offroad = bool(np.asarray(offroad_ts[start_t:, 0] if overlap_ts.shape[0] > start_t else offroad_ts[:, 0]).sum() > 0)
    if offroad_ts.ndim == 2 and offroad_ts.shape[0] < offroad_ts.shape[1]:
        offroad = bool(np.asarray(offroad_ts[0, start_t:]).sum() > 0)
    success = bool(goal_eval["reached"][0]) and (not overlap) and (not offroad)

    collider_idx = None
    collider_dist = None
    pre_overlap = None
    if t_hit is not None:
        collider_idx, collider_dist = colliding_agent_at(replaced, timestep=t_hit, world_idx=0)
        t_replan = last_replan_at_or_before(
            t_hit, start_timestep=start_t, interval=interval, last_replan=last_plan_t
        )
        pre = next((row for row in replans if int(row["t"]) == int(t_replan)), None)
        if pre is not None:
            xy = pre["forecast_xy"]
            h = int(pre["horizon"])
            forecasts = {
                "h": h,
                "gt": xy["gt"],
                "wm": xy["wm"],
                "cv": xy["cv"],
                "gt_mask": xy["gt_mask"],
                "wm_mask": xy["wm_mask"],
                "cv_mask": xy["cv_mask"],
            }
            collider_ade = agent_metrics(
                forecasts,
                sim_state,
                timestep=t_replan,
                nearby_radius_m=float(args.nearby_radius_m),
                collider_idx=collider_idx,
            )
            err_at_hit = None
            dt = int(t_hit) - int(t_replan)
            if collider_idx is not None and 0 <= dt < h:
                gt_xy = xy["gt"][collider_idx, dt]
                wm_xy = xy["wm"][collider_idx, dt]
                cv_xy = xy["cv"][collider_idx, dt]
                err_at_hit = {
                    "dt": int(dt),
                    "wm_err_m": float(np.linalg.norm(wm_xy - gt_xy)),
                    "cv_err_m": float(np.linalg.norm(cv_xy - gt_xy)),
                }
            pre_overlap = {
                "t_replan": int(t_replan),
                "t_hit": int(t_hit),
                "plan_hits_gt": pre["plan_hits_gt"],
                "plan_hits_wm": pre["plan_hits_wm"],
                "plan_hits_cv": pre["plan_hits_cv"],
                "nearby": pre["ade"]["nearby"],
                "collider": collider_ade["collider"],
                "err_at_hit": err_at_hit,
            }

    false_neg = sum(1 for r in replans if (not r["plan_hits_wm"]) and r["plan_hits_gt"])
    false_pos = sum(1 for r in replans if r["plan_hits_wm"] and (not r["plan_hits_gt"]))
    return {
        "tfrecord": tf_name,
        "scenario_idx": scenario_idx,
        "ego_idx": ego_idx,
        "reproduced_overlap": bool(overlap),
        "offroad": bool(offroad),
        "goal_reached": bool(goal_eval["reached"][0]),
        "success": bool(success),
        "t_hit": t_hit,
        "collider_idx": collider_idx,
        "collider_dist_m": collider_dist,
        "n_replans": len(replans),
        "n_wm_miss_gt_hit": int(false_neg),
        "n_wm_hit_gt_miss": int(false_pos),
        "replans": replans,
        "pre_overlap": pre_overlap,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overlap_rows = [r for r in rows if r.get("pre_overlap") is not None]
    all_replans = [rep for r in rows for rep in r.get("replans", [])]
    n_replan = len(all_replans)
    fn = sum(1 for r in all_replans if (not r["plan_hits_wm"]) and r["plan_hits_gt"])
    fp = sum(1 for r in all_replans if r["plan_hits_wm"] and (not r["plan_hits_gt"]))
    tp = sum(1 for r in all_replans if r["plan_hits_wm"] and r["plan_hits_gt"])
    tn = sum(1 for r in all_replans if (not r["plan_hits_wm"]) and (not r["plan_hits_gt"]))

    def _col(key: str, src: str) -> float | None:
        vals = []
        for r in overlap_rows:
            block = (r.get("pre_overlap") or {}).get(src) or {}
            v = block.get(key) if isinstance(block, dict) else None
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    nearby_wm = [r["ade"]["nearby"]["wm_ade_m"] for r in all_replans if r["ade"]["nearby"]["wm_ade_m"] is not None]
    nearby_cv = [r["ade"]["nearby"]["cv_ade_m"] for r in all_replans if r["ade"]["nearby"]["cv_ade_m"] is not None]

    pre_fn = sum(
        1
        for r in overlap_rows
        if (not r["pre_overlap"]["plan_hits_wm"]) and r["pre_overlap"]["plan_hits_gt"]
    )
    pre_safe_both = sum(
        1
        for r in overlap_rows
        if (not r["pre_overlap"]["plan_hits_wm"]) and (not r["pre_overlap"]["plan_hits_gt"])
    )
    return {
        "n_scenes": len(rows),
        "n_reproduced_overlap": sum(1 for r in rows if r.get("reproduced_overlap")),
        "n_with_pre_overlap_dump": len(overlap_rows),
        "all_replan_collision_labels": {
            "n": n_replan,
            "wm_vs_gt_tp": tp,
            "wm_vs_gt_tn": tn,
            "wm_false_negative_safe_in_wm_hits_gt": fn,
            "wm_false_positive_hits_wm_safe_in_gt": fp,
            "fn_rate": float(fn / n_replan) if n_replan else None,
            "fp_rate": float(fp / n_replan) if n_replan else None,
        },
        "nearby_ade_all_replans_m": {
            "wm": float(np.mean(nearby_wm)) if nearby_wm else None,
            "cv": float(np.mean(nearby_cv)) if nearby_cv else None,
        },
        "pre_overlap_replan": {
            "n": len(overlap_rows),
            "plan_safe_in_wm_but_hits_gt": pre_fn,
            "plan_safe_in_both": pre_safe_both,
            "collider_wm_ade_m": _col("wm_ade_m", "collider"),
            "collider_cv_ade_m": _col("cv_ade_m", "collider"),
            "collider_wm_fde_m": _col("wm_fde_m", "collider"),
            "collider_cv_fde_m": _col("cv_fde_m", "collider"),
            "nearby_wm_ade_m": _col("wm_ade_m", "nearby"),
            "nearby_cv_ade_m": _col("cv_ade_m", "nearby"),
            "err_at_hit_wm_m": _mean_ignore_none(
                [
                    ((r["pre_overlap"].get("err_at_hit") or {}).get("wm_err_m"))
                    for r in overlap_rows
                ]
            ),
            "err_at_hit_cv_m": _mean_ignore_none(
                [
                    ((r["pre_overlap"].get("err_at_hit") or {}).get("cv_err_m"))
                    for r in overlap_rows
                ]
            ),
        },
    }


def main() -> None:
    args = parse_args()
    scenes = load_overlap_scenes(Path(args.compare_json))
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT_ROOT) / f"wm_overlap_gap_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] {len(scenes)} GT-only overlap scenes -> {out_dir}")

    planner = build_planner(args)
    rows: list[dict[str, Any]] = []
    jsonl_path = out_dir / "per_scene.jsonl"
    with jsonl_path.open("w") as f:
        for i, scene in enumerate(scenes, start=1):
            print(
                f"[{i}/{len(scenes)}] {scene['tfrecord']} scenario {scene['scenario_idx']}",
                flush=True,
            )
            try:
                row = run_scene(args, planner, scene)
            except Exception as exc:
                row = {
                    "tfrecord": scene["tfrecord"],
                    "scenario_idx": int(scene["scenario_idx"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR: {row['error']}", flush=True)
            rows.append(row)
            f.write(json.dumps(_jsonable(row)) + "\n")
            f.flush()
            pre = row.get("pre_overlap")
            if pre:
                col = pre.get("collider") or {}
                print(
                    f"  overlap t={pre['t_hit']} replan t={pre['t_replan']} "
                    f"hits_wm={pre['plan_hits_wm']} hits_gt={pre['plan_hits_gt']} "
                    f"collider_wm_ade={col.get('wm_ade_m')} collider_cv_ade={col.get('cv_ade_m')}",
                    flush=True,
                )

    ok_rows = [r for r in rows if "error" not in r]
    summary = summarize(ok_rows)
    summary["n_errors"] = sum(1 for r in rows if "error" in r)
    summary["compare_json"] = str(args.compare_json)
    summary["out_dir"] = str(out_dir)
    _save_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
