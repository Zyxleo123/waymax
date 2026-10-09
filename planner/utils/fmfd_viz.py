"""Before/after CFG comparison images for FM-FD failure corrections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from data.failure_detection_loader import (
	FAILURE_REASON_COLLISION,
	FAILURE_REASON_GOAL_NOT_REACHED,
	FAILURE_REASON_OFFROAD,
)
from viz import viz as viz_module
from viz.utils import plot_numpy_bounding_boxes


HIGHLIGHT_RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
UNCOND_TRAJ_COLOR = "#1f77b4"
CFG_TRAJ_COLOR = "#ff7f0e"
VIEW_RANGE_M = 40.0


def _as_numpy(value: Any) -> np.ndarray:
	return np.asarray(value)


def _ego_center_xy(state, timestep: int) -> np.ndarray:
	is_ego = np.asarray(state.object_metadata.is_sdc).astype(bool)
	xy = np.stack(
		[
			np.asarray(state.log_trajectory.x)[:, timestep],
			np.asarray(state.log_trajectory.y)[:, timestep],
		],
		axis=-1,
	)
	if np.any(is_ego):
		return xy[is_ego][0]
	return np.nanmean(xy, axis=0)


def _valid_xy(traj_t5: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
	xy = np.asarray(traj_t5, dtype=np.float32)[:, :2]
	if valid is None:
		return xy
	mask = np.asarray(valid).astype(bool).reshape(-1)
	if mask.shape[0] != xy.shape[0]:
		n = min(mask.shape[0], xy.shape[0])
		return xy[:n][mask[:n]]
	return xy[mask]


def _plot_trajectory_overlay(
	ax,
	traj_t5: np.ndarray,
	valid: np.ndarray | None,
	*,
	color: str,
) -> None:
	xy = _valid_xy(traj_t5, valid)
	if xy.shape[0] == 0:
		return
	ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2, alpha=0.95, zorder=8)
	ax.scatter(xy[:, 0], xy[:, 1], s=14, color=color, zorder=9, edgecolors="white", linewidths=0.4)
	ax.scatter(xy[0, 0], xy[0, 1], s=42, color=color, marker="o", zorder=10, edgecolors="black", linewidths=0.6)
	ax.scatter(xy[-1, 0], xy[-1, 1], s=48, color=color, marker="x", zorder=10, linewidths=1.8)


def _highlight_collision_object(ax, state, timestep: int, object_id: int) -> None:
	num_obj = int(state.log_trajectory.num_objects)
	if object_id < 0 or object_id >= num_obj:
		return
	valid = np.asarray(state.log_trajectory.valid)
	if valid.ndim == 2 and not bool(valid[object_id, timestep]):
		return
	bbox = np.stack(
		[
			np.asarray(state.log_trajectory.x)[object_id, timestep],
			np.asarray(state.log_trajectory.y)[object_id, timestep],
			np.asarray(state.log_trajectory.length)[object_id, timestep],
			np.asarray(state.log_trajectory.width)[object_id, timestep],
			np.asarray(state.log_trajectory.yaw)[object_id, timestep],
		],
		axis=0,
	)[None, :].astype(np.float32)
	plot_numpy_bounding_boxes(ax, bbox, color=HIGHLIGHT_RED, alpha=1.0)


def _highlight_road_segment(ax, state, road_id: int) -> None:
	if road_id < 0:
		return
	rg = state.roadgraph_points
	valid = np.asarray(rg.valid).astype(bool)
	ids = np.asarray(rg.ids)
	xy = np.asarray(rg.xy)
	mask = valid & (ids == int(road_id))
	if not np.any(mask):
		return
	ax.plot(
		xy[mask, 0],
		xy[mask, 1],
		color="red",
		linewidth=3.4,
		alpha=0.95,
		zorder=6,
	)


def _plot_goal(ax, goal_xy: np.ndarray | None) -> None:
	if goal_xy is None:
		return
	xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
	if xy.size < 2:
		return
	ax.scatter(
		float(xy[0]),
		float(xy[1]),
		s=90,
		marker="*",
		color="gold",
		edgecolors="black",
		linewidths=0.6,
		zorder=11,
		label="goal",
	)


def _draw_base_scene(
	ax,
	state,
	timestep: int,
	*,
	reason: int,
	object_id: int,
	road_id: int,
	goal_xy: np.ndarray | None,
) -> None:
	viz_module.plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
	is_ego = np.asarray(state.object_metadata.is_sdc).astype(bool)
	num_obj = int(state.log_trajectory.num_objects)
	is_controlled = np.zeros((num_obj,), dtype=bool)
	is_adv = np.zeros((num_obj,), dtype=bool)
	if int(reason) == FAILURE_REASON_COLLISION and 0 <= int(object_id) < num_obj:
		is_adv[int(object_id)] = True
	viz_module.plot_trajectory(
		ax,
		state.log_trajectory,
		is_controlled=is_controlled,
		time_idx=timestep,
		indices=None,
		past_traj_length=0,
		is_ego=is_ego,
		is_adv=is_adv,
	)
	if int(reason) == FAILURE_REASON_COLLISION:
		_highlight_collision_object(ax, state, timestep, int(object_id))
	elif int(reason) == FAILURE_REASON_OFFROAD:
		_highlight_road_segment(ax, state, int(road_id))
	_plot_goal(ax, goal_xy)

	center = _ego_center_xy(state, timestep)
	ax.axis(
		(
			float(center[0]) - VIEW_RANGE_M,
			float(center[0]) + VIEW_RANGE_M,
			float(center[1]) - VIEW_RANGE_M,
			float(center[1]) + VIEW_RANGE_M,
		)
	)
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, alpha=0.2)


def save_fmfd_cfg_comparison_image(
	*,
	sim_state,
	batch_idx: int,
	timestep: int,
	uncond_traj_t5: np.ndarray,
	cfg_traj_t5: np.ndarray,
	world_t_valid: np.ndarray | None,
	failure_text: str,
	reason: int,
	object_id: int,
	road_id: int,
	goal_xy: np.ndarray | None,
	output_path: Path,
) -> None:
	state_b = viz_module._index_pytree(sim_state, int(batch_idx))
	t = int(np.clip(timestep, 0, int(np.asarray(state_b.log_trajectory.x).shape[-1]) - 1))

	fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2))
	gold_note = ""
	if int(reason) == FAILURE_REASON_COLLISION:
		gold_note = f"  |  gold object_id={int(object_id)}"
	elif int(reason) == FAILURE_REASON_OFFROAD:
		gold_note = f"  |  gold road_id={int(road_id)}"
	fig.suptitle(f"{str(failure_text).strip()}{gold_note}  |  t={t}", fontsize=13, fontweight="bold")

	_draw_base_scene(
		axes[0],
		state_b,
		t,
		reason=int(reason),
		object_id=int(object_id),
		road_id=int(road_id),
		goal_xy=goal_xy,
	)
	_plot_trajectory_overlay(axes[0], uncond_traj_t5, world_t_valid, color=UNCOND_TRAJ_COLOR)
	axes[0].set_title("before CFG")

	_draw_base_scene(
		axes[1],
		state_b,
		t,
		reason=int(reason),
		object_id=int(object_id),
		road_id=int(road_id),
		goal_xy=goal_xy,
	)
	_plot_trajectory_overlay(axes[1], cfg_traj_t5, world_t_valid, color=CFG_TRAJ_COLOR)
	axes[1].set_title("after CFG")

	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=140)
	plt.close(fig)


__all__ = ["save_fmfd_cfg_comparison_image"]
