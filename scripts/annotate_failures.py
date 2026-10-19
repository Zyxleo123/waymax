from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _find_repo_root(start: Path) -> Path:
	for candidate in [start, *start.parents]:
		if (candidate / "README.md").exists() and (candidate / "data").is_dir():
			return candidate
	raise FileNotFoundError("Could not find the repository root.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from waymax.datatypes.object_state import ObjectTypeIds
from waymax.datatypes.roadgraph import MapElementIds


ROAD_EDGE_TYPE_IDS = {
	int(MapElementIds.ROAD_EDGE_BOUNDARY.value),
	int(MapElementIds.ROAD_EDGE_MEDIAN.value),
}
VEHICLE_TYPE_ID = int(ObjectTypeIds.VEHICLE.value)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Annotate failure JSON files with offroad/collision details.")
	parser.add_argument(
		"--exp_dir",
		type=str,
		default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/exp",
		help="Experiment directory containing per-scenario JSON files and a sim_state_cache subdirectory.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing failure annotations in JSON files.",
	)
	return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)
		handle.write("\n")


def _iter_result_json_paths(exp_dir: Path) -> list[Path]:
	paths: list[Path] = []
	for path in sorted(exp_dir.rglob("*.json")):
		if path.name == "summary.json":
			continue
		paths.append(path)
	return paths


def _cache_path_for_result(result_path: Path, tfrecord_name: str, scenario_idx: int) -> Path:
	cache_file_name = f"{tfrecord_name}.scenario_{int(scenario_idx):03d}.sim_state_cache.npz"
	for parent in [result_path.parent, *result_path.parents]:
		candidate = parent / "sim_state_cache" / cache_file_name
		if candidate.exists():
			return candidate
	return result_path.parent / "sim_state_cache" / cache_file_name


def _to_numpy(value: Any) -> np.ndarray:
	return np.asarray(value)


def _ego_index_from_metadata(object_is_sdc: np.ndarray) -> int:
	mask = np.asarray(object_is_sdc).astype(bool)
	if mask.ndim != 1:
		raise ValueError(f"Expected a 1D ego mask, got {mask.shape}.")
	ego_count = int(np.sum(mask))
	if ego_count != 1:
		raise ValueError(f"Expected exactly one ego object, got {ego_count}.")
	return int(np.argmax(mask.astype(np.int32)))


def _polygon_corners_xy(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
	cos_h = float(np.cos(heading))
	sin_h = float(np.sin(heading))
	dx = float(length) / 2.0 * cos_h
	dy = float(length) / 2.0 * sin_h
	wx = float(width) / 2.0 * sin_h
	wy = float(width) / 2.0 * cos_h
	return np.asarray(
		[
			[x + dx - wx, y + dy + wy],
			[x + dx + wx, y + dy - wy],
			[x - dx + wx, y - dy - wy],
			[x - dx - wx, y - dy + wy],
		],
		dtype=np.float32,
	)


def _polygon_axes(polygon_xy: np.ndarray) -> np.ndarray:
	edges = np.roll(polygon_xy, shift=-1, axis=0) - polygon_xy
	axes = np.asarray([[-edge[1], edge[0]] for edge in edges], dtype=np.float32)
	norms = np.linalg.norm(axes, axis=-1, keepdims=True)
	norms = np.maximum(norms, 1e-6)
	return axes / norms


def _polygons_intersect(poly_a_xy: np.ndarray, poly_b_xy: np.ndarray) -> bool:
	for axis_xy in np.concatenate([_polygon_axes(poly_a_xy), _polygon_axes(poly_b_xy)], axis=0):
		proj_a = poly_a_xy @ axis_xy
		proj_b = poly_b_xy @ axis_xy
		if float(np.max(proj_a)) < float(np.min(proj_b)) or float(np.max(proj_b)) < float(np.min(proj_a)):
			return False
	return True


def _extract_offroad_details(cache_npz: np.lib.npyio.NpzFile) -> dict[str, Any] | None:
	object_is_sdc = _to_numpy(cache_npz["object_metadata_is_sdc"])
	ego_idx = _ego_index_from_metadata(object_is_sdc)

	log_x = _to_numpy(cache_npz["log_trajectory_x"])
	log_y = _to_numpy(cache_npz["log_trajectory_y"])
	log_yaw = _to_numpy(cache_npz["log_trajectory_yaw"])
	log_length = _to_numpy(cache_npz["log_trajectory_length"])
	log_width = _to_numpy(cache_npz["log_trajectory_width"])
	log_valid = _to_numpy(cache_npz["log_trajectory_valid"]).astype(bool)

	roadgraph_x = _to_numpy(cache_npz["roadgraph_points_x"])
	roadgraph_y = _to_numpy(cache_npz["roadgraph_points_y"])
	roadgraph_dir_x = _to_numpy(cache_npz["roadgraph_points_dir_x"])
	roadgraph_dir_y = _to_numpy(cache_npz["roadgraph_points_dir_y"])
	roadgraph_ids = _to_numpy(cache_npz["roadgraph_points_ids"])
	roadgraph_types = _to_numpy(cache_npz["roadgraph_points_types"]).astype(np.int32)
	roadgraph_valid = _to_numpy(cache_npz["roadgraph_points_valid"]).astype(bool)

	road_edge_mask = roadgraph_valid & np.isin(roadgraph_types, np.asarray(sorted(ROAD_EDGE_TYPE_IDS), dtype=np.int32))
	if not np.any(road_edge_mask):
		return None

	sampled_xy = np.stack([roadgraph_x[road_edge_mask], roadgraph_y[road_edge_mask]], axis=-1)
	sampled_dir = np.stack([roadgraph_dir_x[road_edge_mask], roadgraph_dir_y[road_edge_mask]], axis=-1)
	sampled_ids = roadgraph_ids[road_edge_mask].astype(np.int64)

	num_timesteps = int(log_x.shape[1])
	for timestep in range(num_timesteps):
		if not bool(log_valid[ego_idx, timestep]):
			continue
		corners = _polygon_corners_xy(
			float(log_x[ego_idx, timestep]),
			float(log_y[ego_idx, timestep]),
			float(log_yaw[ego_idx, timestep]),
			float(log_length[ego_idx, timestep]),
			float(log_width[ego_idx, timestep]),
		)
		square_distances = np.sum((sampled_xy[None, :, :] - corners[:, None, :]) ** 2, axis=-1)
		nearest_indices = np.argmin(square_distances, axis=-1)
		prior_indices = np.maximum(0, nearest_indices - 1)
		nearest_xy = sampled_xy[nearest_indices]
		nearest_dir = sampled_dir[nearest_indices]
		prior_dir = sampled_dir[prior_indices]
		points_to_edge = corners - nearest_xy
		cross_product = points_to_edge[:, 0] * nearest_dir[:, 1] - points_to_edge[:, 1] * nearest_dir[:, 0]
		cross_product_prior = points_to_edge[:, 0] * prior_dir[:, 1] - points_to_edge[:, 1] * prior_dir[:, 0]
		same_curve = sampled_ids[nearest_indices] == sampled_ids[prior_indices]
		offroad_sign = np.sign(np.where(np.logical_and(same_curve, cross_product_prior < cross_product), cross_product_prior, cross_product))
		signed_distances = np.linalg.norm(nearest_xy - corners, axis=-1) * offroad_sign
		positive_mask = signed_distances > 0.0
		if not np.any(positive_mask):
			continue
		corner_idx = int(np.argmin(np.where(positive_mask, signed_distances, np.inf)))
		return {
			"offroad_first_timestep": int(timestep),
			"offroad_road_edge_segment_id": int(sampled_ids[nearest_indices[corner_idx]]),
		}

	return None


def _extract_collision_details(
	cache_npz: np.lib.npyio.NpzFile,
) -> dict[str, Any] | None:
	object_is_sdc = _to_numpy(cache_npz["object_metadata_is_sdc"])
	object_types = _to_numpy(cache_npz["object_metadata_object_types"]).astype(np.int32)
	ego_idx = _ego_index_from_metadata(object_is_sdc)

	log_x = _to_numpy(cache_npz["log_trajectory_x"])
	log_y = _to_numpy(cache_npz["log_trajectory_y"])
	log_yaw = _to_numpy(cache_npz["log_trajectory_yaw"])
	log_length = _to_numpy(cache_npz["log_trajectory_length"])
	log_width = _to_numpy(cache_npz["log_trajectory_width"])
	log_valid = _to_numpy(cache_npz["log_trajectory_valid"]).astype(bool)

	vehicle_mask = object_types == VEHICLE_TYPE_ID
	vehicle_mask[ego_idx] = False
	if not np.any(vehicle_mask):
		return None

	vehicle_indices = np.flatnonzero(vehicle_mask)

	num_timesteps = int(log_x.shape[1])
	for timestep in range(num_timesteps):
		if not bool(log_valid[ego_idx, timestep]):
			continue
		ego_polygon = _polygon_corners_xy(
			float(log_x[ego_idx, timestep]),
			float(log_y[ego_idx, timestep]),
			float(log_yaw[ego_idx, timestep]),
			float(log_length[ego_idx, timestep]),
			float(log_width[ego_idx, timestep]),
		)
		colliding_vehicle_indices: list[int] = []
		for vehicle_idx in vehicle_indices:
			if not bool(log_valid[vehicle_idx, timestep]):
				continue
			vehicle_polygon = _polygon_corners_xy(
				float(log_x[vehicle_idx, timestep]),
				float(log_y[vehicle_idx, timestep]),
				float(log_yaw[vehicle_idx, timestep]),
				float(log_length[vehicle_idx, timestep]),
				float(log_width[vehicle_idx, timestep]),
			)
			if _polygons_intersect(ego_polygon, vehicle_polygon):
				colliding_vehicle_indices.append(int(vehicle_idx))
		if not colliding_vehicle_indices:
			continue

		selected_vehicle_id = int(colliding_vehicle_indices[0])

		return {
			"collision_first_timestep": int(timestep),
			"collision_vehicle_id": int(selected_vehicle_id),
		}

	return None


def _annotate_result_json(
	result_path: Path,
	overwrite: bool,
) -> bool:
	record = _load_json(result_path)
	if bool(record.get("success", True)):
		return False

	collision_flag = bool(record.get("overlap", False)) or bool(record.get("collision", False))
	offroad_flag = bool(record.get("offroad", False))
	if not collision_flag and not offroad_flag:
		return False

	if not overwrite:
		already_annotated = (
			("offroad_first_timestep" in record and "offroad_road_edge_segment_id" in record)
			or ("collision_first_timestep" in record and "collision_vehicle_id" in record)
		)
		if already_annotated:
			return False

	tfrecord_path = str(record.get("tfrecord", ""))
	if not tfrecord_path:
		raise KeyError(f"Missing 'tfrecord' in result JSON: {result_path}")
	scenario_idx = int(record["scenario_idx"])
	cache_path = _cache_path_for_result(result_path, Path(tfrecord_path).name, scenario_idx)
	if not cache_path.exists():
		raise FileNotFoundError(f"Missing sim_state cache file for {result_path}: {cache_path}")

	updates: dict[str, Any] = {}
	with np.load(cache_path, allow_pickle=False) as cache_npz:
		if offroad_flag:
			offroad_details = _extract_offroad_details(cache_npz)
			if offroad_details is not None:
				updates.update(offroad_details)
		if collision_flag:
			collision_details = _extract_collision_details(
					cache_npz,
			)
			if collision_details is not None:
				updates.update(collision_details)

	if not updates:
		return False

	record.update(updates)
	_save_json(result_path, record)
	return True


def run(args: argparse.Namespace) -> list[Path]:
	from tqdm import tqdm
	exp_dir = Path(args.exp_dir)
	if not exp_dir.exists():
		raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

	updated_paths: list[Path] = []
	for result_path in tqdm(_iter_result_json_paths(exp_dir), desc="Annotating results"):
		try:
			if _annotate_result_json(result_path, overwrite=bool(args.overwrite)):
				updated_paths.append(result_path)
		except Exception as exc:
			print(f"Failed to annotate {result_path}: {exc}")

	return updated_paths


def main() -> None:
	args = _parse_args()
	updated_paths = run(args)
	for path in updated_paths:
		print(path)


if __name__ == "__main__":
	main()