from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from annotation.helpers import get_left_lane_id, get_right_lane_id
from lane_graph.lane_graph_utils import LaneGraphData, LaneGraphShardStore, lane_graph_zip_path_for_tfrecord


ANNOTATION_FILE_RE = re.compile(r"^(?P<tfrecord>.+)_t(?P<step>\d+)\.jsonl$")


def _to_jsonable(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, dict):
		return {str(key): _to_jsonable(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(item) for item in value]
	return str(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def _annotation_file_info(path: Path) -> tuple[str, int] | None:
	match = ANNOTATION_FILE_RE.fullmatch(path.name)
	if match is None:
		return None
	return match.group("tfrecord"), int(match.group("step"))


def _group_annotation_files(annotation_dir: Path) -> dict[tuple[str, int], Path]:
	file_map: dict[tuple[str, int], Path] = {}
	for path in sorted(annotation_dir.glob("*.jsonl")):
		if path.name.endswith(".lane_context.jsonl"):
			continue
		info = _annotation_file_info(path)
		if info is None:
			continue
		file_map[info] = path
	return file_map


def _records_by_scenario(records: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
	indexed: dict[tuple[str, int], dict[str, Any]] = {}
	for record in records:
		tfrecord_path = str(record.get("tfrecord_path", ""))
		scenario_index = int(record.get("scenario_index", -1))
		indexed[(tfrecord_path, scenario_index)] = record
	return indexed


def _lane_id_range(lane_graph: LaneGraphData, lane_id: int) -> tuple[int, int] | None:
	return lane_graph.lane_id_to_node_range.get(int(lane_id))


def _lane_pose(lane_graph: LaneGraphData, lane_id: int, which: str) -> tuple[np.ndarray, np.ndarray] | None:
	lane_range = _lane_id_range(lane_graph, lane_id)
	if lane_range is None:
		return None
	start, end = int(lane_range[0]), int(lane_range[1])
	nodes_xyz = np.asarray(lane_graph.nodes_xyz, dtype=np.float32)
	if end - start <= 0 or start < 0 or end > int(nodes_xyz.shape[0]):
		return None

	if which == "start":
		xy = nodes_xyz[start, :2]
		if end - start >= 2:
			dir_xy = nodes_xyz[start + 1, :2] - nodes_xyz[start, :2]
		else:
			dir_xy = np.asarray([1.0, 0.0], dtype=np.float32)
	else:
		xy = nodes_xyz[end - 1, :2]
		if end - start >= 2:
			dir_xy = nodes_xyz[end - 1, :2] - nodes_xyz[end - 2, :2]
		else:
			dir_xy = np.asarray([1.0, 0.0], dtype=np.float32)

	if float(np.linalg.norm(dir_xy)) <= 1e-6:
		dir_xy = np.asarray([1.0, 0.0], dtype=np.float32)
	else:
		dir_xy = dir_xy / np.linalg.norm(dir_xy)
	return np.asarray(xy, dtype=np.float32), np.asarray(dir_xy, dtype=np.float32)


def _signed_angle(reference_dir: np.ndarray, target_dir: np.ndarray) -> float:
	reference_dir = np.asarray(reference_dir, dtype=np.float32)
	target_dir = np.asarray(target_dir, dtype=np.float32)
	reference_norm = float(np.linalg.norm(reference_dir))
	target_norm = float(np.linalg.norm(target_dir))
	if reference_norm <= 1e-6 or target_norm <= 1e-6:
		return 0.0
	reference_dir = reference_dir / reference_norm
	target_dir = target_dir / target_norm
	cross = float(reference_dir[0] * target_dir[1] - reference_dir[1] * target_dir[0])
	dot = float(reference_dir[0] * target_dir[0] + reference_dir[1] * target_dir[1])
	return float(math.atan2(cross, dot))


def _normalize_lane_ids(lane_ids: Iterable[Any]) -> list[int]:
	return sorted({int(lane_id) for lane_id in lane_ids})


def _direct_successor_lane_ids(lane_graph: LaneGraphData, lane_id: int) -> list[int]:
	lane_range = _lane_id_range(lane_graph, lane_id)
	if lane_range is None:
		return []
	start, end = int(lane_range[0]), int(lane_range[1])
	if end <= start:
		return []
	successor_edges = np.asarray(lane_graph.successor_edges, dtype=np.int64)
	if successor_edges.size == 0:
		return []
	next_start_indices = successor_edges[successor_edges[:, 0] == (end - 1)][:, 1]
	if next_start_indices.size == 0:
		return []
	return _normalize_lane_ids(
		lane_id_value for lane_id_value, node_range in lane_graph.lane_id_to_node_range.items() if int(node_range[0]) in set(next_start_indices.tolist())
	)


def _collect_reachable_lane_ids(
	lane_graph: LaneGraphData,
	seed_lane_ids: Iterable[int],
	max_hops: int = 8,
) -> set[int]:
	visited: set[int] = set()
	frontier = _normalize_lane_ids(seed_lane_ids)
	for lane_id in frontier:
		visited.add(int(lane_id))

	for _ in range(max(int(max_hops), 0)):
		next_frontier: list[int] = []
		for current_lane_id in frontier:
			for successor_lane_id in _direct_successor_lane_ids(lane_graph, int(current_lane_id)):
				if successor_lane_id in visited:
					continue
				visited.add(successor_lane_id)
				next_frontier.append(successor_lane_id)
		if not next_frontier:
			break
		frontier = _normalize_lane_ids(next_frontier)
	return visited


def _branch_candidates(lane_graph: LaneGraphData, start_lane_id: int) -> list[dict[str, Any]]:
	current_pose = _lane_pose(lane_graph, int(start_lane_id), which="end")
	if current_pose is None:
		return []
	current_xy, current_dir = current_pose
	candidate_lane_ids = _direct_successor_lane_ids(lane_graph, int(start_lane_id))
	candidates: list[dict[str, Any]] = []
	for lane_id in candidate_lane_ids:
		candidate_pose = _lane_pose(lane_graph, int(lane_id), which="start")
		if candidate_pose is None:
			continue
		candidate_xy, candidate_dir = candidate_pose
		angle = _signed_angle(current_dir, candidate_dir)
		distance = float(np.linalg.norm(candidate_xy - current_xy))
		if angle > 0.15:
			direction = "left"
		elif angle < -0.15:
			direction = "right"
		else:
			direction = "straight"
		candidates.append(
			{
				"lane_id": int(lane_id),
				"signed_angle_rad": float(angle),
				"distance_to_start": float(distance),
				"direction": direction,
			}
		)

	candidates.sort(key=lambda item: (-float(item["signed_angle_rad"]), float(item["distance_to_start"]), int(item["lane_id"])))
	for branch_index, candidate in enumerate(candidates, start=1):
		candidate["branch_index_from_left"] = int(branch_index)
	return candidates


def _match_branch_candidates(
	lane_graph: LaneGraphData,
	start_lane_id: int,
	future_lane_id: int,
	max_hops: int = 8,
) -> dict[str, Any]:
	branch_candidates = _branch_candidates(lane_graph, int(start_lane_id))
	if not branch_candidates:
		return {
			"relation": "none",
			"branch_index_from_left": None,
			"branch_count": 0,
			"selected_branch_lane_id": None,
			"branch_candidates": [],
		}

	matched: list[dict[str, Any]] = []
	for candidate in branch_candidates:
		descendants = _collect_reachable_lane_ids(lane_graph, [int(candidate["lane_id"])], max_hops=max_hops)
		if int(future_lane_id) in descendants or int(future_lane_id) == int(candidate["lane_id"]):
			matched.append(candidate)

	if not matched:
		return {
			"relation": "none",
			"branch_index_from_left": None,
			"branch_count": len(branch_candidates),
			"selected_branch_lane_id": None,
			"branch_candidates": branch_candidates,
		}

	selected = matched[0]
	return {
		"relation": "branch",
		"branch_index_from_left": int(selected["branch_index_from_left"]),
		"branch_count": len(branch_candidates),
		"selected_branch_lane_id": int(selected["lane_id"]),
		"matched_branch_lane_ids": [int(item["lane_id"]) for item in matched],
		"ambiguous": len(matched) > 1,
		"branch_candidates": branch_candidates,
	}


def _match_side_lane_relation(
	lane_graph: LaneGraphData,
	start_lane_id: int,
	future_lane_id: int,
	max_hops: int = 8,
) -> dict[str, Any]:
	left_lane_ids = _normalize_lane_ids(get_left_lane_id(lane_graph, int(start_lane_id)))
	right_lane_ids = _normalize_lane_ids(get_right_lane_id(lane_graph, int(start_lane_id)))
	left_reachable = _collect_reachable_lane_ids(lane_graph, left_lane_ids, max_hops=max_hops) if left_lane_ids else set()
	right_reachable = _collect_reachable_lane_ids(lane_graph, right_lane_ids, max_hops=max_hops) if right_lane_ids else set()

	if int(future_lane_id) in left_reachable:
		relation = "left"
		matched_lane_ids = sorted(left_reachable)
	elif int(future_lane_id) in right_reachable:
		relation = "right"
		matched_lane_ids = sorted(right_reachable)
	else:
		relation = "none"
		matched_lane_ids = []

	return {
		"relation": relation,
		"left_lane_ids": left_lane_ids,
		"right_lane_ids": right_lane_ids,
		"matched_lane_ids": matched_lane_ids,
	}


def build_lane_transition_context(
	current_record: dict[str, Any],
	future_record: dict[str, Any],
	lane_graph: LaneGraphData,
	anchor_step: int,
	future_step: int,
	max_hops: int = 8,
) -> dict[str, Any]:
	current_annotation = dict(current_record.get("annotation", {}))
	future_annotation = dict(future_record.get("annotation", {}))
	current_lane_context = dict(current_annotation.get("lane_context", {}))
	future_lane_context = dict(future_annotation.get("lane_context", {}))

	start_lane_id = int(current_lane_context.get("start_lane_id", -1))
	future_lane_id = int(future_lane_context.get("start_lane_id", -1))
	if start_lane_id < 0 or future_lane_id < 0:
		raise ValueError(
			f"Missing start_lane_id for scenario_index={current_record.get('scenario_index')} at anchor_step={anchor_step} or future_step={future_step}."
		)

	current_lane_ids = _normalize_lane_ids(_collect_reachable_lane_ids(lane_graph, [start_lane_id], max_hops=max_hops))
	current_left_lane_ids = _normalize_lane_ids(get_left_lane_id(lane_graph, start_lane_id))
	current_right_lane_ids = _normalize_lane_ids(get_right_lane_id(lane_graph, start_lane_id))
	direct_successor_lane_ids = _direct_successor_lane_ids(lane_graph, start_lane_id)
	branch_info = _match_branch_candidates(lane_graph, start_lane_id, future_lane_id, max_hops=max_hops)
	side_info = _match_side_lane_relation(lane_graph, start_lane_id, future_lane_id, max_hops=max_hops)

	if int(future_lane_id) == int(start_lane_id):
		transition_type = "same_lane"
		selected_branch_lane_id = None
		branch_index_from_left = None
	elif branch_info["relation"] == "branch":
		transition_type = "branch"
		selected_branch_lane_id = branch_info.get("selected_branch_lane_id")
		branch_index_from_left = branch_info.get("branch_index_from_left")
	elif side_info["relation"] in {"left", "right"}:
		transition_type = f"lane_change_{side_info['relation']}"
		selected_branch_lane_id = None
		branch_index_from_left = None
	elif int(future_lane_id) in current_lane_ids:
		transition_type = "follow_current_lane"
		selected_branch_lane_id = None
		branch_index_from_left = None
	else:
		transition_type = "unknown"
		selected_branch_lane_id = None
		branch_index_from_left = None

	return {
		"anchor_step": int(anchor_step),
		"future_step": int(future_step),
		"start_lane_id": int(start_lane_id),
		"future_lane_id": int(future_lane_id),
		"transition_type": transition_type,
		"branch_index_from_left": branch_index_from_left,
		"selected_branch_lane_id": selected_branch_lane_id,
		"direct_successor_lane_ids": direct_successor_lane_ids,
		"left_lane_ids": current_left_lane_ids,
		"right_lane_ids": current_right_lane_ids,
		"current_lane_ids": current_lane_ids,
		"branch_info": branch_info,
		"side_lane_info": side_info,
		"current_annotation": current_annotation,
		"future_annotation": future_annotation,
	}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build lane transition context from existing annotation JSONL files.")
	parser.add_argument(
		"--annotation_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/annotations/",
		help="Directory containing annotation JSONL files created by annotation.py.",
	)
	parser.add_argument(
		"--lane_graph_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs/",
		help="Directory containing matching lane graph shard zips.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Directory where enriched JSONL files will be written. Defaults to <annotation_dir>/lane_context.",
	)
	parser.add_argument(
		"--horizon",
		type=int,
		default=30,
		help="Future timestep horizon in steps. The script looks for anchor_step + horizon annotation files.",
	)
	parser.add_argument(
		"--max_hops",
		type=int,
		default=8,
		help="Maximum number of successor hops used when tracing branch and side-lane reachability.",
	)
	parser.add_argument(
		"--anchor_steps",
		type=int,
		nargs="*",
		default=None,
		help="Optional list of anchor timestep values to process. Defaults to all annotation files found.",
	)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONL files.")
	return parser.parse_args()


def run(args: argparse.Namespace) -> list[Path]:
	annotation_dir = Path(args.annotation_dir)
	if not annotation_dir.is_dir():
		raise FileNotFoundError(f"annotation_dir does not exist or is not a directory: {annotation_dir}")

	output_dir = Path(args.output_dir) if args.output_dir is not None else annotation_dir / "lane_context"
	output_dir.mkdir(parents=True, exist_ok=True)

	file_map = _group_annotation_files(annotation_dir)
	if not file_map:
		raise FileNotFoundError(f"No annotation JSONL files found in {annotation_dir}")

	if args.anchor_steps is None:
		anchor_steps = sorted({step for _, step in file_map.keys()})
	else:
		anchor_steps = [int(step) for step in args.anchor_steps]

	store_cache: dict[str, LaneGraphShardStore] = {}
	written_paths: list[Path] = []
	horizon = int(args.horizon)
	max_hops = int(args.max_hops)

	for anchor_step in anchor_steps:
		anchor_files = [
			(tfrecord_name, step, path)
			for (tfrecord_name, step), path in file_map.items()
			if int(step) == int(anchor_step)
		]
		if not anchor_files:
			continue

		future_step = int(anchor_step) + horizon
		for tfrecord_name, _, anchor_path in anchor_files:
			future_path = file_map.get((tfrecord_name, future_step))
			if future_path is None:
				continue

			output_path = output_dir / f"{anchor_path.stem}_h{horizon}.lane_context.jsonl"
			if output_path.exists() and not bool(args.overwrite):
				written_paths.append(output_path)
				continue

			current_records = _records_by_scenario(_load_jsonl(anchor_path))
			future_records = _records_by_scenario(_load_jsonl(future_path))
			if tfrecord_name not in store_cache:
				lane_graph_zip = lane_graph_zip_path_for_tfrecord(tfrecord_name, str(args.lane_graph_dir))
				store_cache[tfrecord_name] = LaneGraphShardStore(str(lane_graph_zip))
			store = store_cache[tfrecord_name]

			output_path.parent.mkdir(parents=True, exist_ok=True)
			with output_path.open("w", encoding="utf-8") as handle:
				for scenario_key, current_record in sorted(current_records.items(), key=lambda item: item[0][1]):
					future_record = future_records.get(scenario_key)
					if future_record is None:
						continue
					scenario_index = int(scenario_key[1])
					lane_graph = store.get_by_record_index(scenario_index)
					if lane_graph is None:
						continue
					lane_transition_context = build_lane_transition_context(
						current_record=current_record,
						future_record=future_record,
						lane_graph=lane_graph,
						anchor_step=int(anchor_step),
						future_step=int(future_step),
						max_hops=max_hops,
					)
					enriched_record = dict(current_record)
					enriched_record["future_annotation"] = future_record.get("annotation")
					enriched_record["future_instruction"] = future_record.get("instruction")
					enriched_record["lane_transition_context"] = lane_transition_context
					handle.write(json.dumps(_to_jsonable(enriched_record), ensure_ascii=False))
					handle.write("\n")

			written_paths.append(output_path)

	return written_paths


def main() -> None:
	args = _parse_args()
	written_paths = run(args)
	for path in written_paths:
		print(path)


if __name__ == "__main__":
	main()
