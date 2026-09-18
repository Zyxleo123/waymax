from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable=None, *args, **kwargs):
		return iterable


def _find_repo_root(start: Path) -> Path:
	for candidate in [start, *start.parents]:
		if (candidate / "README.md").exists() and (candidate / "data").is_dir():
			return candidate
	raise FileNotFoundError("Could not find the repository root.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from data.failure_detection_loader import _pick_failure_reason
from data.preprocess_cached import _CACHE_FIELD_NAMES
from model.diffusion.fm_policy import (
	FAILURE_REASON_COLLISION,
	FAILURE_REASON_GOAL_NOT_REACHED,
	FAILURE_REASON_NONE,
	FAILURE_REASON_OFFROAD,
)


_SOURCE_CACHE_RE = re.compile(r"training_tfexample\.tfrecord-(\d+)-of-01000\.sim_state_cache\.npz$")
_REASON_TO_ID = {
	"collision": FAILURE_REASON_COLLISION,
	"offroad": FAILURE_REASON_OFFROAD,
	"goal not reached": FAILURE_REASON_GOAL_NOT_REACHED,
	"success": FAILURE_REASON_NONE,
}
_ANNOTATION_FIELD_NAMES = (
	"failure_reason",
	"failure_step",
	"critical_object_ids",
	"critical_road_segment_ids",
	"collision_vehicle_id",
	"offroad_road_edge_segment_id",
	"collision_first_timestep",
	"offroad_first_timestep",
)


@dataclass(frozen=True)
class FailureAnnotation:
	tfrecord_name: str
	tfrecord_path: str
	scenario_idx: int
	result_path: Path
	reason: str
	reason_id: int
	failure_step: int
	object_id: int
	road_id: int
	collision_first_timestep: int
	offroad_first_timestep: int


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Extract WOMD logged (successful) trajectories for annotated failure scenarios "
			"and pack them with failure_reason / gold ids."
		)
	)
	parser.add_argument(
		"--exp_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_fm_training_collect_failure_20260903_030456",
		help="Experiment directory containing annotated per-scenario failure JSON files.",
	)
	parser.add_argument(
		"--source_cache_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz",
		help="Directory of original per-TFRecord WOMD sim_state_cache npz files.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Output directory. Defaults to <exp_dir>/success_for_failure_cache.",
	)
	parser.add_argument(
		"--chunk_size",
		type=int,
		default=500,
		help="Number of scenarios packed into one output shard.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing shard files.",
	)
	parser.add_argument(
		"--max_scenarios",
		type=int,
		default=None,
		help="Optional cap on the number of annotated failures to pack.",
	)
	return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)
		handle.write("\n")


def _iter_result_json_paths(exp_dir: Path) -> Iterator[Path]:
	for path in sorted(exp_dir.rglob("*.json")):
		if path.name == "summary.json":
			continue
		yield path


def _reason_id(reason: str) -> int:
	return int(_REASON_TO_ID.get(reason, FAILURE_REASON_NONE))


def _usable_failure(reason: str, object_id: int, road_id: int) -> bool:
	if reason == "collision":
		return object_id >= 0
	if reason == "offroad":
		return road_id >= 0
	if reason == "goal not reached":
		return True
	return False


def _as_int(value: Any, default: int = -1) -> int:
	if value is None:
		return int(default)
	try:
		return int(value)
	except (TypeError, ValueError):
		return int(default)


def _parse_tfrecord_index(tfrecord_name: str) -> int:
	match = _SOURCE_CACHE_RE.match(f"{Path(tfrecord_name).name}.sim_state_cache.npz")
	if match is None:
		match = re.search(r"tfrecord-(\d+)-of-", Path(tfrecord_name).name)
	if match is None:
		return -1
	return int(match.group(1))


def _index_source_caches(source_cache_dir: Path) -> dict[str, Path]:
	if not source_cache_dir.exists():
		raise FileNotFoundError(f"source_cache_dir does not exist: {source_cache_dir}")
	index: dict[str, Path] = {}
	for path in sorted(source_cache_dir.rglob("*.sim_state_cache.npz")):
		if ".scenario_" in path.name:
			continue
		stem = path.name.removesuffix(".sim_state_cache.npz")
		index.setdefault(stem, path)
		index.setdefault(path.name, path)
	if not index:
		raise FileNotFoundError(
			f"No per-TFRecord WOMD caches found in {source_cache_dir}. "
			"Expected files like 'training_tfexample.tfrecord-00000-of-01000.sim_state_cache.npz'."
		)
	return index


def _source_cache_for_tfrecord(index: dict[str, Path], tfrecord_name: str) -> Path | None:
	name = Path(tfrecord_name).name
	return index.get(name) or index.get(f"{name}.sim_state_cache.npz")


def _collect_failure_annotations(exp_dir: Path) -> tuple[list[FailureAnnotation], dict[str, int]]:
	counts = {
		"json_files": 0,
		"success_skipped": 0,
		"unusable_gold_id": 0,
		"missing_tfrecord": 0,
		"usable": 0,
	}
	annotations: list[FailureAnnotation] = []
	for result_path in _iter_result_json_paths(exp_dir):
		counts["json_files"] += 1
		try:
			record = _load_json(result_path)
		except (OSError, json.JSONDecodeError):
			continue
		if bool(record.get("success", True)):
			counts["success_skipped"] += 1
			continue
		tfrecord_path = str(record.get("tfrecord", "") or record.get("tfrecord_path", ""))
		if not tfrecord_path:
			counts["missing_tfrecord"] += 1
			continue
		scenario_idx = _as_int(record.get("scenario_idx", record.get("scenario_index")), -1)
		if scenario_idx < 0:
			counts["missing_tfrecord"] += 1
			continue
		reason, failure_step = _pick_failure_reason(record, num_steps=91)
		object_id = _as_int(record.get("collision_vehicle_id"), -1)
		road_id = _as_int(record.get("offroad_road_edge_segment_id"), -1)
		if not _usable_failure(reason, object_id, road_id):
			counts["unusable_gold_id"] += 1
			continue
		tfrecord_name = Path(tfrecord_path).name
		annotations.append(
			FailureAnnotation(
				tfrecord_name=tfrecord_name,
				tfrecord_path=tfrecord_path,
				scenario_idx=int(scenario_idx),
				result_path=result_path,
				reason=reason,
				reason_id=_reason_id(reason),
				failure_step=int(failure_step),
				object_id=object_id if reason == "collision" else -1,
				road_id=road_id if reason == "offroad" else -1,
				collision_first_timestep=_as_int(record.get("collision_first_timestep"), -1),
				offroad_first_timestep=_as_int(record.get("offroad_first_timestep"), -1),
			)
		)
		counts["usable"] += 1
	return annotations, counts


def _scenario_row_map(source_npz: np.lib.npyio.NpzFile) -> dict[int, int]:
	if "scenario_indices" in source_npz.files:
		indices = np.asarray(source_npz["scenario_indices"]).reshape(-1)
		return {int(scenario_idx): int(row_idx) for row_idx, scenario_idx in enumerate(indices.tolist())}
	num_scenarios = 1
	if "num_scenarios" in source_npz.files:
		num_scenarios = int(np.asarray(source_npz["num_scenarios"]).reshape(-1)[0])
	elif "log_trajectory_x" in source_npz.files:
		num_scenarios = int(np.asarray(source_npz["log_trajectory_x"]).shape[0])
	return {row_idx: row_idx for row_idx in range(num_scenarios)}


def _copy_row(value: Any, row_idx: int) -> np.ndarray:
	array = np.asarray(value)
	if array.ndim == 0:
		return np.array(array)
	if array.shape[0] == 1 and row_idx == 0:
		return np.array(array[0])
	if row_idx >= array.shape[0]:
		raise IndexError(f"row_idx={row_idx} is out of range for array with shape {array.shape}")
	return np.array(array[row_idx])


def _pack_rows(arrays: list[np.ndarray]) -> np.ndarray:
	if not arrays:
		raise ValueError("No rows were produced for a cache chunk.")
	try:
		return np.stack(arrays, axis=0)
	except ValueError:
		packed = np.empty((len(arrays),), dtype=object)
		for i, item in enumerate(arrays):
			packed[i] = item
		return packed


def _annotation_payload(item: FailureAnnotation) -> dict[str, np.ndarray]:
	return {
		"failure_reason": np.asarray(item.reason_id, dtype=np.int32),
		"failure_step": np.asarray(item.failure_step, dtype=np.int32),
		"critical_object_ids": np.asarray(item.object_id, dtype=np.int32),
		"critical_road_segment_ids": np.asarray(item.road_id, dtype=np.int32),
		"collision_vehicle_id": np.asarray(item.object_id, dtype=np.int32),
		"offroad_road_edge_segment_id": np.asarray(item.road_id, dtype=np.int32),
		"collision_first_timestep": np.asarray(item.collision_first_timestep, dtype=np.int32),
		"offroad_first_timestep": np.asarray(item.offroad_first_timestep, dtype=np.int32),
	}


def _write_chunk(
	output_path: Path,
	rows: list[dict[str, np.ndarray]],
	overwrite: bool,
) -> Path:
	if output_path.exists() and not overwrite:
		return output_path
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		raise ValueError(f"No rows to write for {output_path}")

	packed: dict[str, np.ndarray] = {
		"num_scenarios": np.asarray(len(rows), dtype=np.int32),
		"cached_field_names": np.asarray(sorted(_CACHE_FIELD_NAMES), dtype=np.str_),
		"tfrecord_path": np.asarray([row["tfrecord_path"] for row in rows], dtype=np.str_),
		"tfrecord_name": np.asarray([row["tfrecord_name"] for row in rows], dtype=np.str_),
		"tfrecord_index": np.asarray([row["tfrecord_index"] for row in rows], dtype=np.int32),
		"scenario_indices": np.asarray([row["scenario_idx"] for row in rows], dtype=np.int32),
		"result_path": np.asarray([row["result_path"] for row in rows], dtype=np.str_),
		"source_cache_path": np.asarray([row["source_cache_path"] for row in rows], dtype=np.str_),
		"failure_reason_name": np.asarray([row["failure_reason_name"] for row in rows], dtype=np.str_),
		"world_idx": np.asarray(0, dtype=np.int32),
	}
	for field_name in _CACHE_FIELD_NAMES:
		packed[field_name] = _pack_rows([row[field_name] for row in rows])
	for field_name in _ANNOTATION_FIELD_NAMES:
		packed[field_name] = np.asarray([row[field_name] for row in rows], dtype=np.int32)
	np.savez_compressed(output_path, **packed)
	return output_path


def _extract_success_rows(
	annotations: list[FailureAnnotation],
	source_path: Path,
	counts: dict[str, int],
) -> list[dict[str, np.ndarray]]:
	rows: list[dict[str, np.ndarray]] = []
	with np.load(source_path, allow_pickle=False, mmap_mode="r") as source_npz:
		row_map = _scenario_row_map(source_npz)
		missing_fields = [name for name in _CACHE_FIELD_NAMES if name not in source_npz.files]
		if missing_fields:
			raise KeyError(f"{source_path} is missing required fields: {missing_fields}")
		for item in annotations:
			row_idx = row_map.get(int(item.scenario_idx))
			if row_idx is None:
				counts["missing_scenario"] += 1
				continue
			try:
				extracted = {name: _copy_row(source_npz[name], row_idx) for name in _CACHE_FIELD_NAMES}
			except IndexError:
				counts["missing_scenario"] += 1
				continue
			num_steps = int(np.asarray(extracted["log_trajectory_x"]).shape[-1])
			failure_step = int(np.clip(item.failure_step, 0, max(0, num_steps - 1)))
			row = {
				**extracted,
				**_annotation_payload(item),
				"failure_step": np.asarray(failure_step, dtype=np.int32),
				"tfrecord_path": str(item.tfrecord_path),
				"tfrecord_name": str(item.tfrecord_name),
				"tfrecord_index": int(_parse_tfrecord_index(item.tfrecord_name)),
				"scenario_idx": int(item.scenario_idx),
				"result_path": str(item.result_path),
				"source_cache_path": str(source_path),
				"failure_reason_name": str(item.reason),
			}
			rows.append(row)
			counts["packed"] += 1
	return rows


def run(args: argparse.Namespace) -> list[Path]:
	exp_dir = Path(args.exp_dir)
	if not exp_dir.exists():
		raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
	source_cache_dir = Path(args.source_cache_dir)
	output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "success_for_failure_cache"
	output_dir.mkdir(parents=True, exist_ok=True)

	annotations, collect_counts = _collect_failure_annotations(exp_dir)
	if args.max_scenarios is not None:
		annotations = annotations[: int(args.max_scenarios)]
	if not annotations:
		raise FileNotFoundError(
			f"No usable annotated failures found in {exp_dir}. "
			"Need non-success JSONs with collision/offroad gold ids or goal-not-reached."
		)

	source_index = _index_source_caches(source_cache_dir)
	grouped: dict[str, list[FailureAnnotation]] = defaultdict(list)
	counts = {
		**collect_counts,
		"missing_source_cache": 0,
		"missing_scenario": 0,
		"packed": 0,
	}
	for item in annotations:
		grouped[item.tfrecord_name].append(item)

	pending_rows: list[dict[str, np.ndarray]] = []
	written_paths: list[Path] = []
	shard_index = 0
	chunk_size = max(1, int(args.chunk_size))

	for tfrecord_name, group in tqdm(sorted(grouped.items()), desc="Extracting success caches"):
		source_path = _source_cache_for_tfrecord(source_index, tfrecord_name)
		if source_path is None:
			counts["missing_source_cache"] += len(group)
			continue
		extracted_rows = _extract_success_rows(group, source_path, counts)
		for row in extracted_rows:
			pending_rows.append(row)
			if len(pending_rows) >= chunk_size:
				output_path = output_dir / f"success_for_failure_shard_{shard_index:05d}.sim_state_cache.npz"
				written_paths.append(_write_chunk(output_path, pending_rows, overwrite=bool(args.overwrite)))
				shard_index += 1
				pending_rows = []

	if pending_rows:
		output_path = output_dir / f"success_for_failure_shard_{shard_index:05d}.sim_state_cache.npz"
		written_paths.append(_write_chunk(output_path, pending_rows, overwrite=bool(args.overwrite)))

	summary = {
		"exp_dir": str(exp_dir),
		"source_cache_dir": str(source_cache_dir),
		"output_dir": str(output_dir),
		"num_shards": len(written_paths),
		"counts": counts,
		"shards": [str(path) for path in written_paths],
	}
	_save_json(output_dir / "summary.json", summary)
	print(json.dumps(summary["counts"], indent=2, sort_keys=True))
	return written_paths


def main() -> None:
	args = _parse_args()
	written_paths = run(args)
	for path in written_paths:
		print(path)


if __name__ == "__main__":
	main()
