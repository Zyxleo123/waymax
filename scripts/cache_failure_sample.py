from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


_RESULT_RE = re.compile(r"^(?P<tfrecord_name>.+)\.scenario_(?P<scenario_idx>\d+)\.json$")
_SOURCE_CACHE_RE = re.compile(r"training_tfexample\.tfrecord-(\d+)-of-01000\.sim_state_cache\.npz$")
_FAILURE_CACHE_RE = re.compile(
	r"training_tfexample\.tfrecord-(\d+)-of-01000(?:\.scenario_(\d+))?\.sim_state_cache\.npz$"
)
_CACHE_FIELD_NAMES = (
	"object_metadata_is_sdc",
	"object_metadata_object_types",
	"log_trajectory_timestamp_micros",
	"log_trajectory_x",
	"log_trajectory_y",
	"log_trajectory_yaw",
	"log_trajectory_valid",
	"log_trajectory_vel_x",
	"log_trajectory_vel_y",
	"log_trajectory_speed",
	"log_trajectory_length",
	"log_trajectory_width",
	"log_traffic_light_state",
	"log_traffic_light_x",
	"log_traffic_light_y",
	"log_traffic_light_lane_ids",
	"log_traffic_light_valid",
	"roadgraph_points_x",
	"roadgraph_points_y",
	"roadgraph_points_ids",
	"roadgraph_points_types",
	"roadgraph_points_valid",
	"roadgraph_points_dir_x",
	"roadgraph_points_dir_y",
)


@dataclass(frozen=True)
class FailureScenarioRecord:
	tfrecord_name: str
	tfrecord_index: int
	scenario_idx: int
	result_path: Path
	cache_path: Path


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Bundle failure scenarios into chunked sim_state cache files."
	)
	parser.add_argument(
		"--exp_dir",
		type=str,
		required=True,
		help="Experiment directory containing per-scenario result JSON files.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
		help="Directory where chunked failure cache npz files will be written.",
	)
	parser.add_argument(
		"--source_cache_dir",
		type=str,
		default=None,
		help="Optional directory containing the original per-TFRecord sim_state cache files.",
	)
	parser.add_argument(
		"--chunk_size",
		type=int,
		default=500,
		help="Number of failure scenarios to pack into one cache file.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing chunk files.",
	)
	return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _iter_result_json_paths(exp_dir: Path) -> list[Path]:
	paths: list[Path] = []
	for path in sorted(exp_dir.rglob("*.json")):
		if path.name == "summary.json":
			continue
		paths.append(path)
	return paths


def _result_path_for_cache(cache_file: Path, tfrecord_name: str, scenario_idx: int) -> Path:
	stem = cache_file.name.removesuffix(".sim_state_cache.npz")
	for parent in [cache_file.parent, *cache_file.parents]:
		candidate = parent / "failure_results" / f"{stem}.scenario_{int(scenario_idx):03d}.json"
		if candidate.exists():
			return candidate
	return cache_file.parent / "failure_results" / f"{stem}.scenario_{int(scenario_idx):03d}.json"


def _cache_path_for_result(result_path: Path, tfrecord_name: str, scenario_idx: int) -> Path:
	cache_file_name = f"{tfrecord_name}.scenario_{int(scenario_idx):03d}.sim_state_cache.npz"
	for parent in [result_path.parent, *result_path.parents]:
		candidate = parent / "sim_state_cache" / cache_file_name
		if candidate.exists():
			return candidate
	return result_path.parent / "sim_state_cache" / cache_file_name


def _parse_tfrecord_index(path: Path) -> int:
	match = _SOURCE_CACHE_RE.match(path.name)
	if match is None:
		match = _FAILURE_CACHE_RE.match(path.name)
	if match is None:
		raise ValueError(f"Could not parse tfrecord index from cache file: {path}")
	return int(match.group(1))


def _extract_failure_records(exp_dir: Path) -> list[FailureScenarioRecord]:
	records: list[FailureScenarioRecord] = []
	for result_path in _iter_result_json_paths(exp_dir):
		record = _load_json(result_path)
		if bool(record.get("success", True)):
			continue
		tfrecord_path = str(record.get("tfrecord", ""))
		scenario_idx = int(record["scenario_idx"])
		if not tfrecord_path:
			continue
		tfrecord_name = Path(tfrecord_path).name
		cache_path = _cache_path_for_result(result_path, tfrecord_name, scenario_idx)
		if not cache_path.exists():
			continue
		tfrecord_index = _parse_tfrecord_index(cache_path)
		records.append(
			FailureScenarioRecord(
				tfrecord_name=tfrecord_name,
				tfrecord_index=tfrecord_index,
				scenario_idx=scenario_idx,
				result_path=result_path,
				cache_path=cache_path,
			)
		)
	return records


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


def _as_python_scalar(value: Any) -> Any:
	array_value = np.asarray(value)
	if array_value.ndim == 0:
		return array_value.item()
	return array_value.reshape(-1)[0].item()


def _extract_cache_arrays(cache_npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
	row: dict[str, np.ndarray] = {}
	for field_name in _CACHE_FIELD_NAMES:
		row[field_name] = np.asarray(cache_npz[field_name])
	return row


def _result_group_key(record: FailureScenarioRecord) -> tuple[str, int]:
	return record.tfrecord_name, record.tfrecord_index


def _write_chunk(output_path: Path, records: list[FailureScenarioRecord], overwrite: bool) -> Path:
	if output_path.exists() and not overwrite:
		return output_path
	output_path.parent.mkdir(parents=True, exist_ok=True)

	feature_chunks: dict[str, list[np.ndarray]] = {field_name: [] for field_name in _CACHE_FIELD_NAMES}
	selected_scenario_indices: list[int] = []
	tfrecord_paths: list[str] = []
	tfrecord_names: list[str] = []
	tfrecord_indices: list[int] = []
	first_cache: dict[str, Any] | None = None

	for record in records:
		with np.load(record.cache_path, allow_pickle=False) as cache_npz:
			if first_cache is None:
				first_cache = {key: cache_npz[key] for key in cache_npz.files}
			for field_name, value in _extract_cache_arrays(cache_npz).items():
				feature_chunks[field_name].append(value)
			selected_scenario_indices.append(int(record.scenario_idx))
			tfrecord_paths.append(str(cache_npz.get("tfrecord_path", record.cache_path)))
			tfrecord_names.append(str(cache_npz.get("tfrecord_name", record.tfrecord_name)))
			tfrecord_indices.append(int(record.tfrecord_index))

	if first_cache is None:
		raise RuntimeError(f"No cache data found for chunk {output_path}")

	world_idx_value = int(_as_python_scalar(first_cache.get("world_idx", 0)))

	packed: dict[str, np.ndarray] = {
		"tfrecord_path": np.asarray(tfrecord_paths, dtype=np.str_),
		"tfrecord_name": np.asarray(tfrecord_names, dtype=np.str_),
		"tfrecord_index": np.asarray(tfrecord_indices, dtype=np.int32),
		"world_idx": np.asarray(world_idx_value, dtype=np.int32),
		"num_scenarios": np.asarray(len(records), dtype=np.int32),
		"scenario_indices": np.asarray(selected_scenario_indices, dtype=np.int32),
		"cached_field_names": np.asarray(sorted(_CACHE_FIELD_NAMES), dtype=np.str_),
	}
	for field_name, values in feature_chunks.items():
		packed[field_name] = _pack_rows(values)

	# Save with the same raw-cache layout as build_cache.py, but chunked across failures.
	np.savez_compressed(output_path, **packed)
	return output_path


def run(args: argparse.Namespace) -> list[Path]:
	exp_dir = Path(args.exp_dir)
	if not exp_dir.exists():
		raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

	output_dir = Path(args.output_dir)
	records = _extract_failure_records(exp_dir)
	if not records:
		raise FileNotFoundError(f"No failure scenarios found in {exp_dir}.")

	record_groups: dict[tuple[str, int], list[FailureScenarioRecord]] = defaultdict(list)
	for record in records:
		record_groups[_result_group_key(record)].append(record)

	written_paths: list[Path] = []
	current_shard: list[FailureScenarioRecord] = []
	current_count = 0
	shard_index = 0
	chunk_size = max(1, int(args.chunk_size))

	for (_, _), group_records in tqdm(sorted(record_groups.items()), desc="Chunking failures"):
		group_records = sorted(group_records, key=lambda item: item.scenario_idx)
		group_size = len(group_records)
		if current_shard and current_count + group_size > chunk_size:
			output_path = output_dir / f"failure_shard_{shard_index:05d}.sim_state_cache.npz"
			written_paths.append(_write_chunk(output_path, current_shard, overwrite=bool(args.overwrite)))
			shard_index += 1
			current_shard = []
			current_count = 0

		current_shard.extend(group_records)
		current_count += group_size

	if current_shard:
		output_path = output_dir / f"failure_shard_{shard_index:05d}.sim_state_cache.npz"
		written_paths.append(_write_chunk(output_path, current_shard, overwrite=bool(args.overwrite)))

	return written_paths


def main() -> None:
	args = _parse_args()
	written_paths = run(args)
	for path in written_paths:
		print(path)


if __name__ == "__main__":
	main()
