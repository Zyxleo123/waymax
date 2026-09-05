from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm


ANNOTATION_FILE_RE = re.compile(r"^(?P<tfrecord>.+)_t(?P<step>\d+)\.jsonl$")


def _annotation_file_info(path: Path) -> tuple[str, int] | None:
	match = ANNOTATION_FILE_RE.fullmatch(path.name)
	if match is None:
		return None
	return match.group("tfrecord"), int(match.group("step"))


def _group_annotation_files(annotation_dir: Path) -> dict[tuple[str, int], Path]:
	file_map: dict[tuple[str, int], Path] = {}
	for path in sorted(annotation_dir.glob("*.jsonl")):
		if path.name.endswith(".idx.json"):
			continue
		if path.name.endswith(".lane_context.jsonl"):
			continue
		info = _annotation_file_info(path)
		if info is None:
			continue
		file_map[info] = path
	return file_map


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def _records_by_scenario(records: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
	indexed: dict[tuple[str, int], dict[str, Any]] = {}
	for record in records:
		tfrecord_path = str(record.get("tfrecord_path", ""))
		scenario_index = int(record.get("scenario_index", -1))
		indexed[(tfrecord_path, scenario_index)] = record
	return indexed


def _to_jsonable(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, dict):
		return {str(key): _to_jsonable(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(item) for item in value]
	return value


def _write_jsonl_with_index(path: Path, records: list[dict[str, Any]]) -> None:
	offsets: list[int] = []
	with path.open("w", encoding="utf-8") as handle:
		offset = 0
		for record in records:
			offsets.append(offset)
			line = json.dumps(_to_jsonable(record), ensure_ascii=False)
			handle.write(line)
			handle.write("\n")
			offset += len(line.encode("utf-8")) + 1
	idx_path = path.with_suffix(".idx.json")
	with idx_path.open("w", encoding="utf-8") as handle:
		json.dump(offsets, handle)
		handle.write("\n")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Add target_lane_id to saved annotation JSONL files.")
	parser.add_argument(
		"--annotation_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/annotations/",
		help="Directory containing annotation JSONL files produced by annotation.py.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Directory where enriched JSONL files will be written. Defaults to <annotation_dir>/target_lane.",
	)
	parser.add_argument(
		"--horizon",
		type=int,
		default=30,
		help="Number of steps ahead to use for target_lane_id. The target step is clamped to 90.",
	)
	parser.add_argument(
		"--max_step",
		type=int,
		default=90,
		help="Maximum timestep to use when clamping the future target step.",
	)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONL files.")
	return parser.parse_args()


def run(args: argparse.Namespace) -> list[Path]:
	annotation_dir = Path(args.annotation_dir)
	if not annotation_dir.is_dir():
		raise FileNotFoundError(f"annotation_dir does not exist or is not a directory: {annotation_dir}")

	output_dir = Path(args.output_dir) if args.output_dir is not None else annotation_dir / "target_lane"
	output_dir.mkdir(parents=True, exist_ok=True)

	file_map = _group_annotation_files(annotation_dir)
	if not file_map:
		raise FileNotFoundError(f"No annotation JSONL files found in {annotation_dir}")

	horizon = int(args.horizon)
	max_step = int(args.max_step)
	written_paths: list[Path] = []

	for (tfrecord_name, step), current_path in tqdm(sorted(file_map.items(), key=lambda item: (item[0][0], item[0][1])), desc="Processing annotation files"):
		future_step = min(int(step) + horizon, max_step)
		future_path = file_map.get((tfrecord_name, future_step))
		if future_path is None:
			continue

		output_path = output_dir / current_path.name
		if output_path.exists() and not bool(args.overwrite):
			written_paths.append(output_path)
			continue

		current_records = _records_by_scenario(_load_jsonl(current_path))
		future_records = _records_by_scenario(_load_jsonl(future_path))

		enriched_records: list[dict[str, Any]] = []
		for scenario_key, current_record in sorted(current_records.items(), key=lambda item: item[0][1]):
			future_record = future_records.get(scenario_key)
			target_lane_id = -1
			if future_record is not None:
				future_annotation = dict(future_record.get("annotation", {}))
				future_lane_context = dict(future_annotation.get("lane_context", {}))
				target_lane_id = int(future_lane_context.get("start_lane_id", -1))

			enriched_record = dict(current_record)
			annotation = dict(enriched_record.get("annotation", {}))
			annotation["target_lane_id"] = int(target_lane_id)
			enriched_record["annotation"] = annotation
			enriched_record["target_lane_id"] = int(target_lane_id)
			enriched_records.append(enriched_record)

		output_path.parent.mkdir(parents=True, exist_ok=True)
		_write_jsonl_with_index(output_path, enriched_records)
		written_paths.append(output_path)

	return written_paths


def main() -> None:
	args = _parse_args()
	written_paths = run(args)
	for path in written_paths:
		print(path)


if __name__ == "__main__":
	main()
