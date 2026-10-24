from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.failure_detection_loader import (
	FAILURE_REASON_COLLISION,
	FAILURE_REASON_GOAL_NOT_REACHED,
	FAILURE_REASON_NONE,
	FAILURE_REASON_OFFROAD,
	_cache_scenario_count,
)
from data.preprocess_cached import preprocess_cached_npz
from data.types import CacheBatch, PreprocessConfig
from data.utils import npz_to_jax, npz_to_torch


DEFAULT_ANCHOR_STEPS = (0, 10, 20, 30, 40)


@dataclass(frozen=True)
class FMGoldLoaderConfig:
	cache_dir: str
	preprocess_cfg: PreprocessConfig
	cache_file_indices: list[int] | None = None
	batch_size: int = 8
	shuffle_seed: int = 0
	anchor_steps: tuple[int, ...] = DEFAULT_ANCHOR_STEPS
	backend: str = "jax"
	drop_last: bool = True


def _resolve_gold_cache_dir(cache_dir: str) -> Path:
	directory = Path(cache_dir)
	nested = directory / "success_for_failure_cache"
	if nested.is_dir():
		return nested
	return directory


def _is_success_for_failure_cache(path: Path) -> bool:
	name = path.name
	if name == "summary.json" or not name.endswith(".npz"):
		return False
	return name.startswith("success_for_failure_shard") or name.endswith(".sim_state_cache.npz")


def _resolve_gold_cache_paths(cache_dir: str, file_indices: list[int] | None) -> tuple[str, ...]:
	directory = _resolve_gold_cache_dir(cache_dir)
	if not directory.exists():
		raise FileNotFoundError(f"success_for_failure cache dir does not exist: {directory}")

	paths = [path for path in sorted(directory.rglob("*.npz")) if _is_success_for_failure_cache(path)]
	if file_indices is not None:
		wanted = set(int(index) for index in file_indices)
		filtered: list[Path] = []
		for path in paths:
			stem = path.name.removesuffix(".sim_state_cache.npz")
			digits = "".join(ch for ch in stem if ch.isdigit())
			if digits and int(digits) in wanted:
				filtered.append(path)
		paths = filtered

	if not paths:
		raise FileNotFoundError(
			"No success_for_failure cache npz files found in "
			f"{directory}. Run scripts/make_success_cache_for_failures.py first."
		)
	return tuple(str(path) for path in paths)


def _as_row_int(value: Any, row_idx: int, default: int = -1) -> int:
	if value is None:
		return int(default)
	array = np.asarray(value)
	if array.size == 0:
		return int(default)
	if array.ndim == 0:
		return int(array.item())
	flat = array.reshape(-1)
	if row_idx < flat.size:
		return int(flat[row_idx])
	if flat.size == 1:
		return int(flat[0])
	return int(default)


def _usable_gold_labels(reason_id: int, object_id: int, road_id: int) -> bool:
	if reason_id == FAILURE_REASON_COLLISION:
		return object_id >= 0
	if reason_id == FAILURE_REASON_OFFROAD:
		return road_id >= 0
	if reason_id == FAILURE_REASON_GOAL_NOT_REACHED:
		return True
	return False


def _sample_anchor_step(rng: random.Random, num_steps: int, anchor_steps: Sequence[int]) -> int:
	max_step = max(0, int(num_steps) - 1)
	candidates = [int(step) for step in anchor_steps if 0 <= int(step) <= max_step]
	if not candidates:
		return min(max_step, int(anchor_steps[0]) if anchor_steps else 0)
	return int(rng.choice(candidates))


class FMGoldCacheDataset(IterableDataset[CacheBatch]):
	def __init__(self, cfg: FMGoldLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg
		if self.cfg.backend == "torch":
			self.to_tensor = npz_to_torch
		elif self.cfg.backend == "jax":
			self.to_tensor = npz_to_jax
		else:
			raise ValueError(f"Unsupported backend: {self.cfg.backend}")
		if not self.cfg.anchor_steps:
			raise ValueError("anchor_steps cannot be empty")
		self.cache_paths = _resolve_gold_cache_paths(self.cfg.cache_dir, self.cfg.cache_file_indices)
		print(
			f"[fm_gold_loader] {len(self.cache_paths)} success_for_failure cache files, "
			f"anchor_steps={tuple(self.cfg.anchor_steps)}"
		)

	def _paths_for_worker(self) -> tuple[str, ...]:
		paths = list(self.cache_paths)
		worker_info = get_worker_info()
		if self.cfg.shuffle_seed:
			random.Random(int(self.cfg.shuffle_seed)).shuffle(paths)
		if worker_info is not None:
			paths = paths[worker_info.id :: worker_info.num_workers]
		return tuple(paths)

	def _iter_labeled_rows(self, rng: random.Random) -> Iterator[dict[str, Any]]:
		anchor_steps = tuple(int(step) for step in self.cfg.anchor_steps)
		for cache_path in self._paths_for_worker():
			cache_file = Path(cache_path)
			with np.load(cache_file, allow_pickle=False) as cache_npz:
				num_steps = int(np.asarray(cache_npz["log_trajectory_x"]).shape[-1])
				scenario_count = _cache_scenario_count(cache_npz)
				row_order = list(range(scenario_count))
				rng.shuffle(row_order)
				grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
				for row_idx in row_order:
					reason_id = _as_row_int(cache_npz.get("failure_reason"), row_idx, FAILURE_REASON_NONE)
					object_id = _as_row_int(
						cache_npz.get("critical_object_ids", cache_npz.get("collision_vehicle_id")),
						row_idx,
						-1,
					)
					road_id = _as_row_int(
						cache_npz.get("critical_road_segment_ids", cache_npz.get("offroad_road_edge_segment_id")),
						row_idx,
						-1,
					)
					if not _usable_gold_labels(reason_id, object_id, road_id):
						continue
					anchor_step = _sample_anchor_step(rng, num_steps, anchor_steps)
					grouped[int(anchor_step)].append(
						{
							"row_idx": int(row_idx),
							"reason_id": int(reason_id),
							"object_id": object_id if reason_id == FAILURE_REASON_COLLISION else -1,
							"road_id": road_id if reason_id == FAILURE_REASON_OFFROAD else -1,
						}
					)

			goal_step = max(0, num_steps - 1)
			for anchor_step, items in grouped.items():
				preprocessed = preprocess_cached_npz(
					cache_file,
					cfg=self.cfg.preprocess_cfg,
					scenario_indices=[item["row_idx"] for item in items],
					anchor_step_override=int(anchor_step),
					goal_step_override=int(goal_step),
					include_inst_features=False,
				)
				features = preprocessed["features"]
				aux = preprocessed["aux"]
				num_objects = int(np.asarray(features["other_states"]).shape[1])
				for batch_row, item in enumerate(items):
					example_features = {
						name: np.asarray(value)[batch_row : batch_row + 1]
						for name, value in features.items()
					}
					example_aux = {
						name: np.asarray(value)[batch_row : batch_row + 1]
						for name, value in aux.items()
					}
					example_features["failure_reason"] = np.asarray([item["reason_id"]], dtype=np.int32)
					example_features["critical_object_ids"] = np.asarray([item["object_id"]], dtype=np.int32)
					example_features["critical_road_segment_ids"] = np.asarray([item["road_id"]], dtype=np.int32)
					example_features["object_ids"] = np.arange(num_objects, dtype=np.int32)[None, :]
					example_features["other_ids"] = example_features["object_ids"]
					example_aux["is_failure"] = np.asarray([True])
					example_aux["anchor_step"] = np.asarray([int(anchor_step)], dtype=np.int32)
					yield {
						"features": example_features,
						"aux": example_aux,
						"key": f"{cache_file.stem}:{item['row_idx']}:{anchor_step}",
					}

	def __iter__(self) -> Iterator[CacheBatch]:
		worker_info = get_worker_info()
		seed = int(self.cfg.shuffle_seed)
		if worker_info is not None:
			seed += int(worker_info.id) * 10007
		rng = random.Random(seed)

		pending_features: list[dict[str, np.ndarray]] = []
		pending_aux: list[dict[str, np.ndarray]] = []
		pending_keys: list[str] = []

		def _flush() -> CacheBatch:
			feature_keys = pending_features[0].keys()
			aux_keys = pending_aux[0].keys()
			batch = CacheBatch(
				features={
					key: np.concatenate([row[key] for row in pending_features], axis=0)
					for key in feature_keys
				},
				aux={
					key: np.concatenate([row[key] for row in pending_aux], axis=0)
					for key in aux_keys
				},
				keys=list(pending_keys),
			)
			return CacheBatch(
				features={key: self.to_tensor(value) for key, value in batch.features.items()},
				aux={key: self.to_tensor(value) for key, value in batch.aux.items()},
				keys=batch.keys,
			)

		first_pass = True
		while True:
			produced = 0
			for example in self._iter_labeled_rows(rng):
				produced += 1
				pending_features.append(example["features"])
				pending_aux.append(example["aux"])
				pending_keys.append(example["key"])
				if len(pending_keys) == int(self.cfg.batch_size):
					yield _flush()
					pending_features = []
					pending_aux = []
					pending_keys = []
			if pending_keys and not self.cfg.drop_last:
				yield _flush()
				pending_features = []
				pending_aux = []
				pending_keys = []
			if produced == 0:
				raise RuntimeError(
					"No gold-id examples were found in "
					f"{self.cfg.cache_dir}. Check success_for_failure_cache shards."
				)
			if first_pass:
				print(f"[fm_gold_loader] first pass produced {produced} labeled examples")
				first_pass = False


def build_dataloader(
	cache_dir: str,
	preprocess_cfg: PreprocessConfig,
	*,
	cache_file_indices: list[int] | None = None,
	batch_size: int = 8,
	shuffle_seed: int = 0,
	anchor_steps: Sequence[int] = DEFAULT_ANCHOR_STEPS,
	num_workers: int = 0,
	pin_memory: bool = False,
	backend: str = "jax",
	drop_last: bool = True,
) -> DataLoader[CacheBatch]:
	cfg = FMGoldLoaderConfig(
		cache_dir=cache_dir,
		preprocess_cfg=preprocess_cfg,
		cache_file_indices=cache_file_indices,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		anchor_steps=tuple(int(step) for step in anchor_steps),
		backend=backend,
		drop_last=drop_last,
	)
	dataset = FMGoldCacheDataset(cfg)
	return DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=pin_memory)
