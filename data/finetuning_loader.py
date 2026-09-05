from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.preprocess_cached import preprocess_cached_npz
from data.types import CacheBatch, PreprocessConfig
from data.utils import npz_to_jax, npz_to_torch, resolve_cache_paths


_SOURCE_CACHE_PATTERN = re.compile(r"training_tfexample\.tfrecord-(\d+)-of-01000\.sim_state_cache\.npz$")
_LEGACY_FAILURE_CACHE_PATTERN = re.compile(
	r"training_tfexample\.tfrecord-(\d+)-of-01000(?:\.scenario_(\d+))?\.sim_state_cache\.npz$"
)


@dataclass(frozen=True)
class FinetuningLoaderConfig:
	source_cache_dir: str
	failure_cache_dir: str
	preprocess_cfg: PreprocessConfig
	source_file_indices: list[int] | None = None
	failure_file_indices: list[int] | None = None
	batch_size: int = 1
	shuffle_seed: int = 42
	failure_sampling_prob: float = 0.5
	backend: str = "torch"
	include_inst_features: bool = False
	drop_last: bool = False


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
	if not rows:
		raise ValueError("No rows were produced for a batch.")
	if len(rows) == 1:
		return rows[0]
	return np.concatenate(rows, axis=0)


def _parse_tfrecord_index(path: Path) -> int:
	match = _SOURCE_CACHE_PATTERN.match(path.name)
	if match is None:
		match = _LEGACY_FAILURE_CACHE_PATTERN.match(path.name)
	if match is None:
		raise ValueError(f"Could not parse tfrecord index from cache file: {path}")
	return int(match.group(1))


def _resolve_failure_cache_paths(failure_cache_dir: str, file_indices: list[int] | None) -> tuple[str, ...]:
	directory = Path(failure_cache_dir)
	if not directory.exists():
		raise FileNotFoundError(f"failure_cache_dir does not exist: {failure_cache_dir}")
	if not directory.is_dir():
		raise NotADirectoryError(f"failure_cache_dir is not a directory: {failure_cache_dir}")

	paths: list[Path] = []
	for path in sorted(directory.rglob("*.npz")):
		if file_indices is None:
			paths.append(path)
			continue

		include_path = False
		match = _LEGACY_FAILURE_CACHE_PATTERN.match(path.name)
		if match is not None and int(match.group(1)) in file_indices:
			include_path = True
		else:
			match = _SOURCE_CACHE_PATTERN.match(path.name)
			if match is not None and int(match.group(1)) in file_indices:
				include_path = True
			else:
				with np.load(path, allow_pickle=False) as npz_data:
					if "tfrecord_index" in npz_data.files:
						indices = np.asarray(npz_data["tfrecord_index"]).reshape(-1)
						include_path = any(int(value) in file_indices for value in indices.tolist())

		if include_path:
			paths.append(path)

	if not paths:
		raise FileNotFoundError(
			f"No failure npz files found in {failure_cache_dir}"
			+ (f" for file_indices={file_indices}" if file_indices else "")
		)
	return tuple(str(path) for path in paths)


def _repeat_value(value: np.ndarray | int | str | bool, size: int) -> np.ndarray:
	array = np.asarray(value)
	if array.ndim == 0:
		return np.repeat(array.reshape(1), size, axis=0)
	if array.shape[0] == size:
		return array
	if array.size == 1:
		return np.repeat(array.reshape(1), size, axis=0)
	raise ValueError(f"Could not broadcast metadata value with shape {array.shape} to size {size}")


def _infer_row_count(tree: dict[str, np.ndarray]) -> int:
	for value in tree.values():
		array = np.asarray(value)
		if array.ndim >= 1:
			return int(array.shape[0])
	raise ValueError("Could not infer row count from cached shard.")


def _slice_tree(tree: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
	return {key: np.asarray(value)[start:end] for key, value in tree.items()}


def _concat_tree_chunks(chunks: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
	return {key: _stack_rows(values) for key, values in chunks.items()}


def _extract_row_metadata(cache_path: Path, metadata: dict[str, np.ndarray], row_count: int) -> dict[str, np.ndarray]:
	row_metadata: dict[str, np.ndarray] = {}

	if "tfrecord_index" in metadata:
		row_metadata["tfrecord_index"] = _repeat_value(metadata["tfrecord_index"], row_count).astype(np.int32, copy=False)
	else:
		row_metadata["tfrecord_index"] = np.repeat(np.asarray(_parse_tfrecord_index(cache_path), dtype=np.int32), row_count)

	if "scenario_indices" in metadata:
		row_metadata["scenario_index"] = _repeat_value(metadata["scenario_indices"], row_count).astype(np.int32, copy=False)
	else:
		row_metadata["scenario_index"] = np.arange(row_count, dtype=np.int32)

	return row_metadata


def _build_sample_keys(tfrecord_index: np.ndarray, scenario_index: np.ndarray) -> list[str]:
	return [f"{int(tf_index):05d}_{int(scenario_idx)}" for tf_index, scenario_idx in zip(tfrecord_index.tolist(), scenario_index.tolist(), strict=True)]


def _materialize_cache_batch(
	features: dict[str, list[np.ndarray]],
	aux: dict[str, list[np.ndarray]],
	keys: list[str],
) -> CacheBatch:
	return CacheBatch(features=_concat_tree_chunks(features), aux=_concat_tree_chunks(aux), keys=keys)


def _convert_cache_batch(batch: CacheBatch, converter) -> CacheBatch:
	return CacheBatch(
		features={key: converter(value) for key, value in batch.features.items()},
		aux={key: converter(value) for key, value in batch.aux.items()},
		keys=batch.keys,
	)


def _merge_cache_batches(left: CacheBatch, right: CacheBatch, rng: np.random.Generator) -> CacheBatch:
	feature_keys = set(left.features) | set(right.features)
	aux_keys = set(left.aux) | set(right.aux)
	if feature_keys != set(left.features) or feature_keys != set(right.features):
		raise ValueError("Source and failure batches must expose the same feature keys.")
	if aux_keys != set(left.aux) or aux_keys != set(right.aux):
		raise ValueError("Source and failure batches must expose the same auxiliary keys.")

	features = {
		key: np.concatenate((np.asarray(left.features[key]), np.asarray(right.features[key])), axis=0)
		for key in left.features
	}
	aux = {key: np.concatenate((np.asarray(left.aux[key]), np.asarray(right.aux[key])), axis=0) for key in left.aux}
	keys = list(left.keys or []) + list(right.keys or [])

	row_count = len(keys)
	if row_count == 0:
		raise ValueError("Cannot merge empty batches.")
	permutation = rng.permutation(row_count)
	features = {key: value[permutation] for key, value in features.items()}
	aux = {key: value[permutation] for key, value in aux.items()}
	keys = [keys[index] for index in permutation.tolist()]
	return CacheBatch(features=features, aux=aux, keys=keys)


class _ShardBatchStream(IterableDataset[CacheBatch]):
	def __init__(
		self,
		cache_paths: tuple[str, ...],
		cfg: FinetuningLoaderConfig,
		batch_size: int,
		*,
		is_failure: bool,
	) -> None:
		super().__init__()
		self.cache_paths = cache_paths
		self.cfg = cfg
		self.batch_size = batch_size
		self.is_failure = is_failure
		if self.cfg.backend == "torch":
			self.to_tensor = npz_to_torch
		elif self.cfg.backend == "jax":
			self.to_tensor = npz_to_jax
		else:
			raise ValueError(f"Unsupported backend: {self.cfg.backend}")

	def _paths_for_worker(self) -> tuple[str, ...]:
		worker_info = get_worker_info()
		if worker_info is None:
			return self.cache_paths
		return self.cache_paths[worker_info.id :: worker_info.num_workers]

	def __iter__(self) -> Iterator[CacheBatch]:
		paths = self._paths_for_worker()
		if not paths:
			return

		worker_info = get_worker_info()
		seed = int(self.cfg.shuffle_seed) + (0 if worker_info is None else int(worker_info.id) * 10007)
		seed += 1 if self.is_failure else 0
		path_rng = random.Random(seed)
		merge_rng = np.random.default_rng(seed)

		pending_feature_chunks: dict[str, list[np.ndarray]] = {}
		pending_aux_chunks: dict[str, list[np.ndarray]] = {}
		pending_keys: list[str] = []
		pending_count = 0

		while True:
			ordered_paths = list(paths)
			path_rng.shuffle(ordered_paths)
			for cache_path_str in ordered_paths:
				cache_path = Path(cache_path_str)
				preprocessed = preprocess_cached_npz(
					cache_path,
					cfg=self.cfg.preprocess_cfg,
					include_inst_features=self.cfg.include_inst_features,
				)
				features = {key: np.asarray(value) for key, value in preprocessed["features"].items()}
				aux = {key: np.asarray(value) for key, value in preprocessed["aux"].items()}
				row_count = _infer_row_count(features)
				row_metadata = _extract_row_metadata(cache_path, {key: np.asarray(value) for key, value in preprocessed["metadata"].items()}, row_count)
				row_metadata["is_failure"] = np.repeat(np.asarray(self.is_failure, dtype=np.bool_), row_count)

				cursor = 0
				while cursor < row_count:
					need = self.batch_size - pending_count
					take = min(need, row_count - cursor)
					stop = cursor + take

					for key, value in _slice_tree(features, cursor, stop).items():
						pending_feature_chunks.setdefault(key, []).append(value)
					for key, value in _slice_tree(aux, cursor, stop).items():
						pending_aux_chunks.setdefault(key, []).append(value)
					for key, value in _slice_tree(row_metadata, cursor, stop).items():
						pending_aux_chunks.setdefault(key, []).append(value)
					pending_keys.extend(_build_sample_keys(row_metadata["tfrecord_index"][cursor:stop], row_metadata["scenario_index"][cursor:stop]))
					pending_count += take
					cursor = stop

					if pending_count == self.batch_size:
						batch = _materialize_cache_batch(pending_feature_chunks, pending_aux_chunks, pending_keys)
						yield batch
						pending_feature_chunks = {}
						pending_aux_chunks = {}
						pending_keys = []
						pending_count = 0

			if self.cfg.drop_last:
				pending_feature_chunks = {}
				pending_aux_chunks = {}
				pending_keys = []
				pending_count = 0


def _split_batch_size(batch_size: int, failure_sampling_prob: float) -> tuple[int, int]:
	if batch_size <= 0:
		raise ValueError(f"batch_size must be positive, got {batch_size}")
	failure_batch_size = int(batch_size * float(failure_sampling_prob) + 0.5)
	failure_batch_size = max(0, min(batch_size, failure_batch_size))
	if 0.0 < failure_sampling_prob < 1.0 and batch_size > 1:
		failure_batch_size = max(1, min(batch_size - 1, failure_batch_size))
	source_batch_size = batch_size - failure_batch_size
	return source_batch_size, failure_batch_size


class FinetuningCacheDataset(IterableDataset[CacheBatch]):
	def __init__(self, cfg: FinetuningLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg
		if self.cfg.backend == "torch":
			self.to_tensor = npz_to_torch
		elif self.cfg.backend == "jax":
			self.to_tensor = npz_to_jax
		else:
			raise ValueError(f"Unsupported backend: {self.cfg.backend}")

		source_paths = resolve_cache_paths(self.cfg.source_cache_dir, self.cfg.source_file_indices)
		self._source_cache_paths = tuple(str(path) for path in source_paths)
		self._failure_cache_paths = _resolve_failure_cache_paths(self.cfg.failure_cache_dir, self.cfg.failure_file_indices)

		if not self._source_cache_paths and not self._failure_cache_paths:
			raise RuntimeError(
				f"No finetuning cache shards found. source_cache_dir={self.cfg.source_cache_dir}, "
				f"failure_cache_dir={self.cfg.failure_cache_dir}"
			)

		self._source_batch_size, self._failure_batch_size = _split_batch_size(
			self.cfg.batch_size,
			self.cfg.failure_sampling_prob,
		)

		if self._source_cache_paths and self._failure_cache_paths and self._source_batch_size == 0 and self._failure_batch_size == 0:
			raise ValueError("Batch split produced empty source and failure batches.")

	def __iter__(self) -> Iterator[CacheBatch]:
		worker_info = get_worker_info()
		seed = int(self.cfg.shuffle_seed) + (0 if worker_info is None else int(worker_info.id) * 10007)
		mix_rng = np.random.default_rng(seed)

		has_source = bool(self._source_cache_paths) and self._source_batch_size > 0
		has_failure = bool(self._failure_cache_paths) and self._failure_batch_size > 0

		source_stream = iter(
			_ShardBatchStream(
				self._source_cache_paths,
				self.cfg,
				self._source_batch_size if has_source else self.cfg.batch_size,
				is_failure=False,
			)
		) if self._source_cache_paths else None
		failure_stream = iter(
			_ShardBatchStream(
				self._failure_cache_paths,
				self.cfg,
				self._failure_batch_size if has_failure else self.cfg.batch_size,
				is_failure=True,
			)
		) if self._failure_cache_paths else None

		if not has_source and not has_failure:
			return

		if has_source and has_failure:
			while True:
				source_batch = next(source_stream)
				failure_batch = next(failure_stream)
				merged = _merge_cache_batches(source_batch, failure_batch, mix_rng)
				yield _convert_cache_batch(merged, self.to_tensor)

		elif has_source:
			while True:
				yield _convert_cache_batch(next(source_stream), self.to_tensor)

		else:
			while True:
				yield _convert_cache_batch(next(failure_stream), self.to_tensor)


def build_dataloader(
	source_cache_dir: str,
	failure_cache_dir: str,
	preprocess_cfg: PreprocessConfig,
	*,
	source_file_indices: list[int] | None = None,
	failure_file_indices: list[int] | None = None,
	batch_size: int = 1,
	shuffle_seed: int = 42,
	failure_sampling_prob: float = 0.5,
	num_workers: int = 0,
	pin_memory: bool = False,
	backend: str = "torch",
	include_inst_features: bool = False,
	drop_last: bool = False,
) -> DataLoader[CacheBatch]:
	cfg = FinetuningLoaderConfig(
		source_cache_dir=source_cache_dir,
		failure_cache_dir=failure_cache_dir,
		preprocess_cfg=preprocess_cfg,
		source_file_indices=source_file_indices,
		failure_file_indices=failure_file_indices,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		failure_sampling_prob=failure_sampling_prob,
		backend=backend,
		include_inst_features=include_inst_features,
		drop_last=drop_last,
	)
	dataset = FinetuningCacheDataset(cfg)
	return DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=pin_memory)