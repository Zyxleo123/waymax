from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.preprocess_cached import preprocess_cached_npz
from data.types import CacheBatch, PreprocessConfig
from data.utils import npz_to_jax, npz_to_torch, resolve_cache_paths


FUTURE_EGO_TRAJECTORY_STEPS = 25
FUTURE_EGO_TRAJECTORY_STRIDE = 2
DEFAULT_SUCCESS_ANSWERS = "success"
DEFAULT_FAILURE_PROMPT = (
	"Given the driving scenario, classify the rollout outcome. "
	"Answer with exactly one of: success, offroad, collision, goal not reached."
)


@dataclass(frozen=True)
class FailureDetectionLoaderConfig:
	success_cache_paths: tuple[str, ...]
	failure_cache_paths: tuple[str, ...]
	failure_result_root: str
	preprocess_cfg: PreprocessConfig = PreprocessConfig()
	batch_size: int = 16
	shuffle_seed: int = 42
	backend: str = "torch"
	include_inst_features: bool = False
	drop_last: bool = False
	success_anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80)
	failure_max_anchor_offset: int = 50
	goal_step_override: int | None = None
	prompt: str = DEFAULT_FAILURE_PROMPT
	answer_success: str = DEFAULT_SUCCESS_ANSWERS


def _load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _stack_numpy_rows(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
	if not rows:
		return {}
	keys = rows[0].keys()
	return {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}


def _cache_scenario_count(cache_npz: np.lib.npyio.NpzFile) -> int:
	if "num_scenarios" in cache_npz.files:
		num_scenarios = np.asarray(cache_npz["num_scenarios"])
		if num_scenarios.ndim == 0:
			return int(num_scenarios)
	if "scenario_indices" in cache_npz.files:
		scenario_indices = np.asarray(cache_npz["scenario_indices"])
		if scenario_indices.ndim >= 1:
			return int(scenario_indices.shape[0])
	return 1


def _has_explicit_scenario_axis(cache_npz: np.lib.npyio.NpzFile) -> bool:
	return "num_scenarios" in cache_npz.files or "scenario_indices" in cache_npz.files


def _rotate_xy(xy: np.ndarray, heading: float) -> np.ndarray:
	cos_h = float(np.cos(-heading))
	sin_h = float(np.sin(-heading))
	x = xy[..., 0]
	y = xy[..., 1]
	xr = x * cos_h - y * sin_h
	yr = x * sin_h + y * cos_h
	return np.stack([xr, yr], axis=-1)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
	return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _extract_ego_index(object_is_sdc: np.ndarray) -> int:
	mask = np.asarray(object_is_sdc).astype(bool)
	if mask.ndim != 1:
		raise ValueError(f"Expected a 1D ego mask, got {mask.shape}.")
	ego_count = int(np.sum(mask))
	if ego_count != 1:
		raise ValueError(f"Expected exactly one ego object, got {ego_count}.")
	return int(np.argmax(mask.astype(np.int32)))


def _build_future_ego_trajectory(
	cache_npz: np.lib.npyio.NpzFile,
	scenario_idx: int,
	anchor_step: int,
	cfg: PreprocessConfig,
	) -> tuple[np.ndarray, np.ndarray]:
	if _has_explicit_scenario_axis(cache_npz):
		object_is_sdc = np.asarray(cache_npz["object_metadata_is_sdc"][scenario_idx])
		log_x = np.asarray(cache_npz["log_trajectory_x"][scenario_idx]).astype(np.float32)
		log_y = np.asarray(cache_npz["log_trajectory_y"][scenario_idx]).astype(np.float32)
		log_yaw = np.asarray(cache_npz["log_trajectory_yaw"][scenario_idx]).astype(np.float32)
		log_vx = np.asarray(cache_npz["log_trajectory_vel_x"][scenario_idx]).astype(np.float32)
		log_vy = np.asarray(cache_npz["log_trajectory_vel_y"][scenario_idx]).astype(np.float32)
		log_valid = np.asarray(cache_npz["log_trajectory_valid"][scenario_idx]).astype(bool)
	else:
		object_is_sdc = np.asarray(cache_npz["object_metadata_is_sdc"])
		log_x = np.asarray(cache_npz["log_trajectory_x"]).astype(np.float32)
		log_y = np.asarray(cache_npz["log_trajectory_y"]).astype(np.float32)
		log_yaw = np.asarray(cache_npz["log_trajectory_yaw"]).astype(np.float32)
		log_vx = np.asarray(cache_npz["log_trajectory_vel_x"]).astype(np.float32)
		log_vy = np.asarray(cache_npz["log_trajectory_vel_y"]).astype(np.float32)
		log_valid = np.asarray(cache_npz["log_trajectory_valid"]).astype(bool)
	ego_idx = _extract_ego_index(object_is_sdc)
	if log_x.ndim == 2:
		log_x = log_x[ego_idx]
		log_y = log_y[ego_idx]
		log_yaw = log_yaw[ego_idx]
		log_vx = log_vx[ego_idx]
		log_vy = log_vy[ego_idx]
		log_valid = log_valid[ego_idx]
	elif log_x.ndim == 3:
		log_x = log_x[scenario_idx, ego_idx]
		log_y = log_y[scenario_idx, ego_idx]
		log_yaw = log_yaw[scenario_idx, ego_idx]
		log_vx = log_vx[scenario_idx, ego_idx]
		log_vy = log_vy[scenario_idx, ego_idx]
		log_valid = log_valid[scenario_idx, ego_idx]

	num_steps = int(log_x.shape[0])
	anchor_step = int(np.clip(anchor_step, 0, max(0, num_steps - 1)))
	anchor_x = float(log_x[anchor_step])
	anchor_y = float(log_y[anchor_step])
	anchor_yaw = float(log_yaw[anchor_step])

	future_traj = np.zeros((FUTURE_EGO_TRAJECTORY_STEPS, 5), dtype=np.float32)
	future_valid = np.zeros((FUTURE_EGO_TRAJECTORY_STEPS,), dtype=bool)
	for waypoint_idx in range(FUTURE_EGO_TRAJECTORY_STEPS):
		step = anchor_step + FUTURE_EGO_TRAJECTORY_STRIDE * (waypoint_idx + 1)
		if step >= num_steps or not bool(log_valid[step]):
			continue
		position_xy = np.asarray([log_x[step] - anchor_x, log_y[step] - anchor_y], dtype=np.float32)
		velocity_xy = np.asarray([log_vx[step], log_vy[step]], dtype=np.float32)
		future_traj[waypoint_idx, :2] = _rotate_xy(position_xy[None, :], anchor_yaw)[0] / float(cfg.ego_range)
		future_traj[waypoint_idx, 2:4] = _rotate_xy(velocity_xy[None, :], anchor_yaw)[0] / float(cfg.max_velocity)
		future_traj[waypoint_idx, 4] = float(_wrap_to_pi(np.asarray(log_yaw[step] - anchor_yaw, dtype=np.float32))) / float(np.pi)
		future_valid[waypoint_idx] = True

	return future_traj[None, ...], future_valid[None, ...]


def _pick_failure_reason(record: Mapping[str, Any], num_steps: int) -> tuple[str, int]:
	offroad_step = record.get("offroad_first_timestep")
	collision_step = record.get("collision_first_timestep")
	goal_reached = bool(record.get("goal_reached", False))
	reached_step = int(record.get("reached_timestep", -1))

	def _normalize_step(value: Any) -> int | None:
		if value is None:
			return None
		step = int(value)
		return step if step >= 0 else None

	offroad_step_i = _normalize_step(offroad_step)
	collision_step_i = _normalize_step(collision_step)
	if offroad_step_i is not None and collision_step_i is not None:
		if offroad_step_i <= collision_step_i:
			return "offroad", offroad_step_i
		return "collision", collision_step_i
	if offroad_step_i is not None or bool(record.get("offroad", False)):
		return "offroad", offroad_step_i if offroad_step_i is not None else max(0, num_steps - 1)
	if collision_step_i is not None or bool(record.get("collision", False)) or bool(record.get("overlap", False)):
		return "collision", collision_step_i if collision_step_i is not None else max(0, num_steps - 1)
	if goal_reached:
		return "success", max(0, reached_step)
	if reached_step >= 0:
		return "goal not reached", reached_step
	return "goal not reached", max(0, num_steps - 1)


def _sample_failure_anchor_step(failure_step: int, num_steps: int, rng: random.Random, max_offset: int) -> int:
	failure_step = int(np.clip(failure_step, 0, max(0, num_steps - 1)))
	if failure_step <= 0:
		return 0
	start = max(0, failure_step - int(max_offset))
	end = max(0, failure_step - 1)
	if end < start:
		return start
	return rng.randint(start, end)


class FailureDetectionCacheDataset(IterableDataset[CacheBatch]):
	def __init__(self, cfg: FailureDetectionLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg
		if self.cfg.backend == "torch":
			self.to_tensor = npz_to_torch
		elif self.cfg.backend == "jax":
			self.to_tensor = npz_to_jax
		else:
			raise ValueError(f"Unsupported backend: {self.cfg.backend}")
		self._failure_result_index = self._build_failure_result_index(Path(self.cfg.failure_result_root))

	@staticmethod
	def _build_failure_result_index(result_root: Path) -> dict[str, Path]:
		if not result_root.exists():
			raise FileNotFoundError(f"failure_result_root does not exist: {result_root}")
		index: dict[str, Path] = {}
		for path in sorted(result_root.rglob("*.json")):
			if path.name == "summary.json":
				continue
			index.setdefault(path.stem, path)
		return index

	def _maybe_shard_paths(self, cache_paths: list[str]) -> list[str]:
		worker_info = get_worker_info()
		paths = list(cache_paths)
		if self.cfg.shuffle_seed:
			random.Random(int(self.cfg.shuffle_seed)).shuffle(paths)
		if worker_info is not None:
			paths = paths[worker_info.id :: worker_info.num_workers]
		return paths

	def _result_path_for_cache(self, cache_file: Path, scenario_idx: int) -> Path:
		stem = cache_file.name.removesuffix(".sim_state_cache.npz")
		result_path = self._failure_result_index.get(stem)
		if result_path is None:
			candidate = cache_file.parent.parent / f"{stem}.json"
			if candidate.exists():
				return candidate
			raise FileNotFoundError(f"Could not find result JSON for {cache_file}")
		return result_path

	def _select_row(self, array: np.ndarray, row_idx: int) -> np.ndarray:
		return np.asarray(array)[row_idx:row_idx + 1]

	def _build_example(
		self,
		preprocessed: dict[str, dict[str, np.ndarray]],
		cache_npz: np.lib.npyio.NpzFile,
		row_idx: int,
		scenario_idx: int,
		anchor_step: int,
		answer: str,
		key: str,
		critical_object_id: int = -1,
		critical_road_segment_id: int = -1,
	) -> dict[str, Any]:
		features = {name: self._select_row(value, row_idx) for name, value in preprocessed["features"].items()}
		aux = {name: self._select_row(value, row_idx) for name, value in preprocessed["aux"].items()}

		if _has_explicit_scenario_axis(cache_npz):
			object_is_sdc = np.asarray(cache_npz["object_metadata_is_sdc"][scenario_idx])
		else:
			object_is_sdc = np.asarray(cache_npz["object_metadata_is_sdc"])
		num_objects = int(object_is_sdc.shape[0])
		object_ids = np.arange(num_objects, dtype=np.int32)[None, :]
		features["object_ids"] = object_ids
		features["other_ids"] = object_ids

		future_traj, future_valid = _build_future_ego_trajectory(
			cache_npz,
			scenario_idx,
			anchor_step,
			self.cfg.preprocess_cfg,
		)
		features["ego_trajectory"] = future_traj
		features["ego_trajectory_valid"] = future_valid
		features["anchor_step"] = np.asarray([[int(anchor_step)]], dtype=np.int32)
		aux["critical_object_ids"] = np.asarray([int(critical_object_id)], dtype=np.int32)
		aux["critical_road_segment_ids"] = np.asarray([int(critical_road_segment_id)], dtype=np.int32)

		return {
			"features": features,
			"aux": aux,
			"prompt": self.cfg.prompt,
			"answer": answer,
			"key": key,
		}

	def _iter_success_examples(self, cache_paths: list[str], rng: random.Random) -> Iterator[dict[str, Any]]:
		for cache_path in cache_paths:
			cache_file = Path(cache_path)
			with np.load(cache_file, allow_pickle=False) as cache_npz:
				goal_step_override = self.cfg.goal_step_override
				if goal_step_override is None:
					goal_step_override = int(np.asarray(cache_npz["log_trajectory_x"]).shape[-1] - 1)
				scenario_count = _cache_scenario_count(cache_npz)
				scenario_order = list(range(scenario_count))
				rng.shuffle(scenario_order)
				anchor_choices = self.cfg.success_anchor_steps or (0,)
				anchor_groups: dict[int, list[int]] = defaultdict(list)
				for scenario_idx in scenario_order:
					anchor_step = int(rng.choice(anchor_choices))
					anchor_groups[anchor_step].append(int(scenario_idx))

				for anchor_step, scenario_indices in anchor_groups.items():
					preprocessed = preprocess_cached_npz(
						cache_file,
						cfg=self.cfg.preprocess_cfg,
						scenario_indices=scenario_indices,
						anchor_step_override=int(anchor_step),
						goal_step_override=goal_step_override,
						include_inst_features=self.cfg.include_inst_features,
					)
					for row_idx, scenario_idx in enumerate(scenario_indices):
						yield self._build_example(
							preprocessed,
							cache_npz,
							row_idx,
							scenario_idx,
							int(anchor_step),
							self.cfg.answer_success,
							"success",
							-1,
							-1,
						)

	def _iter_failure_examples(self, cache_paths: list[str], rng: random.Random) -> Iterator[dict[str, Any]]:
		for cache_path in cache_paths:
			cache_file = Path(cache_path)
			with np.load(cache_file, allow_pickle=False) as cache_npz:
				goal_step_override = self.cfg.goal_step_override
				if goal_step_override is None:
					goal_step_override = int(np.asarray(cache_npz["log_trajectory_x"]).shape[-1] - 1)
				scenario_count = _cache_scenario_count(cache_npz)
				scenario_order = list(range(scenario_count))
				rng.shuffle(scenario_order)
				anchor_groups: dict[int, list[int]] = defaultdict(list)
				for scenario_idx in scenario_order:
					result_path = self._result_path_for_cache(cache_file, scenario_idx)
					record = _load_json(result_path)
					reason, failure_step = _pick_failure_reason(record, int(np.asarray(cache_npz["log_trajectory_x"]).shape[-1]))
					critical_object_id = -1
					critical_road_segment_id = -1
					if reason == "collision":
						critical_object_id = int(record.get("collision_vehicle_id", -1))
					elif reason == "offroad":
						critical_road_segment_id = int(record.get("offroad_road_edge_segment_id", -1))
					anchor_step = _sample_failure_anchor_step(
						failure_step,
						int(np.asarray(cache_npz["log_trajectory_x"]).shape[-1]),
						rng,
						self.cfg.failure_max_anchor_offset,
					)
					anchor_groups[anchor_step].append((int(scenario_idx), reason, critical_object_id, critical_road_segment_id))

				for anchor_step, scenario_items in anchor_groups.items():
					scenario_indices = [item[0] for item in scenario_items]
					preprocessed = preprocess_cached_npz(
						cache_file,
						cfg=self.cfg.preprocess_cfg,
						scenario_indices=scenario_indices,
						anchor_step_override=int(anchor_step),
						goal_step_override=goal_step_override,
						include_inst_features=self.cfg.include_inst_features,
					)
					for row_idx, (scenario_idx, reason, critical_object_id, critical_road_segment_id) in enumerate(scenario_items):
						yield self._build_example(
							preprocessed,
							cache_npz,
							row_idx,
							scenario_idx,
							int(anchor_step),
							reason,
							reason.replace(" ", "_"),
							critical_object_id,
							critical_road_segment_id,
						)

	def __iter__(self) -> Iterator[CacheBatch]:
		success_paths = self._maybe_shard_paths(list(self.cfg.success_cache_paths))
		failure_paths = self._maybe_shard_paths(list(self.cfg.failure_cache_paths))

		worker_info = get_worker_info()
		iterator_seed = int(self.cfg.shuffle_seed)
		if worker_info is not None:
			iterator_seed += int(worker_info.id) * 10007
		rng = random.Random(iterator_seed)

		success_iter = iter(self._iter_success_examples(success_paths, rng))
		failure_iter = iter(self._iter_failure_examples(failure_paths, rng))
		extra_success = True
		has_odd_batch = bool(self.cfg.batch_size % 2)

		while True:
			success_count = self.cfg.batch_size // 2
			failure_count = self.cfg.batch_size - success_count
			if has_odd_batch:
				if extra_success:
					success_count += 1
				else:
					failure_count += 1
				extra_success = not extra_success

			batch_examples: list[dict[str, Any]] = []
			for _ in range(success_count):
				try:
					batch_examples.append(next(success_iter))
				except StopIteration:
					return
			for _ in range(failure_count):
				try:
					batch_examples.append(next(failure_iter))
				except StopIteration:
					return

			rng.shuffle(batch_examples)
			feature_keys = batch_examples[0]["features"].keys()
			aux_keys = batch_examples[0]["aux"].keys()
			batch_features = {
				key: self.to_tensor(np.concatenate([example["features"][key] for example in batch_examples], axis=0))
				for key in feature_keys
			}
			batch_aux = {
				key: self.to_tensor(np.concatenate([example["aux"][key] for example in batch_examples], axis=0))
				for key in aux_keys
			}
			prompts = [str(example["prompt"]) for example in batch_examples]
			answers = [str(example["answer"]) for example in batch_examples]
			keys = [str(example["key"]) for example in batch_examples]

			yield CacheBatch(
				features=batch_features,
				aux=batch_aux,
				prompts=prompts,
				answers=answers,
				keys=keys,
			)


def build_dataloader(
	success_cache_dir: str,
	failure_cache_dir: str,
	preprocess_cfg: PreprocessConfig,
	*,
	success_cache_paths: tuple[str, ...] | None = None,
	failure_cache_paths: tuple[str, ...] | None = None,
	failure_result_root: str | None = None,
	success_file_indices: list[int] | None = None,
	failure_file_indices: list[int] | None = None,
	batch_size: int = 16,
	shuffle_seed: int = 42,
	num_workers: int = 0,
	pin_memory: bool = False,
	backend: str = "torch",
	include_inst_features: bool = False,
	drop_last: bool = False,
	success_anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80),
	failure_max_anchor_offset: int = 50,
	goal_step_override: int | None = None,
	prompt: str = DEFAULT_FAILURE_PROMPT,
	answer_success: str = DEFAULT_SUCCESS_ANSWERS,
) -> DataLoader[CacheBatch]:
	if success_cache_paths is None:
		success_paths = resolve_cache_paths(success_cache_dir, success_file_indices)
	else:
		success_paths = tuple(success_cache_paths)
	if failure_cache_paths is None:
		failure_paths = resolve_cache_paths(failure_cache_dir, failure_file_indices)
	else:
		failure_paths = tuple(failure_cache_paths)
	if failure_result_root is None:
		failure_result_root = str(Path(failure_cache_dir).expanduser().resolve().parent)
	cfg = FailureDetectionLoaderConfig(
		success_cache_paths=success_paths,
		failure_cache_paths=failure_paths,
		failure_result_root=failure_result_root,
		preprocess_cfg=preprocess_cfg,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		backend=backend,
		include_inst_features=include_inst_features,
		drop_last=drop_last,
		success_anchor_steps=success_anchor_steps,
		failure_max_anchor_offset=failure_max_anchor_offset,
		goal_step_override=goal_step_override,
		prompt=prompt,
		answer_success=answer_success,
	)
	dataset = FailureDetectionCacheDataset(cfg)
	return DataLoader(
		dataset,
		batch_size=None,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


__all__ = [
	"FailureDetectionCacheDataset",
	"FailureDetectionLoaderConfig",
	"build_dataloader",
]
