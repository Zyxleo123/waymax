from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.preprocess_cached import preprocess_cached_npz
from data.types import CacheBatch, PreprocessConfig
from data.utils import npz_to_jax, npz_to_torch, resolve_cache_paths


_SIM_STATE_CACHE_PATTERN = re.compile(
    r"training_tfexample\.tfrecord-(\d+)-of-01000\.sim_state_cache\.npz$"
)


@dataclass
class DiffusionCacheLoaderConfig:
    cache_paths: tuple[str, ...]
    preprocess_cfg: PreprocessConfig
    batch_size: int = 1
    shuffle_seed: int = 42
    anchor_steps: tuple[int, ...] | None = None
    backend: str = "torch"
    goal_step_override: int | None = None
    include_inst_features: bool = False
    drop_last: bool = False


class DiffusionCacheDataset(IterableDataset[CacheBatch]):
    def __init__(self, cfg: DiffusionCacheLoaderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if self.cfg.backend == "torch":
            self.to_tensor = npz_to_torch
        elif self.cfg.backend == "jax":
            self.to_tensor = npz_to_jax
        else:
            raise ValueError(f"Unsupported backend: {self.cfg.backend}")

    def _parse_tfrecord_index(self, cache_file: Path) -> int:
        match = _SIM_STATE_CACHE_PATTERN.match(cache_file.name)
        if match is None:
            raise ValueError(f"Could not parse tfrecord index from cache file: {cache_file}")
        return int(match.group(1))

    def __iter__(self) -> Iterator[CacheBatch]:
        worker_info = get_worker_info()
        cache_paths = list(self.cfg.cache_paths)
        if self.cfg.shuffle_seed:
            random.Random(int(self.cfg.shuffle_seed)).shuffle(cache_paths)
        if worker_info is not None:
            cache_paths = cache_paths[worker_info.id :: worker_info.num_workers]

        iterator_seed = int(self.cfg.shuffle_seed)
        if worker_info is not None:
            iterator_seed += int(worker_info.id) * 10007
        iterator_rng = random.Random(iterator_seed)

        for cache_path in cache_paths:
            cache_file = Path(cache_path)
            tfrecord_index = self._parse_tfrecord_index(cache_file)

            cache_seed = iterator_rng.randint(0, 2**31 - 1)

            if self.cfg.anchor_steps is None:
                anchor_step_choices = (None,)
            else:
                anchor_step_choices = self.cfg.anchor_steps

            for anchor_step in anchor_step_choices:
                preprocess_seed = cache_seed + (0 if anchor_step is None else int(anchor_step))
                preprocessed = preprocess_cached_npz(
                    cache_file,
                    cfg=self.cfg.preprocess_cfg,
                    seed=preprocess_seed,
                    anchor_step_override=anchor_step,
                    goal_step_override=self.cfg.goal_step_override,
                    include_inst_features=self.cfg.include_inst_features,
                )

                feature_keys = sorted(preprocessed["features"].keys())
                if not feature_keys:
                    continue
                total_examples = int(preprocessed["features"][feature_keys[0]].shape[0])

                if self.cfg.shuffle_seed:
                    rng = random.Random(int(self.cfg.shuffle_seed) + tfrecord_index + int(anchor_step))
                    perm = list(range(total_examples))
                    rng.shuffle(perm)
                    for key, value in preprocessed["features"].items():
                        preprocessed["features"][key] = value[perm]
                    for key, value in preprocessed["aux"].items():
                        preprocessed["aux"][key] = value[perm]

                for start_index in range(0, total_examples, self.cfg.batch_size):
                    end_index = min(start_index + self.cfg.batch_size, total_examples)
                    if self.cfg.drop_last and end_index == total_examples:
                        continue

                    batch_features = {
                        key: self.to_tensor(value[start_index:end_index])
                        for key, value in preprocessed["features"].items()
                    }
                    batch_aux = {
                        key: self.to_tensor(value[start_index:end_index])
                        for key, value in preprocessed["aux"].items()
                    }

                    yield CacheBatch(features=batch_features, aux=batch_aux)


def build_dataloader(
    cache_dir: str,
    preprocess_cfg: PreprocessConfig,
    *,
    file_indices: list[int] | None = None,
    batch_size: int = 1,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    cache_paths: tuple[str, ...] | None = None,
    drop_last: bool = False,
    anchor_steps: tuple[int, ...] | None = None,
    backend: str = "torch",
    goal_step_override: int | None = None,
    include_inst_features: bool = False,
) -> DataLoader[CacheBatch]:
    if cache_paths is None:
        cache_paths = resolve_cache_paths(cache_dir, file_indices)
    cfg = DiffusionCacheLoaderConfig(
        cache_paths=cache_paths,
        preprocess_cfg=preprocess_cfg,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        anchor_steps=anchor_steps,
        backend=backend,
        goal_step_override=goal_step_override,
        include_inst_features=include_inst_features,
        drop_last=drop_last,
    )
    dataset = DiffusionCacheDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )