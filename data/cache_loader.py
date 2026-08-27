from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from dataclasses import dataclass
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from data.preprocess_cached import preprocess_cached_npz
from data.utils import (
    resolve_cache_paths,
    build_question_bank,
    npz_to_torch,
    npz_to_jax,
    VLA_PROMPT,
    infer_subgoal_from_features,
)
from data.types import PreprocessConfig, CacheLoaderConfig, CacheBatch
from data.hardness_sampler import draw_order, load_risk_table


class CacheDataset(IterableDataset[CacheBatch]):
    def __init__(self, cfg: CacheLoaderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._epoch = 0
        if self.cfg.backend == "torch":
            self.to_tensor = npz_to_torch
        elif self.cfg.backend == "jax":
            self.to_tensor = npz_to_jax
        else:
            raise ValueError(f"Unsupported backend: {self.cfg.backend}")

    def build_annotation_paths(self, cache_file: Path, anchor_step: int) -> tuple[Path, Path]:
        annotation_dir = Path(self.cfg.annotation_dir)
        stem = cache_file.name.removesuffix(".sim_state_cache.npz")
        return (
            annotation_dir / f"{stem}_t{anchor_step}.jsonl",
            annotation_dir / f"{stem}_t{anchor_step}.idx.json",
        )
    
    def load_byte_offsets(self, idx_path: Path) -> list[int]:
        with idx_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        if isinstance(index_data, list):
            return [int(v) for v in index_data]
        if isinstance(index_data, dict):
            if "byte_offsets" in index_data and isinstance(index_data["byte_offsets"], list):
                return [int(v) for v in index_data["byte_offsets"]]
            if len(index_data) == 1:
                value = next(iter(index_data.values()))
                if isinstance(value, list):
                    return [int(v) for v in value]
        raise ValueError(f"Unsupported annotation index format: {idx_path}")
    
    def load_annotation(self, cache_file: Path, scenario_index: int, anchor_step: int) -> dict[str, Any]:
        jsonl_path, idx_path = self.build_annotation_paths(cache_file, anchor_step)
        offsets = self.load_byte_offsets(idx_path)
        offset = offsets[int(scenario_index)]
        with jsonl_path.open("rb") as handle:
            handle.seek(int(offset))
            raw = handle.readline()
        obj = json.loads(raw.decode("utf-8"))
        annotation = obj["annotation"]
        annotation["instruction"] = obj["instruction"].strip()
        return annotation

    def _risk_table(self):
        if not self.cfg.risk_table_path:
            return None
        return load_risk_table(
            str(self.cfg.risk_table_path),
            float(self.cfg.hardness_alpha),
            float(self.cfg.weight_clip),
            bool(self.cfg.exclude_dirty),
        )

    def __iter__(self) -> Iterator[CacheBatch]:
        worker_info = get_worker_info()
        epoch = self._epoch
        self._epoch += 1

        # (shard, anchor) is the work unit, and the two are shuffled TOGETHER.
        # Nesting anchors outside shards -- as this loop used to -- walks the
        # whole 455k-scenario dataset at anchor 0 before it ever sees anchor 10,
        # which is ~7k steps of an unintended curriculum at batch 64.
        blocks = [
            (path, anchor)
            for path in self.cfg.cache_paths
            for anchor in self.cfg.anchor_steps
        ]
        # `is not None`, not truthiness: shuffle_seed defaults to 0 in
        # DiffusionTrainConfig, and `if self.cfg.shuffle_seed:` silently disabled
        # shuffling entirely for every run that took the default.
        if self.cfg.shuffle_seed is not None:
            random.Random(int(self.cfg.shuffle_seed) + epoch).shuffle(blocks)
        if worker_info is not None:
            blocks = blocks[worker_info.id :: worker_info.num_workers]

        risk_table = self._risk_table()
        alpha = float(self.cfg.hardness_alpha) if risk_table is not None else 0.0

        for cache_path, anchor_step in blocks:
            cache_file = Path(cache_path)
            preprocessed = preprocess_cached_npz(
                cache_file, cfg=self.cfg.preprocess_cfg, anchor_step_override=anchor_step,
                include_inst_features=self.cfg.include_inst_features
            )
            feature_keys = sorted(preprocessed["features"].keys())
            total_examples = int(preprocessed["features"][feature_keys[0]].shape[0])
            scenario_indices = preprocessed["metadata"]["scenario_indices"]

            seed = (int(self.cfg.shuffle_seed or 0) + epoch, anchor_step, hash(str(cache_path)) % (2**31))
            rng = np.random.default_rng(seed)
            if risk_table is not None:
                weights = risk_table.weights(str(cache_path), scenario_indices, anchor_step)
            else:
                weights = np.ones(total_examples, dtype=np.float64)
            order = draw_order(weights, rng, alpha)
            # Exclusion can shorten a block, and resampling keeps it the same
            # length; either way the batch loop must follow `order`, not the
            # original shard size.
            total_examples = int(order.size)

            scenario_indices = scenario_indices[order]
            for key, value in preprocessed["features"].items():
                preprocessed["features"][key] = value[order]
            for key, value in preprocessed["aux"].items():
                preprocessed["aux"][key] = value[order]

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
                if self.cfg.language_label is not None:
                    prompts, answers, keys = [], [], []
                    for i, scenario_index in enumerate(scenario_indices[start_index:end_index]):
                        annotation = self.load_annotation(cache_file, scenario_index, anchor_step)
                        if self.cfg.language_label == "qa":
                            qa = build_question_bank(annotation)
                            for item in qa:
                                prompts.append(item["question"])
                                answers.append(item["answer"])
                                keys.append(item["key"])
                        elif self.cfg.language_label == "instruction":
                            prompts.append(VLA_PROMPT)
                            answer = f"{annotation['instruction']}"
                            answers.append(answer)
                    batch = CacheBatch(
                        features=batch_features,
                        aux=batch_aux,
                        prompts=prompts,
                        answers=answers,
                        keys=keys
                    )
                else:
                    batch = CacheBatch(features=batch_features, aux=batch_aux)
                yield batch


def build_dataloader(
    cache_dir: str,
    preprocess_cfg: PreprocessConfig,
    annotation_dir: str | None = None,
    anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80),
    language_label: str | None = None,
    backend: str = "torch",
    include_inst_features: bool = False,
    *,
    file_indices: list[int] | None = None,
    batch_size: int = 1,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    cache_paths: tuple[str, ...] | None = None,
    drop_last: bool = False,
    risk_table_path: str | None = None,
    hardness_alpha: float = 0.0,
    weight_clip: float = 10.0,
    exclude_dirty: bool = True,
) -> DataLoader[CacheBatch]:
    if cache_paths is None:
        cache_paths = resolve_cache_paths(cache_dir, file_indices)
    cfg = CacheLoaderConfig(
        cache_paths=cache_paths,
        annotation_dir=annotation_dir,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        preprocess_cfg=preprocess_cfg,
        anchor_steps=anchor_steps,
        language_label=language_label,
        backend=backend,
        include_inst_features=include_inst_features,
        drop_last=drop_last,
        risk_table_path=risk_table_path,
        hardness_alpha=hardness_alpha,
        weight_clip=weight_clip,
        exclude_dirty=exclude_dirty,
    )
    dataset = CacheDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

