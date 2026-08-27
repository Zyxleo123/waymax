from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import tensorflow as tf
from waymax import config as waymax_config
from waymax import dataloader
from waymax.dataloader import womd_factories

# TensorFlow is used here only to read and parse tfrecords -- never to compute on
# a GPU -- but by default it preallocates most of the device on first use and
# then fights JAX for what is left. On a 24G a5000 that shows up far away from
# here, as "Allocator (GPU_0_bfc) ran out of memory" inside waymax's offroad
# metric while it tries to allocate 1.73GiB for f32[10,512,30000]. The rl/
# entrypoints set TF_FORCE_GPU_ALLOW_GROWTH for the same reason; hiding the
# device outright is stronger and does not depend on being set before import.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:  # already initialised, or no GPU to hide -- either is fine
    pass

"""
This module provides functions for directly loading scenario data (sim_state) from TFRecord files.
"""


class ScenarioIndexOutOfRange(ValueError):
    """The requested scenario indices lie past the end of the tfrecord.

    Distinct from a *parse* failure on purpose: a sweep that walks a directory
    needs to tell "this file is exhausted, move to the next one" apart from
    "this file cannot be read at all". Treating the two alike silently turns an
    unreadable eval set into a zero-scenario pass.
    """


def infer_route_dims(tfrecord_path: str) -> tuple[int, int]:
    """``(num_paths, num_points_per_path)`` for the SDC routes in a tfrecord.

    ``waymax_config.WOD_1_3_1_TRAINING`` hardcodes 45 x 800, which is right for
    WOMD 1.3.1 tf_examples and wrong for anything re-exported with a different
    route budget -- e.g. failures.tfrecord ships 10 x 300, and parsing it under
    the 45x800 contract dies with "Values size: 9000 but output shape:
    [45,800,3]". ``path_samples/on_route`` has one entry per path and
    ``path_samples/valid`` one per sample, which pins both dimensions.
    """
    raw_ds = tf.data.TFRecordDataset([str(tfrecord_path)])
    try:
        serialized = next(iter(raw_ds))
    except StopIteration as exc:
        raise ValueError(f"tfrecord '{tfrecord_path}' is empty.") from exc

    example = tf.train.Example()
    example.ParseFromString(serialized.numpy())
    features = example.features.feature
    for key in ("path_samples/on_route", "path_samples/valid"):
        if key not in features:
            raise ValueError(
                f"tfrecord '{tfrecord_path}' has no '{key}'; it does not carry SDC routes."
            )

    num_paths = len(features["path_samples/on_route"].int64_list.value)
    num_valid = len(features["path_samples/valid"].int64_list.value)
    if num_paths <= 0 or num_valid % num_paths != 0:
        raise ValueError(
            f"tfrecord '{tfrecord_path}': cannot infer route dims from "
            f"on_route={num_paths}, valid={num_valid}."
        )
    return num_paths, num_valid // num_paths


def load_scenario_state_fast(cfg: waymax_config.DatasetConfig, scenario_index: int):
    """Loads one scenario directly from TFRecord by skipping raw records first."""
    raw_ds = tf.data.TFRecordDataset([cfg.path]).skip(int(scenario_index)).take(1)
    iterator = iter(raw_ds)
    try:
        serialized = next(iterator)
    except StopIteration as exc:
        raise ScenarioIndexOutOfRange(
            f"scenario_index {scenario_index} is out of range for tfrecord '{cfg.path}'."
        ) from exc

    serialized = tf.expand_dims(serialized, axis=0)
    processed = dataloader.preprocess_serialized_womd_data(serialized, cfg)
    return womd_factories.simulator_state_from_womd_dict(
        processed, include_sdc_paths=cfg.include_sdc_paths
    )


def load_scenario_state_batch_fast(
    cfg: waymax_config.DatasetConfig, scenario_indices: Iterable[int]
):
    """Loads a batch of scenarios from one TFRecord in a single dataset scan."""
    indices = [int(i) for i in scenario_indices]
    if not indices:
        raise ValueError("scenario_indices must be non-empty.")
    if any(i < 0 for i in indices):
        raise ValueError(f"scenario_indices must be >= 0, got {indices}.")

    unique_indices = sorted(set(indices))
    needed = set(unique_indices)
    selected: list[tf.Tensor] = []
    selected_indices: list[int] = []

    raw_ds = tf.data.TFRecordDataset([cfg.path])
    for raw_idx, serialized in enumerate(raw_ds):
        if raw_idx in needed:
            selected.append(serialized)
            selected_indices.append(raw_idx)
            if len(selected) == len(needed):
                break

    missing = sorted(needed.difference(selected_indices))
    if missing:
        raise ScenarioIndexOutOfRange(
            f"scenario_index values {missing} are out of range for tfrecord '{cfg.path}'."
        )

    serialized_batch = tf.stack(selected, axis=0)
    processed = dataloader.preprocess_serialized_womd_data(serialized_batch, cfg)
    state_batch = womd_factories.simulator_state_from_womd_dict(
        processed, include_sdc_paths=cfg.include_sdc_paths
    )
    scenario_to_batch_idx = {scenario_idx: i for i, scenario_idx in enumerate(selected_indices)}
    return state_batch, scenario_to_batch_idx
