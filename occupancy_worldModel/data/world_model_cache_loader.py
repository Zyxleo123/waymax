from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from waymax.datatypes.object_state import ObjectTypeIds

from data.preprocess_cached import (
    _build_map_features,
    _build_tl_features,
    _load_npz,
    _wrap_to_pi,
)
from data.types import PreprocessConfig
from data.utils import resolve_cache_paths
from model.world_model.config import WorldModelConfig


_DYNAMIC_TYPES = np.asarray(
    [
        int(ObjectTypeIds.VEHICLE.value),
        int(ObjectTypeIds.PEDESTRIAN.value),
        int(ObjectTypeIds.CYCLIST.value),
    ],
    dtype=np.int32,
)


def _rotate_xy_numpy(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Rotate xy by -heading. heading broadcasts over leading dimensions."""
    c = np.cos(-heading)
    s = np.sin(-heading)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def _gather_ego(arr_snt: np.ndarray, ego_idx_s: np.ndarray) -> np.ndarray:
    return arr_snt[np.arange(arr_snt.shape[0]), ego_idx_s]


def _build_world_model_arrays(
    payload: Mapping[str, np.ndarray],
    *,
    anchor_step: int,
    config: WorldModelConfig,
) -> dict[str, np.ndarray]:
    """Build one anchor-time training set from one cached WOMD shard.

    This function deliberately uses only history/current scene state as model
    input. Logged future states are read only to create supervised targets.
    """
    x = np.asarray(payload["log_trajectory_x"], dtype=np.float32)
    y = np.asarray(payload["log_trajectory_y"], dtype=np.float32)
    vx = np.asarray(payload["log_trajectory_vel_x"], dtype=np.float32)
    vy = np.asarray(payload["log_trajectory_vel_y"], dtype=np.float32)
    yaw = np.asarray(payload["log_trajectory_yaw"], dtype=np.float32)
    length = np.asarray(payload["log_trajectory_length"], dtype=np.float32)
    width = np.asarray(payload["log_trajectory_width"], dtype=np.float32)
    valid = np.asarray(payload["log_trajectory_valid"]).astype(bool)
    is_sdc = np.asarray(payload["object_metadata_is_sdc"]).astype(bool)
    object_types = np.asarray(payload["object_metadata_object_types"], dtype=np.int32)

    if x.ndim != 3:
        raise ValueError(f"Expected cached log trajectories [S,N,T], got {x.shape}")

    s, n_total, num_steps = x.shape
    ego_idx_full = np.argmax(is_sdc.astype(np.int32), axis=1)
    n = min(int(config.max_agents), int(n_total))
    if np.any(ego_idx_full >= n):
        raise ValueError(
            "WorldModelConfig.max_agents truncates away the SDC for at least one scenario. "
            "Use max_agents >= the Waymax max_num_objects setting (normally 128)."
        )
    x = x[:, :n]
    y = y[:, :n]
    vx = vx[:, :n]
    vy = vy[:, :n]
    yaw = yaw[:, :n]
    length = length[:, :n]
    width = width[:, :n]
    valid = valid[:, :n]
    is_sdc = is_sdc[:, :n]
    object_types = object_types[:, :n]

    anchor = int(np.clip(anchor_step, 0, num_steps - 1))
    ego_idx = ego_idx_full
    ego_x = x[np.arange(s), ego_idx, anchor]
    ego_y = y[np.arange(s), ego_idx, anchor]
    ego_yaw = yaw[np.arange(s), ego_idx, anchor]
    origin_xy = np.stack([ego_x, ego_y], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # History input: H samples ending at the anchor.
    # ------------------------------------------------------------------
    hist_offsets = np.arange(config.history_steps - 1, -1, -1, dtype=np.int32)
    hist_raw_idx = anchor - hist_offsets * int(config.history_stride)
    hist_idx = np.clip(hist_raw_idx, 0, num_steps - 1)
    hist_in_range = hist_raw_idx >= 0

    hx = x[:, :, hist_idx]
    hy = y[:, :, hist_idx]
    hvx = vx[:, :, hist_idx]
    hvy = vy[:, :, hist_idx]
    hyaw = yaw[:, :, hist_idx]
    hlen = length[:, :, hist_idx]
    hwid = width[:, :, hist_idx]
    hvalid = valid[:, :, hist_idx] & hist_in_range[None, None, :]

    hist_xy_world = np.stack([hx, hy], axis=-1)
    hist_vel_world = np.stack([hvx, hvy], axis=-1)
    centered = hist_xy_world - origin_xy[:, None, None, :]
    hist_xy_local = _rotate_xy_numpy(centered, ego_yaw[:, None, None])
    hist_vel_local = _rotate_xy_numpy(hist_vel_world, ego_yaw[:, None, None])
    hist_yaw_rel = _wrap_to_pi(hyaw - ego_yaw[:, None, None])

    type_safe = np.clip(object_types, 0, int(config.num_object_types) - 1)
    type_oh = np.eye(int(config.num_object_types), dtype=np.float32)[type_safe]
    type_hist = np.broadcast_to(
        type_oh[:, :, None, :],
        (s, n, config.history_steps, config.num_object_types),
    )

    agent_history = np.concatenate(
        [
            hist_xy_local / float(config.max_range),
            hist_vel_local / float(config.max_velocity),
            np.sin(hist_yaw_rel)[..., None],
            np.cos(hist_yaw_rel)[..., None],
            hlen[..., None] / float(config.max_width),
            hwid[..., None] / float(config.max_width),
            type_hist,
        ],
        axis=-1,
    ).astype(np.float32)
    agent_history = np.where(hvalid[..., None], agent_history, 0.0)

    current_valid = valid[:, :, anchor]
    agent_valid = current_valid.astype(bool)

    current_xy_world = np.stack([x[:, :, anchor], y[:, :, anchor]], axis=-1)
    current_xy_local = _rotate_xy_numpy(
        current_xy_world - origin_xy[:, None, :],
        ego_yaw[:, None],
    ).astype(np.float32)
    current_yaw = yaw[:, :, anchor]

    # ------------------------------------------------------------------
    # Future supervision. We predict displacement relative to each object's
    # current state, not an absolute global position.
    # ------------------------------------------------------------------
    future_offsets = np.arange(1, config.future_steps + 1, dtype=np.int32)
    future_raw_idx = anchor + future_offsets * int(config.future_stride)
    future_idx = np.clip(future_raw_idx, 0, num_steps - 1)
    future_in_range = future_raw_idx < num_steps

    fx = x[:, :, future_idx]
    fy = y[:, :, future_idx]
    fyaw = yaw[:, :, future_idx]
    fvalid = valid[:, :, future_idx] & future_in_range[None, None, :]

    future_xy_world = np.stack([fx, fy], axis=-1)
    future_xy_local = _rotate_xy_numpy(
        future_xy_world - origin_xy[:, None, None, :],
        ego_yaw[:, None, None],
    )
    target_delta_xy = (
        future_xy_local - current_xy_local[:, :, None, :]
    ) / float(config.max_range)

    yaw_delta = _wrap_to_pi(fyaw - current_yaw[:, :, None])
    target_yaw_vec = np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], axis=-1)

    dynamic = np.isin(object_types, _DYNAMIC_TYPES)
    not_ego = ~is_sdc
    target_valid = (
        fvalid
        & current_valid[:, :, None]
        & dynamic[:, :, None]
        & not_ego[:, :, None]
    )

    # ------------------------------------------------------------------
    # Static scene context. Reuse the exact map/TL representation already used
    # by the planner/VLA stack.
    # ------------------------------------------------------------------
    preprocess_cfg = PreprocessConfig(
        max_range=float(config.max_range),
        max_velocity=float(config.max_velocity),
        max_width=float(config.max_width),
        max_tl_points=int(config.max_tl_points),
        num_map_type_classes=int(config.num_map_type_classes),
        map_unknown_type_index=int(config.map_unknown_type_index),
        max_segments=int(config.max_map_segments),
        max_points_per_segment=int(config.max_points_per_segment),
        num_object_types=int(config.num_object_types),
    )
    anchor_step_b = np.full((s,), anchor, dtype=np.int32)
    map_features, map_valid, _ = _build_map_features(
        payload,
        origin_xy,
        ego_yaw,
        preprocess_cfg,
        max_segments=int(config.max_map_segments),
    )
    tl_features, tl_valid = _build_tl_features(
        payload,
        anchor_step_b,
        origin_xy,
        ego_yaw,
        preprocess_cfg,
    )

    return {
        "agent_history": agent_history,
        "agent_history_valid": hvalid.astype(bool),
        "agent_valid": agent_valid.astype(bool),
        "map_features": map_features.astype(np.float32),
        "map_valid": map_valid.astype(bool),
        "traffic_light_features": tl_features.astype(np.float32),
        "traffic_light_valid": tl_valid.astype(bool),
        "target_delta_xy": target_delta_xy.astype(np.float32),
        "target_yaw_vec": target_yaw_vec.astype(np.float32),
        "target_valid": target_valid.astype(bool),
        "current_local_xy": current_xy_local.astype(np.float32),
        "current_yaw_rel": _wrap_to_pi(current_yaw - ego_yaw[:, None]).astype(np.float32),
        "current_length": length[:, :, anchor].astype(np.float32),
        "current_width": width[:, :, anchor].astype(np.float32),
        "object_types": object_types.astype(np.int32),
        "is_sdc": is_sdc.astype(bool),
    }


def _slice_to_torch(tree: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(value[indices]))
        for key, value in tree.items()
    }


@dataclass
class WorldModelCacheLoader:
    cache_paths: tuple[str, ...]
    config: WorldModelConfig
    batch_size: int = 16
    anchor_steps: tuple[int, ...] = (20, 30, 40)
    shuffle: bool = True
    seed: int = 0
    drop_last: bool = False

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(int(self.seed))
        paths = list(self.cache_paths)
        if self.shuffle:
            rng.shuffle(paths)

        for cache_path in paths:
            payload = _load_npz(cache_path)
            anchor_steps = list(self.anchor_steps)
            if self.shuffle:
                rng.shuffle(anchor_steps)

            for anchor_step in anchor_steps:
                arrays = _build_world_model_arrays(
                    payload,
                    anchor_step=int(anchor_step),
                    config=self.config,
                )
                count = int(arrays["agent_history"].shape[0])
                order = np.arange(count, dtype=np.int32)
                if self.shuffle:
                    np_rng = np.random.default_rng(
                        int(self.seed) + int(anchor_step) + sum(Path(cache_path).name.encode("utf-8"))
                    )
                    np_rng.shuffle(order)

                for start in range(0, count, int(self.batch_size)):
                    end = min(start + int(self.batch_size), count)
                    if self.drop_last and end - start < int(self.batch_size):
                        continue
                    yield _slice_to_torch(arrays, order[start:end])


def build_world_model_cache_loader(
    cache_dir: str,
    config: WorldModelConfig,
    *,
    file_indices: list[int] | None = None,
    cache_paths: tuple[str, ...] | None = None,
    batch_size: int = 16,
    anchor_steps: tuple[int, ...] = (20, 30, 40),
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
) -> WorldModelCacheLoader:
    if cache_paths is None:
        cache_paths = resolve_cache_paths(cache_dir, file_indices)
    return WorldModelCacheLoader(
        cache_paths=cache_paths,
        config=config,
        batch_size=batch_size,
        anchor_steps=anchor_steps,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


__all__ = [
    "WorldModelCacheLoader",
    "build_world_model_cache_loader",
    "_build_world_model_arrays",
]
