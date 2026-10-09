"""JAX Flow Matching <-> PyTorch Failure Detection VLA conversions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import torch

from data.failure_detection_loader import (
	DEFAULT_FAILURE_PROMPT,
	FAILURE_REASON_COLLISION,
	FAILURE_REASON_GOAL_NOT_REACHED,
	FAILURE_REASON_NONE,
	FAILURE_REASON_OFFROAD,
	FUTURE_EGO_TRAJECTORY_STEPS,
	FUTURE_EGO_TRAJECTORY_STRIDE,
)


FAILURE_REASON_NAMES = {
	FAILURE_REASON_NONE: "success",
	FAILURE_REASON_COLLISION: "collision",
	FAILURE_REASON_OFFROAD: "offroad",
	FAILURE_REASON_GOAL_NOT_REACHED: "goal not reached",
}

_REASON_PHRASES: tuple[tuple[str, int], ...] = (
	("goal not reached", FAILURE_REASON_GOAL_NOT_REACHED),
	("goalnot reached", FAILURE_REASON_GOAL_NOT_REACHED),
	("collision", FAILURE_REASON_COLLISION),
	("offroad", FAILURE_REASON_OFFROAD),
	("off road", FAILURE_REASON_OFFROAD),
	("success", FAILURE_REASON_NONE),
)


@dataclass(frozen=True)
class FDPrediction:
	texts: tuple[str, ...]
	reasons: np.ndarray
	critical_object_ids: np.ndarray
	critical_road_segment_ids: np.ndarray
	should_correct: np.ndarray

	@property
	def batch_size(self) -> int:
		return int(self.reasons.shape[0])

	@property
	def any_correct(self) -> bool:
		return bool(np.any(self.should_correct))


def _as_numpy(value: Any) -> np.ndarray:
	if isinstance(value, np.ndarray):
		return value
	return np.asarray(value)


def _rotate_xy(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
	cos_h = np.cos(-heading)
	sin_h = np.sin(-heading)
	x = xy[..., 0]
	y = xy[..., 1]
	xr = x * cos_h - y * sin_h
	yr = x * sin_h + y * cos_h
	return np.stack([xr, yr], axis=-1)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
	return (angle + np.pi) % (2.0 * np.pi) - np.pi


def normalize_failure_text(text: str) -> str:
	return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def parse_failure_reason_text(text: str) -> int:
	"""Map FD VLA answer text to ``FAILURE_REASON_*``. Unknown → success."""
	normalized = normalize_failure_text(text)
	for phrase, reason in _REASON_PHRASES:
		if phrase in normalized:
			return int(reason)
	return int(FAILURE_REASON_NONE)


def should_apply_cfg(reason: int, object_id: int, road_id: int) -> bool:
	if int(reason) == FAILURE_REASON_COLLISION:
		return int(object_id) >= 0
	if int(reason) == FAILURE_REASON_OFFROAD:
		return int(road_id) >= 0
	if int(reason) == FAILURE_REASON_GOAL_NOT_REACHED:
		return True
	return False


def jax_features_to_torch(
	features: Mapping[str, Any],
	*,
	device: torch.device,
) -> dict[str, torch.Tensor]:
	moved: dict[str, torch.Tensor] = {}
	for key, value in features.items():
		array = _as_numpy(value)
		tensor = torch.as_tensor(array, device=device)
		if tensor.dtype.is_floating_point:
			tensor = tensor.to(dtype=torch.float32)
		moved[key] = tensor
	return moved


def build_fd_prompt_batch(
	tokenizer: Any,
	*,
	batch_size: int,
	device: torch.device,
	prompt: str = DEFAULT_FAILURE_PROMPT,
	max_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
	texts = [prompt] * int(batch_size)
	tokenized = tokenizer(
		texts,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=int(max_length),
		add_special_tokens=True,
	)
	prompt_ids = tokenized["input_ids"].to(device)
	prompt_mask = tokenized["attention_mask"].to(device)
	return prompt_ids, prompt_mask


def parse_fd_predictions(predictions: Mapping[str, Any]) -> FDPrediction:
	texts = tuple(str(text).strip() for text in predictions.get("failure_reason", predictions.get("answer", ())))
	reasons = np.asarray([parse_failure_reason_text(text) for text in texts], dtype=np.int32)
	object_ids = np.asarray(predictions.get("critical_object_ids", [-1] * len(texts)), dtype=np.int32).reshape(-1)
	road_ids = np.asarray(predictions.get("critical_road_segment_ids", [-1] * len(texts)), dtype=np.int32).reshape(-1)
	if object_ids.shape[0] != reasons.shape[0]:
		raise ValueError(
			f"critical_object_ids batch {object_ids.shape[0]} != answers {reasons.shape[0]}"
		)
	if road_ids.shape[0] != reasons.shape[0]:
		raise ValueError(
			f"critical_road_segment_ids batch {road_ids.shape[0]} != answers {reasons.shape[0]}"
		)
	should_correct = np.asarray(
		[
			should_apply_cfg(int(reason), int(object_id), int(road_id))
			for reason, object_id, road_id in zip(reasons, object_ids, road_ids)
		],
		dtype=bool,
	)
	return FDPrediction(
		texts=texts,
		reasons=reasons,
		critical_object_ids=object_ids,
		critical_road_segment_ids=road_ids,
		should_correct=should_correct,
	)


def world_plan_to_fd_ego_trajectory(
	world_bt5: np.ndarray | jax.Array,
	origin_xy: np.ndarray | jax.Array,
	anchor_yaw: np.ndarray | jax.Array,
	world_t_valid: np.ndarray | jax.Array | None = None,
	*,
	ego_range: float,
	max_velocity: float,
	num_steps: int = FUTURE_EGO_TRAJECTORY_STEPS,
	stride: int = FUTURE_EGO_TRAJECTORY_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
	"""Convert a world-dt plan into the FD VLA ``ego_trajectory`` convention.

	``world_bt5`` is ``[B, Tw, 5]`` with layout ``[x, y, yaw, vx, vy]`` and
	index 0 at the current anchor. FD training samples 25 future waypoints at
	stride 2 (world steps 2, 4, ..., 50) in the ego frame.
	"""
	world_bt5 = np.asarray(world_bt5, dtype=np.float32)
	origin_xy = np.asarray(origin_xy, dtype=np.float32).reshape(-1, 2)
	anchor_yaw = np.asarray(anchor_yaw, dtype=np.float32).reshape(-1)
	if world_bt5.ndim != 3 or world_bt5.shape[-1] != 5:
		raise ValueError(f"world_bt5 must have shape [B, T, 5], got {world_bt5.shape}")
	batch_size, num_world_steps, _ = world_bt5.shape
	if origin_xy.shape[0] != batch_size or anchor_yaw.shape[0] != batch_size:
		raise ValueError(
			f"origin/yaw batch mismatch: world={batch_size}, "
			f"origin={origin_xy.shape[0]}, yaw={anchor_yaw.shape[0]}"
		)

	if world_t_valid is None:
		valid_bt = np.ones((batch_size, num_world_steps), dtype=bool)
	else:
		valid_bt = np.asarray(world_t_valid).astype(bool)
		if valid_bt.shape != (batch_size, num_world_steps):
			valid_bt = np.broadcast_to(valid_bt, (batch_size, num_world_steps))

	future = np.zeros((batch_size, int(num_steps), 5), dtype=np.float32)
	future_valid = np.zeros((batch_size, int(num_steps)), dtype=bool)
	ego_range = float(ego_range)
	max_velocity = float(max_velocity)

	for waypoint_idx in range(int(num_steps)):
		step = int(stride) * (waypoint_idx + 1)
		if step >= num_world_steps:
			continue
		step_valid = valid_bt[:, step]
		position_xy = world_bt5[:, step, :2] - origin_xy
		velocity_xy = world_bt5[:, step, 3:5]
		yaw = world_bt5[:, step, 2]
		future[:, waypoint_idx, :2] = _rotate_xy(position_xy, anchor_yaw) / ego_range
		future[:, waypoint_idx, 2:4] = _rotate_xy(velocity_xy, anchor_yaw) / max_velocity
		future[:, waypoint_idx, 4] = _wrap_to_pi(yaw - anchor_yaw) / np.pi
		future[:, waypoint_idx] = np.where(step_valid[:, None], future[:, waypoint_idx], 0.0)
		future_valid[:, waypoint_idx] = step_valid

	return future, future_valid


def attach_object_axis_ids(features: dict[str, Any]) -> dict[str, Any]:
	"""Set ``object_ids`` / ``other_ids`` to the other-agent axis index.

	Collision gold ids are object-axis indices, matching gold finetune and
	FD VLA training.
	"""
	out = dict(features)
	other_states = jnp.asarray(out["other_states"])
	batch_size = int(other_states.shape[0])
	num_objects = int(other_states.shape[1])
	object_ids = jnp.broadcast_to(
		jnp.arange(num_objects, dtype=jnp.int32)[None, :],
		(batch_size, num_objects),
	)
	out["object_ids"] = object_ids
	out["other_ids"] = object_ids
	return out


def attach_unconditional_failure_fields(features: dict[str, Any]) -> dict[str, Any]:
	out = attach_object_axis_ids(features)
	batch_size = int(jnp.asarray(out["ego_state"]).shape[0])
	out["failure_reason"] = jnp.full((batch_size,), FAILURE_REASON_NONE, dtype=jnp.int32)
	out["critical_object_ids"] = jnp.full((batch_size,), -1, dtype=jnp.int32)
	out["critical_road_segment_ids"] = jnp.full((batch_size,), -1, dtype=jnp.int32)
	return out


def inject_failure_condition(
	features: Mapping[str, Any],
	prediction: FDPrediction,
) -> dict[str, Any]:
	out = attach_object_axis_ids(dict(features))
	batch_size = int(jnp.asarray(out["ego_state"]).shape[0])
	if prediction.batch_size != batch_size:
		raise ValueError(
			f"FD batch {prediction.batch_size} != feature batch {batch_size}"
		)
	out["failure_reason"] = jnp.asarray(prediction.reasons, dtype=jnp.int32)
	out["critical_object_ids"] = jnp.asarray(prediction.critical_object_ids, dtype=jnp.int32)
	out["critical_road_segment_ids"] = jnp.asarray(
		prediction.critical_road_segment_ids,
		dtype=jnp.int32,
	)
	return out


def guidance_scale_from_prediction(
	prediction: FDPrediction,
	*,
	guidance_scale: float,
) -> jnp.ndarray:
	scales = np.where(prediction.should_correct, float(guidance_scale), 1.0).astype(np.float32)
	return jnp.asarray(scales, dtype=jnp.float32)


def reason_name(reason: int) -> str:
	return FAILURE_REASON_NAMES.get(int(reason), "success")


__all__ = [
	"DEFAULT_FAILURE_PROMPT",
	"FDPrediction",
	"FAILURE_REASON_NAMES",
	"attach_object_axis_ids",
	"attach_unconditional_failure_fields",
	"build_fd_prompt_batch",
	"guidance_scale_from_prediction",
	"inject_failure_condition",
	"jax_features_to_torch",
	"parse_failure_reason_text",
	"parse_fd_predictions",
	"reason_name",
	"should_apply_cfg",
	"world_plan_to_fd_ego_trajectory",
]
