from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

from data.failure_detection_loader import DEFAULT_FAILURE_PROMPT
from data.postprocess import postprocess_predictions
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessBatch, PreprocessConfig
from model.diffusion.fm_policy import FlowMatchingPolicy
from model.vla.failure_detection_vla import FailureDetectionVLA
from planner.abstract_planner import AbstractPlanner, PlannerResult
from planner.utils.fd_bridge import (
	attach_unconditional_failure_fields,
	build_fd_prompt_batch,
	guidance_scale_from_prediction,
	inject_failure_condition,
	jax_features_to_torch,
	parse_fd_predictions,
	world_plan_to_fd_ego_trajectory,
)
from planner.utils.fmfd_viz import save_fmfd_cfg_comparison_image


@dataclass
class FMFDPlannerResult(PlannerResult):
	instruction_texts: Sequence[str]
	failure_reasons: Sequence[str]
	critical_object_ids: Sequence[int]
	critical_road_segment_ids: Sequence[int]
	used_cfg: Sequence[bool]


class FMFDPlanner(AbstractPlanner):
	"""Uncond FM candidate → FD VLA → CFG-correct failures.

	Pass 1 samples with ``failure_mask=0`` (FiLM identity). Pass 2 resamples
	or samples only the worlds FD marked as collision / offroad / GNR, using
	per-sample ``guidance_scale``. Success worlds keep the pass-1 trajectory.
	"""

	def __init__(
		self,
		policy: FlowMatchingPolicy,
		fd_vla: FailureDetectionVLA | str | Path,
		preprocess_cfg: PreprocessConfig,
		population_size: int = 1,
		num_worlds: int = 1,
		num_steps: int | None = None,
		solver: str = "euler",
		*,
		guidance_scale: float = 2.0,
		cfg_t_start: float = 0.2,
		correction_mode: str = "resample",
		torch_device: str | torch.device | None = None,
		jax_device: Any | None = None,
		prompt_text: str = DEFAULT_FAILURE_PROMPT,
		max_new_tokens: int = 8,
		max_prompt_length: int = 128,
		save_cfg_viz: bool = True,
		**kwargs: Any,
	) -> None:
		del kwargs
		if correction_mode not in ("resample", "sample"):
			raise ValueError(f"Unknown correction_mode: {correction_mode}")

		self.policy = policy
		self.preprocess_cfg = preprocess_cfg
		self.num_worlds = int(num_worlds)
		self.population_size = int(population_size)
		self.num_steps = num_steps
		self.solver = solver
		self.guidance_scale = float(guidance_scale)
		self.cfg_t_start = float(cfg_t_start)
		self.correction_mode = correction_mode
		self.prompt_text = prompt_text
		self.max_new_tokens = int(max_new_tokens)
		self.max_prompt_length = int(max_prompt_length)
		self.save_cfg_viz = bool(save_cfg_viz)
		self._viz_output_dir: Path | None = None
		self._viz_tfrecord_name: str | None = None
		self._viz_scenario_indices: list[int] | None = None

		if torch_device is None:
			if torch.cuda.is_available():
				torch_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
			else:
				torch_device = torch.device("cpu")
		self.torch_device = torch.device(torch_device)

		if isinstance(fd_vla, (str, Path)):
			self.fd_vla = FailureDetectionVLA.from_pretrained(
				fd_vla,
				device=self.torch_device,
				dtype=torch.bfloat16,
			)
		else:
			self.fd_vla = fd_vla.to(self.torch_device)
		self.fd_vla.eval()

		if jax_device is None:
			gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
			jax_device = gpu_devices[0] if gpu_devices else jax.devices()[0]
		self.jax_device = jax_device

		self._compute_condition_jit = jax.jit(self.policy.compute_condition)
		self._sample_uncond_jit = jax.jit(
			lambda *, cond_bf, rng: self._sample_population(
				cond_bf=cond_bf,
				rng=rng,
				guidance_scale=1.0,
			),
		)
		self._sample_guided_jit = jax.jit(
			lambda *, cond_bf, rng, guidance_scale: self._sample_population(
				cond_bf=cond_bf,
				rng=rng,
				guidance_scale=guidance_scale,
			),
		)
		self._resample_guided_jit = jax.jit(
			lambda *, cond_bf, proposals_btd, rng, guidance_scale: self._resample_population(
				cond_bf=cond_bf,
				proposals_btd=proposals_btd,
				rng=rng,
				guidance_scale=guidance_scale,
			),
		)

	def set_episode_context(
		self,
		*,
		output_dir: str | Path | None,
		tfrecord_path: str | None,
		scenario_indices: Sequence[int] | None,
	) -> None:
		self._viz_output_dir = Path(output_dir) if output_dir is not None else None
		if tfrecord_path is None:
			self._viz_tfrecord_name = None
		else:
			self._viz_tfrecord_name = Path(tfrecord_path).name
		self._viz_scenario_indices = (
			[int(idx) for idx in scenario_indices] if scenario_indices is not None else None
		)

	def plan_trajectory(
		self,
		sim_state,
		goal,
		*,
		rng: jax.Array,
		timestep: int = 0,
		mask_goal: bool = False,
		**kwargs: Any,
	) -> FMFDPlannerResult:
		del kwargs

		rng, key_pre, key_uncond, key_cfg = jax.random.split(rng, 4)
		pre_batch, _ = preprocess_simulator_state(
			sim_state,
			key_pre,
			self.preprocess_cfg,
			anchor_step_override=timestep,
			goal_step_override=90,
			goal_xy_override=goal,
		)

		features = attach_unconditional_failure_fields(dict(pre_batch.features))
		if mask_goal:
			features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
			features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])

		with jax.default_device(self.jax_device):
			cond_bf = self._compute_condition_jit(features)
			current_norm_bktd = self._sample_uncond_jit(
				cond_bf=cond_bf,
				rng=key_uncond,
			)

		current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
			current_norm_bktd,
			pre_batch,
		)
		best_indices = jnp.zeros((cond_bf.shape[0], 1), dtype=jnp.int32)
		best_norm_btd = jnp.squeeze(self._gather_population(current_norm_bktd, best_indices), axis=1)
		best_world_bt5 = jnp.squeeze(self._gather_population(current_world_bkt5, best_indices), axis=1)
		uncond_world_bt5 = best_world_bt5

		fd_prediction = self._detect_failures(
			features,
			best_world_bt5,
			pre_batch,
		)

		used_cfg = np.asarray(fd_prediction.should_correct, dtype=bool)
		if fd_prediction.any_correct:
			guided_features = inject_failure_condition(features, fd_prediction)
			guidance = guidance_scale_from_prediction(
				fd_prediction,
				guidance_scale=self.guidance_scale,
			)
			with jax.default_device(self.jax_device):
				guided_cond = self._compute_condition_jit(
					guided_features
				)
				if self.correction_mode == "resample":
					corrected_norm_btd = self._resample_guided_jit(
						cond_bf=guided_cond,
						proposals_btd=best_norm_btd,
						rng=key_cfg,
						guidance_scale=guidance,
					)
				else:
					corrected_norm_bktd = self._sample_guided_jit(
						cond_bf=guided_cond,
						rng=key_cfg,
						guidance_scale=guidance,
					)
					corrected_norm_btd = jnp.squeeze(
						self._gather_population(corrected_norm_bktd, best_indices),
						axis=1,
					)

			correct_mask = jnp.asarray(used_cfg)[:, None, None]
			best_norm_btd = jnp.where(correct_mask, corrected_norm_btd, best_norm_btd)
			best_world_bt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_single(
				best_norm_btd,
				pre_batch,
			)
			self._maybe_save_cfg_viz(
				sim_state=sim_state,
				timestep=timestep,
				uncond_world_bt5=uncond_world_bt5,
				cfg_world_bt5=best_world_bt5,
				world_t_valid_bt=world_t_valid_bt,
				fd_prediction=fd_prediction,
				used_cfg=used_cfg,
				goal=goal,
			)

		aux = dict(pre_batch.aux)
		aux["fd_failure_reason"] = jnp.asarray(fd_prediction.reasons, dtype=jnp.int32)
		aux["fd_critical_object_ids"] = jnp.asarray(fd_prediction.critical_object_ids, dtype=jnp.int32)
		aux["fd_critical_road_segment_ids"] = jnp.asarray(
			fd_prediction.critical_road_segment_ids,
			dtype=jnp.int32,
		)
		aux["used_cfg"] = jnp.asarray(used_cfg, dtype=bool)

		return FMFDPlannerResult(
			start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
			trajectory_norm_btd=best_norm_btd,
			trajectory_world_bt5=best_world_bt5,
			world_t_seconds_bt=world_t_seconds_bt,
			world_t_valid_bt=world_t_valid_bt,
			aux=aux,
			instruction_texts=list(fd_prediction.texts),
			failure_reasons=list(fd_prediction.texts),
			critical_object_ids=[int(value) for value in fd_prediction.critical_object_ids.tolist()],
			critical_road_segment_ids=[
				int(value) for value in fd_prediction.critical_road_segment_ids.tolist()
			],
			used_cfg=[bool(value) for value in used_cfg.tolist()],
		)

	def _detect_failures(
		self,
		features: dict[str, Any],
		best_world_bt5: jax.Array,
		pre_batch: PreprocessBatch,
	):
		fd_traj, fd_valid = world_plan_to_fd_ego_trajectory(
			best_world_bt5,
			pre_batch.aux["origin_xy"],
			pre_batch.aux["anchor_yaw"],
			pre_batch.aux["world_t_valid"],
			ego_range=self.preprocess_cfg.ego_range,
			max_velocity=self.preprocess_cfg.max_velocity,
		)
		fd_features = dict(features)
		fd_features["ego_trajectory"] = fd_traj
		fd_features["ego_trajectory_valid"] = fd_valid
		torch_features = jax_features_to_torch(fd_features, device=self.torch_device)
		prompt_ids, prompt_mask = build_fd_prompt_batch(
			self.fd_vla.tokenizer,
			batch_size=int(fd_traj.shape[0]),
			device=self.torch_device,
			prompt=self.prompt_text,
			max_length=self.max_prompt_length,
		)

		with torch.inference_mode():
			amp_enabled = self.torch_device.type == "cuda"
			with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16, enabled=amp_enabled):
				predictions = self.fd_vla.generate_predictions(
					input_features=torch_features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					max_new_tokens=self.max_new_tokens,
				)
		return parse_fd_predictions(predictions)

	def _maybe_save_cfg_viz(
		self,
		*,
		sim_state,
		timestep: int,
		uncond_world_bt5: jax.Array,
		cfg_world_bt5: jax.Array,
		world_t_valid_bt: jax.Array,
		fd_prediction,
		used_cfg: np.ndarray,
		goal,
	) -> None:
		if not self.save_cfg_viz or self._viz_output_dir is None:
			return
		uncond = np.asarray(uncond_world_bt5, dtype=np.float32)
		cfg = np.asarray(cfg_world_bt5, dtype=np.float32)
		valid = np.asarray(world_t_valid_bt)
		goal_xy = None if goal is None else np.asarray(goal, dtype=np.float32)
		tfrecord_name = self._viz_tfrecord_name or "tfrecord"
		for batch_idx, should_viz in enumerate(used_cfg.tolist()):
			if not bool(should_viz):
				continue
			if self._viz_scenario_indices is not None and batch_idx < len(self._viz_scenario_indices):
				scenario_idx = int(self._viz_scenario_indices[batch_idx])
			else:
				scenario_idx = int(batch_idx)
			image_name = (
				f"{tfrecord_name}.scenario_{scenario_idx:03d}.t{int(timestep):03d}.fmfd.png"
			)
			goal_b = None
			if goal_xy is not None:
				if goal_xy.ndim == 1:
					goal_b = goal_xy
				elif batch_idx < int(goal_xy.shape[0]):
					goal_b = goal_xy[batch_idx]
			valid_b = valid[batch_idx] if valid.ndim == 2 and batch_idx < valid.shape[0] else None
			try:
				save_fmfd_cfg_comparison_image(
					sim_state=sim_state,
					batch_idx=batch_idx,
					timestep=int(timestep),
					uncond_traj_t5=uncond[batch_idx],
					cfg_traj_t5=cfg[batch_idx],
					world_t_valid=valid_b,
					failure_text=fd_prediction.texts[batch_idx],
					reason=int(fd_prediction.reasons[batch_idx]),
					object_id=int(fd_prediction.critical_object_ids[batch_idx]),
					road_id=int(fd_prediction.critical_road_segment_ids[batch_idx]),
					goal_xy=goal_b,
					output_path=self._viz_output_dir / image_name,
				)
			except Exception as exc:
				print(f"Failed to save FM-FD CFG viz {image_name}: {exc}")

	def _sample_population(
		self,
		*,
		cond_bf: jax.Array,
		rng: jax.Array,
		guidance_scale: float | jax.Array,
	) -> jax.Array:
		batch_size = int(cond_bf.shape[0])
		cond_bkf = jnp.repeat(cond_bf, repeats=self.population_size, axis=0)
		del guidance_scale
		samples_btd = self.policy.sample_from_condition(
			cond_bkf,
			rng=rng,
			num_steps=self.num_steps,
			solver=self.solver,
		)
		horizon = int(samples_btd.shape[1])
		target_dim = int(samples_btd.shape[2])
		return samples_btd.reshape(batch_size, self.population_size, horizon, target_dim)

	def _resample_population(
		self,
		*,
		cond_bf: jax.Array,
		proposals_btd: jax.Array,
		rng: jax.Array,
		guidance_scale: float | jax.Array,
	) -> jax.Array:
		batch_size = int(cond_bf.shape[0])
		del guidance_scale
		return self.policy.resample_from_condition(
			cond_bf,
			proposals_btd,
			rng=rng,
			t_start=self.cfg_t_start,
			num_steps=self.num_steps,
			solver=self.solver,
		)

	def _postprocess_single(
		self,
		proposals_btd: jax.Array,
		pre_batch: PreprocessBatch,
	) -> tuple[jax.Array, jax.Array, jax.Array]:
		world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
			proposals_btd[:, None, ...],
			pre_batch,
		)
		return world_bkt5[:, 0], world_t_seconds_bt, world_t_valid_bt

	def _postprocess_population(
		self,
		proposals_bktd: jax.Array,
		pre_batch: PreprocessBatch,
	) -> tuple[jax.Array, jax.Array, jax.Array]:
		batch_size, population_size, horizon, target_dim = proposals_bktd.shape
		flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
		repeated_aux = self._repeat_dict_leading_axis(pre_batch.aux, repeats=population_size)
		post = postprocess_predictions(flat_proposals, repeated_aux, self.preprocess_cfg)

		world_flat = post["trajectory_world_world_dt"]
		world_steps = int(world_flat.shape[1])
		world_dim = int(world_flat.shape[2])
		trajectories_world_bkt5 = world_flat.reshape(
			batch_size,
			population_size,
			world_steps,
			world_dim,
		)
		world_t_seconds_bt = jnp.asarray(pre_batch.aux["world_t_seconds"], dtype=jnp.float32)
		world_t_valid_bt = jnp.asarray(pre_batch.aux["world_t_valid"])
		return trajectories_world_bkt5, world_t_seconds_bt, world_t_valid_bt

	@staticmethod
	def _gather_population(population_bk: jax.Array, indices_bm: jax.Array) -> jax.Array:
		gather_indices = jnp.asarray(indices_bm, dtype=jnp.int32)
		trailing_shape = population_bk.shape[2:]
		reshape_shape = gather_indices.shape + (1,) * len(trailing_shape)
		broadcast_shape = gather_indices.shape + trailing_shape
		gather_indices = jnp.broadcast_to(
			gather_indices.reshape(reshape_shape),
			broadcast_shape,
		)
		return jnp.take_along_axis(population_bk, gather_indices, axis=1)

	@staticmethod
	def _repeat_dict_leading_axis(
		values: dict[str, jax.Array],
		*,
		repeats: int,
	) -> dict[str, jax.Array]:
		return jax.tree_util.tree_map(
			lambda x: jnp.repeat(jnp.asarray(x), repeats=repeats, axis=0),
			values,
		)


__all__ = ["FMFDPlanner", "FMFDPlannerResult"]
