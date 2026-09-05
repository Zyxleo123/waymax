from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from data.postprocess import postprocess_predictions
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessBatch, PreprocessConfig
from model.diffusion_torch.diffusion_policy import DiffusionPolicy
from planner.abstract_planner import AbstractPlanner, PlannerResult


class DiffusionTorchPlanner(AbstractPlanner):
    def __init__(
        self,
        policy: DiffusionPolicy,
        preprocess_cfg: PreprocessConfig,
        *,
        population_size: int = 1,
        num_worlds: int = 1,
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.policy = policy
        self.preprocess_cfg = preprocess_cfg
        self.population_size = int(population_size)
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device) if device is not None else next(policy.parameters()).device
        self.policy.eval()

    def plan_trajectory(
        self,
        sim_state,
        goal,
        *,
        rng: jax.Array,
        instruction: jnp.ndarray | None = None,
        instruction_mask: jnp.ndarray | None = None,
        timestep: int = 0,
        mask_goal: bool = False,
        **kwargs: Any,
    ) -> PlannerResult:
        del kwargs
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            rng,
            self.preprocess_cfg,
            anchor_step_override=timestep,
            goal_step_override=90,
            goal_xy_override=goal,
        )

        features = dict(pre_batch.features)
        if instruction is not None or instruction_mask is not None:
            raise NotImplementedError("Instruction conditioning is not implemented in DiffusionTorchPlanner.")

        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])

        cond_bkf = self._build_population_condition(features)

        with torch.no_grad():
            pred_norm_bktd = self._sample_population(cond_bkf)

        pred_norm_bktd_jax = jnp.asarray(pred_norm_bktd)
        world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(pred_norm_bktd_jax, pre_batch)

        best_indices = jnp.zeros((world_bkt5.shape[0], 1), dtype=jnp.int32)
        best_norm_bt1d = self._gather_population(pred_norm_bktd_jax, best_indices)
        best_world_bt15 = self._gather_population(world_bkt5, best_indices)

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=pre_batch.aux,
        )

    def _build_population_condition(self, features: dict[str, jax.Array]) -> torch.Tensor:
        torch_features = {
            key: self._jax_to_torch(value)
            for key, value in features.items()
            if key
            in {
                "ego_state",
                "goal_xy",
                "remaining_timesteps",
                "other_states",
                "other_valid",
                "map_features",
                "map_valid",
                "traffic_light_features",
                "traffic_light_valid",
            }
        }

        cond_bf = self.policy.compute_condition(torch_features)
        if self.population_size == 1:
            return cond_bf
        return cond_bf.repeat_interleave(self.population_size, dim=0)

    def _sample_population(self, cond_bkf: torch.Tensor) -> np.ndarray:
        samples_btd = self.policy.sample_from_condition(cond_bkf)
        batch_size = int(samples_btd.shape[0]) // self.population_size
        horizon = int(samples_btd.shape[1])
        target_dim = int(samples_btd.shape[2])
        return (
            samples_btd.reshape(batch_size, self.population_size, horizon, target_dim)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _postprocess_population(
        self,
        proposals_bktd: jax.Array,
        pre_batch: PreprocessBatch,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_size, population_size, horizon, target_dim = proposals_bktd.shape
        flat = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
        repeated_aux = self._repeat_dict_leading_axis(pre_batch.aux, repeats=population_size)
        post = postprocess_predictions(flat, repeated_aux, self.preprocess_cfg)

        world_flat = post["trajectory_world_world_dt"]
        world_steps = int(world_flat.shape[1])
        world_dim = int(world_flat.shape[2])
        world_bkt5 = world_flat.reshape(batch_size, population_size, world_steps, world_dim)
        world_t_seconds_bt = jnp.asarray(pre_batch.aux["world_t_seconds"], dtype=jnp.float32)
        world_t_valid_bt = jnp.asarray(pre_batch.aux["world_t_valid"])
        return world_bkt5, world_t_seconds_bt, world_t_valid_bt

    def _jax_to_torch(self, array: jax.Array) -> torch.Tensor:
        np_array = np.asarray(array)
        tensor = torch.from_numpy(np_array)
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor.to(self.device)

    @staticmethod
    def _gather_population(population_bk: jax.Array, indices_bm: jax.Array) -> jax.Array:
        gather_indices = jnp.asarray(indices_bm, dtype=jnp.int32)
        trailing_shape = population_bk.shape[2:]
        reshape_shape = gather_indices.shape + (1,) * len(trailing_shape)
        broadcast_shape = gather_indices.shape + trailing_shape
        gather_indices = jnp.broadcast_to(gather_indices.reshape(reshape_shape), broadcast_shape)
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
