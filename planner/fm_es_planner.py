from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scores.scorer_lane_graph import Scorer
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessConfig
from model.diffusion.fm_policy import FlowMatchingPolicy
from planner.abstract_planner import PlannerResult
from planner.fm_planner import FMPlanner


class FMESPlanner(FMPlanner):
    """Flow-matching evolutionary search planner over batched Waymax scenarios."""

    def __init__(
        self,
        policy: FlowMatchingPolicy,
        preprocess_cfg: PreprocessConfig,
        population_size: int = 64,
        num_worlds: int = 1,
        metrics: list[str] | None = None,
        weights: dict[str, float] | None = None,
        elite_size: int = 8,
        num_iterations: int = 4,
        resample_t_start: float = 0.2,
        resample_num_steps: int | None = None,
        num_steps: int | None = None,
        solver: str = "euler",
        other_agent_forecast: str = "cv",
        world_model_forecaster: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy=policy,
            preprocess_cfg=preprocess_cfg,
            population_size=population_size,
            num_worlds=num_worlds,
            num_steps=num_steps,
            solver=solver,
            **kwargs,
        )
        self.elite_size = int(elite_size)
        self.num_iterations = int(num_iterations)
        self.resample_t_start = float(resample_t_start)
        self.resample_num_steps = resample_num_steps
        self.other_agent_forecast = str(other_agent_forecast).strip().lower()
        self.world_model_forecaster = world_model_forecaster
        if metrics is None:
            metrics = ["collision", "offroad"]
        self.scorers = [
            Scorer(
                metrics=metrics,
                weights=weights,
                other_agent_forecast=self.other_agent_forecast,
                world_model_forecaster=self.world_model_forecaster,
            )
            for _ in range(num_worlds)
        ]

        self._resample_population_jit = jax.jit(
            lambda *, cond_bf, inst_cond_bf, inst_cond_mask_bf, proposals_bktd, rng: self._resample_population(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=inst_cond_mask_bf,
                proposals_bktd=proposals_bktd,
                rng=rng,
            ),
        )

    def add_metric(self, metric_name: str, target: Any = None, world_idx: int = 0, weight: float = 1.0):
        self.scorers[world_idx].add_metric(metric_name, target=target, weight=weight)

    def plan_trajectory(
        self,
        sim_state,
        goal,
        *,
        rng: jax.Array,
        instruction: jnp.ndarray = None,
        instruction_mask: jnp.ndarray = None,
        timestep: int = 0,
        mask_goal: bool = False,
        **kwargs: Any,
    ) -> PlannerResult:
        del kwargs

        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            anchor_step_override=timestep,
            goal_step_override=90,
            goal_xy_override=goal,
        )

        features = pre_batch.features
        if instruction is None:
            instruction = jnp.zeros(
                (self.num_worlds, self.preprocess_cfg.inst_dim),
                dtype=jnp.float32,
            )
            instruction_mask = jnp.zeros((self.num_worlds,), dtype=jnp.float32)
        elif instruction_mask is None:
            instruction_mask = jnp.ones((instruction.shape[0],), dtype=jnp.float32)

        features["inst_features"] = instruction
        features["inst_valid"] = instruction_mask

        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])

        features["subgoal_xy"] = jnp.zeros((self.num_worlds, 2), dtype=jnp.float32)
        features["subgoal_valid"] = jnp.zeros((self.num_worlds,), dtype=jnp.float32)

        cond_bf, inst_cond_bf = self._compute_condition_jit(features)

        current_norm_bktd = self._sample_population_jit(
            cond_bf=cond_bf,
            inst_cond_bf=inst_cond_bf,
            inst_cond_mask_bf=instruction_mask,
            rng=key_sample,
        )
        current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
            current_norm_bktd,
            pre_batch,
        )
        current_scores_bk = []
        for world_idx in range(current_world_bkt5.shape[0]):
            current_scores_bk.append(
                self.scorers[world_idx].compute_score(
                    trajectories=current_world_bkt5[world_idx, :, :],
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx,
                )
            )
        current_scores_bk = jnp.stack(current_scores_bk)

        for iteration_index in range(self.num_iterations + 1):
            elite_indices_bk, elite_scores_bk = self._select_topk_per_scenario(
                current_scores_bk,
                top_k=self.elite_size,
            )
            if iteration_index == self.num_iterations:
                break

            elite_norm_betd = self._gather_population(
                current_norm_bktd,
                elite_indices_bk,
            )
            elite_world_bet5, _, _ = self._postprocess_population(
                elite_norm_betd,
                pre_batch,
            )
            replicated_norm_bktd = self._replicate_elites(
                elite_norm_betd,
                population_size=self.population_size,
            )

            rng, key_resample = jax.random.split(rng)
            current_norm_bktd = self._resample_population_jit(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=instruction_mask,
                proposals_bktd=replicated_norm_bktd,
                rng=key_resample,
            )
            current_world_bkt5, _, _ = self._postprocess_population(
                current_norm_bktd,
                pre_batch,
            )
            current_scores_bk = []
            for world_idx in range(current_world_bkt5.shape[0]):
                current_scores_k = self.scorers[world_idx].compute_score(
                    trajectories=current_world_bkt5[world_idx, :, :],
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx,
                )
                current_scores_bk.append(current_scores_k)
            current_scores_bk = jnp.stack(current_scores_bk)
            current_norm_bktd = jnp.concat([elite_norm_betd, current_norm_bktd], axis=1)
            current_scores_bk = jnp.concat([elite_scores_bk, current_scores_bk], axis=1)
            current_world_bkt5 = jnp.concat([elite_world_bet5, current_world_bkt5], axis=1)

        best_indices_b1, best_scores_b1 = self._select_topk_per_scenario(
            current_scores_bk,
            top_k=1,
        )
        best_norm_bt1d = self._gather_population(current_norm_bktd, best_indices_b1)
        best_world_bt15 = self._gather_population(current_world_bkt5, best_indices_b1)

        elite_indices_bk, elite_scores_bk = self._select_topk_per_scenario(
            current_scores_bk,
            top_k=self.elite_size,
        )
        elite_world_bet5 = self._gather_population(current_world_bkt5, elite_indices_bk)
        # Full scored pool after the last ES concat (typically elite_size + population_size).
        pop_order_bk = jnp.argsort(-current_scores_bk, axis=-1)
        pop_world_bkt5 = jnp.take_along_axis(
            current_world_bkt5,
            pop_order_bk[:, :, None, None],
            axis=1,
        )
        pop_scores_bk = jnp.take_along_axis(current_scores_bk, pop_order_bk, axis=-1)
        aux = dict(pre_batch.aux)
        aux["es_elite_world_bet5"] = elite_world_bet5
        aux["es_elite_scores_be"] = elite_scores_bk
        aux["es_pop_world_bkt5"] = pop_world_bkt5
        aux["es_pop_scores_bk"] = pop_scores_bk
        aux["es_best_score_b"] = jnp.squeeze(best_scores_b1, axis=1)

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=aux,
        )

    def _resample_population(
        self,
        *,
        cond_bf: jax.Array,
        inst_cond_bf: jax.Array,
        inst_cond_mask_bf: jax.Array,
        proposals_bktd: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        batch_size, population_size, horizon, target_dim = proposals_bktd.shape
        cond_bkf = jnp.repeat(cond_bf, repeats=population_size, axis=0)
        inst_cond_bkf = jnp.repeat(inst_cond_bf, repeats=population_size, axis=0)
        inst_cond_mask_bkf = jnp.repeat(inst_cond_mask_bf, repeats=population_size, axis=0)
        flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
        samples_btd = self.policy.resample_from_condition(
            cond_bkf,
            inst_cond_bkf,
            inst_cond_mask_bkf,
            flat_proposals,
            rng=rng,
            t_start=self.resample_t_start,
            num_steps=self.resample_num_steps,
            solver=self.solver,
        )
        return samples_btd.reshape(batch_size, population_size, horizon, target_dim)

    @staticmethod
    def _select_topk_per_scenario(
        scores_bk: jax.Array,
        *,
        top_k: int,
    ) -> tuple[jax.Array, jax.Array]:
        top_scores_bk, indices_bk = jax.lax.top_k(scores_bk, top_k)
        return (
            jnp.asarray(indices_bk, dtype=jnp.int32),
            jnp.asarray(top_scores_bk, dtype=jnp.float32),
        )

    @staticmethod
    def _replicate_elites(elite_betd: jax.Array, *, population_size: int) -> jax.Array:
        batch_size, elite_size, horizon, target_dim = elite_betd.shape
        del batch_size, horizon, target_dim

        base = population_size // elite_size
        remainder = population_size % elite_size
        selector = np.concatenate(
            [
                np.full((base + (1 if elite_idx < remainder else 0),), elite_idx, dtype=np.int32)
                for elite_idx in range(elite_size)
            ],
            axis=0,
        )
        return jnp.take(elite_betd, jnp.asarray(selector, dtype=jnp.int32), axis=1)
