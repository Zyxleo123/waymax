from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scores.scorer_lane_graph import Scorer
from data.postprocess import postprocess_predictions
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessBatch, PreprocessConfig

from model.diffusion.diffusion_policy import DiffusionPolicy
from planner.abstract_planner import AbstractPlanner, PlannerResult
from planner.diffusion_planner import DiffusionPlanner

class DiffusionESPlanner(DiffusionPlanner):
    def __init__(
        self,
        policy: DiffusionPolicy,
        preprocess_cfg: PreprocessConfig,
        population_size: int = 64,
        num_worlds: int = 1,
        metrics: list[str] = None,
        weights: dict[str, float] = None,
        resample_timesteps: int = 3,
        elite_size: int = 4,
        num_iterations: int = 3,
        other_agent_forecast: str = "cv",
        world_model_forecaster: Any = None,
        offroad_mode: str = "binary",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy=policy,
            preprocess_cfg=preprocess_cfg,
            population_size=population_size,
            num_worlds=num_worlds,
        )
        self.resample_timesteps = resample_timesteps
        self.elite_size = elite_size
        self.num_iterations = num_iterations
        self.other_agent_forecast = str(other_agent_forecast).strip().lower()
        self.world_model_forecaster = world_model_forecaster
        # Optional es_discriminator.es_candidate_recorder.ESCandidateRecorder; read-only.
        self.candidate_recorder = None
        # Optional es_discriminator.real_es.closed_loop.LearnedESScorer; when set it replaces the
        # handcrafted scorers for every population (hence every elite and the final selection).
        self.learned_scorer = None
        # Optional (sim_state, timestep) -> (xy [B,M,T,2], yaw [B,M,T], ...) world-frame ego trajectories at the
        # policy's model steps, appended to the initial population (anchor_wm.es_debug.AnchorEgoSeeder).
        self.ego_seed_fn = None
        # Optional (sim_state, timestep, goal, info: dict) callback after every replan (anchor_wm.es_debug).
        self.debug_dump_fn = None
        self.scorers = [
            Scorer(
                metrics=metrics,
                weights=weights,
                other_agent_forecast=self.other_agent_forecast,
                world_model_forecaster=self.world_model_forecaster,
                offroad_mode=offroad_mode,
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
        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            anchor_step_override=timestep,
            goal_step_override=90,
            goal_xy_override=goal,
            world_dt_method="median",
        )
        features = pre_batch.features
        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
        cond_bf, inst_cond_bf, instruction_mask = self._compute_condition(features, instruction)

        current_norm_bktd = self._sample_population_jit(
            cond_bf=cond_bf,
            inst_cond_bf=inst_cond_bf,
            inst_cond_mask_bf=instruction_mask,
            rng=key_sample,
        )
        num_seeds = 0
        if self.ego_seed_fn is not None:
            seed_xy, seed_yaw = self.ego_seed_fn(sim_state, timestep)[:2]
            seeds_bmtd = self._world_to_norm(seed_xy, seed_yaw, pre_batch.aux, current_norm_bktd.shape)
            num_seeds = int(seeds_bmtd.shape[1])
            current_norm_bktd = jnp.concatenate([current_norm_bktd, seeds_bmtd.astype(current_norm_bktd.dtype)], axis=1)
        # 1 = descends from an ego seed (through elite resampling), 0 = from the diffusion samples.
        origin_bk = np.zeros(current_norm_bktd.shape[:2], np.int8)
        origin_bk[:, origin_bk.shape[1] - num_seeds:] = 1
        current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
            current_norm_bktd,
            pre_batch,
        )
        if self.learned_scorer is not None:
            current_scores_bk = self.learned_scorer(sim_state, timestep, goal, current_world_bkt5)
        else:
            current_scores_bk = []
            for world_idx in range(current_world_bkt5.shape[0]):
                current_scores_bk.append(self.scorers[world_idx].compute_score(
                    trajectories=current_world_bkt5[world_idx, :, :],
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx
                ))
            current_scores_bk = jnp.stack(current_scores_bk)
        debug = None
        if self.debug_dump_fn is not None:
            debug = {"num_seeds": num_seeds, "init_world": np.asarray(current_world_bkt5),
                     "init_scores": np.asarray(current_scores_bk), "init_origin": origin_bk.copy(), "elite_seed_frac": []}
        recorder = self.candidate_recorder
        if recorder is not None and num_seeds:
            raise ValueError("ego seeding changes the initial population size; not with the candidate recorder")
        if recorder is not None:
            recorder.begin_replan(sim_state, int(timestep), goal, self.world_model_forecaster)
            recorder.record_stage(0, current_world_bkt5, self._stage_components(current_scores_bk), None)
        # print(f"Initial scores: {current_scores_bk} \n Shape: {current_scores_bk.shape} \n Mean: {jnp.mean(current_scores_bk)}, Std: {jnp.std(current_scores_bk)}")
        for iteration_index in range(self.num_iterations + 1):
            elite_indices_bk, elite_scores_bk = self._select_topk_per_scenario(
                current_scores_bk,
                top_k=self.elite_size,
            )
            if recorder is not None:
                recorder.record_elites(iteration_index, elite_indices_bk)
            elite_origin_be = np.take_along_axis(origin_bk, np.asarray(elite_indices_bk), axis=1)
            if debug is not None:
                debug["elite_seed_frac"].append(elite_origin_be.mean(1))
            if iteration_index == self.num_iterations:
                break
            origin_bk = np.concatenate(
                [elite_origin_be, elite_origin_be[:, self._elite_selector(self.elite_size, self.population_size)]], axis=1)

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
            if self.learned_scorer is not None:
                current_scores_bk = self.learned_scorer(sim_state, timestep, goal, current_world_bkt5)
            else:
                current_scores_bk = []
                for world_idx in range(current_world_bkt5.shape[0]):
                    current_scores_k = self.scorers[world_idx].compute_score(
                        trajectories=current_world_bkt5[world_idx, :, :],
                        sim_state=sim_state,
                        timestep=timestep,
                        world_idx=world_idx
                    )
                    current_scores_bk.append(current_scores_k)
                current_scores_bk = jnp.stack(current_scores_bk)
            if recorder is not None:
                recorder.record_stage(
                    iteration_index + 1,
                    current_world_bkt5,
                    self._stage_components(current_scores_bk),
                    self._elite_selector(self.elite_size, self.population_size),
                )
            current_norm_bktd = jnp.concat([elite_norm_betd, current_norm_bktd], axis=1)
            current_scores_bk = jnp.concat([elite_scores_bk, current_scores_bk], axis=1)
            current_world_bkt5 = jnp.concat([elite_world_bet5, current_world_bkt5], axis=1)

        # print(f"Final scores: {current_scores_bk} \n Shape: {current_scores_bk.shape} \n Mean: {jnp.mean(current_scores_bk)}, Std: {jnp.std(current_scores_bk)}")
        best_indices_b1, best_scores_b1 = self._select_topk_per_scenario(
            current_scores_bk,
            top_k=1,
        )
        best_norm_bt1d = self._gather_population(current_norm_bktd, best_indices_b1)
        best_world_bt15 = self._gather_population(current_world_bkt5, best_indices_b1)
        if recorder is not None:
            recorder.end_replan(best_indices_b1[:, 0])

        # these chnages are for plotting population only
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
        if debug is not None:
            best_idx = np.asarray(best_indices_b1)
            debug.update(final_world=np.asarray(pop_world_bkt5), final_scores=np.asarray(pop_scores_bk),
                         final_origin=np.take_along_axis(origin_bk, np.asarray(pop_order_bk), axis=1),
                         best_world=np.asarray(best_world_bt15[:, 0]),
                         best_origin=np.take_along_axis(origin_bk, best_idx, axis=1)[:, 0],
                         elite_seed_frac=np.stack(debug["elite_seed_frac"], 1))
            self.debug_dump_fn(sim_state, timestep, goal, debug)

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=aux,
        )

    def _world_to_norm(self, xy_bmt2, yaw_bmt, aux: dict, pop_shape) -> jax.Array:
        """World ego trajectories at the model steps -> the policy's normalized [x, y, vx, vy, yaw] (inverse of
        data.postprocess.ego_normalized_to_world); velocity by backward differences from the anchor pose."""
        cfg = self.preprocess_cfg
        horizon, target_dim = int(pop_shape[2]), int(pop_shape[3])
        if xy_bmt2.shape[2] != horizon or target_dim != 5:
            raise ValueError(f"ego seeds have {xy_bmt2.shape[2]} steps, policy {horizon} x {target_dim}")
        origin = np.asarray(aux["origin_xy"], np.float64)[:, None, None]                     # [B,1,1,2]
        heading = np.asarray(aux["anchor_yaw"], np.float64)[:, None, None]                   # [B,1,1]
        rot = lambda v: np.stack([v[..., 0] * np.cos(heading) + v[..., 1] * np.sin(heading),  # noqa: E731
                                  -v[..., 0] * np.sin(heading) + v[..., 1] * np.cos(heading)], -1)
        pos = rot(np.asarray(xy_bmt2, np.float64) - origin)
        prev = np.concatenate([np.zeros_like(pos[:, :, :1]), pos[:, :, :-1]], axis=2)
        vel = (pos - prev) / float(cfg.model_dt)
        yaw = np.angle(np.exp(1j * (np.asarray(yaw_bmt, np.float64) - heading)))
        out = np.concatenate([pos / cfg.ego_range, vel / cfg.max_velocity, (yaw / np.pi)[..., None]], -1)
        return jnp.asarray(out, jnp.float32)

    def _stage_components(self, scores_bk) -> list[dict]:
        """Per-world score components of the population just scored, for the candidate recorder: the
        handcrafted scorers' factors, or with a learned scorer its score as `total` (collision / offroad NaN)."""
        if self.learned_scorer is None:
            return [sc.last_components for sc in self.scorers]
        scores = np.asarray(jax.device_get(scores_bk), dtype=np.float32)
        nan = np.full(scores.shape[1], np.nan, dtype=np.float32)
        return [{"total": scores[w], "collision": nan, "offroad": nan} for w in range(scores.shape[0])]

    def _resample_population(
        self,
        *,
        cond_bf: jax.Array,
        inst_cond_bf: jax.Array | None,
        inst_cond_mask_bf: jax.Array | None,
        proposals_bktd: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        batch_size, population_size, horizon, target_dim = proposals_bktd.shape
        cond_bkf = jnp.repeat(cond_bf, repeats=population_size, axis=0)
        flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
        if inst_cond_bf is None:
            inst_args = ()
        else:
            inst_args = (jnp.repeat(inst_cond_bf, repeats=population_size, axis=0),
                         jnp.repeat(inst_cond_mask_bf, repeats=population_size, axis=0))
        samples_btd = self.policy.resample_from_condition(
            cond_bkf,
            *inst_args,
            flat_proposals,
            self.resample_timesteps,
            rng=rng,
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
    def _elite_selector(elite_size: int, population_size: int) -> np.ndarray:
        """Elite slot that each of the `population_size` resampled proposals starts from."""
        base = population_size // elite_size
        remainder = population_size % elite_size
        return np.concatenate(
            [
                np.full((base + (1 if elite_idx < remainder else 0),), elite_idx, dtype=np.int32)
                for elite_idx in range(elite_size)
            ],
            axis=0,
        )

    @staticmethod
    def _replicate_elites(elite_betd: jax.Array, *, population_size: int) -> jax.Array:
        batch_size, elite_size, horizon, target_dim = elite_betd.shape
        del batch_size, horizon, target_dim

        selector = DiffusionESPlanner._elite_selector(elite_size, population_size)
        return jnp.take(elite_betd, jnp.asarray(selector, dtype=jnp.int32), axis=1)