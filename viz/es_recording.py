"""Diffusion-ES planner that records every ES iteration, plus a closed-loop batch runner.

``RecordingDiffusionESPlanner.plan_trajectory`` follows the exact sampling / RNG path of
``DiffusionESPlanner.plan_trajectory``; ``simulate_batch`` and ``evaluate_batch`` mirror
``simulation/planning_utils.predict_planner_trajectories_with_periodic_replan`` and the
success logic of ``simulation/runner.py``, so a batch re-run reproduces the original run.
"""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from data.preprocess import preprocess_simulator_state
from planner.abstract_planner import PlannerResult
from planner.diffusion_es_planner import DiffusionESPlanner
from scores import helpers as scorer_helpers


def _instrument_scorer(scorer) -> None:
    orig_coll = scorer.compute_collision_score
    orig_off = scorer.compute_offroad_score_v2

    def coll(*a, **k):
        out = orig_coll(*a, **k)
        scorer._rec["collision"] = np.asarray(out, dtype=np.float32)
        return out

    def off(ego_trajectories, *a, **k):
        out = orig_off(ego_trajectories, *a, **k)
        scorer._rec["offroad"] = np.asarray(out, dtype=np.float32)
        scorer._rec["horizon"] = int(np.asarray(ego_trajectories).shape[1])
        return out

    scorer.compute_collision_score = coll
    scorer.compute_offroad_score_v2 = off
    scorer._orig_offroad = orig_off
    scorer._rec = {}


class RecordingDiffusionESPlanner(DiffusionESPlanner):
    """DiffusionESPlanner that stores population / verdicts / elites of every ES iteration."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for sc in self.scorers:
            _instrument_scorer(sc)
        self.last_record: dict[str, Any] | None = None

    def set_forecast(self, mode: str, world_model_forecaster: Any = None) -> None:
        mode = str(mode).strip().lower()
        self.other_agent_forecast = mode
        self.world_model_forecaster = world_model_forecaster
        for sc in self.scorers:
            sc.other_agent_forecast = mode
            sc.world_model_forecaster = world_model_forecaster

    def _score_all(self, world_bkt5, sim_state, timestep):
        scores, coll, off, horizon = [], [], [], []
        for w in range(world_bkt5.shape[0]):
            sc = self.scorers[w]
            sc._rec = {}
            s = sc.compute_score(
                trajectories=world_bkt5[w, :, :],
                sim_state=sim_state,
                timestep=timestep,
                world_idx=w,
            )
            scores.append(s)
            k = int(s.shape[0])
            coll.append(sc._rec.get("collision", np.ones((k,), np.float32)) < 0.5)
            off.append(sc._rec.get("offroad", np.ones((k,), np.float32)) < 0.5)
            horizon.append(sc._rec.get("horizon", int(world_bkt5.shape[2])))
        return jnp.stack(scores), np.stack(coll), np.stack(off), np.asarray(horizon)

    def per_step_offroad(self, plan_bt5: np.ndarray, sim_state, timestep: int) -> np.ndarray:
        """Scorer's offroad verdict for each timestep of a plan, [B, H]."""
        out = []
        t = int(timestep)
        for w in range(plan_bt5.shape[0]):
            sc = self.scorers[w]
            ego = int(np.argmax(np.asarray(sim_state.object_metadata.is_sdc[w]).astype(bool)))
            length = float(np.asarray(sim_state.log_trajectory.length[w, ego, t]))
            width = float(np.asarray(sim_state.log_trajectory.width[w, ego, t]))
            plan = np.asarray(plan_bt5[w], dtype=np.float32)
            n = plan.shape[0]
            ego_h15 = np.stack(
                [plan[:, 0], plan[:, 1], plan[:, 2], np.full((n,), length, np.float32), np.full((n,), width, np.float32)],
                axis=-1,
            )[:, None, :]
            flags = sc._orig_offroad(ego_h15, sim_state=sim_state, timestep=t, world_idx=w)
            out.append(np.asarray(flags, dtype=np.float32) < 0.5)
        return np.stack(out)

    def plan_trajectory(
        self,
        sim_state,
        goal,
        *,
        rng,
        instruction=None,
        instruction_mask=None,
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
        )
        features = pre_batch.features
        if instruction is None:
            instruction = jnp.zeros((self.num_worlds, self.preprocess_cfg.inst_dim), dtype=jnp.float32)
            instruction_mask = jnp.zeros((self.num_worlds,), dtype=jnp.float32)
        else:
            raise NotImplementedError("Conditioning on instruction is not yet implemented.")
        features["inst_features"] = instruction
        features["inst_valid"] = instruction_mask
        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
        features["subgoal_xy"] = jnp.zeros((self.num_worlds, 2), dtype=jnp.float32)
        features["subgoal_valid"] = jnp.zeros((self.num_worlds,), dtype=jnp.float32)
        cond_bf, inst_cond_bf = self._compute_condition_jit(pre_batch.features)

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
        current_scores_bk, cur_coll, cur_off, horizon = self._score_all(current_world_bkt5, sim_state, timestep)
        stage = {
            "traj": np.asarray(current_world_bkt5[..., :3], dtype=np.float32),
            "score": np.asarray(current_scores_bk, dtype=np.float32),
            "coll": cur_coll,
            "off": cur_off,
            "n_carried": 0,
        }
        stages: list[dict[str, Any]] = []
        for iteration_index in range(self.num_iterations + 1):
            elite_indices_bk, elite_scores_bk = self._select_topk_per_scenario(
                current_scores_bk,
                top_k=self.elite_size,
            )
            stage["elite_idx"] = np.asarray(elite_indices_bk, dtype=np.int32)
            stages.append(stage)
            if iteration_index == self.num_iterations:
                break

            elite_norm_betd = self._gather_population(current_norm_bktd, elite_indices_bk)
            elite_world_bet5, _, _ = self._postprocess_population(elite_norm_betd, pre_batch)
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
            current_world_bkt5, _, _ = self._postprocess_population(current_norm_bktd, pre_batch)
            new_scores_bk, new_coll, new_off, _ = self._score_all(current_world_bkt5, sim_state, timestep)
            elite_np = np.asarray(elite_indices_bk)
            elite_coll = np.take_along_axis(cur_coll, elite_np, axis=1)
            elite_off = np.take_along_axis(cur_off, elite_np, axis=1)
            current_norm_bktd = jnp.concat([elite_norm_betd, current_norm_bktd], axis=1)
            current_scores_bk = jnp.concat([elite_scores_bk, new_scores_bk], axis=1)
            current_world_bkt5 = jnp.concat([elite_world_bet5, current_world_bkt5], axis=1)
            cur_coll = np.concatenate([elite_coll, new_coll], axis=1)
            cur_off = np.concatenate([elite_off, new_off], axis=1)
            stage = {
                "traj": np.asarray(current_world_bkt5[..., :3], dtype=np.float32),
                "score": np.asarray(current_scores_bk, dtype=np.float32),
                "coll": cur_coll,
                "off": cur_off,
                "n_carried": int(self.elite_size),
            }

        best_indices_b1, _ = self._select_topk_per_scenario(current_scores_bk, top_k=1)
        best_norm_bt1d = self._gather_population(current_norm_bktd, best_indices_b1)
        best_world_bt15 = self._gather_population(current_world_bkt5, best_indices_b1)
        self.last_record = {
            "stages": stages,
            "best_idx": np.asarray(best_indices_b1, dtype=np.int32)[:, 0],
            "horizon_scored": horizon,
        }
        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=dict(pre_batch.aux),
        )


def build_recording_planner(
    *,
    diffusion_ckpt_path: str,
    seed: int = 0,
    population_size: int = 64,
    elite_size: int = 8,
    num_iterations: int = 4,
    resample_timesteps: int = 5,
    num_worlds: int = 5,
    other_agent_forecast: str = "gt",
    world_model_forecaster: Any = None,
) -> RecordingDiffusionESPlanner:
    from flax import nnx

    from train_diffusion.utils.infer import load_model_for_inference

    inference = load_model_for_inference(
        checkpoint_path=diffusion_ckpt_path,
        use_ema=True,
        metadata_path=None,
        seed=int(seed),
        skip_checkpoint_load=False,
    )
    policy = nnx.merge(inference.graphdef, inference.params_state, inference.nonparam_state)
    return RecordingDiffusionESPlanner(
        policy=policy,
        preprocess_cfg=inference.preprocess_cfg,
        population_size=int(population_size),
        resample_timesteps=int(resample_timesteps),
        elite_size=int(elite_size),
        num_iterations=int(num_iterations),
        num_worlds=int(num_worlds),
        metrics=["collision", "offroad"],
        other_agent_forecast=str(other_agent_forecast),
        world_model_forecaster=world_model_forecaster,
    )


def load_deterministic_world_model(ckpt: str, *, device: str = "auto", refresh_steps: int = 10):
    """Production (deterministic) WM forecaster, as used by the es_diffusion_wm* runs."""
    from simulation.run_simulation import _load_world_model_forecaster

    return _load_world_model_forecaster(
        argparse.Namespace(
            other_agent_forecast="wm",
            world_model_ckpt=ckpt,
            world_model_device=device,
            wm_backend="production",
            wm_refresh_steps=int(refresh_steps),
            num_collision_worlds=1,
            collision_reduce="mean",
            collision_sample_seed=0,
        )
    )


def load_batch(tfrecord_path: str, indices: list[int], *, max_num_objects: int = 128):
    """Load one batch of scenarios exactly like simulation/runner.py."""
    import dataclasses

    from waymax import config as waymax_config

    from data.scenario_loader import load_scenario_state_batch_fast

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord_path),
        max_num_objects=int(max_num_objects),
        batch_dims=(len(indices),),
        shuffle_seed=0,
    )
    sim_state, _, scenario_ids = load_scenario_state_batch_fast(ds_cfg, indices, return_scenario_ids=True)
    return sim_state, scenario_ids


def attach_lane_graphs(planner, sim_state, tfrecord_path: str, indices: list[int], lane_graph_dir: str) -> None:
    from lane_graph.lane_graph_utils import lane_graph_zip_path_for_tfrecord
    from scores.lane_graph_utils import LaneGraphShardStore

    store = LaneGraphShardStore(lane_graph_zip_path_for_tfrecord(str(tfrecord_path), lane_graph_dir).as_posix())
    for w, idx in enumerate(indices):
        planner.scorers[w].set_lane_graph(store.get_by_record_index(int(idx)))
        scorer_helpers.update_lane_points(planner.scorers[w], sim_state, w)


def simulate_batch(
    planner: RecordingDiffusionESPlanner,
    sim_state,
    goal_xy_b2,
    *,
    start_timestep: int = 10,
    replan_interval_steps: int = 10,
    seed: int = 0,
    record_forecasts: bool = False,
) -> tuple[Any, dict[int, dict[str, Any]]]:
    """Closed-loop periodic replanning; returns the ego-replaced state and per-replan records.

    With ``record_forecasts`` each record also holds the other-agent forecast the scorer
    used at that replan (``forecast_traj`` [B, N, H, 5], ``forecast_mask`` [B, N, H]).
    """
    from simulation.planning_utils import apply_ego_replacements_to_expanded_state

    num_steps = int(sim_state.log_trajectory.x.shape[-1])
    num_worlds = int(sim_state.log_trajectory.x.shape[0])
    plan_rng = jax.random.PRNGKey(int(seed))
    replaced = sim_state
    records: dict[int, dict[str, Any]] = {}
    for step in range(int(start_timestep), num_steps - 1, int(replan_interval_steps)):
        plan_rng, key_plan = jax.random.split(plan_rng)
        res = planner.plan_trajectory(
            replaced,
            goal_xy_b2,
            rng=key_plan,
            timestep=step,
            mask_goal=False,
            instruction_texts=None,
            use_subgoal=False,
        )
        plan = np.asarray(res.trajectory_world_bt5, dtype=np.float32)
        rec = planner.last_record
        rec["chosen"] = plan
        rec["chosen_offroad_steps"] = planner.per_step_offroad(plan, replaced, step)
        if record_forecasts:
            trajs, masks = [], []
            for w in range(num_worlds):
                traj, mask, _, _ = scorer_helpers.forecast_other_agents(
                    replaced,
                    timestep=step,
                    traj_length=int(plan.shape[1]),
                    world_idx=w,
                    mode=planner.other_agent_forecast,
                    world_model_forecaster=planner.world_model_forecaster,
                )
                trajs.append(np.asarray(traj, dtype=np.float32))
                masks.append(np.asarray(mask).astype(bool))
            h = min(t.shape[1] for t in trajs)
            rec["forecast_traj"] = np.stack([t[:, :h] for t in trajs])
            rec["forecast_mask"] = np.stack([m[:, :h] for m in masks])
        records[step] = rec
        replaced = apply_ego_replacements_to_expanded_state(replaced, start_t_b=step, traj_bt5=plan)
    return replaced, records


def _first_true(arr: np.ndarray, offset: int) -> int:
    idx = np.flatnonzero(np.asarray(arr) > 0)
    return int(idx[0]) + int(offset) if idx.size else -1


def evaluate_batch(replaced, goal_xy_b2, *, start_timestep: int = 10, seed: int = 0) -> list[dict[str, Any]]:
    """Per-world overlap / offroad / goal exactly as simulation/runner.py, plus first-event timesteps."""
    from simulation.evaluation_utils import check_goal_reaching, rollout_predicted_trajectories_with_metrics

    start = int(start_timestep)
    rollout = rollout_predicted_trajectories_with_metrics(replaced, rng_key=jax.random.PRNGKey(int(seed)))
    goal_eval = check_goal_reaching(replaced, goal_xy_b2, start_timestep=start)
    reached = np.asarray(goal_eval["reached"])
    reached_step = np.asarray(goal_eval["reached_step"])
    ov_all = np.asarray(rollout["metric_timeseries"]["overlap"])
    off_all = np.asarray(rollout["metric_timeseries"]["offroad"])
    is_sdc = np.asarray(replaced.object_metadata.is_sdc).astype(bool)
    xy = np.asarray(replaced.log_trajectory.xy, dtype=np.float32)
    goals = np.asarray(goal_xy_b2, dtype=np.float32)

    out = []
    for w in range(ov_all.shape[1]):
        ov_ts = ov_all[:, w].astype(np.float32)
        off_ts = off_all[:, w].astype(np.float32)
        ep_len = int(reached_step[w]) + 1 if bool(reached[w]) else ov_ts[start:].shape[0]
        overlap = bool(ov_ts[start:][:ep_len].sum() > 0)
        offroad = bool(off_ts[start:][:ep_len].sum() > 0)
        ego = int(np.argmax(is_sdc[w]))
        dists = np.linalg.norm(xy[w, ego, start:] - goals[w][None, :], axis=-1)
        out.append(
            {
                "overlap": overlap,
                "offroad": offroad,
                "goal_reached": bool(reached[w]),
                "success": bool(reached[w]) and not overlap and not offroad,
                "reached_timestep": int(reached_step[w]),
                "first_overlap_t": _first_true(ov_ts[start:][:ep_len], start),
                "first_offroad_t": _first_true(off_ts[start:][:ep_len], start),
                "min_goal_distance_m": float(dists.min()),
                "final_goal_distance_m": float(dists[-1]),
                "overlap_ts": ov_ts,
                "offroad_ts": off_ts,
            }
        )
    return out
