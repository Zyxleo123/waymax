"""Learned Diffusion-ES reward: a trained WM-conditioned discriminator replaces the handcrafted scorer.

`DiffusionESPlanner.learned_scorer` (None by default = handcrafted reward, unchanged) is set to a
`LearnedESScorer`; the planner then scores the initial population, every ES iteration's
resampled population, and therefore every elite and final selection, with the discriminator only.

The discriminator always gets the frozen multi-future Gaussian WM prediction (one forward per
replan, shared by the 5 scoring calls) and road-edge tokens, independent of the run's
`--other_agent_forecast`:
  wm (multi-future, same checkpoint)  the planner's own forecaster object is reused; with the
                                      handcrafted scorer bypassed it runs no other forward
  gt / cv / other WM                  a separate frozen copy of the multi-future WM is loaded

Per replan it checks that the population's anchor is the executed ego pose (coordinate
alignment) and counts scoring calls; `<output_dir>/discriminator_stats.json` holds the call
counts, timings (WM forward + inputs, scene encoding, candidate scoring) and alignment errors.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from .deploy import DiscriminatorScorer

DEFAULT_CHECKPOINT = ("/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/safety_progress/"
                      "safety_progress_full_gaussian_wm_20261013_090539_81674/best.pt")
ANCHOR_TOL_M = 0.05
ANCHOR_TOL_RAD = 0.01


class LearnedESScorer:
    def __init__(self, checkpoint: str, *, wm_checkpoint: str, planner_forecaster=None,
                 planner_wm_checkpoint: str | None = None, device: str = "auto",
                 expected_calls_per_replan: int = 0, stats_path: str | None = None) -> None:
        reuse = (planner_forecaster is not None and planner_wm_checkpoint is not None
                 and os.path.realpath(planner_wm_checkpoint) == os.path.realpath(wm_checkpoint)
                 and hasattr(planner_forecaster, "_prepare_features"))
        self.scorer = DiscriminatorScorer(checkpoint, device=device, wm_checkpoint=wm_checkpoint,
                                          forecaster=planner_forecaster if reuse else None)
        self.wm_source = "planner_forecaster" if reuse else "separate_frozen_copy"
        self.checkpoint, self.wm_checkpoint = checkpoint, wm_checkpoint
        self.expected_calls = int(expected_calls_per_replan)
        self.stats_path = stats_path
        self._replan_key = None
        self._calls = 0
        self.calls_per_replan: Counter = Counter()
        self.replans = 0
        self.t_context: list[float] = []
        self.t_score: list[float] = []
        self.t_replan: list[float] = []
        self.t_replan_span: list[float] = []
        self._in_call = 0.0
        self.anchor_err_m = 0.0
        self.anchor_err_rad = 0.0
        self.pool_sizes: Counter = Counter()

    def describe(self) -> dict:
        return {"checkpoint": self.checkpoint, "variant": self.scorer.variant, "method": self.scorer.method,
                "wm_checkpoint": self.wm_checkpoint, "wm_source": self.wm_source,
                "device": str(self.scorer.device), "expected_calls_per_replan": self.expected_calls}

    def _finish_replan(self) -> None:
        if self._replan_key is None:
            return
        self.calls_per_replan[self._calls] += 1
        self.replans += 1
        if self.expected_calls and self._calls != self.expected_calls:
            raise RuntimeError(f"discriminator scored {self._calls} pools in replan {self._replan_key}, "
                               f"expected {self.expected_calls}")
        if self.stats_path and self.replans % 50 == 0:
            self.write_stats()

    def _check_anchor(self, sim_state, timestep: int, pop: np.ndarray, valid: np.ndarray) -> None:
        log = sim_state.log_trajectory
        sdc = np.argmax(np.asarray(sim_state.object_metadata.is_sdc).astype(bool), axis=1)
        w = np.arange(len(sdc))
        ego = np.stack([np.asarray(log.x)[w, sdc, timestep], np.asarray(log.y)[w, sdc, timestep],
                        np.asarray(log.yaw)[w, sdc, timestep]], -1)
        d = pop[:, :, 0, :2] - ego[:, None, :2]
        dyaw = np.abs((pop[:, :, 0, 2] - ego[:, None, 2] + np.pi) % (2 * np.pi) - np.pi)
        em = float(np.linalg.norm(d, axis=-1)[valid].max(initial=0.0))
        er = float(dyaw[valid].max(initial=0.0))
        self.anchor_err_m, self.anchor_err_rad = max(self.anchor_err_m, em), max(self.anchor_err_rad, er)
        if em > ANCHOR_TOL_M or er > ANCHOR_TOL_RAD:
            raise RuntimeError(f"population anchor is {em:.3f} m / {er:.4f} rad from the executed ego at t={timestep}")

    def __call__(self, sim_state, timestep: int, goal, world_bkt5) -> jax.Array:
        key = (id(sim_state), int(timestep))
        t0 = time.perf_counter()
        if key != self._replan_key:
            self._finish_replan()
            self._replan_key, self._calls = key, 0
            st = jax.device_get(sim_state)
            self.scorer.contexts(sim_state, int(timestep), np.asarray(goal))
            self.t_context.append(time.perf_counter() - t0)
            self._state_np = st
            self._replan_t0 = t0
            self._in_call = 0.0
        pop = np.asarray(jax.device_get(world_bkt5), dtype=np.float64)
        _, _, _, valid = self.scorer.contexts(sim_state, int(timestep), np.asarray(goal))
        if self._calls == 0:
            self._check_anchor(self._state_np, int(timestep), pop, valid)
        t1 = time.perf_counter()
        scores = self.scorer.score(sim_state, int(timestep), goal, pop)
        now = time.perf_counter()
        self.t_score.append(now - t1)
        self._calls += 1
        self.pool_sizes[pop.shape[1]] += 1
        self._in_call += now - t0
        if self.expected_calls and self._calls == self.expected_calls:
            self.t_replan.append(self._in_call)
            self.t_replan_span.append(now - self._replan_t0)
        return jnp.asarray(scores, dtype=jnp.float32)

    def write_stats(self) -> dict:
        ms = lambda x: {"mean": 1000 * float(np.mean(x)), "median": 1000 * float(np.median(x)),  # noqa: E731
                        "p90": 1000 * float(np.percentile(x, 90)), "n": len(x)} if x else None
        out = {**self.describe(), "replans_completed": self.replans,
               "scoring_calls_per_replan": {str(k): v for k, v in sorted(self.calls_per_replan.items())},
               "pool_sizes": {str(k): v for k, v in sorted(self.pool_sizes.items())},
               "context_ms_per_replan": ms(self.t_context),
               "wm_forward_ms_per_replan": ms(self.scorer.wm_forward_s),
               "score_ms_per_call": ms(self.t_score),
               "discriminator_ms_per_replan": ms(self.t_replan),
               "es_loop_span_ms_per_replan": ms(self.t_replan_span),
               "max_anchor_err_m": self.anchor_err_m, "max_anchor_err_rad": self.anchor_err_rad}
        if self.stats_path:
            tmp = self.stats_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(out, f, indent=1)
            os.replace(tmp, self.stats_path)
        return out

    def close(self) -> dict:
        self._finish_replan()
        self._replan_key = None
        return self.write_stats()
