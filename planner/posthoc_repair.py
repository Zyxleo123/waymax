"""Post-hoc trajectory optimization of every ES candidate, before elites are chosen.

The repair is o_am from yixiz/waymax_rs (``rl/diffusion_ft/README_posthoc_opt.md``):
one loss per candidate, ``Safety(d, s) + Comfort(d) + Comfort(s)`` over a lateral
offset ``d_t`` and an along-path offset ``s_t`` on the next ``window`` world steps,
minimized by 32 steps of Hessian-preconditioned descent. A candidate already clear
of every margin has zero loss at the origin and comes back unchanged.

Here it runs inside Diffusion-ES on the whole population of every round: sample ->
postprocess -> REPAIR -> score -> elites. Elites are therefore chosen on the plans
that would actually be executed, and the executed plan is the best repaired one.
Resampling still starts from the elites' raw diffusion samples (the repair edits
world coordinates; the population lives in the model's normalized space), so the
search is over raw samples judged by their repaired versions.

The other cars come from the SAME forecast the ES scorer ranks with -- one predictor
per decision:

* ``cv``     -- ``scores.helpers.forecast_other_agents(mode="cv")``: the PDM-style
  constant velocity with its 0.8x-speed copies (the scorer's collision ensemble);
* ``gt``     -- the logged future (an oracle under log replay);
* ``wm``     -- the scheduled deterministic / Gaussian WM's collision boxes;
* ``wm_mf``  -- the multi-future WM's K mode means, each a pseudo-agent weighted by
  its mode probability, so the barrier is the per-agent expectation over modes
  (the additive analogue of ``expected_independent``).

Forecasts are requested at the scorer's own horizon so the WM caches are shared,
then sliced to steps 1..window.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.lateral_opt import LateralOptConfig, edit_world, make_lateral_optimizer
from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig
from scores import helpers as scorer_helpers

O_AM_CFG = "lon=1,w_lon_prox=0.05,lon_back_m=1000"
WORLD_STEP_S = 0.1          # one simulator step; the plan is executed one pose per step


class PosthocRepair:
    def __init__(self, *, other_agent_forecast: str, world_model_forecaster: Any = None,
                 window: int = 20, cfg: str = O_AM_CFG, map_batch: int = 8):
        self.forecast = str(other_agent_forecast).strip().lower()
        self.fc = world_model_forecaster
        self.window = int(window)
        self.cfg = LateralOptConfig.parse(cfg)
        if not self.cfg.lon:
            raise ValueError("the o_am repair optimizes lateral AND along-path offsets (lon=1)")
        self.mode_weighted = (self.forecast == "wm" and self.fc is not None
                              and scorer_helpers.uses_independent_mode_collision(self.fc))
        self.source = "wm_mf" if self.mode_weighted else self.forecast
        self.map_batch = int(map_batch)
        self._opt = make_lateral_optimizer(SafetyObjectiveConfig(), self.cfg, dt=WORLD_STEP_S,
                                           neighbor_pred="external")
        self._programs = {}
        self.stats = {"calls": 0, "candidates": 0, "moved": 0}

    # ------------------------------------------------------------------ #
    def neighbor_boxes(self, sim_state, timestep: int, traj_length: int, n: int):
        """``(box [B, n, M, 5] = x, y, length, width, yaw, valid [B, n, M], weight [B, n, M])``."""
        B = int(np.asarray(sim_state.object_metadata.is_sdc).shape[0])
        boxes, valids, weights = [], [], []
        for b in range(B):
            if self.mode_weighted:
                fm = self.fc.forecast_modes(sim_state, timestep=int(timestep),
                                            traj_length=int(traj_length), world_idx=b)
                traj = np.asarray(fm["traj_k"])                      # [N, K, H, 5]
                N, K = traj.shape[:2]
                traj = traj.reshape(N * K, *traj.shape[2:])
                mask = np.repeat(np.asarray(fm["masks"]), K, axis=0)  # [N*K, H]
                w = np.asarray(fm["mode_probs"], np.float32).reshape(N * K)
                w = np.broadcast_to(w[:, None], mask.shape)
            else:
                _, _, traj, mask = scorer_helpers.forecast_other_agents(
                    sim_state, timestep=int(timestep), traj_length=int(traj_length),
                    world_idx=b, mode=self.forecast, world_model_forecaster=self.fc)
                traj, mask = np.asarray(traj), np.asarray(mask)
                w = np.ones(mask.shape, np.float32)
            boxes.append(traj[:, 1:n + 1][..., [0, 1, 3, 4, 2]])     # -> x, y, len, wid, yaw
            valids.append(mask[:, 1:n + 1].astype(bool))
            weights.append(np.asarray(w, np.float32)[:, 1:n + 1])
        M = max(x.shape[0] for x in boxes)

        def pad(a, fill):
            out = np.full((M,) + a.shape[1:], fill, a.dtype)
            out[:a.shape[0]] = a
            return np.swapaxes(out, 0, 1)                            # [n, M, ...]
        return (jnp.asarray(np.stack([pad(x, 0.0) for x in boxes])),
                jnp.asarray(np.stack([pad(x, False) for x in valids])),
                jnp.asarray(np.stack([pad(x, 0.0) for x in weights])))

    def _program(self, n: int):
        if n not in self._programs:
            opt, cfg = self._opt, self.cfg

            def run(world_kbt5, state, t0, nb):
                B = world_kbt5.shape[1]
                act = jnp.ones((B,), jnp.float32)
                x0 = jnp.zeros((B, 2 * n), jnp.float32)

                def one(w):
                    x, _, _ = opt(w, x0, state, t0, act, n, nb)
                    moved = jnp.any(x != 0.0, axis=1)
                    ed = edit_world(w, x, n=n, dt=WORLD_STEP_S, lon=cfg.lon)
                    return jnp.where(moved[:, None, None], ed, w), moved
                return jax.lax.map(one, world_kbt5, batch_size=self.map_batch)
            self._programs[n] = jax.jit(run)
        return self._programs[n]

    def __call__(self, world_bkt5, sim_state, timestep: int):
        """Repaired population ``[B, K, T, 5]`` (index 0 of T is the current pose)."""
        B, K, T, _ = world_bkt5.shape
        horizon = int(np.asarray(sim_state.log_trajectory.x).shape[-1])
        n = int(min(self.window, T - 1, horizon - 1 - int(timestep)))
        if n <= 0:
            return world_bkt5
        nb = self.neighbor_boxes(sim_state, timestep, T, n)
        rep_kbt5, moved_kb = self._program(n)(jnp.swapaxes(world_bkt5, 0, 1), sim_state,
                                              jnp.asarray(timestep, jnp.int32), nb)
        self.stats["calls"] += 1
        self.stats["candidates"] += B * K
        self.stats["moved"] += int(np.asarray(moved_kb).sum())
        return jnp.swapaxes(rep_kbt5, 0, 1)
