"""NAVSIM v1 PDM score (PDMS) as the Diffusion-ES reward (`run_simulation.py --planner es --es_reward pdms`).

Each candidate is scored as NAVSIM scores a submitted trajectory (navsim/evaluate/pdm_score.py, v1):
  - its first 4 s (PROPOSAL_STEPS at 10 Hz) of box-centre poses, as rear-axle poses, are tracked from the current
    ego state by PDM's LQR tracker and kinematic bicycle model;
  - the simulated states are scored with PDMScorer against the constant-velocity observation, and paired with
    PDM-Closed's own plan for the same replan, which sets the progress normalisation:
        PDMS = NC * DAC * (5 EP + 5 TTC + 2 C) / 12,
        EP = progress * NC * DAC / max(that, PDM-Closed's), or 1 (0 if NC * DAC = 0) when both are below
        PROGRESS_DISTANCE_THRESHOLD.
NAVSIM v1's PDMS has no driving-direction term; PDM-Closed's internal score (used to pick the reference) does.

Everything WOMD lacks (route, lane polygons, speed limits, intersections) is replaced as in planner.pdm_closed_planner,
so the reward is consistent with the PDM-Closed baseline. Other agents are PDM's constant-velocity forecast
(`--pdm_cv_velocity_ratios`), whatever `--other_agent_forecast` the planner's handcrafted scorer would have used.
"""

from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from planner.pdm_closed_planner import (
    PROGRESS_DISTANCE_THRESHOLD, PROPOSAL_STEPS, WEIGHT_COMFORT, WEIGHT_PROGRESS, WEIGHT_TTC, PDMClosedPlanner,
    _score_components, _simulate,
)

SUB_SCORES = ("no_collision", "drivable", "ego_progress", "ttc", "comfort")


class PDMScoreReward:
    """Callable as DiffusionESPlanner.learned_scorer: (sim_state, timestep, goal [B,2] world, world_bkt5 [B,K,T,5]
    box-centre (x, y, yaw, vel_x, vel_y), index 0 = now, one world step apart) -> PDMS [B,K] (higher is better).

    `scorers` are the planner's (the runner sets their lane graphs; their road-edge test is the offroad check).
    PDM-Closed runs once per world and replan (the reference plan and the scoring context); the ES iterations of
    that replan reuse it.
    """

    def __init__(self, scorers, num_worlds: int, cv_velocity_ratios=(1.0,)) -> None:
        self.pdm = PDMClosedPlanner(num_worlds=num_worlds, cv_velocity_ratios=cv_velocity_ratios)
        self.pdm.scorers = scorers
        self.num_worlds = int(num_worlds)
        self._cache: list[tuple | None] = [None] * self.num_worlds
        self.last_components: list[dict | None] = [None] * self.num_worlds
        self.calls = self.replans = 0
        self.seconds = 0.0
        self._sums: dict[str, float] = {}
        self._n_worlds_scored = 0

    def __call__(self, sim_state, timestep, goal, world_bkt5) -> jnp.ndarray:
        t0 = time.time()
        goal = np.asarray(goal, dtype=np.float64).reshape(-1, 2)
        trajs = np.asarray(world_bkt5, dtype=np.float64)
        scores = [self._score_world(sim_state, int(timestep), goal[w], trajs[w], w) for w in range(trajs.shape[0])]
        for c in self.last_components:  # population statistics and the best candidate's sub-scores
            best = int(np.argmax(c["pdms"]))
            for k, v in (("pdms_mean", c["pdms"].mean()), ("pdms_best", c["pdms"][best]),
                         ("frac_zero", (c["pdms"] == 0).mean()),
                         *((f"best_{k}", c[k][best]) for k in SUB_SCORES)):
                self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._n_worlds_scored += 1
        self.calls += 1
        self.seconds += time.time() - t0
        return jnp.asarray(np.stack(scores), dtype=jnp.float32)

    def _context(self, sim_state, t, goal_xy, w):
        key = (id(self.pdm.scorers[w]._lane_graph), float(goal_xy[0]), float(goal_xy[1]), t)
        if self._cache[w] is not None and self._cache[w][0] == key:
            return self._cache[w][1:]
        self.pdm._plan_world(sim_state, goal_xy, t, w)
        ctx = self.pdm.last_context[w]
        ref = _score_components(ctx.executed[None, : PROPOSAL_STEPS + 1], ctx.obs, ctx.centerline, ctx.lane_map,
                                ctx.vehicle, ctx.offroad_fn)
        ref_raw = float(ref["progress"][0] * ref["no_collision"][0] * ref["drivable"][0])
        self._cache[w] = (key, ctx, ref_raw)
        self.replans += 1
        return ctx, ref_raw

    def _score_world(self, sim_state, t, goal_xy, trajs_kt5, w) -> np.ndarray:
        ctx, ref_raw = self._context(sim_state, t, goal_xy, w)
        n = min(PROPOSAL_STEPS + 1, trajs_kt5.shape[1])
        r = ctx.vehicle.rear_axle_to_center
        x, y, yaw = trajs_kt5[:, :n, 0], trajs_kt5[:, :n, 1], trajs_kt5[:, :n, 2]
        poses = np.stack([x - r * np.cos(yaw), y - r * np.sin(yaw), yaw], -1)                   # rear axle [K,n,3]
        states = _simulate(poses, ctx.ego_state, ctx.vehicle)
        c = _score_components(states, ctx.obs, ctx.centerline, ctx.lane_map, ctx.vehicle, ctx.offroad_fn)
        multiplicative = c["no_collision"] * c["drivable"]
        raw = c["progress"] * multiplicative
        denom = np.maximum(raw, ref_raw)
        ep = np.where(denom > PROGRESS_DISTANCE_THRESHOLD, raw / np.maximum(denom, 1e-9),
                      np.where(multiplicative == 0.0, 0.0, 1.0))
        weighted = (WEIGHT_PROGRESS * ep + WEIGHT_TTC * c["ttc"] + WEIGHT_COMFORT * c["comfort"]) / (
            WEIGHT_PROGRESS + WEIGHT_TTC + WEIGHT_COMFORT)
        pdms = multiplicative * weighted
        self.last_components[w] = {"pdms": pdms, "no_collision": c["no_collision"], "drivable": c["drivable"],
                                   "ego_progress": ep, "ttc": c["ttc"], "comfort": c["comfort"]}
        return pdms

    def describe(self) -> dict[str, Any]:
        return {"reward": "pdms_navsim_v1", "horizon_s": PROPOSAL_STEPS * 0.1,
                "weights": {"ego_progress": WEIGHT_PROGRESS, "ttc": WEIGHT_TTC, "comfort": WEIGHT_COMFORT},
                "progress_distance_threshold_m": PROGRESS_DISTANCE_THRESHOLD,
                "cv_velocity_ratios": list(self.pdm.cv_velocity_ratios)}

    def stats(self) -> dict[str, Any]:
        means = {k: round(v / max(1, self._n_worlds_scored), 4) for k, v in self._sums.items()}
        return {"calls": self.calls, "replans": self.replans, "seconds": round(self.seconds, 1),
                "seconds_per_call": round(self.seconds / max(1, self.calls), 3), **means}
