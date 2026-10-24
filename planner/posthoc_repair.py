"""Post-hoc repair of the plan a planner picks (yixiz's arm ``o_am``, rl/diffusion_ft/README_posthoc_opt.md).

The next ``window`` world steps of the chosen plan are edited (lateral offset + along-path retiming) to minimise
one loss, safety barrier (road edge 0.3 m, agents 0.75 m) + comfort of the edit, with constant-velocity
neighbours. A plan already clear of every margin has zero loss and runs unchanged. Two starts per row, keeping the
lower final loss: no edit (``init0``) and the tail of the plan executed at the previous replan projected onto the
fresh plan (``initprev``). Mirrors ``rl.diffusion_ft.proposal_select.ProposalSelectEval._argmin``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

O_AM_CFG = "lon=1,w_lon_prox=0.05,lon_back_m=1000"


class PosthocRepair:
    def __init__(self, cfg: str = O_AM_CFG, window: int = 20, neighbor_pred: str = "cv", dt: float = 0.1):
        from rl.diffusion_ft.lateral_opt import LateralOptConfig, make_lateral_optimizer
        from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig

        if neighbor_pred == "log":
            raise ValueError("neighbor_pred='log' reads the recorded future (an oracle); use 'cv' or 'none'")
        self.window, self.dt = int(window), float(dt)
        self.cfg = LateralOptConfig.parse(cfg)
        if not self.cfg.lon:
            raise ValueError("PosthocRepair expects a lon=1 config (edits are [d_1..d_n, s_1..s_n])")
        self._opt = make_lateral_optimizer(SafetyObjectiveConfig(), self.cfg, dt=self.dt,
                                           neighbor_pred=neighbor_pred, world_dt_method="median")
        self._prev = None  # (executed world plan [B, W, 5], its timestep)

    def __call__(self, world_bw5, state, timestep: int):
        """``world_bw5`` [B, W, 5] (x, y, yaw, vx, vy; index 0 = now) -> (plan to execute [B, W, 5], moved [B])."""
        from rl.diffusion_ft.lateral_opt import edit_world, project_onto_path

        t0 = int(timestep)
        world = jnp.asarray(world_bw5, jnp.float32)
        B, W = int(world.shape[0]), int(world.shape[1])
        if self._prev is not None and (t0 <= self._prev[1] or self._prev[0].shape[0] != B):
            self._prev = None  # a new episode (batch) started
        horizon = int(state.log_trajectory.x.shape[-1])
        n = int(min(self.window, horizon - 1 - t0, W - 1))
        moved = np.zeros(B, dtype=bool)
        if n > 0:
            starts = [np.zeros((B, 2 * n), np.float32)]  # init0
            if self._prev is not None:  # initprev: previous executed plan from the current state on
                prev_world, prev_t = self._prev
                tail = prev_world[:, t0 - prev_t:]
                if tail.shape[1] >= 2:
                    tail = jnp.concatenate([tail, jnp.repeat(tail[:, -1:], W - tail.shape[1], axis=1)], axis=1)
                    starts.append(np.asarray(project_onto_path(world, tail, n), np.float32))
            t0j, act = jnp.asarray(t0, jnp.int32), jnp.ones((B,), jnp.float32)
            best_x, best_l = None, np.full(B, np.inf)
            for x0 in starts:
                x, loss, _ = self._opt(world, jnp.asarray(x0), state, t0j, act, n)
                x, loss = np.asarray(x), np.asarray(loss)
                take = loss < best_l  # ties keep the earlier start, so a zero-loss row keeps init0's x = 0
                best_x = x if best_x is None else np.where(take[:, None], x, best_x)
                best_l = np.where(take, loss, best_l)
            moved = np.any(best_x != 0.0, axis=1)
            edited = edit_world(world, jnp.asarray(best_x), n=n, dt=self.dt, lon=1)
            world = jnp.where(jnp.asarray(moved)[:, None, None], edited, world)
        self._prev = (world, t0)
        return world, moved
