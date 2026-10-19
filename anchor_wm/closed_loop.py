"""AnchorWM in closed-loop Diffusion-ES (simulation/run_simulation.py).

AnchorForecaster      `--other_agent_forecast wm --wm_backend anchor`: the anchor world model as the
                      planner's forecaster. Same interface as the MF-Gaussian WorldModelForecaster
                      (forecast / score_independent_mode_collision / forecast_modes): the 6 planner
                      modes (endpoint NMS, probabilities renormalized over them) are decoded to world
                      boxes like MF's `_decode_boxes` (current pose + delta, linear interpolation to
                      the planner's dt, other objects held at their current pose), and ES collision
                      scoring reuses MF's `planning.score_independent_agent_collision` unchanged.
AnchorLearnedScorer   `--anchor_scorer_ckpt`: the AnchorWM scorer head replaces the handcrafted ES
                      reward (initial population, every ES iteration, final selection), like
                      es_discriminator.real_es.closed_loop.LearnedESScorer. One encoder / decoder
                      pass per replan; candidates are the population at 0.2 s (world steps
                      t+2 .. t+50) in the ego frame, as es_discriminator's `pool_world_to_ego`.

Inputs come from `data.build_arrays` on the closed-loop state (whose log trajectory holds the
executed ego, as MF's forecaster and the discriminator assume), i.e. exactly the training
preprocessing.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anchor_wm.data import build_arrays, payload_from_sim_state, rotate_xy_np, wrap_to_pi
from anchor_wm.eval_bank import load_model
from anchor_wm.model import gather_k, select_modes

REPO = Path(__file__).resolve().parents[1]
DYNAMIC_TYPES = np.asarray([1, 2, 3], np.int32)
ANCHOR_TOL_M = 0.05
ANCHOR_TOL_RAD = 0.01


def _mf_planning():
    """MF-Gaussian's planning module (collision scoring), imported read-only under a private alias."""
    alias = "_anchor_wm_mf_planning"
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [str(REPO / "multiFuture_GaussianWorldModel" / "model" / "world_model")]
        sys.modules[alias] = pkg
    return importlib.import_module(f"{alias}.planning")


def _inputs(model, sim_state, timestep: int) -> tuple[dict, dict]:
    import jax
    payload = payload_from_sim_state(jax.device_get(sim_state))
    arrays = build_arrays(payload, anchor_step=int(timestep), cfg=model.config, with_targets=False)
    dev = next(model.parameters()).device
    f = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in arrays.items()
         if k not in ("origin_xy", "origin_yaw")}
    return f, {"payload": payload, "origin_xy": arrays["origin_xy"], "origin_yaw": arrays["origin_yaw"]}


class AnchorForecaster:
    num_collision_worlds = 1
    collision_reduce = "any"

    def __init__(self, model, *, planning_mode: str = "expected_independent", mode_temperature: float = 1.0,
                 checkpoint: str = "") -> None:
        self.model = model.eval()
        self.config = model.config
        self.device = next(model.parameters()).device
        self.planning = _mf_planning()
        self.planning_mode = self.planning.canonicalize_planning_mode(planning_mode)
        self.checkpoint = checkpoint
        self._mode_temperature = float(mode_temperature)
        self._cache: dict = {}
        self._last_collision_debug = None
        self.stats_path: str | None = None
        self.num_forecasts = 0
        self.forward_s: list[float] = []

    @classmethod
    def from_checkpoint(cls, path: str, *, device: str = "cpu", planning_mode: str | None = None,
                        mode_temperature: float = 1.0) -> "AnchorForecaster":
        return cls(load_model(path, torch.device(device)), planning_mode=planning_mode or "expected_independent",
                   mode_temperature=mode_temperature, checkpoint=str(path))

    @property
    def mode_temperature(self) -> float:
        return self._mode_temperature

    @mode_temperature.setter
    def mode_temperature(self, value: float) -> None:
        if not float(value) > 0:
            raise ValueError(f"mode_temperature must be > 0, got {value}")
        self._mode_temperature = float(value)
        self._cache.clear()

    def clear_cache(self) -> None:
        self._cache.clear()

    # ---- prediction ---------------------------------------------------------------------------

    @torch.no_grad()
    def _predict(self, sim_state, timestep: int, traj_length: int, dt: float) -> dict:
        key = (id(sim_state), int(timestep), int(traj_length), float(dt))
        if key in self._cache:
            return self._cache[key]
        cfg = self.config
        t0 = time.perf_counter()
        f, meta = _inputs(self.model, sim_state, timestep)
        _, dec = self.model(f)
        sel = select_modes(dec, cfg.num_out_modes, cfg.nms_dist_m, cfg.max_range)                 # [B,n,6]
        delta = gather_k(dec["pred_xy_k"].float(), sel).cpu().numpy() * cfg.max_range              # ego frame
        yaw_vec = gather_k(dec["yaw_vec_k"].float(), sel).cpu().numpy()
        p = dec["probs"].float().gather(-1, sel)
        if self._mode_temperature != 1.0:
            p = p.clamp_min(1e-12) ** (1.0 / self._mode_temperature)
        p = (p / p.sum(-1, keepdim=True).clamp_min(1e-12)) * f["agent_valid"][..., None].float()
        self.forward_s.append(time.perf_counter() - t0)

        pl = meta["payload"]
        x, y = pl["log_trajectory_x"], pl["log_trajectory_y"]
        yaw = pl["log_trajectory_yaw"]
        valid = pl["log_trajectory_valid"].astype(bool)
        is_sdc = pl["object_metadata_is_sdc"].astype(bool)
        types_ = pl["object_metadata_object_types"].astype(np.int32)
        t = int(timestep)
        b, n_total, _ = x.shape
        n = delta.shape[1]
        cur_xy = np.stack([x[:, :, t], y[:, :, t]], -1).astype(np.float64)
        cur_yaw = yaw[:, :, t].astype(np.float64)
        cur_valid = valid[:, :, t]
        modeled = np.zeros((b, n_total), bool)
        modeled[:, :n] = cur_valid[:, :n] & np.isin(types_[:, :n], DYNAMIC_TYPES) & ~is_sdc[:, :n]

        # Positions at 0 (now) and the model steps, world frame: [B,n,6,T+1,2] / yaw [B,n,6,T+1].
        ego_yaw = meta["origin_yaw"]
        world_delta = rotate_xy_np(delta, -ego_yaw[:, None, None, None])        # rotate by +heading
        src_xy = np.concatenate([np.zeros_like(world_delta[..., :1, :]), world_delta], -2) + cur_xy[:, :n, None, None]
        dyaw = np.arctan2(yaw_vec[..., 0], yaw_vec[..., 1])
        src_yaw = cur_yaw[:, :n, None, None] + np.concatenate([np.zeros_like(dyaw[..., :1]), dyaw], -1)

        h = int(traj_length)
        fdt = float(cfg.future_dt)
        tmax = cfg.future_steps
        q = np.arange(h) * float(dt) / fdt                                       # source index (float)
        i0 = np.clip(np.floor(q).astype(int), 0, tmax)
        i1 = np.minimum(i0 + 1, tmax)
        w = np.clip(q - i0, 0.0, 1.0)
        w[i0 == tmax] = 0.0
        lerp = lambda a: a[..., i0, :] * (1 - w)[:, None] + a[..., i1, :] * w[:, None]  # noqa: E731
        k_xy = lerp(src_xy)                                                      # [B,n,6,H,2]
        sc = lerp(np.stack([np.sin(src_yaw), np.cos(src_yaw)], -1))
        k_yaw = np.arctan2(sc[..., 0], sc[..., 1])

        m = cfg.num_out_modes
        out_xy = np.broadcast_to(cur_xy[:, :, None, None], (b, n_total, m, h, 2)).copy()
        out_yaw = np.broadcast_to(cur_yaw[:, :, None, None], (b, n_total, m, h)).copy()
        mod = modeled[:, :n]
        out_xy[:, :n][mod] = k_xy[mod]
        out_yaw[:, :n][mod] = k_yaw[mod]
        lw = np.stack([pl["log_trajectory_length"][:, :, t], pl["log_trajectory_width"][:, :, t]], -1)
        lw = np.broadcast_to(lw[:, :, None, None], (b, n_total, m, h, 2))
        traj_k = np.concatenate([out_xy, wrap_to_pi(out_yaw)[..., None], lw], -1).astype(np.float32)
        probs = np.zeros((b, n_total, m), np.float32)
        probs[:, :n] = p.cpu().numpy()
        top1 = traj_k[:, :, 0]                                                   # picks are sorted by probability
        masks = np.broadcast_to((cur_valid & ~is_sdc)[:, :, None], (b, n_total, h)).copy()
        out = {"top1_traj": top1, "masks": masks, "traj_k": traj_k, "mode_probs": probs,
               "top1_mode": np.zeros((b, n_total), np.int32)}
        self._cache.clear()
        self._cache[key] = out
        self.num_forecasts += 1
        if self.stats_path and self.num_forecasts % 50 == 0:
            self.write_forecast_stats(self.stats_path)
        return out

    # ---- planner interface (as MF's WorldModelForecaster) --------------------------------------

    def forecast(self, sim_state, *, timestep: int, traj_length: int, world_idx: int = 0, dt: float = 0.1,
                 planning_mode: str | None = None):
        import jax.numpy as jnp
        c = self._predict(sim_state, timestep, traj_length, dt)
        i = int(world_idx)
        top, msk = c["top1_traj"][i], c["masks"][i]
        return (jnp.asarray(top, jnp.float32), jnp.asarray(msk).astype(jnp.bool_),
                jnp.asarray(top, jnp.float32), jnp.asarray(msk).astype(jnp.bool_))

    def score_independent_mode_collision(self, ego_trajectories_world, *, world_idx: int = 0,
                                         planning_mode: str | None = None):
        if not self._cache:
            raise RuntimeError("Call forecast() before independent-mode scoring.")
        c = next(iter(self._cache.values()))
        i = int(world_idx)
        mode = self.planning.canonicalize_planning_mode(planning_mode or self.planning_mode)
        score, debug = self.planning.score_independent_agent_collision(
            np.asarray(ego_trajectories_world, np.float32), c["traj_k"][i], c["masks"][i], c["mode_probs"][i],
            planning_mode=mode)
        self._last_collision_debug = debug
        return score, debug

    def forecast_modes(self, sim_state, *, timestep: int, traj_length: int, world_idx: int = 0,
                       dt: float = 0.1) -> dict:
        c = self._predict(sim_state, timestep, traj_length, dt)
        i = int(world_idx)
        return {k: c[k][i] for k in ("top1_traj", "masks", "traj_k", "mode_probs", "top1_mode")}

    def forecast_stats_summary(self) -> dict[str, Any]:
        f = self.forward_s
        return {"backend": "anchor", "checkpoint": self.checkpoint, "planning_mode": self.planning_mode,
                "mode_temperature": self._mode_temperature, "num_forecasts": self.num_forecasts,
                "forward_ms_mean": 1000 * float(np.mean(f)) if f else None}

    def write_forecast_stats(self, path: str) -> None:
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.forecast_stats_summary(), fh, indent=2)
        os.replace(tmp, path)


class AnchorLearnedScorer:
    """Callable as DiffusionESPlanner.learned_scorer: (sim_state, timestep, goal [B,2] world,
    population [B,K,T,>=3] world at 0.1 s) -> scores [B,K] (higher is better)."""

    def __init__(self, checkpoint: str, *, device: str = "cuda", expected_calls_per_replan: int = 0,
                 stats_path: str | None = None, chunk: int = 64, model=None, rule_scorers=None,
                 rule_lambda: float = 0.0, rule_terms: tuple[str, ...] = ("collision", "offroad")) -> None:
        """`rule_scorers` (the planner's handcrafted Scorers, one per world) with `rule_lambda` > 0 makes a
        hybrid reward: learned score + rule_lambda * sum_t log(clip(rule factor t, 1e-3, 1)) over
        `rule_terms` (diagnostic: hand the safety terms back to the rule scorer)."""
        self.rule_scorers = rule_scorers
        self.rule_lambda = float(rule_lambda)
        self.rule_terms = tuple(rule_terms)
        if self.rule_lambda > 0 and not rule_scorers:
            raise ValueError("rule_lambda > 0 needs the planner's rule scorers")
        self.model = model if model is not None else load_model(checkpoint, torch.device(device))
        self.model.eval()
        self.checkpoint = checkpoint
        self.device = next(self.model.parameters()).device
        self.expected_calls = int(expected_calls_per_replan)
        self.stats_path, self.chunk = stats_path, int(chunk)
        self._key = None
        self._calls = 0
        self._ctx = None
        self.calls_per_replan: Counter = Counter()
        self.replans = 0
        self.t_context: list[float] = []
        self.t_score: list[float] = []
        self.anchor_err_m = 0.0
        self.anchor_err_rad = 0.0
        self.pool_sizes: Counter = Counter()

    def describe(self) -> dict:
        return {"checkpoint": self.checkpoint, "type": "anchor_wm_scorer", "device": str(self.device),
                "expected_calls_per_replan": self.expected_calls, "rule_lambda": self.rule_lambda,
                "rule_terms": list(self.rule_terms) if self.rule_lambda > 0 else []}

    def _rule_penalty(self, sim_state, timestep: int, world_bkt5) -> np.ndarray:
        """[B,K] sum of log rule factors (collision = WM P(safe), offroad gate) from the handcrafted scorers."""
        out = []
        for w, scorer in enumerate(self.rule_scorers):
            scorer.compute_score(trajectories=world_bkt5[w], sim_state=sim_state, timestep=timestep, world_idx=w)
            comps = scorer.last_components
            pen = np.zeros(world_bkt5.shape[1], np.float64)
            for t in self.rule_terms:
                if t not in comps:
                    raise KeyError(f"rule scorer has no '{t}' component (has {sorted(comps)})")
                pen += np.log(np.clip(np.asarray(comps[t], np.float64), 1e-3, 1.0))
            out.append(pen)
        return np.stack(out)

    def _finish_replan(self) -> None:
        if self._key is None:
            return
        self.calls_per_replan[self._calls] += 1
        self.replans += 1
        if self.expected_calls and self._calls != self.expected_calls:
            raise RuntimeError(f"anchor scorer scored {self._calls} pools in replan {self._key}, "
                               f"expected {self.expected_calls}")
        if self.stats_path and self.replans % 50 == 0:
            self.write_stats()

    @torch.no_grad()
    def _context(self, sim_state, timestep: int, goal_world: np.ndarray) -> None:
        f, meta = _inputs(self.model, sim_state, timestep)
        origin, heading = meta["origin_xy"], meta["origin_yaw"]
        goal = rotate_xy_np(np.asarray(goal_world, np.float64) - origin, heading).astype(np.float32)
        f["goal_xy"] = torch.from_numpy(goal).to(self.device)
        enc, dec = self.model(f)
        dec = {k: v.float() for k, v in dec.items()}
        ctx = self.model.scorer.context(enc, dec, f, 0.0)
        pl = meta["payload"]
        sdc = np.argmax(pl["object_metadata_is_sdc"].astype(bool), axis=1)
        w = np.arange(len(sdc))
        valid = pl["log_trajectory_valid"][w, sdc, int(timestep)].astype(bool)
        ego = np.stack([pl["log_trajectory_x"][w, sdc, timestep], pl["log_trajectory_y"][w, sdc, timestep],
                        pl["log_trajectory_yaw"][w, sdc, timestep]], -1).astype(np.float64)
        self._ctx = (ctx, origin, heading, valid, ego)

    def _check_anchor(self, pop: np.ndarray, ego: np.ndarray, valid: np.ndarray) -> None:
        d = np.linalg.norm(pop[:, :, 0, :2] - ego[:, None, :2], axis=-1)
        dyaw = np.abs(wrap_to_pi(pop[:, :, 0, 2] - ego[:, None, 2]))
        em = float(d[valid].max(initial=0.0))
        er = float(dyaw[valid].max(initial=0.0))
        self.anchor_err_m, self.anchor_err_rad = max(self.anchor_err_m, em), max(self.anchor_err_rad, er)
        if em > ANCHOR_TOL_M or er > ANCHOR_TOL_RAD:
            raise RuntimeError(f"population anchor is {em:.3f} m / {er:.4f} rad from the executed ego")

    @torch.no_grad()
    def __call__(self, sim_state, timestep: int, goal, world_bkt5):
        import jax
        import jax.numpy as jnp
        key = (id(sim_state), int(timestep))
        t0 = time.perf_counter()
        if key != self._key:
            self._finish_replan()
            self._key, self._calls = key, 0
            self._context(sim_state, int(timestep), np.asarray(jax.device_get(goal)))
            self.t_context.append(time.perf_counter() - t0)
        ctx, origin, heading, valid, ego = self._ctx
        pop = np.asarray(jax.device_get(world_bkt5), dtype=np.float64)
        if self._calls == 0:
            self._check_anchor(pop, ego, valid)
        t1 = time.perf_counter()
        cfg = self.model.config
        idx = cfg.future_stride * np.arange(1, cfg.future_steps + 1)
        if pop.shape[2] <= idx[-1]:
            raise ValueError(f"population has {pop.shape[2]} steps, need {idx[-1] + 1}")
        sub = pop[:, :, idx]
        xy = rotate_xy_np(sub[..., :2] - origin[:, None, None], heading[:, None, None])
        yaw = wrap_to_pi(sub[..., 2] - heading[:, None, None])
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)  # noqa: E731
        total, _ = self.model.scorer(ctx, to(xy), to(yaw), chunk=self.chunk, heading=to(yaw))
        scores = total.float().cpu().numpy()
        if self.rule_lambda > 0:
            scores = scores + self.rule_lambda * self._rule_penalty(sim_state, int(timestep), world_bkt5)
        scores[~valid] = 0.0
        self.t_score.append(time.perf_counter() - t1)
        self._calls += 1
        self.pool_sizes[pop.shape[1]] += 1
        return jnp.asarray(scores, dtype=jnp.float32)

    def write_stats(self) -> dict:
        ms = lambda x: {"mean": 1000 * float(np.mean(x)), "median": 1000 * float(np.median(x)),  # noqa: E731
                        "n": len(x)} if x else None
        out = {**self.describe(), "replans_completed": self.replans,
               "scoring_calls_per_replan": {str(k): v for k, v in sorted(self.calls_per_replan.items())},
               "pool_sizes": {str(k): v for k, v in sorted(self.pool_sizes.items())},
               "context_ms_per_replan": ms(self.t_context), "score_ms_per_call": ms(self.t_score),
               "max_anchor_err_m": self.anchor_err_m, "max_anchor_err_rad": self.anchor_err_rad}
        if self.stats_path:
            tmp = self.stats_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(out, fh, indent=1)
            os.replace(tmp, self.stats_path)
        return out

    def close(self) -> dict:
        self._finish_replan()
        self._key = None
        return self.write_stats()
