"""Record every Diffusion-ES candidate evaluated in closed loop, for discriminator training.

Attached to `DiffusionESPlanner.candidate_recorder` by `simulation/runner.py` when
`--save_es_candidates` is set; the planner calls the hooks with values it already
computed, so planning (RNG, scores, selection) is unchanged.

One compressed .npz per runner batch (`num_worlds` scenarios), written atomically
before the batch's per-scenario result JSONs, so the runner's `--start_from`
resume never leaves a scenario JSON without its candidate file.

Candidate IDs within a replan: stage s (0 = initial diffusion samples, 1..I = ES
mutation rounds) owns IDs s*P .. s*P+P-1 (P = population size). Carried elites
are not stored again; `cand_parent` is the elite each mutation was resampled from
and `cand_elite_bits` bit i marks membership in the elite set chosen at
iteration i (bit I = final elites). Per-replan arrays are indexed by group
g = (world, replan); per-candidate arrays by row c with `cand_group[c] = g`.

Array reference (G groups, C candidates, S scenarios in the batch, T = 51 future
0.1 s steps after the anchor, A = max_track_agents, M = WM modes):

  scenario level [S]
    scen_tfrecord, scen_index, scen_id, scen_goal_world [S,2],
    scen_expert_world [S,91,5] (x,y,yaw,vx,vy log), scen_expert_valid [S,91],
    scen_executed_world [S,91,5] (closed-loop ego), scen_overlap, scen_offroad,
    scen_goal_reached, scen_reached_step, scen_success, scen_min_goal_dist,
    scen_final_goal_dist
  group level [G]
    group_scen [G] (row in scenario arrays), group_t [G] (replan step),
    group_valid, group_origin [G,2] (world, f64), group_heading [G],
    group_ego_state [G,7] (x,y,yaw,vx,vy,length,width world), group_horizon
    [G] (steps the ES scorer scored, incl. anchor), group_selected_id [G],
    discriminator context: agents, agent_mask, map, map_mask, lights,
    lights_mask, goal, ego_vel (es_discriminator.data.extract_context format),
    expert_future [G,T,3] / expert_future_valid [G,T] (logged ego, ego frame),
    track_agent_idx [G,A] (object index, -1 pad), track_size [G,A,2],
    log_future [G,A,T+1,3] / log_future_valid [G,A,T+1] (logged, incl. t),
    wm_traj [G,A,M,T+1,3] f16, wm_mode_probs [G,A,M] f16, wm_valid [G,A,T+1]
  candidate level [C]
    cand_group, cand_id, cand_stage, cand_parent (-1 for stage 0),
    cand_traj [C,T,3] (x,y,yaw ego frame, steps 1..T), cand_selected,
    cand_elite_bits, cand_total, cand_<component> for each scorer factor
    (collision, offroad, ...), cand_wm_max_agent_collision_prob,
    cand_wm_argmax_risk_agent
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .data import CONTEXT_KEYS, DataConfig, _nearest_indices, to_ego_frame, wrap_pi
from .es_adapter import context_from_sim_state

FORMAT_VERSION = 1
EPISODE_STEPS = 91


def _sdc_index(is_sdc_row: np.ndarray) -> int:
    return int(np.argmax(np.asarray(is_sdc_row).astype(bool)))


def _local(xyyaw: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    xy = to_ego_frame(xyyaw[..., :2], origin, heading)
    yaw = wrap_pi(np.asarray(xyyaw[..., 2], dtype=np.float64) - heading)
    return np.concatenate([xy, yaw[..., None]], axis=-1)


class ESCandidateRecorder:
    def __init__(
        self,
        out_dir: str | os.PathLike,
        *,
        population_size: int,
        num_iterations: int,
        future_steps: int = 51,
        max_track_agents: int = 32,
        track_radius_m: float = 100.0,
        cfg: DataConfig = DataConfig(),
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.population_size = int(population_size)
        self.num_iterations = int(num_iterations)
        self.future_steps = int(future_steps)
        self.max_track_agents = int(max_track_agents)
        self.track_radius_m = float(track_radius_m)
        self.cfg = cfg
        self._batch: dict[str, Any] | None = None
        self._replan: dict[str, Any] | None = None

    # ---- runner hooks -------------------------------------------------------------

    def write_metadata(self, args: Any) -> None:
        try:
            repo = Path(__file__).resolve().parents[1]
            git_hash = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip() or None
        except Exception:
            git_hash = None
        meta = {
            "format_version": FORMAT_VERSION,
            "git_commit_hash": git_hash,
            "population_size": self.population_size,
            "num_iterations": self.num_iterations,
            "future_steps": self.future_steps,
            "world_dt": 0.1,
            "max_track_agents": self.max_track_agents,
            "track_radius_m": self.track_radius_m,
            "context_config": {k: getattr(self.cfg, k) for k in (
                "history", "history_stride", "max_agents", "max_map_points", "map_point_stride",
                "max_lights", "max_agent_distance_m", "max_map_distance_m", "horizon", "future_stride")},
            "args": {k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v))
                     for k, v in sorted(vars(args).items())},
        }
        path = self.out_dir / "run_config.json"
        if path.exists():
            prev = json.loads(path.read_text())
            changed = {k for k in ("population_size", "num_iterations", "future_steps", "max_track_agents")
                       if prev.get(k) != meta[k]}
            if changed:
                raise RuntimeError(f"Resuming into {self.out_dir} with different settings: {sorted(changed)}")
            path = self.out_dir / f"run_config.resume_{len(list(self.out_dir.glob('run_config*.json')))}.json"
        path.write_text(json.dumps(meta, indent=2))

    def batch_path(self, tfrecord_name: str, scenario_indices: list[int]) -> Path:
        tag = "_".join(f"{int(i):03d}" for i in scenario_indices)
        return self.out_dir / f"{tfrecord_name}.scen_{tag}.npz"

    def start_batch(self, sim_state, tfrecord_name: str, scenario_indices, scenario_ids, goal_xy_b2) -> None:
        log = jax.device_get(sim_state.log_trajectory)
        is_sdc = np.asarray(jax.device_get(sim_state.object_metadata.is_sdc)).astype(bool)
        nw = is_sdc.shape[0]
        ego = np.array([_sdc_index(is_sdc[w]) for w in range(nw)])
        w_idx = np.arange(nw)
        fields = [np.asarray(getattr(log, f))[w_idx, ego] for f in ("x", "y", "yaw", "vel_x", "vel_y")]
        ids = [s.decode() if isinstance(s, bytes) else str(s) for s in scenario_ids]
        self._batch = {
            "tfrecord": str(tfrecord_name),
            "indices": [int(i) for i in scenario_indices],
            "ids": ids,
            "goal": np.asarray(goal_xy_b2, dtype=np.float64),
            "log": log,
            "ego": ego,
            "expert_world": np.stack(fields, axis=-1).astype(np.float32),
            "expert_valid": np.asarray(log.valid)[w_idx, ego].astype(bool),
            "groups": [],
            "cands": [],
        }

    def finish_batch(self, replaced_state, outcomes: dict[str, np.ndarray]) -> Path | None:
        if self._batch is None:
            return None
        b = self._batch
        self._batch = None
        log = jax.device_get(replaced_state.log_trajectory)
        w_idx = np.arange(len(b["indices"]))
        executed = np.stack([np.asarray(getattr(log, f))[w_idx, b["ego"]]
                             for f in ("x", "y", "yaw", "vel_x", "vel_y")], axis=-1).astype(np.float32)
        out: dict[str, np.ndarray] = {
            "format_version": np.asarray(FORMAT_VERSION),
            "scen_tfrecord": np.asarray([b["tfrecord"]] * len(b["indices"])),
            "scen_index": np.asarray(b["indices"], dtype=np.int32),
            "scen_id": np.asarray(b["ids"]),
            "scen_goal_world": b["goal"],
            "scen_expert_world": b["expert_world"],
            "scen_expert_valid": b["expert_valid"],
            "scen_executed_world": executed,
        }
        for key, value in outcomes.items():
            out[f"scen_{key}"] = np.asarray(value)
        groups, cands = b["groups"], b["cands"]
        if groups:
            for key in groups[0]:
                out[key] = np.stack([g[key] for g in groups])
            ckeys = set().union(*(c.keys() for c in cands))
            for key in sorted(ckeys):
                is_idx = "argmax" in key
                parts = [c[key] if key in c else np.full((len(c["cand_id"]),), -1 if is_idx else np.nan,
                                                          dtype=np.int16 if is_idx else np.float32)
                         for c in cands]
                out[key] = np.concatenate(parts)
        path = self.batch_path(b["tfrecord"], b["indices"])
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        return path

    # ---- planner hooks ------------------------------------------------------------

    def begin_replan(self, sim_state, timestep: int, goal, forecaster: Any = None) -> None:
        if self._batch is None:
            self._replan = None
            return
        b = self._batch
        t = int(timestep)
        np_state = jax.device_get(sim_state)
        log = np_state.log_trajectory
        nw = len(b["indices"])
        T = self.future_steps
        groups = []
        for w in range(nw):
            e = int(b["ego"][w])
            g: dict[str, np.ndarray] = {"group_scen": np.int16(w), "group_t": np.int16(t)}
            ctx = context_from_sim_state(np_state, w, t, b["goal"][w], self.cfg, use_sim_trajectory=False)
            g["group_valid"] = np.bool_(ctx is not None)
            if ctx is None:
                origin = np.array([log.x[w, e, t], log.y[w, e, t]], dtype=np.float64)
                origin = np.nan_to_num(origin)
                heading = float(np.nan_to_num(log.yaw[w, e, t]))
                ctx = {k: np.zeros_like(v) for k, v in self._empty_context().items()}
            else:
                origin, heading = ctx["origin"], float(ctx["heading"])
            for key in CONTEXT_KEYS:
                g[key] = ctx[key]
            g["group_origin"] = origin
            g["group_heading"] = np.float64(heading)
            g["group_ego_state"] = np.array([
                log.x[w, e, t], log.y[w, e, t], log.yaw[w, e, t], log.vel_x[w, e, t], log.vel_y[w, e, t],
                log.length[w, e, t], log.width[w, e, t]], dtype=np.float32)

            fut = np.arange(t + 1, t + 1 + T)
            ok = fut < EPISODE_STEPS
            fut_c = np.minimum(fut, EPISODE_STEPS - 1)
            exp = b["expert_world"][w, fut_c, :3].astype(np.float64)
            g["expert_future"] = np.where(ok[:, None], _local(exp, origin, heading), 0.0).astype(np.float32)
            g["expert_future_valid"] = ok & b["expert_valid"][w, fut_c]

            # Other agents: logged futures (unchanged by the ego in log replay) and WM forecasts.
            orig = b["log"]
            at = np.stack([orig.x[w, :, t], orig.y[w, :, t]], axis=-1) - origin
            active = np.asarray(orig.valid[w, :, t]).astype(bool).copy()
            active[e] = False
            track = _nearest_indices(at, active, self.max_track_agents, self.track_radius_m)
            A = self.max_track_agents
            steps = np.arange(t, t + 1 + T)
            s_ok = steps < EPISODE_STEPS
            s_c = np.minimum(steps, EPISODE_STEPS - 1)
            idx = np.full((A,), -1, dtype=np.int16)
            idx[:len(track)] = track
            size = np.zeros((A, 2), dtype=np.float32)
            lf = np.zeros((A, T + 1, 3), dtype=np.float32)
            lv = np.zeros((A, T + 1), dtype=bool)
            if len(track):
                size[:len(track)] = np.stack([orig.length[w, track, t], orig.width[w, track, t]], axis=-1)
                xyyaw = np.stack([orig.x[w][track][:, s_c], orig.y[w][track][:, s_c], orig.yaw[w][track][:, s_c]],
                                 axis=-1).astype(np.float64)
                valid = np.asarray(orig.valid[w])[track][:, s_c].astype(bool) & s_ok[None]
                lf[:len(track)] = np.where(valid[..., None], _local(xyyaw, origin, heading), 0.0)
                lv[:len(track)] = valid
            g["track_agent_idx"] = idx
            g["track_size"] = size
            g["log_future"] = lf
            g["log_future_valid"] = lv
            g.update(self._wm_prediction(forecaster, w, track, origin, heading, T + 1))
            groups.append(g)

        P = self.population_size
        self._replan = {
            "groups": groups,
            "pool_ids": np.tile(np.arange(P, dtype=np.int16), (nw, 1)),
            "elite_ids": None,
            "cands": [dict(ids=[], stage=[], parent=[], traj=[], comps=[]) for _ in range(nw)],
            "elite_bits": [dict() for _ in range(nw)],
        }

    def record_stage(self, stage: int, world_bkt5, components: list[dict], selector) -> None:
        r = self._replan
        if r is None:
            return
        world = np.asarray(jax.device_get(world_bkt5), dtype=np.float64)
        nw, k = world.shape[:2]
        P = self.population_size
        new_ids = (stage * P + np.arange(k)).astype(np.int16)
        if stage == 0:
            parents = np.full((nw, k), -1, dtype=np.int16)
        else:
            parents = r["elite_ids"][:, np.asarray(selector)]
            r["pool_ids"] = np.concatenate([r["elite_ids"], np.tile(new_ids, (nw, 1))], axis=1)
        T = self.future_steps
        for w in range(nw):
            g = r["groups"][w]
            c = r["cands"][w]
            traj = world[w, :, 1:1 + T, :3]
            local = _local(traj, g["group_origin"], float(g["group_heading"])).astype(np.float32)
            if local.shape[1] < T:
                local = np.pad(local, ((0, 0), (0, T - local.shape[1]), (0, 0)), constant_values=np.nan)
            comps = {}
            for key, value in (components[w] or {}).items():
                if key == "horizon":
                    g["group_horizon"] = np.int16(value)
                    continue
                arr = np.asarray(jax.device_get(value))
                if arr.shape[:1] != (k,):
                    continue
                if key == "wm_argmax_risk_agent":
                    comps[f"cand_{key}"] = arr.astype(np.int16)
                else:
                    comps[f"cand_{key}"] = arr.astype(np.float32)
            c["ids"].append(new_ids)
            c["stage"].append(np.full((k,), stage, dtype=np.int8))
            c["parent"].append(parents[w])
            c["traj"].append(local)
            c["comps"].append(comps)

    def record_elites(self, iteration: int, elite_indices_bk) -> None:
        r = self._replan
        if r is None:
            return
        idx = np.asarray(jax.device_get(elite_indices_bk))
        r["elite_ids"] = np.take_along_axis(r["pool_ids"], idx, axis=1)
        for w in range(idx.shape[0]):
            bits = r["elite_bits"][w]
            for cid in r["elite_ids"][w]:
                bits[int(cid)] = bits.get(int(cid), 0) | (1 << int(iteration))

    def end_replan(self, best_indices_b) -> None:
        r = self._replan
        self._replan = None
        if r is None or self._batch is None:
            return
        best = np.asarray(jax.device_get(best_indices_b)).reshape(-1)
        for w, g in enumerate(r["groups"]):
            c = r["cands"][w]
            ids = np.concatenate(c["ids"])
            sel_id = int(r["pool_ids"][w, int(best[w])])
            g["group_selected_id"] = np.int16(sel_id)
            g.setdefault("group_horizon", np.int16(-1))
            gi = len(self._batch["groups"])
            self._batch["groups"].append(g)
            bits = r["elite_bits"][w]
            cand = {
                "cand_group": np.full((len(ids),), gi, dtype=np.int32),
                "cand_id": ids,
                "cand_stage": np.concatenate(c["stage"]),
                "cand_parent": np.concatenate(c["parent"]),
                "cand_traj": np.concatenate(c["traj"]),
                "cand_selected": ids == sel_id,
                "cand_elite_bits": np.array([bits.get(int(i), 0) for i in ids], dtype=np.uint8),
            }
            for key in set().union(*(d.keys() for d in c["comps"])):
                cand[key] = np.concatenate([
                    d[key] if key in d else np.full((len(s),), -1 if "argmax" in key else np.nan,
                                                   dtype=np.int16 if "argmax" in key else np.float32)
                    for d, s in zip(c["comps"], c["ids"])])
            self._batch["cands"].append(cand)

    # ---- helpers ------------------------------------------------------------------

    def _empty_context(self) -> dict[str, np.ndarray]:
        c = self.cfg
        return {
            "agents": np.zeros((c.max_agents, c.history, 8), np.float32),
            "agent_mask": np.zeros((c.max_agents, c.history), bool),
            "map": np.zeros((c.max_map_points, 5), np.float32),
            "map_mask": np.zeros((c.max_map_points,), bool),
            "lights": np.zeros((c.max_lights, 4), np.float32),
            "lights_mask": np.zeros((c.max_lights,), bool),
            "goal": np.zeros((2,), np.float32),
            "ego_vel": np.zeros((2,), np.float32),
        }

    def _wm_prediction(self, forecaster, w: int, track: np.ndarray, origin, heading: float, steps: int) -> dict:
        """WM forecast of the tracked agents from the forecaster's per-replan cache (if any)."""
        cache = getattr(forecaster, "_cache", None)
        if not cache:
            return {}
        cached = next(iter(cache.values()))
        if not isinstance(cached, dict) or "masks" not in cached:
            return {}
        if "traj_k" in cached:
            traj_k = np.asarray(cached["traj_k"][w])
            probs = np.asarray(cached["mode_probs"][w])
        elif "top1_traj" in cached:
            traj_k = np.asarray(cached["top1_traj"][w])[:, None]
            probs = np.ones(traj_k.shape[:2], np.float32)
        else:
            return {}
        masks = np.asarray(cached["masks"][w]).astype(bool)
        A, M = self.max_track_agents, traj_k.shape[1]
        h = min(steps, traj_k.shape[2])
        wt = np.zeros((A, M, steps, 3), dtype=np.float16)
        wp = np.zeros((A, M), dtype=np.float16)
        wv = np.zeros((A, steps), dtype=bool)
        if len(track):
            sel = traj_k[track, :, :h, :3].astype(np.float64)
            wt[:len(track), :, :h] = _local(sel, origin, heading).astype(np.float16)
            wp[:len(track)] = probs[track]
            wv[:len(track), :h] = masks[track, :h]
        return {"wm_traj": wt, "wm_mode_probs": wp, "wm_valid": wv}
