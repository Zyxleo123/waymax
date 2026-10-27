"""AnchorWM ego modes in Diffusion-ES: seeding the initial population, and per-replan dumps for plots.

AnchorEgoSeeder   `--anchor_ego_seed K`: the K ego modes the AnchorWM forecasts (closed_loop.ego_modes_world) are
                  appended to the diffusion samples of every replan's initial population; ES then scores and
                  resamples them like any other candidate.
ESDebugDumper     `--es_debug_dir DIR`: one npz per replan with the scene at the replan step (agents, roadgraph
                  once per batch, with the whole agent log), the AnchorWM ego modes (NMS top k and every decoded
                  anchor), the initial and final ES pools with scores, the chosen plan, whether each candidate
                  descends from an ego seed, and for the expert-vs-planner diagnosis (notebooks/anchor_wm_failures):
                    expert_world     the logged ego from the replan step, 0.1 s like the pools [B,T,5] (the tail
                                     past the episode end holds the last logged pose), expert_valid [B,T]
                    expert_aligned   the same path moved rigidly onto the executed ego pose; expert_offset_m is
                                     how far the executed ego is from the log, expert_score its learned score
                    *_overlap_step / *_offroad_step  log-replay labels of init / final / best / expert_aligned:
                                     first step (1-based, 0.1 s) overlapping a logged agent / off the road, 0 = none
                                     within the episode
                  Plotted by anchor_wm.viz_failures and anchor_wm.viz_ego_seed.
"""

from __future__ import annotations

import os

import numpy as np

from anchor_wm.closed_loop import ego_modes_world

SCENE_KEYS = ("log_trajectory_x", "log_trajectory_y", "log_trajectory_yaw", "log_trajectory_valid",
              "log_trajectory_length", "log_trajectory_width", "object_metadata_is_sdc",
              "object_metadata_object_types")
ROAD_KEYS = ("roadgraph_points_x", "roadgraph_points_y", "roadgraph_points_types", "roadgraph_points_valid")
EXPERT_KEYS = ("log_trajectory_x", "log_trajectory_y", "log_trajectory_yaw", "log_trajectory_vel_x",
               "log_trajectory_vel_y", "log_trajectory_valid")
LABELLED = ("init", "final", "best", "expert_aligned")


class AnchorEgoSeeder:
    def __init__(self, model, k: int) -> None:
        self.model, self.k = model, int(k)

    def __call__(self, sim_state, timestep: int):
        return ego_modes_world(self.model, sim_state, timestep, self.k)


def expert_future(expert: np.ndarray, t: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """expert [B,6,T_ep] (x, y, yaw, vx, vy, valid of the logged ego) -> world [B,steps,5] from step t and
    valid [B,steps]; steps past the episode end or after the log ends hold the last valid pose at zero speed."""
    b, _, t_ep = expert.shape
    idx = np.minimum(t + np.arange(steps), t_ep - 1)
    valid = expert[:, 5, idx].astype(bool) & (t + np.arange(steps) < t_ep)[None]
    out = np.moveaxis(expert[:, :5, idx], 1, 2).astype(np.float64)                      # [B,steps,5]
    for w in range(b):
        last = np.flatnonzero(valid[w])
        if len(last) and last[-1] + 1 < steps:
            out[w, last[-1] + 1:] = out[w, last[-1]] * np.array([1, 1, 1, 0, 0])
    return out, valid


def align_to(path: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Move paths [B,T,5] (x, y, yaw, vx, vy) rigidly so their first pose is `pose` [B,3]."""
    dyaw = pose[:, 2] - path[:, 0, 2]
    c, s = np.cos(dyaw)[:, None], np.sin(dyaw)[:, None]
    d = path[..., :2] - path[:, :1, :2]
    out = path.copy()
    out[..., 0] = pose[:, None, 0] + c * d[..., 0] - s * d[..., 1]
    out[..., 1] = pose[:, None, 1] + s * d[..., 0] + c * d[..., 1]
    out[..., 2] = path[..., 2] + dyaw[:, None]
    out[..., 3] = c * path[..., 3] - s * path[..., 4]
    out[..., 4] = s * path[..., 3] + c * path[..., 4]
    return out


class LogReplayLabeller:
    """First overlap / offroad step of world trajectories [K,T,5] (index 0 = the replan step) against the logged
    agents and the road edges, as es_discriminator.label_es_candidates labels the ES bank."""

    def __init__(self, chunk: int = 2048) -> None:
        from scores.scorer_lane_graph import Scorer
        self.scorer = Scorer(metrics=["collision", "offroad"], other_agent_forecast="gt")
        self.chunk = chunk

    def __call__(self, sim_state, t: int, w: int, traj: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import jax.numpy as jnp
        from es_discriminator.label_es_candidates import EPISODE_STEPS, collision_steps, first_step, offroad_steps
        from scores.helpers import forecast_other_agents_log_futures

        k, steps = traj.shape[:2]
        h = min(steps - 1, EPISODE_STEPS - 1 - t)
        ego = np.concatenate([traj[:, 1:, :3], np.broadcast_to(size, (k, steps - 1, 2))], -1).astype(np.float32)
        obj, mask, _, _ = forecast_other_agents_log_futures(sim_state, t, steps, w)
        obj, mask = np.asarray(obj), np.asarray(mask)
        pad = steps - obj.shape[1]
        if pad > 0:
            obj, mask = np.pad(obj, ((0, 0), (0, pad), (0, 0))), np.pad(mask, ((0, 0), (0, pad)))
        coll = np.array(collision_steps(jnp.asarray(ego), jnp.asarray(obj[:, 1:]), jnp.asarray(mask[:, 1:])))
        off = offroad_steps(self.scorer, ego, sim_state, t, w, self.chunk)   # whole horizon: one shape per K
        coll[:, h:] = off[:, h:] = False
        return first_step(coll), first_step(off)


class ESDebugDumper:
    def __init__(self, model, out_dir: str, k: int = 6, scorer=None, labels: bool = True) -> None:
        """`scorer`: the planner's learned scorer (closed_loop.AnchorLearnedScorer) to score the expert with."""
        self.model, self.out_dir, self.k, self.scorer = model, out_dir, int(k), scorer
        self.labeller = LogReplayLabeller() if labels else None
        os.makedirs(out_dir, exist_ok=True)
        self.batch, self.last_t, self.calls = -1, None, 0
        self.expert = None

    def __call__(self, sim_state, timestep: int, goal, info: dict) -> None:
        import jax
        from anchor_wm.data import payload_from_sim_state

        t = int(timestep)
        new_batch = self.last_t is None or t <= self.last_t
        self.batch += int(new_batch)
        self.last_t = t
        payload = payload_from_sim_state(jax.device_get(sim_state))
        sdc = np.argmax(payload["object_metadata_is_sdc"].astype(bool), axis=1)
        worlds = np.arange(len(sdc))
        if new_batch:   # the runner writes executed plans into the ego log; the first replan still sees the expert
            self.expert = np.stack([payload[k][worlds, sdc] for k in EXPERT_KEYS], 1).astype(np.float64)
        xy, yaw, p = ego_modes_world(self.model, sim_state, t, self.k)
        all_xy, all_yaw, all_p = ego_modes_world(self.model, sim_state, t, 0)
        out = {k: np.asarray(v) for k, v in info.items()}
        out.update({k: np.asarray(payload[k])[..., t] if k.startswith("log_trajectory") else np.asarray(payload[k])
                    for k in SCENE_KEYS})
        if new_batch:                                          # the whole log (other agents' futures) and the map
            out.update({k: np.asarray(payload[k]) for k in ROAD_KEYS})
            out.update({f"full_{k}": np.asarray(payload[k]) for k in SCENE_KEYS if k.startswith("log_trajectory")})
        out.update(timestep=t, batch=self.batch, goal=np.asarray(goal), awm_ego_xy=xy, awm_ego_yaw=yaw, awm_ego_probs=p,
                   awm_ego_all_xy=all_xy, awm_ego_all_yaw=all_yaw, awm_ego_all_probs=all_p)

        steps = out["best_world"].shape[1]
        ego_now = np.stack([payload[k][worlds, sdc, t] for k in EXPERT_KEYS[:3]], -1).astype(np.float64)
        expert, expert_valid = expert_future(self.expert, t, steps)
        aligned = align_to(expert, ego_now)
        out.update(expert_world=expert, expert_valid=expert_valid, expert_aligned=aligned,
                   expert_offset_m=np.linalg.norm(expert[:, 0, :2] - ego_now[:, :2], axis=-1))
        if self.scorer is not None:
            out["expert_score"] = np.asarray(self.scorer.score_extra(sim_state, t, aligned[:, None]))[:, 0]
        if self.labeller is not None:
            size = np.stack([payload["log_trajectory_length"][worlds, sdc, t],
                             payload["log_trajectory_width"][worlds, sdc, t]], -1)
            trajs = {"init": out["init_world"], "final": out["final_world"], "best": out["best_world"][:, None],
                     "expert_aligned": aligned[:, None]}
            for name in LABELLED:
                lab = [self.labeller(sim_state, t, w, np.asarray(trajs[name][w], np.float64), size[w]) for w in worlds]
                out[f"{name}_overlap_step"] = np.stack([a for a, _ in lab])
                out[f"{name}_offroad_step"] = np.stack([b for _, b in lab])
        np.savez_compressed(os.path.join(self.out_dir, f"b{self.batch:03d}_t{t:03d}.npz"), **out)
        self.calls += 1
