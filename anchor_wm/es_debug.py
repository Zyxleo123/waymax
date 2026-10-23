"""AnchorWM ego modes in Diffusion-ES: seeding the initial population, and per-replan dumps for plots.

AnchorEgoSeeder   `--anchor_ego_seed K`: the K ego modes the AnchorWM forecasts (closed_loop.ego_modes_world) are
                  appended to the diffusion samples of every replan's initial population; ES then scores and
                  resamples them like any other candidate.
ESDebugDumper     `--es_debug_dir DIR`: one npz per replan with the scene at the replan step (agents, roadgraph
                  once per batch, with the whole agent log), the AnchorWM ego modes, the initial and final ES pools with scores, the chosen
                  plan, and whether each candidate descends from an ego seed. Plotted by anchor_wm.viz_ego_seed.
"""

from __future__ import annotations

import os

import numpy as np

from anchor_wm.closed_loop import ego_modes_world

SCENE_KEYS = ("log_trajectory_x", "log_trajectory_y", "log_trajectory_yaw", "log_trajectory_valid",
              "log_trajectory_length", "log_trajectory_width", "object_metadata_is_sdc",
              "object_metadata_object_types")
ROAD_KEYS = ("roadgraph_points_x", "roadgraph_points_y", "roadgraph_points_types", "roadgraph_points_valid")


class AnchorEgoSeeder:
    def __init__(self, model, k: int) -> None:
        self.model, self.k = model, int(k)

    def __call__(self, sim_state, timestep: int):
        return ego_modes_world(self.model, sim_state, timestep, self.k)


class ESDebugDumper:
    def __init__(self, model, out_dir: str, k: int = 6) -> None:
        self.model, self.out_dir, self.k = model, out_dir, int(k)
        os.makedirs(out_dir, exist_ok=True)
        self.batch, self.last_t, self.calls = -1, None, 0

    def __call__(self, sim_state, timestep: int, goal, info: dict) -> None:
        import jax
        from anchor_wm.data import payload_from_sim_state

        t = int(timestep)
        new_batch = self.last_t is None or t <= self.last_t
        self.batch += int(new_batch)
        self.last_t = t
        payload = payload_from_sim_state(jax.device_get(sim_state))
        xy, yaw, p = ego_modes_world(self.model, sim_state, t, self.k)
        out = {k: np.asarray(v) for k, v in info.items()}
        out.update({k: np.asarray(payload[k])[..., t] if k.startswith("log_trajectory") else np.asarray(payload[k])
                    for k in SCENE_KEYS})
        if new_batch:                                          # the whole log (other agents' futures) and the map
            out.update({k: np.asarray(payload[k]) for k in ROAD_KEYS})
            out.update({f"full_{k}": np.asarray(payload[k]) for k in SCENE_KEYS if k.startswith("log_trajectory")})
        out.update(timestep=t, batch=self.batch, goal=np.asarray(goal), awm_ego_xy=xy, awm_ego_yaw=yaw, awm_ego_probs=p)
        np.savez_compressed(os.path.join(self.out_dir, f"b{self.batch:03d}_t{t:03d}.npz"), **out)
        self.calls += 1
