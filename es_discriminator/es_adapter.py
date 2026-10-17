"""Bridge between Diffusion-ES outputs / Waymax states and discriminator inputs.

Diffusion-ES candidates (`aux["es_pop_world_bkt5"]`, `trajectory_world_bt5`) are
world-frame [x, y, yaw, vx, vy] at 0.1 s with index 0 = the anchor pose. The
discriminator wants anchor-ego-frame x, y, yaw at 0.2 s, excluding the anchor:
world indices 2, 4, ..., 2*horizon.

NumPy only; `sim_state` is any Waymax-like SimulatorState (JAX or NumPy leaves).
"""

from __future__ import annotations

import numpy as np

from .data import DataConfig, extract_context, to_ego_frame, wrap_pi


def pool_world_to_ego(
    pool_world: np.ndarray, origin_xy: np.ndarray, heading: float, cfg: DataConfig = DataConfig()
) -> np.ndarray:
    """[...,T_world,>=3] world 0.1 s trajectories -> [...,horizon,3] ego-frame 0.2 s."""
    pool = np.asarray(pool_world, dtype=np.float64)
    idx = cfg.future_stride * np.arange(1, cfg.horizon + 1)
    if pool.shape[-2] <= idx[-1]:
        raise ValueError(f"Need at least {idx[-1] + 1} world steps, got {pool.shape[-2]}")
    sub = pool[..., idx, :]
    xy = to_ego_frame(sub[..., :2], np.asarray(origin_xy, dtype=np.float64), heading)
    yaw = wrap_pi(sub[..., 2] - heading)
    return np.concatenate([xy, yaw[..., None]], axis=-1).astype(np.float32)


def scene_from_sim_state(sim_state, world_idx: int, *, use_sim_trajectory: bool = True) -> dict[str, np.ndarray]:
    """Per-world dict with the cache field names. In closed loop, the simulated
    trajectory (what actually happened, incl. IDM agents) replaces the log."""
    traj = sim_state.sim_trajectory if use_sim_trajectory else sim_state.log_trajectory
    w = int(world_idx)
    get = lambda leaf: np.asarray(leaf)[w]  # noqa: E731
    scene = {
        "object_metadata_is_sdc": get(sim_state.object_metadata.is_sdc),
        "log_trajectory_x": get(traj.x), "log_trajectory_y": get(traj.y), "log_trajectory_yaw": get(traj.yaw),
        "log_trajectory_vel_x": get(traj.vel_x), "log_trajectory_vel_y": get(traj.vel_y),
        "log_trajectory_valid": get(traj.valid),
        "log_trajectory_length": get(traj.length), "log_trajectory_width": get(traj.width),
        "roadgraph_points_x": get(sim_state.roadgraph_points.x),
        "roadgraph_points_y": get(sim_state.roadgraph_points.y),
        "roadgraph_points_valid": get(sim_state.roadgraph_points.valid),
        "roadgraph_points_dir_x": get(sim_state.roadgraph_points.dir_x),
        "roadgraph_points_dir_y": get(sim_state.roadgraph_points.dir_y),
        "roadgraph_points_types": get(sim_state.roadgraph_points.types),
    }
    lights = sim_state.log_traffic_light
    scene.update({
        "log_traffic_light_x": get(lights.x), "log_traffic_light_y": get(lights.y),
        "log_traffic_light_state": get(lights.state), "log_traffic_light_valid": get(lights.valid),
    })
    return scene


def context_from_sim_state(
    sim_state, world_idx: int, timestep: int, goal_xy: np.ndarray, cfg: DataConfig = DataConfig(),
    *, use_sim_trajectory: bool = True,
) -> dict | None:
    """Discriminator context for one world at `timestep`, plus `origin` and `heading`
    (needed by `pool_world_to_ego`). None if the ego is invalid at `timestep`."""
    scene = scene_from_sim_state(sim_state, world_idx, use_sim_trajectory=use_sim_trajectory)
    is_sdc = scene["object_metadata_is_sdc"].astype(bool)
    if np.count_nonzero(is_sdc) != 1:
        return None
    ego = int(np.flatnonzero(is_sdc)[0])
    return extract_context(scene, ego, int(timestep), np.asarray(goal_xy, dtype=np.float64), cfg)
