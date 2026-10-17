"""Discriminator scene context (agents, map, traffic lights, goal) in the anchor ego frame.

NumPy only. `extract_context` reads per-world dicts with the sim_state cache field names
(`log_trajectory_*`, `roadgraph_points_*`, `object_metadata_is_sdc`, optionally
`log_traffic_light_*`); `es_adapter.scene_from_sim_state` builds them from a SimulatorState.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class DataConfig:
    anchors: tuple[int, ...] = (10, 20, 30, 40)
    anchors_per_scene: int = 3
    horizon: int = 25
    future_stride: int = 2
    history: int = 5
    history_stride: int = 2
    max_agents: int = 16
    max_map_points: int = 256
    # Keep every Nth roadgraph point (about 0.5 m apart) before picking the nearest
    # max_map_points; with 64 unstrided points the map only reached about 6 m.
    map_point_stride: int = 6
    max_lights: int = 16
    max_agent_distance_m: float = 80.0
    max_map_distance_m: float = 80.0
    # The fields below are unused, but recorded run configs and checkpoints store them.
    negative_per_anchor: int = 2
    benign_probability: float = 0.25
    negative_strength: float = 1.0
    # Fraction of negatives drawn from the diffusion residual model (if given).
    residual_fraction: float = 0.7
    # Residual negatives below this deviation percentile of real diffusion samples are
    # scaled up to it; None = no floor.
    residual_min_percentile: float | None = 25.0
    benign_residual_scale: tuple[float, float] = (0.1, 0.2)
    dt: float = 0.2


CONTEXT_KEYS = ("agents", "agent_mask", "map", "map_mask", "lights", "lights_mask", "goal", "ego_vel")

def _rotate(xy: np.ndarray, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([c * x + s * y, -s * x + c * y], axis=-1)


def to_ego_frame(xy_world: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    return _rotate(np.asarray(xy_world, dtype=np.float64) - origin, heading)


def _nearest_indices(
    xy: np.ndarray, mask: np.ndarray, count: int, distance: float
) -> np.ndarray:
    dsq = np.sum(xy.astype(np.float64) ** 2, axis=-1)
    valid = np.flatnonzero(mask.astype(bool) & np.isfinite(dsq) & (dsq <= distance**2))
    if len(valid) == 0:
        return valid
    dsq_valid = dsq[valid]
    if len(valid) > count:
        chosen = np.argpartition(dsq_valid, count - 1)[:count]
        valid = valid[chosen]
        dsq_valid = dsq_valid[chosen]
    return valid[np.argsort(dsq_valid)].astype(np.int32)


def extract_context(
    scene: dict[str, np.ndarray], ego: int, anchor: int, goal_world: np.ndarray, cfg: DataConfig
) -> dict | None:
    """Scene features at `anchor` in the ego frame; uses only data up to `anchor`.

    `scene` uses the cache field names. `log_trajectory_*` may hold either the log
    (training) or the simulated trajectory (closed-loop inference, via es_adapter).
    """
    x, y = scene["log_trajectory_x"], scene["log_trajectory_y"]
    yaw = scene["log_trajectory_yaw"]
    vx, vy = scene["log_trajectory_vel_x"], scene["log_trajectory_vel_y"]
    valid = scene["log_trajectory_valid"].astype(bool)
    if not valid[ego, anchor]:
        return None
    origin = np.array([x[ego, anchor], y[ego, anchor]], dtype=np.float64)
    heading = float(yaw[ego, anchor])
    if not np.isfinite(origin).all() or not np.isfinite(heading):
        return None

    goal = to_ego_frame(np.asarray(goal_world, dtype=np.float64), origin, heading)
    goal = np.clip(goal / 100.0, -5.0, 5.0).astype(np.float32)

    hist_steps = anchor - cfg.history_stride * np.arange(cfg.history - 1, -1, -1)
    if hist_steps[0] < 0:
        return None
    at_xy = np.stack([x[:, anchor], y[:, anchor]], axis=-1) - origin
    active = valid[:, anchor].copy()
    active[ego] = False
    neighbors = _nearest_indices(at_xy, active, cfg.max_agents, cfg.max_agent_distance_m)
    agents = np.zeros((cfg.max_agents, cfg.history, 8), dtype=np.float32)
    agent_mask = np.zeros((cfg.max_agents, cfg.history), dtype=bool)
    lengths, widths = scene["log_trajectory_length"], scene["log_trajectory_width"]
    for slot, agent_id in enumerate(neighbors):
        mask = valid[agent_id, hist_steps]
        points = to_ego_frame(np.stack([x[agent_id, hist_steps], y[agent_id, hist_steps]], axis=-1), origin, heading)
        velocities = _rotate(np.stack([vx[agent_id, hist_steps], vy[agent_id, hist_steps]], axis=-1), heading)
        relative_yaw = yaw[agent_id, hist_steps] - heading
        token = np.column_stack([
            points / 100.0, velocities / 25.0,
            np.sin(relative_yaw), np.cos(relative_yaw),
            np.asarray(lengths[agent_id, hist_steps]) / 10.0,
            np.asarray(widths[agent_id, hist_steps]) / 10.0,
        ]).astype(np.float32)
        good = mask & np.all(np.isfinite(token), axis=-1)
        agents[slot, good] = token[good]
        agent_mask[slot, good] = True

    rg_local = to_ego_frame(np.stack([scene["roadgraph_points_x"], scene["roadgraph_points_y"]], axis=-1), origin, heading)
    rg_valid = scene["roadgraph_points_valid"].astype(bool).copy()
    if cfg.map_point_stride > 1:
        keep = np.zeros_like(rg_valid)
        keep[np.flatnonzero(rg_valid)[::cfg.map_point_stride]] = True
        rg_valid = keep
    chosen = _nearest_indices(rg_local, rg_valid, cfg.max_map_points, cfg.max_map_distance_m)
    map_tokens = np.zeros((cfg.max_map_points, 5), dtype=np.float32)
    map_mask = np.zeros((cfg.max_map_points,), dtype=bool)
    if len(chosen):
        dirs = _rotate(np.stack([
            scene["roadgraph_points_dir_x"][chosen], scene["roadgraph_points_dir_y"][chosen]
        ], axis=-1), heading)
        map_tokens[:len(chosen)] = np.column_stack([
            rg_local[chosen] / 100.0,
            dirs,
            np.clip(scene["roadgraph_points_types"][chosen], 0, 32) / 32.0,
        ]).astype(np.float32)
        map_mask[:len(chosen)] = np.all(np.isfinite(map_tokens[:len(chosen)]), axis=-1)
        map_tokens[~map_mask] = 0.0

    lights = np.zeros((cfg.max_lights, 4), dtype=np.float32)
    lights_mask = np.zeros((cfg.max_lights,), dtype=bool)
    if all(f"log_traffic_light_{key}" in scene for key in ("x", "y", "state", "valid")):
        lv = scene["log_traffic_light_valid"][:, anchor].astype(bool)
        local_lights = to_ego_frame(np.stack([
            scene["log_traffic_light_x"][:, anchor], scene["log_traffic_light_y"][:, anchor]
        ], axis=-1), origin, heading)
        lids = _nearest_indices(local_lights, lv, cfg.max_lights, cfg.max_map_distance_m)
        if len(lids):
            lights[:len(lids), :2] = local_lights[lids] / 100.0
            lights[:len(lids), 2] = np.clip(scene["log_traffic_light_state"][lids, anchor], 0, 16) / 16.0
            lights[:len(lids), 3] = 1.0
            lights_mask[:len(lids)] = True

    ego_vel = _rotate(np.array([vx[ego, anchor], vy[ego, anchor]]), heading) / 25.0
    if not np.all(np.isfinite(ego_vel)):
        ego_vel[:] = 0.0
    return {
        "agents": agents,
        "agent_mask": agent_mask,
        "map": map_tokens,
        "map_mask": map_mask,
        "lights": lights,
        "lights_mask": lights_mask,
        "goal": goal,
        "ego_vel": ego_vel.astype(np.float32),
        "origin": origin,
        "heading": heading,
    }
