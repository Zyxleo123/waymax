"""Occupancy-risk scorer for ego candidates against a cached BEV occupancy grid.

SAT binary collision in `scores/scorer.py` is unchanged. This module only
exposes a continuous alternative:

  risk_t = max occupancy under the ego footprint at timestep t
  trajectory_risk = 1 - prod_t (1 - risk_t)   (or max_t risk_t)
  collision_score = 1 - trajectory_risk       (same 1=safe / 0=collision as SAT)

The occupancy grid is not ego-conditioned, so it is computed once per
world-model refresh and reused for every ES candidate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .grid import BEVGrid
from .rasterize import rasterize_oriented_boxes


def _wrap_to_pi_np(yaw: np.ndarray) -> np.ndarray:
    return (yaw + np.pi) % (2.0 * np.pi) - np.pi


def world_boxes_to_ego_frame(
    boxes_world: np.ndarray,
    origin_xy: np.ndarray,
    ego_yaw: float,
) -> np.ndarray:
    """boxes_world: [E, T, 5] world (x,y,yaw,l,w) -> ego-centric at WM refresh."""
    origin = np.asarray(origin_xy, dtype=np.float32).reshape(2)
    yaw0 = float(ego_yaw)
    xy = boxes_world[..., :2] - origin
    c = math.cos(-yaw0)
    s = math.sin(-yaw0)
    x = xy[..., 0] * c - xy[..., 1] * s
    y = xy[..., 0] * s + xy[..., 1] * c
    yaw = _wrap_to_pi_np(boxes_world[..., 2] - yaw0)
    out = np.concatenate(
        [x[..., None], y[..., None], yaw[..., None], boxes_world[..., 3:5]],
        axis=-1,
    ).astype(np.float32)
    return out


def align_occupancy_to_horizon(
    occ_bt: np.ndarray,
    *,
    traj_length: int,
    dt: float,
    future_dt: float,
) -> np.ndarray:
    """Map WM occupancy steps (future_dt) onto an ES horizon (dt)."""
    t_wm = occ_bt.shape[0]
    src_t = (np.arange(t_wm, dtype=np.float32) + 1.0) * float(future_dt)
    dst_t = np.arange(int(traj_length), dtype=np.float32) * float(dt)
    idx = np.searchsorted(src_t, dst_t, side="left")
    idx = np.clip(idx, 0, t_wm - 1)
    return occ_bt[idx]


@dataclass
class OccupancyRiskScorer:
    grid: BEVGrid
    tau: float = 0.25
    window_m: float = 8.0
    reduce: str = "or"  # "or" -> 1-prod(1-r_t); "max" -> max_t r_t
    last_refresh_ms: float = 0.0
    last_score_ms: float = 0.0

    def score_ego_candidates(
        self,
        ego_trajectories_world: np.ndarray,
        occupancy_thw: np.ndarray,
        *,
        origin_xy: np.ndarray,
        ego_yaw: float,
        dt: float,
        future_dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score many ego rollouts against one cached occupancy grid.

        ego_trajectories_world: [E, T, 5]
        occupancy_thw: [T_wm, H, W] in the refresh ego frame
        Returns (collision_score [E], trajectory_risk [E]) with SAT-like
        collision_score = 1 - risk.
        """
        t0 = time.perf_counter()
        ego = np.asarray(ego_trajectories_world, dtype=np.float32)
        if ego.ndim != 3 or ego.shape[-1] != 5:
            raise ValueError(f"ego_trajectories must be [E,T,5], got {ego.shape}")
        occ = align_occupancy_to_horizon(
            np.asarray(occupancy_thw, dtype=np.float32),
            traj_length=int(ego.shape[1]),
            dt=float(dt),
            future_dt=float(future_dt),
        )
        ego_local = world_boxes_to_ego_frame(ego, origin_xy, ego_yaw)
        e, t, _ = ego_local.shape
        xy = torch.from_numpy(ego_local[:, None, :, :2]).contiguous()
        yaw = torch.from_numpy(ego_local[:, None, :, 2]).contiguous()
        length = torch.from_numpy(ego_local[:, None, :, 3]).contiguous()
        width = torch.from_numpy(ego_local[:, None, :, 4]).contiguous()
        valid = torch.ones(e, 1, t, dtype=torch.bool)
        ego_mask = rasterize_oriented_boxes(
            xy,
            yaw,
            length,
            width,
            valid,
            self.grid,
            sigma_xy_m=None,
            sigma_scale=0.0,
            tau=self.tau,
            hard=True,
            window_m=self.window_m,
        )  # [E, T, H, W]
        occ_t = torch.from_numpy(occ)[None]
        under = ego_mask * occ_t
        risk_t = under.amax(dim=(-2, -1)).clamp(0.0, 1.0)  # [E, T]
        if self.reduce == "max":
            traj_risk = risk_t.amax(dim=-1)
        else:
            traj_risk = 1.0 - torch.prod(1.0 - risk_t, dim=-1)
        collision_score = 1.0 - traj_risk
        self.last_score_ms = 1e3 * (time.perf_counter() - t0)
        return collision_score.numpy(), traj_risk.numpy()
