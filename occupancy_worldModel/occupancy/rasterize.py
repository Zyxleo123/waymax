"""Rasterize oriented agent boxes onto an ego-centric BEV occupancy grid.

This is the occupancy "head": it is not a BEV CNN. Predicted Gaussian
trajectories (μ, σ, yaw) plus logged length/width are splatted into
[B, T, H, W] occupancy. Surrounding agents are aggregated with a max
(approach A), which is cheap enough to cache once per world-model refresh.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .grid import BEVGrid


def _wrap_to_pi(yaw: torch.Tensor) -> torch.Tensor:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def future_boxes_from_outputs(
    *,
    pred_delta_xy_norm: torch.Tensor,
    pred_yaw_vec: torch.Tensor,
    current_local_xy: torch.Tensor,
    current_yaw_rel: torch.Tensor,
    current_length: torch.Tensor,
    current_width: torch.Tensor,
    max_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode ego-frame future boxes from the trajectory head.

    Returns xy [B,N,T,2], yaw [B,N,T], length [B,N,T], width [B,N,T] in meters.
    """
    xy = current_local_xy[:, :, None, :] + pred_delta_xy_norm * float(max_range)
    yaw_delta = torch.atan2(pred_yaw_vec[..., 0], pred_yaw_vec[..., 1])
    yaw = _wrap_to_pi(current_yaw_rel[:, :, None] + yaw_delta)
    t = pred_delta_xy_norm.shape[2]
    length = current_length[:, :, None].expand(-1, -1, t)
    width = current_width[:, :, None].expand(-1, -1, t)
    return xy, yaw, length, width


def rasterize_oriented_boxes(
    xy: torch.Tensor,
    yaw: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    valid: torch.Tensor,
    grid: BEVGrid,
    *,
    sigma_xy_m: torch.Tensor | None = None,
    sigma_scale: float = 1.0,
    tau: float = 0.25,
    hard: bool = False,
    window_m: float = 8.0,
) -> torch.Tensor:
    """Aggregate all agents into occupancy [B, T, H, W] in [0, 1].

    Soft occupancy uses a signed-distance kernel around each oriented box.
    Optional diagonal σ expands the box before the kernel, so 10–20 cm
    trajectory uncertainty becomes a thicker occupied region.

    Memory: each agent is splatted in a local window, not the full grid.
    """
    if xy.ndim != 4 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be [B,N,T,2], got {tuple(xy.shape)}")
    b, n, t, _ = xy.shape
    h, w = grid.height, grid.width
    device = xy.device
    dtype = xy.dtype
    cell = float(grid.cell_size)
    radius = max(1, int(math.ceil(float(window_m) / cell)))

    half_l = 0.5 * length
    half_w = 0.5 * width
    if sigma_xy_m is not None:
        extra = float(sigma_scale) * 0.5 * (sigma_xy_m[..., 0] + sigma_xy_m[..., 1])
        extra = extra.clamp(max=float(window_m))
        half_l = half_l + extra
        half_w = half_w + extra

    offsets = torch.arange(-radius, radius + 1, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    # Cell index of the box center, then local window.
    cx = torch.floor((xy[..., 0] - float(grid.x_min)) / cell).long()
    cy = torch.floor((xy[..., 1] - float(grid.y_min)) / cell).long()
    ix = cx[..., None, None] + ox.view(1, 1, 1, *ox.shape)
    iy = cy[..., None, None] + oy.view(1, 1, 1, *oy.shape)
    in_b = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

    cell_x = float(grid.x_min) + (ix.to(dtype) + 0.5) * cell
    cell_y = float(grid.y_min) + (iy.to(dtype) + 0.5) * cell
    dx = cell_x - xy[..., 0, None, None]
    dy = cell_y - xy[..., 1, None, None]
    c = torch.cos(yaw)[..., None, None]
    s = torch.sin(yaw)[..., None, None]
    # Agent body frame: x along length, y along width.
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    qx = local_x.abs() - half_l[..., None, None]
    qy = local_y.abs() - half_w[..., None, None]
    dist_out = torch.sqrt(qx.clamp_min(0.0).square() + qy.clamp_min(0.0).square() + 1e-12)
    dist_in = torch.minimum(qx, qy).clamp_max(0.0)
    sdf = dist_out + dist_in

    if hard or float(tau) <= 0:
        prob = (sdf <= 0).to(dtype)
    else:
        prob = torch.sigmoid(-sdf / float(tau))
    prob = prob * valid.to(dtype)[..., None, None] * in_b.to(dtype)

    occ = xy.new_zeros(b, t, h, w)
    batch_idx = torch.arange(b, device=device).view(b, 1, 1, 1, 1).expand_as(ix)
    time_idx = torch.arange(t, device=device).view(1, 1, t, 1, 1).expand_as(ix)
    ix_c = ix.clamp(0, w - 1)
    iy_c = iy.clamp(0, h - 1)
    linear = ((batch_idx * t + time_idx) * h + iy_c) * w + ix_c
    occ.view(-1).scatter_reduce_(
        0,
        linear.reshape(-1),
        prob.reshape(-1),
        reduce="amax",
        include_self=True,
    )
    return occ.clamp(0.0, 1.0)


def occupancy_from_trajectory_head(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    max_range: float,
    grid: BEVGrid,
    sigma_scale: float,
    tau: float,
    window_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize predicted (soft) and GT (hard) occupancy from trajectory tensors."""
    pred_xy, pred_yaw, length, width = future_boxes_from_outputs(
        pred_delta_xy_norm=out["pred_delta_xy_norm"],
        pred_yaw_vec=out["pred_yaw_vec"],
        current_local_xy=batch["current_local_xy"].float(),
        current_yaw_rel=batch["current_yaw_rel"].float(),
        current_length=batch["current_length"].float(),
        current_width=batch["current_width"].float(),
        max_range=max_range,
    )
    gt_xy, gt_yaw, gt_len, gt_wid = future_boxes_from_outputs(
        pred_delta_xy_norm=batch["target_delta_xy"].float(),
        pred_yaw_vec=F.normalize(batch["target_yaw_vec"].float(), dim=-1, eps=1e-6),
        current_local_xy=batch["current_local_xy"].float(),
        current_yaw_rel=batch["current_yaw_rel"].float(),
        current_length=batch["current_length"].float(),
        current_width=batch["current_width"].float(),
        max_range=max_range,
    )
    valid = batch["target_valid"].bool()
    sigma = out.get("pred_sigma_xy_m")
    pred_occ = rasterize_oriented_boxes(
        pred_xy,
        pred_yaw,
        length,
        width,
        valid,
        grid,
        sigma_xy_m=sigma,
        sigma_scale=sigma_scale,
        tau=tau,
        hard=False,
        window_m=window_m,
    )
    gt_occ = rasterize_oriented_boxes(
        gt_xy,
        gt_yaw,
        gt_len,
        gt_wid,
        valid,
        grid,
        sigma_xy_m=None,
        sigma_scale=0.0,
        tau=tau,
        hard=True,
        window_m=window_m,
    )
    return pred_occ, gt_occ
