from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_occupancy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    boundary_boost: float = 0.0,
) -> torch.Tensor:
    """Binary focal loss on sparse BEV occupancy. pred/target: [B,T,H,W] in [0,1]."""
    p = pred.clamp(1e-4, 1.0 - 1e-4)
    tgt = target.clamp(0.0, 1.0)
    pt = torch.where(tgt > 0.5, p, 1.0 - p)
    w = torch.where(tgt > 0.5, float(alpha), 1.0 - float(alpha))
    if float(boundary_boost) > 0:
        # Cheap stand-in for a distance transform: upweight the 1-cell ring
        # around GT occupied cells so near-miss 10–20 cm errors matter more.
        dilated = F.max_pool2d(tgt, kernel_size=3, stride=1, padding=1)
        ring = (dilated - tgt).clamp_min(0.0)
        w = w * (1.0 + float(boundary_boost) * ring)
    loss = -w * (1.0 - pt).pow(float(gamma)) * pt.log()
    return loss.mean()


def occupancy_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    pred_bin = pred >= float(threshold)
    tgt_bin = target >= 0.5
    inter = (pred_bin & tgt_bin).sum()
    union = (pred_bin | tgt_bin).sum().clamp_min(1)
    pred_pos = pred_bin.sum().clamp_min(1)
    tgt_pos = tgt_bin.sum().clamp_min(1)
    return {
        "occ_iou": inter.to(pred.dtype) / union.to(pred.dtype),
        "occ_precision": inter.to(pred.dtype) / pred_pos.to(pred.dtype),
        "occ_recall": inter.to(pred.dtype) / tgt_pos.to(pred.dtype),
        "occ_pos_rate": tgt_bin.to(pred.dtype).mean(),
    }
