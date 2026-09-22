from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model.config import WorldModelConfig
from model.world_model.network import LatentMultiAgentWorldModel
from occupancy.grid import BEVGrid
from occupancy.loss import occupancy_metrics
from occupancy.rasterize import occupancy_from_trajectory_head
from occupancy.scorer import OccupancyRiskScorer


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Occupancy WM vs CV sanity check.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--world_model_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20, 30, 40))
    p.add_argument("--val_fraction", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.world_model_ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = WorldModelConfig.from_dict(dict(ckpt["config"]))
    model = LatentMultiAgentWorldModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    grid = BEVGrid.from_config(cfg)
    scorer = OccupancyRiskScorer(grid=grid, tau=float(cfg.occ_tau), window_m=float(cfg.occ_window_m))

    paths = resolve_cache_paths(args.cache_dir, None)
    _, val_paths = split_cache_paths(paths, val_fraction=args.val_fraction)
    loader = build_world_model_cache_loader(
        args.cache_dir,
        cfg,
        cache_paths=val_paths,
        batch_size=args.batch_size,
        anchor_steps=args.anchor_steps,
        shuffle=False,
    )

    sums = {
        "ade": 0.0,
        "fde": 0.0,
        "iou": 0.0,
        "prec": 0.0,
        "rec": 0.0,
        "raster_ms": 0.0,
        "score_ms": 0.0,
    }
    batches = 0
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            t0 = time.perf_counter()
            out = model(batch)
            pred_occ, gt_occ = occupancy_from_trajectory_head(
                out,
                batch,
                max_range=float(cfg.max_range),
                grid=grid,
                sigma_scale=float(cfg.occ_sigma_scale),
                tau=float(cfg.occ_tau),
                window_m=float(cfg.occ_window_m),
            )
            raster_ms = 1e3 * (time.perf_counter() - t0)
            m = occupancy_metrics(pred_occ, gt_occ, threshold=float(cfg.occ_threshold))

            target_xy = batch["target_delta_xy"].float()
            valid = batch["target_valid"].bool()
            wm_err = torch.linalg.vector_norm(out["pred_delta_xy_norm"] - target_xy, dim=-1) * cfg.max_range
            valid_f = valid.float()
            ade = (wm_err * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            t = wm_err.shape[-1]
            idx = torch.arange(t, device=wm_err.device).view(1, 1, t)
            last = torch.where(valid, idx, torch.full_like(idx, -1)).amax(-1)
            has = last >= 0
            fde = torch.gather(wm_err, -1, last.clamp_min(0)[..., None]).squeeze(-1)
            fde = (fde * has.float()).sum() / has.float().sum().clamp_min(1.0)

            # Dummy ES call: one ego box at the origin, times occupancy-score runtime.
            ego = np_ego_at_origin(int(cfg.future_steps), float(cfg.future_dt))
            occ0 = pred_occ[0].detach().cpu().numpy()
            scorer.score_ego_candidates(
                ego,
                occ0,
                origin_xy=np.zeros(2, dtype=float),
                ego_yaw=0.0,
                dt=float(cfg.future_dt),
                future_dt=float(cfg.future_dt),
            )

            sums["ade"] += float(ade.cpu())
            sums["fde"] += float(fde.cpu())
            sums["iou"] += float(m["occ_iou"].cpu())
            sums["prec"] += float(m["occ_precision"].cpu())
            sums["rec"] += float(m["occ_recall"].cpu())
            sums["raster_ms"] += float(raster_ms)
            sums["score_ms"] += float(scorer.last_score_ms)
            batches += 1

    if batches == 0:
        raise RuntimeError("No validation batches were produced.")
    n = batches
    print(f"batches={n}")
    print(f"WM ADE={sums['ade']/n:.3f} m  FDE={sums['fde']/n:.3f} m")
    print(f"Occupancy IoU={sums['iou']/n:.3f}  P={sums['prec']/n:.3f}  R={sums['rec']/n:.3f}")
    print(f"runtime rasterize/refresh={sums['raster_ms']/n:.1f} ms  occupancy-score={sums['score_ms']/n:.1f} ms")
    print("SAT binary collision in scores/scorer.py is unchanged; occupancy-risk is an add-on scorer.")


def np_ego_at_origin(t: int, dt: float):
    ego = np.zeros((1, t, 5), dtype=np.float32)
    ego[..., 3] = 4.0
    ego[..., 4] = 2.0
    return ego


if __name__ == "__main__":
    main()
