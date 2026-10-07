from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model.config import WorldModelConfig
from model.world_model.network import LatentMultiAgentWorldModel
from train_world_model import _loss_and_metrics


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline multimodal WM-vs-CV sanity check.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--world_model_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=100)
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
        "min_ade_m": 0.0,
        "min_fde_m": 0.0,
        "top1_ade_m": 0.0,
        "top1_fde_m": 0.0,
        "mode_entropy": 0.0,
        "top1_conf": 0.0,
        "sigma_x_m": 0.0,
        "sigma_y_m": 0.0,
        "cv_ade": 0.0,
        "cv_fde": 0.0,
    }
    batches = 0

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _, metrics = _loss_and_metrics(model, batch, cfg)

            target_xy = batch["target_delta_xy"].float()
            valid = batch["target_valid"].bool()
            current_vel_norm = batch["agent_history"][:, :, -1, 2:4].float()
            times = (
                torch.arange(1, cfg.future_steps + 1, device=device, dtype=torch.float32)
                * float(cfg.future_dt)
            )
            cv_delta_norm = (
                current_vel_norm[:, :, None, :]
                * float(cfg.max_velocity)
                * times[None, None, :, None]
                / float(cfg.max_range)
            )
            cv_err = torch.linalg.vector_norm(cv_delta_norm - target_xy, dim=-1) * cfg.max_range
            valid_f = valid.float()
            cv_ade = (cv_err * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            t = cv_err.shape[-1]
            idx = torch.arange(t, device=cv_err.device).view(1, 1, t)
            last_idx = torch.where(valid, idx, torch.full_like(idx, -1)).amax(dim=-1)
            has = last_idx >= 0
            safe = last_idx.clamp_min(0)
            cv_fde = torch.gather(cv_err, -1, safe[..., None]).squeeze(-1)
            cv_fde = (cv_fde * has.float()).sum() / has.float().sum().clamp_min(1.0)

            for key in (
                "min_ade_m",
                "min_fde_m",
                "top1_ade_m",
                "top1_fde_m",
                "mode_entropy",
                "top1_conf",
                "sigma_x_m",
                "sigma_y_m",
            ):
                sums[key] += float(metrics[key])
            sums["cv_ade"] += float(cv_ade.cpu())
            sums["cv_fde"] += float(cv_fde.cpu())
            batches += 1

    if batches == 0:
        raise RuntimeError("No validation batches were produced.")

    k = int(cfg.num_modes)
    print(f"batches={batches}  K={k}")
    print(f"minADE@{k}={sums['min_ade_m'] / batches:.3f} m  minFDE@{k}={sums['min_fde_m'] / batches:.3f} m")
    print(f"top-1 ADE={sums['top1_ade_m'] / batches:.3f} m  top-1 FDE={sums['top1_fde_m'] / batches:.3f} m")
    print(f"mode entropy={sums['mode_entropy'] / batches:.3f}  top-1 conf={sums['top1_conf'] / batches:.3f}")
    print(f"mean sigma_x={sums['sigma_x_m'] / batches:.3f} m  sigma_y={sums['sigma_y_m'] / batches:.3f} m")
    print(f"Constant Vel: ADE={sums['cv_ade'] / batches:.3f} m  FDE={sums['cv_fde'] / batches:.3f} m")


if __name__ == "__main__":
    main()
