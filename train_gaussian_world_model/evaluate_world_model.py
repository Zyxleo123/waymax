from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model.config import WorldModelConfig
from model.world_model.network import LatentMultiAgentWorldModel


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline WM-vs-CV future-forecast sanity check.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--world_model_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20, 30, 40))
    p.add_argument("--val_fraction", type=float, default=0.1)
    return p.parse_args()


def _ade_fde(error_m: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid_f = valid.float()
    ade = (error_m * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    t = error_m.shape[-1]
    idx = torch.arange(t, device=error_m.device).view(1, 1, t)
    last_idx = torch.where(valid, idx, torch.full_like(idx, -1)).amax(dim=-1)
    has = last_idx >= 0
    safe = last_idx.clamp_min(0)
    fde_all = torch.gather(error_m, -1, safe[..., None]).squeeze(-1)
    fde = (fde_all * has.float()).sum() / has.float().sum().clamp_min(1.0)
    return ade, fde


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

    wm_ade_sum = wm_fde_sum = cv_ade_sum = cv_fde_sum = 0.0
    yaw_sum = 0.0
    sigma_x_sum = sigma_y_sum = 0.0
    batches = 0

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            target_xy = batch["target_delta_xy"].float()
            target_yaw = F.normalize(batch["target_yaw_vec"].float(), dim=-1, eps=1e-6)
            valid = batch["target_valid"].bool()

            wm_err = torch.linalg.vector_norm(out["pred_delta_xy_norm"] - target_xy, dim=-1) * cfg.max_range
            wm_ade, wm_fde = _ade_fde(wm_err, valid)

            # Constant-velocity baseline in the same ego-relative coordinate system.
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
            cv_ade, cv_fde = _ade_fde(cv_err, valid)

            yaw_cos = (out["pred_yaw_vec"] * target_yaw).sum(dim=-1).clamp(-1.0, 1.0)
            yaw_deg = torch.rad2deg(torch.acos(yaw_cos))
            yaw_mae = (yaw_deg * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

            wm_ade_sum += float(wm_ade.cpu())
            wm_fde_sum += float(wm_fde.cpu())
            cv_ade_sum += float(cv_ade.cpu())
            cv_fde_sum += float(cv_fde.cpu())
            yaw_sum += float(yaw_mae.cpu())
            w = valid.float()
            sigma = out["pred_sigma_xy_m"]
            sigma_x_sum += float((sigma[..., 0] * w).sum() / w.sum().clamp_min(1.0))
            sigma_y_sum += float((sigma[..., 1] * w).sum() / w.sum().clamp_min(1.0))
            batches += 1

    if batches == 0:
        raise RuntimeError("No validation batches were produced.")

    print(f"batches={batches}")
    print(f"World Model: ADE={wm_ade_sum / batches:.3f} m  FDE={wm_fde_sum / batches:.3f} m")
    print(f"Constant Vel: ADE={cv_ade_sum / batches:.3f} m  FDE={cv_fde_sum / batches:.3f} m")
    print(f"World Model yaw MAE={yaw_sum / batches:.2f} deg")
    print(f"World Model mean sigma_x={sigma_x_sum / batches:.3f} m  sigma_y={sigma_y_sum / batches:.3f} m")
    print("Target criterion before FM-ES integration: WM should beat CV, especially on nearby dynamic agents.")


if __name__ == "__main__":
    main()
