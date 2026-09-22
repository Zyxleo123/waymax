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
    p = argparse.ArgumentParser(description="Unconditional vs ego-conditioned surrounding-agent ADE.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--world_model_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20, 30, 40))
    p.add_argument("--val_fraction", type=float, default=0.1)
    return p.parse_args()


def _ade_fde(err_m: torch.Tensor, valid: torch.Tensor) -> tuple[float, float]:
    valid_f = valid.float()
    ade = (err_m * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    t = err_m.shape[-1]
    idx = torch.arange(t, device=err_m.device).view(1, 1, t)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).amax(-1)
    has = last >= 0
    fde = torch.gather(err_m, -1, last.clamp_min(0)[..., None]).squeeze(-1)
    fde = (fde * has.float()).sum() / has.float().sum().clamp_min(1.0)
    return float(ade.cpu()), float(fde.cpu())


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

    sums = {"cond_ade": 0.0, "cond_fde": 0.0, "uncond_ade": 0.0, "uncond_fde": 0.0, "cv_ade": 0.0, "cv_fde": 0.0}
    batches = 0
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            scene = model.encode_scene(batch)
            out_uncond = model.decode_from_scene(scene, ego_future=None)
            out_cond = model.decode_from_scene(
                scene,
                ego_future=batch.get("ego_future"),
                ego_future_valid=batch.get("ego_future_valid"),
            )
            target = batch["target_delta_xy"].float()
            valid = batch["target_valid"].bool()
            uncond_err = torch.linalg.vector_norm(out_uncond["pred_delta_xy_norm"] - target, dim=-1) * cfg.max_range
            cond_err = torch.linalg.vector_norm(out_cond["pred_delta_xy_norm"] - target, dim=-1) * cfg.max_range
            ua, uf = _ade_fde(uncond_err, valid)
            ca, cf = _ade_fde(cond_err, valid)

            current_vel_norm = batch["agent_history"][:, :, -1, 2:4].float()
            times = torch.arange(1, cfg.future_steps + 1, device=device, dtype=torch.float32) * float(cfg.future_dt)
            cv_delta = current_vel_norm[:, :, None, :] * float(cfg.max_velocity) * times[None, None, :, None] / float(cfg.max_range)
            cv_err = torch.linalg.vector_norm(cv_delta - target, dim=-1) * cfg.max_range
            cva, cvf = _ade_fde(cv_err, valid)

            sums["uncond_ade"] += ua
            sums["uncond_fde"] += uf
            sums["cond_ade"] += ca
            sums["cond_fde"] += cf
            sums["cv_ade"] += cva
            sums["cv_fde"] += cvf
            batches += 1

    if batches == 0:
        raise RuntimeError("No validation batches were produced.")
    n = batches
    print(f"batches={n}  ego_conditioning={cfg.ego_conditioning}")
    print(f"Unconditional WM: ADE={sums['uncond_ade']/n:.3f} m  FDE={sums['uncond_fde']/n:.3f} m")
    print(f"Ego-conditioned WM (logged ego future): ADE={sums['cond_ade']/n:.3f} m  FDE={sums['cond_fde']/n:.3f} m")
    print(f"Constant Vel: ADE={sums['cv_ade']/n:.3f} m  FDE={sums['cv_fde']/n:.3f} m")
    print(
        "Training only sees one realized ego future per scene; this eval conditions on that "
        "logged ego trajectory, not on ES candidate rollouts."
    )


if __name__ == "__main__":
    main()
