#!/usr/bin/env python3
"""Offline CV vs World-Model forecast comparison against logged GT.

No ego planner / Diffusion-ES / FM-ES. For each cached scenario + anchor time:
  - CV: constant-velocity extrapolation in ego frame
  - WM: learned world-model future
  - GT: logged future displacements (supervised targets)

Reports ADE / FDE (meters) overall and for nearby dynamic agents.

Uses the existing sim_state NPZ cache (fast, scalable). Run from waymax_rs with
the world_model package on PYTHONPATH (see slurm_compare_cv_vs_wm.sh).

Example::

  python scripts/compare_cv_vs_wm_forecast.py \\
    --cache_dir /zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \\
    --world_model_ckpt /zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt \\
    --max_batches 0 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
WM_ROOT = Path(os.environ.get("WORLD_MODEL_ROOT", "/zfsauton2/home/sbellad/world_model"))

# Prefer the training package for cache loading + matching network code.
for root in (WM_ROOT, REPO_ROOT):
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

from data.utils import resolve_cache_paths, split_cache_paths  # noqa: E402
from data.world_model_cache_loader import build_world_model_cache_loader  # noqa: E402
from model.world_model.config import WorldModelConfig  # noqa: E402
from model.world_model.network import LatentMultiAgentWorldModel  # noqa: E402


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache_dir",
        type=str,
        default="/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz",
    )
    p.add_argument(
        "--world_model_ckpt",
        type=str,
        default="/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="0 = use the full loader (all selected shards).",
    )
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20, 30, 40))
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "all"],
        help="Which cache split to evaluate.",
    )
    p.add_argument(
        "--nearby_radius_m",
        type=float,
        default=20.0,
        help="Extra ADE/FDE for dynamic agents within this ego-frame radius at t=0.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/zfsauton/scratch/sbellad/immediate_exp",
    )
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def _ade_fde(error_m: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """error_m / valid: [B, N, T]."""
    valid_f = valid.float()
    denom = valid_f.sum().clamp_min(1.0)
    ade = (error_m * valid_f).sum() / denom

    t = error_m.shape[-1]
    idx = torch.arange(t, device=error_m.device).view(1, 1, t)
    last_idx = torch.where(valid, idx, torch.full_like(idx, -1)).amax(dim=-1)
    has = last_idx >= 0
    safe = last_idx.clamp_min(0)
    fde_all = torch.gather(error_m, -1, safe[..., None]).squeeze(-1)
    fde = (fde_all * has.float()).sum() / has.float().sum().clamp_min(1.0)
    return ade, fde


def _nearby_valid(
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,
    *,
    radius_m: float,
    max_range: float,
) -> torch.Tensor:
    """Mask agents currently within radius_m of ego (ego-frame xy from last history step)."""
    # agent_history: [B, N, H, D], xy are indices 0:2, already / max_range.
    xy_now = batch["agent_history"][:, :, -1, 0:2].float() * float(max_range)
    dist = torch.linalg.vector_norm(xy_now, dim=-1)  # [B, N]
    nearby = dist <= float(radius_m)
    # Exclude ego if marked invalid in target; agent_valid already drops invalids.
    return valid & nearby[:, :, None]


class MetricAccum:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def add(self, value: float) -> None:
        self.sum += float(value)
        self.count += 1

    def mean(self) -> float:
        return self.sum / self.count if self.count else float("nan")


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.world_model_ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = WorldModelConfig.from_dict(dict(ckpt["config"]))
    model = LatentMultiAgentWorldModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    paths = resolve_cache_paths(args.cache_dir, None)
    train_paths, val_paths = split_cache_paths(paths, val_fraction=args.val_fraction)
    if args.split == "val":
        cache_paths = val_paths
    elif args.split == "train":
        cache_paths = train_paths
    else:
        cache_paths = paths

    loader = build_world_model_cache_loader(
        args.cache_dir,
        cfg,
        cache_paths=cache_paths,
        batch_size=int(args.batch_size),
        anchor_steps=tuple(args.anchor_steps),
        shuffle=False,
    )

    metrics = {
        "wm_ade": MetricAccum(),
        "wm_fde": MetricAccum(),
        "cv_ade": MetricAccum(),
        "cv_fde": MetricAccum(),
        "wm_ade_nearby": MetricAccum(),
        "wm_fde_nearby": MetricAccum(),
        "cv_ade_nearby": MetricAccum(),
        "cv_fde_nearby": MetricAccum(),
        "wm_yaw_mae_deg": MetricAccum(),
        "batches_wm_better_ade": 0,
        "batches_cv_better_ade": 0,
        "batches_tie_ade": 0,
    }

    batches = 0
    print(
        f"[compare_cv_vs_wm] ckpt={ckpt_path} device={device} split={args.split} "
        f"shards={len(cache_paths)} anchors={args.anchor_steps} batch_size={args.batch_size}",
        flush=True,
    )

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= int(args.max_batches):
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)

            target_xy = batch["target_delta_xy"].float()
            target_yaw = F.normalize(batch["target_yaw_vec"].float(), dim=-1, eps=1e-6)
            valid = batch["target_valid"].bool()

            wm_err = (
                torch.linalg.vector_norm(out["pred_delta_xy_norm"] - target_xy, dim=-1)
                * float(cfg.max_range)
            )
            wm_ade, wm_fde = _ade_fde(wm_err, valid)

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
            cv_err = torch.linalg.vector_norm(cv_delta_norm - target_xy, dim=-1) * float(
                cfg.max_range
            )
            cv_ade, cv_fde = _ade_fde(cv_err, valid)

            nearby = _nearby_valid(
                batch,
                valid,
                radius_m=float(args.nearby_radius_m),
                max_range=float(cfg.max_range),
            )
            wm_ade_n, wm_fde_n = _ade_fde(wm_err, nearby)
            cv_ade_n, cv_fde_n = _ade_fde(cv_err, nearby)

            yaw_cos = (out["pred_yaw_vec"] * target_yaw).sum(dim=-1).clamp(-1.0, 1.0)
            yaw_deg = torch.rad2deg(torch.acos(yaw_cos))
            yaw_mae = (yaw_deg * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

            metrics["wm_ade"].add(float(wm_ade))
            metrics["wm_fde"].add(float(wm_fde))
            metrics["cv_ade"].add(float(cv_ade))
            metrics["cv_fde"].add(float(cv_fde))
            metrics["wm_ade_nearby"].add(float(wm_ade_n))
            metrics["wm_fde_nearby"].add(float(wm_fde_n))
            metrics["cv_ade_nearby"].add(float(cv_ade_n))
            metrics["cv_fde_nearby"].add(float(cv_fde_n))
            metrics["wm_yaw_mae_deg"].add(float(yaw_mae))

            if float(wm_ade) < float(cv_ade) - 1e-6:
                metrics["batches_wm_better_ade"] += 1
            elif float(cv_ade) < float(wm_ade) - 1e-6:
                metrics["batches_cv_better_ade"] += 1
            else:
                metrics["batches_tie_ade"] += 1

            batches += 1
            if batches % 50 == 0:
                print(
                    f"  batch {batches}: WM ADE={metrics['wm_ade'].mean():.3f} "
                    f"CV ADE={metrics['cv_ade'].mean():.3f}",
                    flush=True,
                )

    if batches == 0:
        raise RuntimeError("No batches produced — check --cache_dir / --split.")

    summary: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "world_model_ckpt": str(ckpt_path),
        "cache_dir": str(args.cache_dir),
        "device": str(device),
        "split": args.split,
        "num_shards": len(cache_paths),
        "anchor_steps": list(args.anchor_steps),
        "batch_size": int(args.batch_size),
        "batches": batches,
        "nearby_radius_m": float(args.nearby_radius_m),
        "future_steps": int(cfg.future_steps),
        "future_dt": float(cfg.future_dt),
        "horizon_s": float(cfg.future_steps) * float(cfg.future_dt),
        "metrics": {
            "wm_ade_m": metrics["wm_ade"].mean(),
            "wm_fde_m": metrics["wm_fde"].mean(),
            "cv_ade_m": metrics["cv_ade"].mean(),
            "cv_fde_m": metrics["cv_fde"].mean(),
            f"wm_ade_nearby_{int(args.nearby_radius_m)}m": metrics["wm_ade_nearby"].mean(),
            f"wm_fde_nearby_{int(args.nearby_radius_m)}m": metrics["wm_fde_nearby"].mean(),
            f"cv_ade_nearby_{int(args.nearby_radius_m)}m": metrics["cv_ade_nearby"].mean(),
            f"cv_fde_nearby_{int(args.nearby_radius_m)}m": metrics["cv_fde_nearby"].mean(),
            "wm_yaw_mae_deg": metrics["wm_yaw_mae_deg"].mean(),
            "ade_improvement_m": metrics["cv_ade"].mean() - metrics["wm_ade"].mean(),
            "fde_improvement_m": metrics["cv_fde"].mean() - metrics["wm_fde"].mean(),
            "nearby_ade_improvement_m": metrics["cv_ade_nearby"].mean()
            - metrics["wm_ade_nearby"].mean(),
            "batches_wm_better_ade": metrics["batches_wm_better_ade"],
            "batches_cv_better_ade": metrics["batches_cv_better_ade"],
            "batches_tie_ade": metrics["batches_tie_ade"],
            "frac_batches_wm_better_ade": metrics["batches_wm_better_ade"] / batches,
        },
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"split{args.split}_anch{'-'.join(map(str, args.anchor_steps))}"
    out_dir = Path(args.out_dir) / f"cv_vs_wm_{tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    m = summary["metrics"]
    print("\n========== CV vs World Model (vs GT futures) ==========")
    print(f"batches={batches}  horizon={summary['horizon_s']:.1f}s  split={args.split}")
    print(f"World Model: ADE={m['wm_ade_m']:.3f} m   FDE={m['wm_fde_m']:.3f} m")
    print(f"Constant Vel: ADE={m['cv_ade_m']:.3f} m   FDE={m['cv_fde_m']:.3f} m")
    print(
        f"Nearby (<={args.nearby_radius_m:.0f}m): "
        f"WM ADE={m[f'wm_ade_nearby_{int(args.nearby_radius_m)}m']:.3f}  "
        f"CV ADE={m[f'cv_ade_nearby_{int(args.nearby_radius_m)}m']:.3f}  "
        f"ΔADE={m['nearby_ade_improvement_m']:+.3f} m"
    )
    print(
        f"WM better ADE on {m['batches_wm_better_ade']}/{batches} batches "
        f"({100 * m['frac_batches_wm_better_ade']:.1f}%)"
    )
    print(f"WM yaw MAE={m['wm_yaw_mae_deg']:.2f} deg")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
