from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model import LatentMultiAgentWorldModel, WorldModelConfig


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return tuple(int(x) for x in items)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Gaussian multi-agent world model.")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./checkpoints/world_model")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--steps_per_epoch",
        type=int,
        default=2000,
        help="Gradient updates per epoch before validation. "
        "0 = full pass over the training loader (very slow on ~500k scenarios).",
    )
    p.add_argument("--val_steps", type=int, default=100, help="0 = consume the whole validation loader")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20,30,40,50,60,70,80))
    p.add_argument("--file_indices", type=_parse_int_tuple, default=None)

    p.add_argument("--history_steps", type=int, default=10)
    p.add_argument("--history_stride", type=int, default=2)
    p.add_argument("--future_steps", type=int, default=25)
    p.add_argument("--future_stride", type=int, default=2)
    p.add_argument("--world_dt", type=float, default=0.1)
    p.add_argument("--max_agents", type=int, default=128)
    p.add_argument("--max_range", type=float, default=50.0)
    p.add_argument("--max_velocity", type=float, default=25.0)
    p.add_argument("--max_width", type=float, default=10.0)
    p.add_argument("--max_map_segments", type=int, default=64)
    p.add_argument("--max_points_per_segment", type=int, default=32)
    p.add_argument("--max_tl_points", type=int, default=16)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_scene_layers", type=int, default=3)
    p.add_argument("--num_future_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--sigma_eps", type=float, default=1e-3)
    p.add_argument("--yaw_loss_weight", type=float, default=0.5)
    p.add_argument("--collision_focus_radius_m", type=float, default=20.0)
    p.add_argument("--collision_focus_weight", type=float, default=2.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--wandb_project", type=str, default="world Model")
    p.add_argument("--wandb_entity", type=str, default="scbellad2025-carnegie-mellon-university")
    p.add_argument("--wandb_name", type=str, default="gaussian_world_model")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def _make_config(args: argparse.Namespace) -> WorldModelConfig:
    return WorldModelConfig(
        history_steps=args.history_steps,
        history_stride=args.history_stride,
        future_steps=args.future_steps,
        future_stride=args.future_stride,
        world_dt=args.world_dt,
        max_agents=args.max_agents,
        max_range=args.max_range,
        max_velocity=args.max_velocity,
        max_width=args.max_width,
        max_map_segments=args.max_map_segments,
        max_points_per_segment=args.max_points_per_segment,
        max_tl_points=args.max_tl_points,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_scene_layers=args.num_scene_layers,
        num_future_layers=args.num_future_layers,
        dropout=args.dropout,
        sigma_eps=args.sigma_eps,
        yaw_loss_weight=args.yaw_loss_weight,
        collision_focus_radius_m=args.collision_focus_radius_m,
        collision_focus_weight=args.collision_focus_weight,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}


def _horizon_index(seconds: float, config: WorldModelConfig) -> int:
    """Index of the future step nearest to `seconds` after the anchor.

    Step 0 is +future_dt (default 0.2 s), so 1 s -> index 4 when future_dt=0.2.
    """
    dt = float(config.future_dt)
    idx = int(round(float(seconds) / max(dt, 1e-6))) - 1
    return int(max(0, min(int(config.future_steps) - 1, idx)))


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _gaussian_nll_xy(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    sigma_xy: torch.Tensor,
) -> torch.Tensor:
    """Diagonal 2D Gaussian NLL per (batch, agent, time). Drops the 2π constant.

    NLL = 0.5 * [((x-mu)/sx)^2 + ((y-mu)/sy)^2 + 2 log sx + 2 log sy]
        = 0.5 (zx^2 + zy^2) + log sx + log sy
    """
    sigma = sigma_xy.clamp_min(1e-6)
    z = (pred_xy - target_xy) / sigma
    return 0.5 * (z.square().sum(dim=-1)) + sigma.log().sum(dim=-1)


def _loss_and_metrics(
    model: LatentMultiAgentWorldModel,
    batch: dict[str, torch.Tensor],
    config: WorldModelConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(batch)
    pred_xy = out["pred_delta_xy_norm"]
    pred_yaw = out["pred_yaw_vec"]
    target_xy = batch["target_delta_xy"].float()
    target_yaw = F.normalize(batch["target_yaw_vec"].float(), dim=-1, eps=1e-6)
    valid = batch["target_valid"].bool()

    yaw_loss = 1.0 - (pred_yaw * target_yaw).sum(dim=-1).clamp(-1.0, 1.0)

    current_dist_m = torch.linalg.vector_norm(batch["current_local_xy"].float(), dim=-1)
    near = (current_dist_m <= float(config.collision_focus_radius_m)).float()
    agent_weight = 1.0 + float(config.collision_focus_weight) * near
    weights = valid.float() * agent_weight[:, :, None]
    denom = weights.sum().clamp_min(1.0)

    sigma_xy = out["pred_sigma_xy_norm"]
    xy_loss = _gaussian_nll_xy(pred_xy, target_xy, sigma_xy)

    xy_term = (xy_loss * weights).sum() / denom
    yaw_term = (yaw_loss * weights).sum() / denom
    loss = xy_term + float(config.yaw_loss_weight) * yaw_term

    with torch.no_grad():
        valid_f = valid.float()
        xy_error_m = torch.linalg.vector_norm(pred_xy - target_xy, dim=-1) * float(config.max_range)
        ade = (xy_error_m * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        t = xy_error_m.shape[-1]
        idx = torch.arange(t, device=xy_error_m.device).view(1, 1, t)
        last_idx = torch.where(valid, idx, torch.full_like(idx, -1)).amax(dim=-1)
        has_future = last_idx >= 0
        safe_idx = last_idx.clamp_min(0)
        fde_all = torch.gather(xy_error_m, dim=-1, index=safe_idx[..., None]).squeeze(-1)
        fde = (fde_all * has_future.float()).sum() / has_future.float().sum().clamp_min(1.0)

        yaw_deg = torch.rad2deg(torch.acos((pred_yaw * target_yaw).sum(dim=-1).clamp(-1.0, 1.0)))
        yaw_mae = (yaw_deg * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        sigma_m = out["pred_sigma_xy_m"]
        w = valid_f
        sigma_x_m = float(_masked_mean(sigma_m[..., 0], w).detach().cpu())
        sigma_y_m = float(_masked_mean(sigma_m[..., 1], w).detach().cpu())
        sigma_horizon = {}
        for sec, tag in ((1.0, "1s"), (3.0, "3s"), (5.0, "5s")):
            ti = _horizon_index(sec, config)
            wt = w[:, :, ti]
            sigma_horizon[f"sigma_x_m_{tag}"] = float(_masked_mean(sigma_m[:, :, ti, 0], wt).detach().cpu())
            sigma_horizon[f"sigma_y_m_{tag}"] = float(_masked_mean(sigma_m[:, :, ti, 1], wt).detach().cpu())

    metrics = {
        "loss": float(loss.detach().cpu()),
        "xy_loss": float(xy_term.detach().cpu()),
        "nll": float(xy_term.detach().cpu()),
        "yaw_loss": float(yaw_term.detach().cpu()),
        "ade_m": float(ade.detach().cpu()),
        "fde_m": float(fde.detach().cpu()),
        "yaw_deg": float(yaw_mae.detach().cpu()),
        "sigma_x_m": sigma_x_m,
        "sigma_y_m": sigma_y_m,
        **sigma_horizon,
    }
    return loss, metrics


def _save_checkpoint(
    path: Path,
    *,
    model: LatentMultiAgentWorldModel,
    optimizer: torch.optim.Optimizer,
    config: WorldModelConfig,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "train_args": vars(args),
            "epoch": int(epoch),
            "global_step": int(global_step),
        },
        path,
    )


def _run_validation(
    model: LatentMultiAgentWorldModel,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    config: WorldModelConfig,
    max_steps: int,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "xy_loss": 0.0,
        "nll": 0.0,
        "yaw_loss": 0.0,
        "ade_m": 0.0,
        "fde_m": 0.0,
        "yaw_deg": 0.0,
        "sigma_x_m": 0.0,
        "sigma_y_m": 0.0,
        "sigma_x_m_1s": 0.0,
        "sigma_y_m_1s": 0.0,
        "sigma_x_m_3s": 0.0,
        "sigma_y_m_3s": 0.0,
        "sigma_x_m_5s": 0.0,
        "sigma_y_m_5s": 0.0,
    }
    count = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_steps > 0 and step >= max_steps:
                break
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, metrics = _loss_and_metrics(model, batch, config)
            for k in totals:
                v = metrics[k]
                totals[k] += 0.0 if v != v else float(v)
            count += 1
    model.train()
    if count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / count for k, v in totals.items()}


def _init_wandb(args: argparse.Namespace, config: WorldModelConfig, save_dir: Path):
    wandb_name = f"{args.wandb_name}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = save_dir / wandb_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = args.wandb_mode
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        dir=str(run_dir),
        config={"world_model": config.to_dict(), "train_args": vars(args)},
    )
    return run, run_dir, wandb_name


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    config = _make_config(args)
    cache_paths = resolve_cache_paths(
        args.cache_dir,
        None if args.file_indices is None else list(args.file_indices),
    )
    if len(cache_paths) < 2:
        raise ValueError("Need at least two cache shards for a train/validation split.")
    train_paths, val_paths = split_cache_paths(cache_paths, val_fraction=float(args.val_fraction))

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    run, run_dir, wandb_name = _init_wandb(args, config, save_root)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"world_model": config.to_dict(), "train_args": vars(args)}, f, indent=2, default=str)

    model = LatentMultiAgentWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "latest.pt"
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        resume_cfg = WorldModelConfig.from_dict(dict(ckpt["config"]))
        if resume_cfg.to_dict() != config.to_dict():
            raise ValueError(
                "Resume checkpoint config does not match current CLI config. "
                "Use the same model/data shape arguments."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metric_keys = (
        "loss",
        "xy_loss",
        "nll",
        "yaw_loss",
        "ade_m",
        "fde_m",
        "yaw_deg",
        "sigma_x_m",
        "sigma_y_m",
        "sigma_x_m_1s",
        "sigma_y_m_1s",
        "sigma_x_m_3s",
        "sigma_y_m_3s",
        "sigma_x_m_5s",
        "sigma_y_m_5s",
    )
    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        train_loader = build_world_model_cache_loader(
            args.cache_dir,
            config,
            cache_paths=train_paths,
            batch_size=args.batch_size,
            anchor_steps=args.anchor_steps,
            shuffle=True,
            seed=args.seed + epoch,
            drop_last=False,
        )
        model.train()
        running = {k: 0.0 for k in metric_keys}
        running_count = 0
        epoch_totals = {k: 0.0 for k in metric_keys}
        epoch_count = 0

        train_total = args.steps_per_epoch if args.steps_per_epoch > 0 else None
        train_bar = tqdm(
            train_loader,
            desc=f"train {epoch:03d}",
            leave=False,
            unit="batch",
            total=train_total,
        )
        for step, batch in enumerate(train_bar):
            if args.steps_per_epoch > 0 and step >= args.steps_per_epoch:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, metrics = _loss_and_metrics(model, batch, config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            )
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running_count += 1
            epoch_count += 1
            for k in metric_keys:
                v = metrics[k]
                if v == v:
                    running[k] += v
                    epoch_totals[k] += v

            if args.log_every > 0 and global_step % args.log_every == 0:
                avg = {k: v / max(running_count, 1) for k, v in running.items()}
                wandb.log(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train/loss": avg["loss"],
                        "train/xy_loss": avg["xy_loss"],
                        "train/nll": avg["nll"],
                        "train/yaw_loss": avg["yaw_loss"],
                        "train/ade_m": avg["ade_m"],
                        "train/fde_m": avg["fde_m"],
                        "train/yaw_deg": avg["yaw_deg"],
                        "train/sigma_x_m": avg["sigma_x_m"],
                        "train/sigma_y_m": avg["sigma_y_m"],
                        "train/sigma_x_m_1s": avg["sigma_x_m_1s"],
                        "train/sigma_y_m_1s": avg["sigma_y_m_1s"],
                        "train/sigma_x_m_3s": avg["sigma_x_m_3s"],
                        "train/sigma_y_m_3s": avg["sigma_y_m_3s"],
                        "train/sigma_x_m_5s": avg["sigma_x_m_5s"],
                        "train/sigma_y_m_5s": avg["sigma_y_m_5s"],
                        "train/grad_norm": grad_norm,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )
                running = {k: 0.0 for k in metric_keys}
                running_count = 0

        val_loader = build_world_model_cache_loader(
            args.cache_dir,
            config,
            cache_paths=val_paths,
            batch_size=args.batch_size,
            anchor_steps=args.anchor_steps,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
        )
        val_metrics = _run_validation(
            model,
            val_loader,
            device=device,
            config=config,
            max_steps=args.val_steps,
            use_amp=use_amp,
        )
        train_epoch = {k: v / max(epoch_count, 1) for k, v in epoch_totals.items()}
        wandb.log(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_epoch/loss": train_epoch["loss"],
                "train_epoch/nll": train_epoch["nll"],
                "train_epoch/ade_m": train_epoch["ade_m"],
                "train_epoch/fde_m": train_epoch["fde_m"],
                "train_epoch/yaw_deg": train_epoch["yaw_deg"],
                "train_epoch/sigma_x_m": train_epoch["sigma_x_m"],
                "train_epoch/sigma_y_m": train_epoch["sigma_y_m"],
                "val/loss": val_metrics["loss"],
                "val/xy_loss": val_metrics["xy_loss"],
                "val/nll": val_metrics["nll"],
                "val/yaw_loss": val_metrics["yaw_loss"],
                "val/ade_m": val_metrics["ade_m"],
                "val/fde_m": val_metrics["fde_m"],
                "val/yaw_deg": val_metrics["yaw_deg"],
                "val/sigma_x_m": val_metrics["sigma_x_m"],
                "val/sigma_y_m": val_metrics["sigma_y_m"],
                "val/sigma_x_m_1s": val_metrics["sigma_x_m_1s"],
                "val/sigma_y_m_1s": val_metrics["sigma_y_m_1s"],
                "val/sigma_x_m_3s": val_metrics["sigma_x_m_3s"],
                "val/sigma_y_m_3s": val_metrics["sigma_y_m_3s"],
                "val/sigma_x_m_5s": val_metrics["sigma_x_m_5s"],
                "val/sigma_y_m_5s": val_metrics["sigma_y_m_5s"],
            },
            step=global_step,
        )
        epoch_bar.set_postfix(
            train=f"{train_epoch['loss']:.3f}",
            val=f"{val_metrics['loss']:.3f}",
            ade=f"{val_metrics['ade_m']:.2f}m",
        )

        _save_checkpoint(
            run_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                run_dir / f"epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

    run.finish()


if __name__ == "__main__":
    main()
