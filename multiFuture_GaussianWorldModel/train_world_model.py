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

# Load Triton (imported lazily by torch.optim) before jax/tensorflow come in via
# waymax: loading its LLVM after theirs segfaults in some environments.
try:
    import triton  # noqa: F401
except ImportError:
    pass

import wandb
from tqdm import tqdm

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model import LatentMultiAgentWorldModel, WorldModelConfig, gather_mode, rotate_xy


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return tuple(int(x) for x in items)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the multimodal Gaussian world model.")
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
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20, 30, 40, 50, 60, 70, 80))
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

    p.add_argument("--num_modes", type=int, default=6)
    p.add_argument("--sigma_eps", type=float, default=1e-3)
    p.add_argument(
        "--agent_frame_decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode in each agent's heading frame (sigma = longitudinal/lateral).",
    )
    p.add_argument(
        "--decoder_type",
        type=str,
        default="direct",
        choices=["per_step", "direct"],
        help="direct = whole-trajectory head per (agent, mode) with per-mode classifier.",
    )
    p.add_argument(
        "--mode_embed_scale",
        type=float,
        default=None,
        help="Init norm of each mode embedding row. Default sqrt(hidden_dim); 1.0 = legacy init.",
    )
    p.add_argument("--yaw_loss_weight", type=float, default=0.5)
    p.add_argument("--mode_cls_weight", type=float, default=1.0)
    p.add_argument("--collision_focus_radius_m", type=float, default=20.0)
    p.add_argument("--collision_focus_weight", type=float, default=2.0)
    p.add_argument(
        "--planning_mode",
        type=str,
        default="top1",
        choices=["top1", "expected", "worst_case"],
        help="Stored on the checkpoint for later ES packing. Training is unchanged.",
    )
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--wandb_project", type=str, default="world Model")
    p.add_argument("--wandb_entity", type=str, default=None, help="Empty/None = logged-in user's default entity.")
    p.add_argument("--wandb_name", type=str, default="multiFuture_GaussianWorldModel")
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
        num_modes=args.num_modes,
        sigma_eps=args.sigma_eps,
        agent_frame_decoding=args.agent_frame_decoding,
        decoder_type=args.decoder_type,
        mode_embed_scale=args.mode_embed_scale,
        yaw_loss_weight=args.yaw_loss_weight,
        mode_cls_weight=args.mode_cls_weight,
        collision_focus_radius_m=args.collision_focus_radius_m,
        collision_focus_weight=args.collision_focus_weight,
        planning_mode=args.planning_mode,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}


def _horizon_index(seconds: float, config: WorldModelConfig) -> int:
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
    sigma = sigma_xy.clamp_min(1e-6)
    z = (pred_xy - target_xy) / sigma
    return 0.5 * (z.square().sum(dim=-1)) + sigma.log().sum(dim=-1)


def _per_mode_ade_fde_m(
    pred_xy_k: torch.Tensor,
    target_xy: torch.Tensor,
    valid: torch.Tensor,
    max_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ADE/FDE in meters for every mode.

    Returns
    -------
    ade_k: [B, N, K]
    fde_k: [B, N, K]
    has_future: [B, N]
    """
    err = torch.linalg.vector_norm(pred_xy_k - target_xy[:, :, None, :, :], dim=-1) * float(max_range)
    valid_f = valid.float()
    ade_k = (err * valid_f[:, :, None, :]).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)[:, :, None]

    t = err.shape[-1]
    idx = torch.arange(t, device=err.device).view(1, 1, t)
    last_idx = torch.where(valid, idx, torch.full_like(idx, -1)).amax(dim=-1)
    has_future = last_idx >= 0
    safe = last_idx.clamp_min(0)
    k = err.shape[2]
    fde_k = err.gather(-1, safe[:, :, None, None].expand(-1, -1, k, 1)).squeeze(-1)
    fde_k = fde_k * has_future[:, :, None].to(fde_k.dtype)
    return ade_k, fde_k, has_future


def _loss_and_metrics(
    model: LatentMultiAgentWorldModel,
    batch: dict[str, torch.Tensor],
    config: WorldModelConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(batch)
    pred_xy_k = out["pred_delta_xy_norm_k"]
    pred_yaw_k = out["pred_yaw_vec_k"]
    sigma_k = out["pred_sigma_xy_norm_k"]
    mode_logits = out["mode_logits"]
    mode_probs = out["mode_probs"]
    target_xy = batch["target_delta_xy"].float()
    target_yaw = F.normalize(batch["target_yaw_vec"].float(), dim=-1, eps=1e-6)
    valid = batch["target_valid"].bool()

    current_dist_m = torch.linalg.vector_norm(batch["current_local_xy"].float(), dim=-1)
    near = (current_dist_m <= float(config.collision_focus_radius_m)).float()
    agent_weight = 1.0 + float(config.collision_focus_weight) * near
    time_weights = valid.float() * agent_weight[:, :, None]
    time_denom = time_weights.sum().clamp_min(1.0)

    ade_k, fde_k, has_future = _per_mode_ade_fde_m(
        pred_xy_k, target_xy, valid, float(config.max_range)
    )
    best_mode = ade_k.argmin(dim=-1)
    pred_yaw = gather_mode(pred_yaw_k, best_mode)
    sigma_xy = gather_mode(sigma_k, best_mode)

    # NLL in the frame sigma lives in (agent frame when agent_frame_decoding).
    heading_sc = out["agent_heading_sc"]
    target_xy_agent = rotate_xy(target_xy, -heading_sc[:, :, None, 0], heading_sc[:, :, None, 1])
    pred_xy_agent = gather_mode(out["pred_delta_xy_agent_norm_k"], best_mode)
    xy_loss = _gaussian_nll_xy(pred_xy_agent, target_xy_agent, sigma_xy)
    yaw_loss = 1.0 - (pred_yaw * target_yaw).sum(dim=-1).clamp(-1.0, 1.0)
    xy_term = (xy_loss * time_weights).sum() / time_denom
    yaw_term = (yaw_loss * time_weights).sum() / time_denom

    agent_w = has_future.float() * agent_weight
    agent_denom = agent_w.sum().clamp_min(1.0)
    ce = F.cross_entropy(
        mode_logits.reshape(-1, mode_logits.shape[-1]),
        best_mode.reshape(-1),
        reduction="none",
    ).reshape(best_mode.shape)
    cls_term = (ce * agent_w).sum() / agent_denom

    loss = xy_term + float(config.yaw_loss_weight) * yaw_term + float(config.mode_cls_weight) * cls_term

    with torch.no_grad():
        valid_f = valid.float()
        min_ade = _masked_mean(ade_k.min(dim=-1).values, has_future.float())
        min_fde = _masked_mean(fde_k.min(dim=-1).values, has_future.float())

        top1 = out["top1_mode"]
        top1_ade = _masked_mean(ade_k.gather(-1, top1[..., None]).squeeze(-1), has_future.float())
        top1_fde = _masked_mean(fde_k.gather(-1, top1[..., None]).squeeze(-1), has_future.float())

        entropy = -(mode_probs * mode_probs.clamp_min(1e-8).log()).sum(dim=-1)
        mode_entropy = _masked_mean(entropy, has_future.float())
        top1_conf = _masked_mean(mode_probs.gather(-1, top1[..., None]).squeeze(-1), has_future.float())

        yaw_deg = torch.rad2deg(torch.acos((pred_yaw * target_yaw).sum(dim=-1).clamp(-1.0, 1.0)))
        yaw_mae = (yaw_deg * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        sigma_m = gather_mode(out["pred_sigma_xy_m_k"], best_mode)
        sigma_x_m = float(_masked_mean(sigma_m[..., 0], valid_f).detach().cpu())
        sigma_y_m = float(_masked_mean(sigma_m[..., 1], valid_f).detach().cpu())
        sigma_horizon = {}
        for sec, tag in ((1.0, "1s"), (3.0, "3s"), (5.0, "5s")):
            ti = _horizon_index(sec, config)
            wt = valid_f[:, :, ti]
            sigma_horizon[f"sigma_x_m_{tag}"] = float(_masked_mean(sigma_m[:, :, ti, 0], wt).detach().cpu())
            sigma_horizon[f"sigma_y_m_{tag}"] = float(_masked_mean(sigma_m[:, :, ti, 1], wt).detach().cpu())

    metrics = {
        "loss": float(loss.detach().cpu()),
        "xy_loss": float(xy_term.detach().cpu()),
        "nll": float(xy_term.detach().cpu()),
        "yaw_loss": float(yaw_term.detach().cpu()),
        "cls_loss": float(cls_term.detach().cpu()),
        "min_ade_m": float(min_ade.detach().cpu()),
        "min_fde_m": float(min_fde.detach().cpu()),
        "top1_ade_m": float(top1_ade.detach().cpu()),
        "top1_fde_m": float(top1_fde.detach().cpu()),
        "yaw_deg": float(yaw_mae.detach().cpu()),
        "mode_entropy": float(mode_entropy.detach().cpu()),
        "top1_conf": float(top1_conf.detach().cpu()),
        "sigma_x_m": sigma_x_m,
        "sigma_y_m": sigma_y_m,
        **sigma_horizon,
    }
    return loss, metrics


METRIC_KEYS = (
    "loss",
    "xy_loss",
    "nll",
    "yaw_loss",
    "cls_loss",
    "min_ade_m",
    "min_fde_m",
    "top1_ade_m",
    "top1_fde_m",
    "yaw_deg",
    "mode_entropy",
    "top1_conf",
    "sigma_x_m",
    "sigma_y_m",
    "sigma_x_m_1s",
    "sigma_y_m_1s",
    "sigma_x_m_3s",
    "sigma_y_m_3s",
    "sigma_x_m_5s",
    "sigma_y_m_5s",
)


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
    totals = {k: 0.0 for k in METRIC_KEYS}
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
        entity=args.wandb_entity or None,
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
        running = {k: 0.0 for k in METRIC_KEYS}
        running_count = 0
        epoch_totals = {k: 0.0 for k in METRIC_KEYS}
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
            for k in METRIC_KEYS:
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
                        "train/nll": avg["nll"],
                        "train/yaw_loss": avg["yaw_loss"],
                        "train/cls_loss": avg["cls_loss"],
                        "train/minADE@K": avg["min_ade_m"],
                        "train/minFDE@K": avg["min_fde_m"],
                        "train/top1_ade_m": avg["top1_ade_m"],
                        "train/top1_fde_m": avg["top1_fde_m"],
                        "train/yaw_deg": avg["yaw_deg"],
                        "train/mode_entropy": avg["mode_entropy"],
                        "train/top1_conf": avg["top1_conf"],
                        "train/sigma_x_m": avg["sigma_x_m"],
                        "train/sigma_y_m": avg["sigma_y_m"],
                        "train/grad_norm": grad_norm,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )
                running = {k: 0.0 for k in METRIC_KEYS}
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
                "train_epoch/cls_loss": train_epoch["cls_loss"],
                "train_epoch/minADE@K": train_epoch["min_ade_m"],
                "train_epoch/minFDE@K": train_epoch["min_fde_m"],
                "train_epoch/top1_ade_m": train_epoch["top1_ade_m"],
                "train_epoch/top1_fde_m": train_epoch["top1_fde_m"],
                "train_epoch/mode_entropy": train_epoch["mode_entropy"],
                "train_epoch/top1_conf": train_epoch["top1_conf"],
                "val/loss": val_metrics["loss"],
                "val/nll": val_metrics["nll"],
                "val/yaw_loss": val_metrics["yaw_loss"],
                "val/cls_loss": val_metrics["cls_loss"],
                "val/minADE@K": val_metrics["min_ade_m"],
                "val/minFDE@K": val_metrics["min_fde_m"],
                "val/top1_ade_m": val_metrics["top1_ade_m"],
                "val/top1_fde_m": val_metrics["top1_fde_m"],
                "val/yaw_deg": val_metrics["yaw_deg"],
                "val/mode_entropy": val_metrics["mode_entropy"],
                "val/top1_conf": val_metrics["top1_conf"],
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
            minADE=f"{val_metrics['min_ade_m']:.2f}m",
            top1=f"{val_metrics['top1_ade_m']:.2f}m",
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
