from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data.diffusion_cache_loader import build_dataloader
from data.types import PreprocessConfig
from model.diffusion_torch.diffusion_policy import DiffusionPolicy
from .configs import DiffusionTorchPretrainConfig, build_preprocess_cfg, config_to_dict, parse_args


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def init_wandb_run(args: DiffusionTorchPretrainConfig, config_dict: Mapping[str, Any]):
    if args.wandb_mode == "disabled" or not args.wandb_project:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required when wandb logging is enabled") from exc

    wandb_name = f"{args.wandb_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_dir = os.path.join(args.save_dir, wandb_name)
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_MODE"] = args.wandb_mode

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        dir=wandb_dir,
        config=dict(config_dict),
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("epoch/*", step_metric="train/step")
    return run


def wandb_log(run, payload: Mapping[str, Any], *, step: int) -> None:
    if run is None:
        return
    run.log({key: _jsonable(value) for key, value in payload.items()}, step=step)


def build_model(args: DiffusionTorchPretrainConfig) -> DiffusionPolicy:
    return DiffusionPolicy(
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        map_attr_dim=args.map_attr_dim,
        tl_attr_dim=args.tl_attr_dim,
        ego_dim=args.ego_dim,
        other_dim=args.other_dim,
        predict_type=args.predict_type,
        predict_horizon=args.predict_horizon,
    )


def make_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value



def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, ema_param in ema_params.items():
            model_param = model_params[name]
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)

        ema_buffers = dict(ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_buffers.items():
            if name in model_buffers and torch.is_floating_point(ema_buffer):
                ema_buffer.data.copy_(model_buffers[name].data)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    config: DiffusionTorchPretrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": _jsonable(config_to_dict(config)),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    ema_model.load_state_dict(payload["ema_model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("epoch", 0)) + 1, int(payload.get("global_step", 0))


def write_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: _jsonable(value) for key, value in record.items()}, sort_keys=True) + "\n")


def build_data_loader(args: DiffusionTorchPretrainConfig, preprocess_cfg: PreprocessConfig):
    return build_dataloader(
        cache_dir=args.cache_dir,
        preprocess_cfg=preprocess_cfg,
        file_indices=args.file_indices,
        batch_size=args.cache_batch_size,
        shuffle_seed=args.shuffle_seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        anchor_steps=None,
        goal_step_override=None,
        backend="torch",
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess_cfg = build_preprocess_cfg(args)
    model = build_model(args).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    total_steps = max(1, int(args.epochs) * int(args.steps_per_epoch))
    scheduler = make_lr_scheduler(optimizer, total_steps, int(args.warmup_steps))

    start_epoch = 1
    global_step = 0
    if args.resume_path:
        start_epoch, global_step = load_checkpoint(
            args.resume_path,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    dataloader = build_data_loader(args, preprocess_cfg)
    data_iter = iter(dataloader)
    jsonl_path = Path(args.log_jsonl_path)

    config_dict = config_to_dict(args)
    wandb_run = init_wandb_run(args, config_dict)
    wandb_log(
        wandb_run,
        {
            "event": "start",
            "device": str(device),
            "config": config_dict,
        },
        step=global_step,
    )
    write_jsonl(jsonl_path, {"event": "start", "config": config_dict, "device": str(device)})

    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            progress = tqdm(total=int(args.steps_per_epoch), desc=f"Epoch {epoch:04d}")

            for step_in_epoch in range(int(args.steps_per_epoch)):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch_features = move_to_device(batch.features, device)
                # for key, value in batch_features.items():
                #     print(f"Feature {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")

                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(batch_features)
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                optimizer.step()
                scheduler.step()
                update_ema(ema_model, model, float(args.ema_decay))

                loss_value = float(loss.detach().cpu())
                grad_norm_value = float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                lr_value = float(optimizer.param_groups[0]["lr"])

                epoch_loss += loss_value
                epoch_steps += 1
                global_step += 1
                progress.update(1)

                if global_step % int(args.log_every) == 0:
                    record = {
                        "event": "train_step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_in_epoch": step_in_epoch,
                        "loss": loss_value,
                        "grad_norm": grad_norm_value,
                        "lr": lr_value,
                    }
                    write_jsonl(jsonl_path, record)
                    wandb_log(
                        wandb_run,
                        {
                            "train/step": global_step,
                            "train/loss": loss_value,
                            "train/grad_norm": grad_norm_value,
                            "train/lr": lr_value,
                            "train/epoch": epoch,
                            "train/step_in_epoch": step_in_epoch,
                        },
                        step=global_step,
                    )
                    progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr_value:.2e}")

            progress.close()

            mean_loss = epoch_loss / max(epoch_steps, 1)
            epoch_payload = {
                "event": "epoch_end",
                "epoch": epoch,
                "global_step": global_step,
                "mean_loss": mean_loss,
            }
            write_jsonl(jsonl_path, epoch_payload)
            wandb_log(
                wandb_run,
                {
                    "epoch/mean_loss": mean_loss,
                    "epoch/epoch": epoch,
                    "epoch/global_step": global_step,
                },
                step=global_step,
            )

            latest_path = Path(args.save_dir) / "latest.pt"
            save_checkpoint(
                latest_path,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=args,
            )
            if epoch % int(args.save_every) == 0:
                epoch_path = Path(args.save_dir) / f"epoch_{epoch:04d}.pt"
                save_checkpoint(
                    epoch_path,
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config=args,
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()