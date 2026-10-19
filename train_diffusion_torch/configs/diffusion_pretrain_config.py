from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

from data.types import PreprocessConfig


@dataclass
class DiffusionTorchPretrainConfig:
    cache_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/"
    file_indices: list[int] | None = None
    cache_batch_size: int = 64
    shuffle_seed: int = 0
    num_workers: int = 16
    pin_memory: bool = False
    drop_last: bool = False

    epochs: int = 500
    steps_per_epoch: int = 1000
    save_every: int = 50
    log_every: int = 50
    seed: int = 0

    save_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/train_diffusion_torch"
    resume_path: str | None = None

    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999

    hidden_dim: int = 512
    cond_dim: int = 256
    target_dim: int = 5
    predict_horizon: int = 25
    map_attr_dim: int = 25
    tl_attr_dim: int = 11
    other_dim: int = 15
    ego_dim: int = 5
    predict_type: str = "v"

    model_dt: float = 0.2
    max_range: float = 30.0
    ego_range: float = 100.0
    max_velocity: float = 25.0
    max_width: float = 10.0
    max_tl_points: int = 16
    num_map_type_classes: int = 21
    max_num_segments: int = 128
    max_points_per_segment: int = 128

    log_jsonl_path: str = "./train_logs.jsonl"
    wandb_project: str = "waymax_vla"
    wandb_name: str = "pretrain_diffusion_torch"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    tag: str | None = None


def build_preprocess_cfg(args: DiffusionTorchPretrainConfig) -> PreprocessConfig:
    return PreprocessConfig(
        model_dt=args.model_dt,
        world_dt_fallback=0.1,
        max_range=args.max_range,
        ego_range=args.ego_range,
        max_velocity=args.max_velocity,
        max_width=args.max_width,
        max_tl_points=args.max_tl_points,
        num_map_type_classes=args.num_map_type_classes,
        predict_horizon=args.predict_horizon,
        max_segments=args.max_num_segments,
        max_points_per_segment=args.max_points_per_segment,
        num_object_types=8,
        inst_dim=768,
    )


def parse_args() -> DiffusionTorchPretrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, default=DiffusionTorchPretrainConfig.cache_dir)
    parser.add_argument("--file_indices", type=int, nargs="*", default=None)
    parser.add_argument("--cache_batch_size", type=int, default=DiffusionTorchPretrainConfig.cache_batch_size)
    parser.add_argument("--shuffle_seed", type=int, default=DiffusionTorchPretrainConfig.shuffle_seed)
    parser.add_argument("--num_workers", type=int, default=DiffusionTorchPretrainConfig.num_workers)
    parser.add_argument("--pin_memory", action="store_true", default=DiffusionTorchPretrainConfig.pin_memory)
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument("--drop_last", action="store_true", default=DiffusionTorchPretrainConfig.drop_last)
    parser.add_argument("--no_drop_last", action="store_false", dest="drop_last")

    parser.add_argument("--epochs", type=int, default=DiffusionTorchPretrainConfig.epochs)
    parser.add_argument("--steps_per_epoch", type=int, default=DiffusionTorchPretrainConfig.steps_per_epoch)
    parser.add_argument("--save_every", type=int, default=DiffusionTorchPretrainConfig.save_every)
    parser.add_argument("--log_every", type=int, default=DiffusionTorchPretrainConfig.log_every)
    parser.add_argument("--seed", type=int, default=DiffusionTorchPretrainConfig.seed)

    parser.add_argument("--save_dir", type=str, default=DiffusionTorchPretrainConfig.save_dir)
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=DiffusionTorchPretrainConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=DiffusionTorchPretrainConfig.weight_decay)
    parser.add_argument("--warmup_steps", type=int, default=DiffusionTorchPretrainConfig.warmup_steps)
    parser.add_argument("--grad_clip_norm", type=float, default=DiffusionTorchPretrainConfig.grad_clip_norm)
    parser.add_argument("--ema_decay", type=float, default=DiffusionTorchPretrainConfig.ema_decay)

    parser.add_argument("--hidden_dim", type=int, default=DiffusionTorchPretrainConfig.hidden_dim)
    parser.add_argument("--cond_dim", type=int, default=DiffusionTorchPretrainConfig.cond_dim)
    parser.add_argument("--target_dim", type=int, default=DiffusionTorchPretrainConfig.target_dim)
    parser.add_argument("--predict_horizon", type=int, default=DiffusionTorchPretrainConfig.predict_horizon)
    parser.add_argument("--map_attr_dim", type=int, default=DiffusionTorchPretrainConfig.map_attr_dim)
    parser.add_argument("--tl_attr_dim", type=int, default=DiffusionTorchPretrainConfig.tl_attr_dim)
    parser.add_argument("--other_dim", type=int, default=DiffusionTorchPretrainConfig.other_dim)
    parser.add_argument("--ego_dim", type=int, default=DiffusionTorchPretrainConfig.ego_dim)
    parser.add_argument("--predict_type", type=str, default=DiffusionTorchPretrainConfig.predict_type, choices=["eps", "mu", "v"])

    parser.add_argument("--model_dt", type=float, default=DiffusionTorchPretrainConfig.model_dt)
    parser.add_argument("--max_range", type=float, default=DiffusionTorchPretrainConfig.max_range)
    parser.add_argument("--ego_range", type=float, default=DiffusionTorchPretrainConfig.ego_range)
    parser.add_argument("--max_velocity", type=float, default=DiffusionTorchPretrainConfig.max_velocity)
    parser.add_argument("--max_width", type=float, default=DiffusionTorchPretrainConfig.max_width)
    parser.add_argument("--max_tl_points", type=int, default=DiffusionTorchPretrainConfig.max_tl_points)
    parser.add_argument("--num_map_type_classes", type=int, default=DiffusionTorchPretrainConfig.num_map_type_classes)
    parser.add_argument("--max_num_segments", type=int, default=DiffusionTorchPretrainConfig.max_num_segments)
    parser.add_argument("--max_points_per_segment", type=int, default=DiffusionTorchPretrainConfig.max_points_per_segment)

    parser.add_argument("--log_jsonl_path", type=str, default=DiffusionTorchPretrainConfig.log_jsonl_path)
    parser.add_argument("--wandb_project", type=str, default=DiffusionTorchPretrainConfig.wandb_project)
    parser.add_argument("--wandb_name", type=str, default=DiffusionTorchPretrainConfig.wandb_name)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=DiffusionTorchPretrainConfig.wandb_mode, choices=["online", "offline", "disabled"])
    parser.add_argument("--tag", type=str, default=None)

    args = parser.parse_args()
    if args.tag is not None:
        args.wandb_name = f"{args.wandb_name}_{args.tag}"
    return DiffusionTorchPretrainConfig(**vars(args))


def config_to_dict(config: DiffusionTorchPretrainConfig) -> dict:
    return asdict(config)