from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class FlowMatchingFinetuningConfig:
	seed: int = 0
	source_cache_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz"
	failure_cache_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/failure_sim_state_cache/"
	source_file_indices: list[int] | None = None
	failure_file_indices: list[int] | None = None
	failure_sampling_prob: float = 0.5
	pretrained_ckpt: str | None = None
	resume_path: str | None = None
	use_ema: bool = True

	batch_size: int = 64
	num_epochs: int = 20
	steps_per_epoch: int = 1000
	save_every: int = 5
	log_every: int = 50
	max_steps: int | None = None
	shuffle_seed: int = 0
	num_workers: int = 0
	pin_memory: bool = False

	output_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/finetune_fm_policy"
	save_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/finetune_fm_policy"
	wandb_project: str | None = "fm_policy"
	wandb_name: str = "finetune_fm_policy"
	wandb_entity: str | None = None
	wandb_mode: str = "online"
	tag: str | None = None
	data_parallel: bool = False

	lr: float = 1e-4
	weight_decay: float = 1e-4
	warmup_steps: int = 500
	grad_clip_norm: float = 1.0
	ema_decay: float = 0.999
	goal_mask_prob: float = 0.3

	target_dim: int = 5
	hidden_dim: int = 512
	cond_dim: int = 256
	predict_horizon: int = 25
	map_attr_dim: int = 25
	tl_attr_dim: int = 11
	other_dim: int = 15
	inst_dim: int = 768
	subgoal_conditioned: bool = False
	default_num_steps: int = 100

	max_range: float = 30.0
	ego_range: float = 100.0
	max_velocity: float = 25.0
	max_width: float = 10.0
	max_tl_points: int = 16
	num_map_type_classes: int = 21
	max_num_segments: int = 128
	max_points_per_segment: int = 128
	model_dt: float = 0.2
	world_dt_fallback: float = 0.1

	jax_compilation_cache_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/.jax_compilation_cache"


def parse_args() -> FlowMatchingFinetuningConfig:
	parser = argparse.ArgumentParser()

	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--source_cache_dir", type=str, default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz")
	parser.add_argument(
		"--failure_cache_dir",
		type=str,
		default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/failure_sim_state_cache/",
	)
	parser.add_argument("--source_file_indices", type=int, nargs="*", default=None)
	parser.add_argument("--failure_file_indices", type=int, nargs="*", default=None)
	parser.add_argument("--failure_sampling_prob", type=float, default=0.5)
	parser.add_argument("--pretrained_ckpt", type=str, default=None)
	parser.add_argument("--resume_path", type=str, default=None)
	parser.add_argument("--use_ema", action="store_true", default=True)
	parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")

	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--num_epochs", type=int, default=1000)
	parser.add_argument("--steps_per_epoch", type=int, default=1000)
	parser.add_argument("--save_every", type=int, default=50)
	parser.add_argument("--log_every", type=int, default=50)
	parser.add_argument("--max_steps", type=int, default=None)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--pin_memory", action="store_true")

	parser.add_argument("--output_dir", type=str, default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/finetune_fm_policy")
	parser.add_argument("--save_dir", type=str, default=None)
	parser.add_argument("--wandb_project", type=str, default="fm_policy")
	parser.add_argument("--wandb_name", type=str, default="finetune_fm_policy")
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
	parser.add_argument("--tag", type=str, default=None)
	parser.add_argument("--data_parallel", action="store_true")

	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--warmup_steps", type=int, default=500)
	parser.add_argument("--grad_clip_norm", type=float, default=1.0)
	parser.add_argument("--ema_decay", type=float, default=0.999)
	parser.add_argument("--goal_mask_prob", type=float, default=0.3)

	parser.add_argument("--target_dim", type=int, default=5)
	parser.add_argument("--hidden_dim", type=int, default=512)
	parser.add_argument("--cond_dim", type=int, default=256)
	parser.add_argument("--predict_horizon", type=int, default=25)
	parser.add_argument("--map_attr_dim", type=int, default=25)
	parser.add_argument("--tl_attr_dim", type=int, default=11)
	parser.add_argument("--other_dim", type=int, default=15)
	parser.add_argument("--inst_dim", type=int, default=768)
	parser.add_argument("--subgoal_conditioned", action="store_true", default=False)
	parser.add_argument("--default_num_steps", type=int, default=100)

	parser.add_argument("--max_range", type=float, default=30.0)
	parser.add_argument("--ego_range", type=float, default=100.0)
	parser.add_argument("--max_velocity", type=float, default=25.0)
	parser.add_argument("--max_width", type=float, default=10.0)
	parser.add_argument("--max_tl_points", type=int, default=16)
	parser.add_argument("--num_map_type_classes", type=int, default=21)
	parser.add_argument("--max_num_segments", type=int, default=128)
	parser.add_argument("--max_points_per_segment", type=int, default=128)
	parser.add_argument("--model_dt", type=float, default=0.2)
	parser.add_argument("--world_dt_fallback", type=float, default=0.1)

	parser.add_argument("--jax_compilation_cache_dir", type=str, default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/.jax_compilation_cache")

	args = parser.parse_args()

	if not 0.0 <= args.failure_sampling_prob <= 1.0:
		parser.error("--failure_sampling_prob must be in [0, 1]")
	if args.warmup_steps < 0:
		parser.error("--warmup_steps must be >= 0")
	if args.max_steps is not None and args.max_steps <= 0:
		parser.error("--max_steps must be > 0 when provided")
	if not args.failure_cache_dir:
		parser.error("Provide --failure_cache_dir.")
	if args.tag is not None:
		args.wandb_name = f"{args.wandb_name}_{args.tag}"
	if args.save_dir is None:
		args.save_dir = args.output_dir
	if args.pretrained_ckpt is None and args.resume_path is None:
		parser.error("Provide --pretrained_ckpt to start finetuning or --resume_path to continue a finetuning run.")

	return FlowMatchingFinetuningConfig(**vars(args))


def config_to_dict(config: FlowMatchingFinetuningConfig) -> dict:
	return asdict(config)
