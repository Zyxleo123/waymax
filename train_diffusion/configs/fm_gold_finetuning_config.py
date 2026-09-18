from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


DEFAULT_CACHE_DIR = (
	"/zfsauton/scratch/mineuih/waymax_rs/exp/"
	"simulation_fm_training_collect_failure_20260903_030456/success_for_failure_cache"
)
DEFAULT_ANCHOR_STEPS = (0, 10, 20, 30, 40)


def _parse_anchor_steps(raw_anchor_steps: list[str] | None) -> tuple[int, ...]:
	if not raw_anchor_steps:
		return DEFAULT_ANCHOR_STEPS
	steps: list[int] = []
	for token in raw_anchor_steps:
		for part in str(token).split(","):
			part = part.strip()
			if not part:
				continue
			steps.append(int(part))
	if not steps:
		raise ValueError("anchor_steps cannot be empty")
	return tuple(steps)


@dataclass
class FlowMatchingGoldFinetuningConfig:
	seed: int = 0
	cache_dir: str = DEFAULT_CACHE_DIR
	cache_file_indices: list[int] | None = None
	pretrained_ckpt: str | None = None
	resume_path: str | None = None
	use_ema: bool = True

	batch_size: int = 32
	num_epochs: int = 20
	steps_per_epoch: int = 1000
	save_every: int = 5
	log_every: int = 50
	max_steps: int | None = None
	shuffle_seed: int = 0
	num_workers: int = 0
	pin_memory: bool = False
	anchor_steps: tuple[int, ...] = DEFAULT_ANCHOR_STEPS

	output_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/checkpoints/finetune_fm_gold_policy"
	save_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/checkpoints/finetune_fm_gold_policy"
	wandb_project: str | None = "fm_policy"
	wandb_name: str = "finetune_fm_gold_policy"
	wandb_entity: str | None = None
	wandb_mode: str = "online"
	tag: str | None = None
	data_parallel: bool = False

	lr: float = 1e-4
	weight_decay: float = 1e-4
	warmup_steps: int = 500
	grad_clip_norm: float = 1.0
	ema_decay: float = 0.999

	target_dim: int = 5
	hidden_dim: int = 512
	cond_dim: int = 256
	predict_horizon: int = 25
	map_attr_dim: int = 25
	tl_attr_dim: int = 11
	other_dim: int = 15
	inst_dim: int = 768
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

	jax_compilation_cache_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/.jax_compilation_cache"


def parse_args() -> FlowMatchingGoldFinetuningConfig:
	parser = argparse.ArgumentParser(
		description="Finetune gold-id encoders and FiLM using success_for_failure_cache."
	)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument(
		"--cache_dir",
		type=str,
		default=DEFAULT_CACHE_DIR,
		help="Directory of success_for_failure_shard_*.sim_state_cache.npz files, or the parent exp dir.",
	)
	parser.add_argument(
		"--success_for_failure_cache_dir",
		type=str,
		default=None,
		help="Alias for --cache_dir.",
	)
	parser.add_argument("--cache_file_indices", type=int, nargs="*", default=None)
	parser.add_argument("--pretrained_ckpt", type=str, default=None)
	parser.add_argument("--resume_path", type=str, default=None)
	parser.add_argument("--use_ema", action="store_true", default=True)
	parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")

	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_epochs", type=int, default=100)
	parser.add_argument("--steps_per_epoch", type=int, default=1000)
	parser.add_argument("--save_every", type=int, default=5)
	parser.add_argument("--log_every", type=int, default=50)
	parser.add_argument("--max_steps", type=int, default=None)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--pin_memory", action="store_true")
	parser.add_argument(
		"--anchor_steps",
		type=str,
		nargs="*",
		default=["0,10,20,30,40"],
		help="Anchor timesteps sampled uniformly per scenario for target/scene features.",
	)

	parser.add_argument(
		"--output_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/finetune_fm_gold_policy",
	)
	parser.add_argument("--save_dir", type=str, default=None)
	parser.add_argument("--wandb_project", type=str, default="fm_policy")
	parser.add_argument("--wandb_name", type=str, default="finetune_fm_gold_policy")
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
	parser.add_argument("--tag", type=str, default=None)
	parser.add_argument("--data_parallel", action="store_true")

	parser.add_argument("--lr", type=float, default=2e-6)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--warmup_steps", type=int, default=500)
	parser.add_argument("--grad_clip_norm", type=float, default=1.0)
	parser.add_argument("--ema_decay", type=float, default=0.999)

	parser.add_argument("--target_dim", type=int, default=5)
	parser.add_argument("--hidden_dim", type=int, default=512)
	parser.add_argument("--cond_dim", type=int, default=256)
	parser.add_argument("--predict_horizon", type=int, default=25)
	parser.add_argument("--map_attr_dim", type=int, default=25)
	parser.add_argument("--tl_attr_dim", type=int, default=11)
	parser.add_argument("--other_dim", type=int, default=15)
	parser.add_argument("--inst_dim", type=int, default=768)
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
	parser.add_argument(
		"--jax_compilation_cache_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/.jax_compilation_cache",
	)

	args = parser.parse_args()
	if args.warmup_steps < 0:
		parser.error("--warmup_steps must be >= 0")
	if args.max_steps is not None and args.max_steps <= 0:
		parser.error("--max_steps must be > 0 when provided")
	if args.success_for_failure_cache_dir:
		args.cache_dir = args.success_for_failure_cache_dir
	if not args.cache_dir:
		parser.error("Provide --cache_dir.")
	if args.pretrained_ckpt is None and args.resume_path is None:
		parser.error("Provide --pretrained_ckpt to start gold-id finetuning or --resume_path to continue.")
	try:
		args.anchor_steps = _parse_anchor_steps(args.anchor_steps)
	except ValueError as exc:
		parser.error(str(exc))
	if args.tag is not None:
		args.wandb_name = f"{args.wandb_name}_{args.tag}"
	if args.save_dir is None:
		args.save_dir = args.output_dir
	del args.success_for_failure_cache_dir
	return FlowMatchingGoldFinetuningConfig(**vars(args))


def config_to_dict(config: FlowMatchingGoldFinetuningConfig) -> dict:
	return asdict(config)
