from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from data.types import PreprocessConfig


@dataclass
class DiffusionTrainConfig:
	cache_dir: str
	preprocess_cfg: PreprocessConfig = PreprocessConfig()
	file_indices: list[int] | None = None
	cache_batch_size: int = 64
	shuffle_seed: int = 0
	instruction_seed: int = 0
	num_workers: int = 0
	pin_memory: bool = False

	# Hardness reweighting. risk_table=None (or alpha=0) is the unweighted control.
	risk_table: str | None = None
	hardness_alpha: float = 0.0
	weight_clip: float = 10.0
	exclude_dirty: bool = True

	pretrained_ckpt: str | None = None
	resume_path: str | None = None
	use_ema: bool = True
	goal_mask_prob: float = 0.5

	# Closed-loop safe-goal-reach eval. Same criterion that produced the failure
	# set: success = goal_reached & ~offroad & ~overlap & ~tl_violation, computed
	# by simulation.runner.run. eval_every=0 disables it.
	eval_every: int = 0
	# Primary eval set (logged under eval/<eval_tag>/...).
	eval_tag: str = "failures"
	eval_tfrecord_dir: str | None = None
	eval_lane_graph_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
	eval_scenario_indices: str | None = None
	eval_num_scenarios: int = 200
	# Optional secondary eval set (logged under eval/<eval2_tag>/...); set
	# eval2_tfrecord_dir to enable it. The rollout settings below are shared.
	eval2_tag: str = "womd"
	eval2_tfrecord_dir: str | None = None
	eval2_lane_graph_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
	eval2_scenario_indices: str | None = None
	eval2_num_scenarios: int = 200
	eval_num_worlds: int = 10
	eval_start_timestep: int = 10
	eval_replan_interval_steps: int = 10
	eval_instruction_interval_steps: int = 40
	eval_max_num_objects: int = 128
	eval_mask_goal: bool = False
	eval_use_ema: bool = True
	eval_seed: int = 0
	eval_visualize_mode: str = "none"

	save_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/vla/train_diffusion"
	epochs: int = 20
	steps_per_epoch: int = 1000
	save_every: int = 5
	log_every: int = 50
	seed: int = 0

	pbar_every: int = 10
	log_jsonl_path: str = "./train_logs.jsonl"

	lr: float = 1e-4
	weight_decay: float = 1e-4
	warmup_steps: int = 500
	grad_clip_norm: float = 1.0
	ema_decay: float = 0.999

	jax_compilation_cache_dir: str = "./.jax_compilation_cache"

	# wandb options
	wandb_project: str = "waymax_vla"
	wandb_name: str = "train_diffusion"
	wandb_entity: str | None = None
	wandb_mode: str = "online"


def parse_args() -> DiffusionTrainConfig:
	parser = argparse.ArgumentParser()

	parser.add_argument("--cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/")
	parser.add_argument("--file_indices", type=int, nargs="*", default=None)
	parser.add_argument("--cache_batch_size", type=int, default=64)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--instruction_seed", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--pin_memory", action="store_true")

	parser.add_argument("--risk_table", type=str, default=None,
		help="risk.npz from train_diffusion.hardness.risk_model; omit for the unweighted run")
	parser.add_argument("--hardness_alpha", type=float, default=0.0,
		help="0 = unweighted (exact control), 1 = sample in proportion to predicted failure risk")
	parser.add_argument("--weight_clip", type=float, default=10.0)
	parser.add_argument("--exclude_dirty", action="store_true", default=True,
		help="drop scenes where the logged expert itself leaves the drivable area")
	parser.add_argument("--no_exclude_dirty", action="store_false", dest="exclude_dirty")

	parser.add_argument("--pretrained_ckpt", type=str, required=True)
	parser.add_argument("--resume_path", type=str, default=None)
	parser.add_argument("--use_ema", action="store_true", default=True)
	parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")

	parser.add_argument("--eval_every", type=int, default=0,
		help="run the closed-loop safe-goal-reach eval every N epochs (0 disables)")
	parser.add_argument("--eval_tag", type=str, default="failures",
		help="wandb sub-namespace for the primary eval set: eval/<tag>/...")
	parser.add_argument("--eval_tfrecord_dir", type=str, default=None,
		help="dir of tfrecords to eval on (e.g. the failure set); required when eval_every>0")
	parser.add_argument("--eval_lane_graph_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs")
	parser.add_argument("--eval_scenario_indices", type=str, default=None,
		help="comma-separated scenario indices; omit to sweep the first eval_num_scenarios")
	parser.add_argument("--eval_num_scenarios", type=int, default=200)
	parser.add_argument("--eval2_tag", type=str, default="womd",
		help="wandb sub-namespace for the optional secondary eval set")
	parser.add_argument("--eval2_tfrecord_dir", type=str, default=None,
		help="enable a second eval set (e.g. held-out WOMD shards); omit to disable")
	parser.add_argument("--eval2_lane_graph_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs")
	parser.add_argument("--eval2_scenario_indices", type=str, default=None)
	parser.add_argument("--eval2_num_scenarios", type=int, default=200)
	parser.add_argument("--eval_num_worlds", type=int, default=10)
	parser.add_argument("--eval_start_timestep", type=int, default=10)
	parser.add_argument("--eval_replan_interval_steps", type=int, default=10)
	parser.add_argument("--eval_instruction_interval_steps", type=int, default=40)
	parser.add_argument("--eval_max_num_objects", type=int, default=128)
	parser.add_argument("--eval_mask_goal", action="store_true", default=False)
	parser.add_argument("--eval_use_ema", action="store_true", default=True)
	parser.add_argument("--no_eval_use_ema", action="store_false", dest="eval_use_ema")
	parser.add_argument("--eval_seed", type=int, default=0)
	parser.add_argument("--eval_visualize_mode", type=str, default="none",
		choices=["all", "success", "failure", "none"])

	parser.add_argument("--save_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/vla/train_diffusion")
	parser.add_argument("--epochs", type=int, default=500)
	parser.add_argument("--steps_per_epoch", type=int, default=1000)
	parser.add_argument("--save_every", type=int, default=10)
	parser.add_argument("--log_every", type=int, default=50)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--pbar_every", type=int, default=10)

	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--warmup_steps", type=int, default=500)
	parser.add_argument("--grad_clip_norm", type=float, default=1.0)
	parser.add_argument("--ema_decay", type=float, default=0.999)

	parser.add_argument("--jax_compilation_cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/vla/.jax_compilation_cache")

	parser.add_argument("--wandb_project", type=str, default="train_diffusion")
	parser.add_argument("--wandb_name", type=str, default="train_diffusion")
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
	args = parser.parse_args()
	return DiffusionTrainConfig(**vars(args))


def config_to_dict(config: DiffusionTrainConfig) -> dict:
	return asdict(config)