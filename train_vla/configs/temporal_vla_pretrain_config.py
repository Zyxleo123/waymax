from __future__ import annotations

import argparse
from dataclasses import dataclass

from data.types import PreprocessConfig


@dataclass(frozen=True)
class TemporalVLAPretrainConfig:
	cache_dir: str
	annotation_dir: str
	file_indices: list[int] | None
	output_dir: str

	wandb_project: str | None = "pretrain_temporal_vla"
	wandb_run_name: str = "pretrain_temporal_vla"
	wandb_entity: str | None = None
	wandb_mode: str = "online"

	model_type: str = "temporal_gemma"
	gemma_name: str = "google/gemma-4-E2B-it"

	batch_size: int = 1
	learning_rate: float = 1e-5
	warmup_steps: int = 1000
	weight_decay: float = 0.01
	num_epochs: int = 1
	max_steps: int | None = None
	grad_accum_steps: int = 1
	max_grad_norm: float = 1.0

	shuffle_seed: int = 42
	max_num_objects: int = 128
	freeze_llm: bool = False
	use_gradient_checkpointing: bool = True
	dtype: str = "bf16"
	log_every: int = 10
	save_every: int = 1000
	eval_num_samples: int = 64
	validation_fraction: float = 0.2
	max_prompt_length: int = 128
	max_answer_length: int = 8
	num_workers: int = 0
	pin_memory: bool = True
	add_eos: bool = True

	anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40)
	num_scene_tokens_per_step: int = 8
	num_scene_tokens: int = 32

	ego_dim: int = 5
	goal_dim: int = 3
	other_dim: int = 15
	map_dim: int = 25
	tl_dim: int = 11
	scene_hidden_dim: int = 512
	num_heads: int = 4
	num_attn_layers: int = 4

	use_lora: bool = False
	lora_r: int = 16
	lora_alpha: int = 32
	lora_dropout: float = 0.05
	lora_target_modules: str | None = None
	predict_target_lane: bool = False

	preprocess_cfg: PreprocessConfig = PreprocessConfig()
	tag: str | None = None


def _parse_file_indices(raw_file_indices: list[str] | None) -> list[int] | None:
	if raw_file_indices is None:
		return None
	parsed: list[int] = []
	for token in raw_file_indices:
		for part in token.split(","):
			part = part.strip()
			if not part:
				continue
			parsed.append(int(part))
	return parsed or None


def _parse_anchor_steps(raw_anchor_steps: list[str] | None) -> tuple[int, ...]:
	if not raw_anchor_steps:
		return (0, 10, 20, 30, 40)
	parsed: list[int] = []
	for token in raw_anchor_steps:
		for part in token.split(","):
			part = part.strip()
			if not part:
				continue
			parsed.append(int(part))
	if not parsed:
		raise ValueError("anchor_steps cannot be empty")
	return tuple(parsed)


def parse_args() -> TemporalVLAPretrainConfig:
	parser = argparse.ArgumentParser(description="Pretrain TemporalGemmaVLA on temporal QA data.")
	parser.add_argument("--cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/")
	parser.add_argument("--tfrecord_dir", type=str, default=None, help="Deprecated alias for --cache_dir.")
	parser.add_argument("--annotation_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/annotations/target_lane/")
	parser.add_argument("--file_indices", type=str, nargs="*", default=None)
	parser.add_argument("--output_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_temporal_vla")

	parser.add_argument("--wandb_project", type=str, default="pretrain_temporal_vla")
	parser.add_argument("--wandb_run_name", type=str, default="pretrain_temporal_vla")
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))

	parser.add_argument("--gemma_name", type=str, default="google/gemma-4-E2B-it")
	parser.add_argument("--anchor_steps", type=str, nargs="*", default=["0,10,20,30,40"])
	parser.add_argument("--num_scene_tokens_per_step", type=int, default=8)
	parser.add_argument("--num_scene_tokens", type=int, default=32)

	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--learning_rate", type=float, default=5e-5)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--num_epochs", type=int, default=1000)
	parser.add_argument("--warmup_steps", type=int, default=1000)
	parser.add_argument("--max_steps", type=int, default=None)
	parser.add_argument("--grad_accum_steps", type=int, default=1)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)

	parser.add_argument("--shuffle_seed", type=int, default=42)
	parser.add_argument("--max_num_objects", type=int, default=128)
	parser.add_argument("--freeze_llm", action="store_true", default=True)
	parser.add_argument("--no_gradient_checkpointing", action="store_true")
	parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16"))
	parser.add_argument("--log_every", type=int, default=10)
	parser.add_argument("--save_every", type=int, default=10000)
	parser.add_argument("--eval_num_samples", type=int, default=500)
	parser.add_argument("--validation_fraction", type=float, default=0.001)
	parser.add_argument("--max_prompt_length", type=int, default=128)
	parser.add_argument("--max_answer_length", type=int, default=8)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--no_pin_memory", action="store_true")
	parser.add_argument("--add_eos", action="store_true", default=True)

	parser.add_argument("--ego_dim", type=int, default=5)
	parser.add_argument("--goal_dim", type=int, default=3)
	parser.add_argument("--other_dim", type=int, default=15)
	parser.add_argument("--map_dim", type=int, default=25)
	parser.add_argument("--tl_dim", type=int, default=11)
	parser.add_argument("--scene_hidden_dim", type=int, default=512)
	parser.add_argument("--num_heads", type=int, default=4)
	parser.add_argument("--num_attn_layers", type=int, default=4)

	parser.add_argument("--use_lora", action="store_true", default=False)
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument("--lora_target_modules", type=str, default=None)
	parser.add_argument("--predict_target_lane", action="store_true", default=False)

	parser.add_argument("--tag", type=str, default=None)

	args = parser.parse_args()
	anchor_steps = _parse_anchor_steps(args.anchor_steps)

	return TemporalVLAPretrainConfig(
		cache_dir=args.cache_dir or args.tfrecord_dir,
		annotation_dir=args.annotation_dir,
		file_indices=_parse_file_indices(args.file_indices),
		output_dir=args.output_dir,
		wandb_project=args.wandb_project,
		wandb_run_name=args.wandb_run_name,
		wandb_entity=args.wandb_entity,
		wandb_mode=args.wandb_mode,
		gemma_name=args.gemma_name,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		num_epochs=args.num_epochs,
		max_steps=args.max_steps,
		warmup_steps=args.warmup_steps,
		grad_accum_steps=args.grad_accum_steps,
		max_grad_norm=args.max_grad_norm,
		shuffle_seed=args.shuffle_seed,
		max_num_objects=args.max_num_objects,
		freeze_llm=args.freeze_llm,
		use_gradient_checkpointing=not args.no_gradient_checkpointing,
		dtype=args.dtype,
		log_every=args.log_every,
		save_every=args.save_every,
		eval_num_samples=args.eval_num_samples,
		validation_fraction=args.validation_fraction,
		max_prompt_length=args.max_prompt_length,
		max_answer_length=args.max_answer_length,
		num_workers=args.num_workers,
		pin_memory=not args.no_pin_memory,
		add_eos=args.add_eos,
		anchor_steps=anchor_steps,
		num_scene_tokens_per_step=args.num_scene_tokens_per_step,
		num_scene_tokens=args.num_scene_tokens,
		ego_dim=args.ego_dim,
		goal_dim=args.goal_dim,
		other_dim=args.other_dim,
		map_dim=args.map_dim,
		tl_dim=args.tl_dim,
		scene_hidden_dim=args.scene_hidden_dim,
		num_heads=args.num_heads,
		num_attn_layers=args.num_attn_layers,
		use_lora=args.use_lora,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		lora_target_modules=args.lora_target_modules,
		predict_target_lane=args.predict_target_lane,
		tag=args.tag,
	)
