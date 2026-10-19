from __future__ import annotations

import argparse
import dataclasses
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data.failure_detection_loader import build_dataloader
from data.types import PreprocessConfig
from data.utils import resolve_cache_paths
from train_vla.utils.utils import (
	build_vla_model,
	move_features_to_device,
	resolve_device,
	resolve_dtype,
	save_checkpoint,
	save_training_config,
	tokenize_text_batch,
)


DEFAULT_SUCCESS_CACHE_DIR = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz"
DEFAULT_FAILURE_CACHE_DIR = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz"
DEFAULT_FAILURE_RESULT_ROOT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/exp"
DEFAULT_SUCCESS_ANCHOR_STEPS = (0, 10, 20, 30, 40, 50, 60, 70, 80)


@dataclass(frozen=True)
class FDTrainingConfig:
	success_cache_dir: str = DEFAULT_SUCCESS_CACHE_DIR
	failure_cache_dir: str = DEFAULT_FAILURE_CACHE_DIR
	failure_result_root: str | None = DEFAULT_FAILURE_RESULT_ROOT
	success_file_indices: list[int] | None = None
	failure_file_indices: list[int] | None = None
	output_dir: str = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/train_fd_vla"
	wandb_project: str | None = "train_fd_vla"
	wandb_run_name: str | None = "train_fd_vla"
	wandb_entity: str | None = None
	wandb_mode: str = "online"
	model_type: str = "failure_detection"
	gemma_name: str = "google/gemma-4-E2B-it"
	qwen_name: str = "Qwen/Qwen3-0.6B"
	batch_size: int = 8
	learning_rate: float = 5e-5
	weight_decay: float = 0.01
	num_epochs: int = 1
	max_steps: int | None = None
	grad_accum_steps: int = 1
	warmup_steps: int = 1000
	max_grad_norm: float = 1.0
	shuffle_seed: int = 0
	eval_every: int = 1000
	eval_num_samples: int = 256
	log_every: int = 10
	save_every: int = 1000
	max_prompt_length: int = 128
	max_answer_length: int = 16
	num_workers: int = 0
	pin_memory: bool = True
	add_eos: bool = True
	dtype: str = "bf16"
	freeze_llm: bool = True
	use_gradient_checkpointing: bool = True
	use_lora: bool = True
	lora_r: int = 16
	lora_alpha: int = 32
	lora_dropout: float = 0.05
	lora_target_modules: str | None = None
	num_scene_tokens: int = 32
	ego_dim: int = 5
	goal_dim: int = 3
	other_dim: int = 15
	map_dim: int = 25
	tl_dim: int = 11
	scene_hidden_dim: int = 512
	success_anchor_steps: tuple[int, ...] = DEFAULT_SUCCESS_ANCHOR_STEPS
	failure_max_anchor_offset: int = 50
	num_success_val_files: int = 1
	num_failure_val_files: int = 1
	tag: str | None = None
	preprocess_cfg: PreprocessConfig = PreprocessConfig()


def _parse_int_list(values: list[str] | None) -> list[int] | None:
	if values is None:
		return None
	parsed: list[int] = []
	for token in values:
		for part in token.split(","):
			part = part.strip()
			if part:
				parsed.append(int(part))
	return parsed or None


def _parse_int_tuple(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
	if raw is None:
		return default
	parts = [part.strip() for part in raw.split(",") if part.strip()]
	if not parts:
		return default
	return tuple(int(part) for part in parts)


def _normalize_answer(text: str) -> str:
	return text.strip().lower()


def _move_numeric_dict_to_device(values: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
	return move_features_to_device(values, device, dtype)


def _flatten_batch(batch) -> tuple[dict[str, torch.Tensor], list[str], list[str], dict[str, Any]]:
	if batch.prompts is None or batch.answers is None:
		raise ValueError("batch.prompts and batch.answers are required for failure detection training")
	return batch.features, list(batch.prompts), list(batch.answers), batch.aux or {}


def _build_token_inputs(
	model,
	prompts: list[str],
	answers: list[str],
	*,
	device: torch.device,
	max_prompt_length: int,
	max_answer_length: int,
	add_eos: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	return tokenize_text_batch(
		model.tokenizer,
		prompts,
		answers,
		add_eos=add_eos,
		device=device,
		max_prompt_length=max_prompt_length,
		max_answer_length=max_answer_length,
	)


def evaluate_failure_detection(
	model,
	loader,
	*,
	device: torch.device,
	dtype: torch.dtype,
	max_samples: int,
	max_prompt_length: int,
	max_answer_length: int,
	add_eos: bool,
) -> dict[str, float]:
	if max_samples <= 0:
		return {key: 0.0 for key in (
			"loss",
			"language_loss",
			"object_loss",
			"road_segment_loss",
			"reason_exact_match",
			"failure_reason_exact_match",
			"success_exact_match",
			"collision_object_id_accuracy",
			"offroad_road_segment_accuracy",
			"object_accuracy",
			"road_segment_accuracy",
			"num_samples",
		)}

	was_training = model.training
	model.eval()

	loss_sum = 0.0
	language_loss_sum = 0.0
	object_loss_sum = 0.0
	road_loss_sum = 0.0
	object_accuracy_sum = 0.0
	road_accuracy_sum = 0.0
	count = 0

	reason_correct = 0
	failure_reason_correct = 0
	success_correct = 0
	failure_total = 0
	success_total = 0

	collision_object_correct = 0
	collision_object_total = 0
	offroad_road_correct = 0
	offroad_road_total = 0

	with torch.inference_mode():
		for batch in loader:
			features, prompts, answers, aux = _flatten_batch(batch)
			remaining = max_samples - count
			if remaining <= 0:
				break
			if len(prompts) > remaining:
				prompts = prompts[:remaining]
				answers = answers[:remaining]
				features = {key: value[:remaining] for key, value in features.items()}
				aux = {key: value[:remaining] for key, value in aux.items()}

			prompt_ids, prompt_mask, answer_ids, answer_mask = _build_token_inputs(
				model,
				prompts,
				answers,
				device=device,
				max_prompt_length=max_prompt_length,
				max_answer_length=max_answer_length,
				add_eos=add_eos,
			)
			features = move_features_to_device(features, device, dtype)
			aux_tensors = _move_numeric_dict_to_device(aux, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				outputs = model(
					features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					answer_ids=answer_ids,
					answer_mask=answer_mask,
					critical_object_ids=aux_tensors.get("critical_object_ids"),
					critical_road_segment_ids=aux_tensors.get("critical_road_segment_ids"),
				)
				predictions = model.generate_predictions(
					input_features=features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					max_new_tokens=max_answer_length,
				)

			batch_size = len(answers)
			loss_sum += float(outputs["loss"].detach().cpu()) * batch_size
			language_loss_sum += float(outputs["language_loss"].detach().cpu()) * batch_size
			object_loss_sum += float(outputs["object_loss"].detach().cpu()) * batch_size
			road_loss_sum += float(outputs["road_segment_loss"].detach().cpu()) * batch_size
			object_accuracy_sum += float(outputs["object_accuracy"].detach().cpu()) * batch_size
			road_accuracy_sum += float(outputs["road_segment_accuracy"].detach().cpu()) * batch_size

			pred_answers = predictions["answer"]
			pred_object_ids = np.asarray(predictions["critical_object_ids"]).reshape(-1)
			pred_road_ids = np.asarray(predictions["critical_road_segment_ids"]).reshape(-1)
			gold_object_ids = aux_tensors.get("critical_object_ids")
			gold_road_ids = aux_tensors.get("critical_road_segment_ids")
			gold_object_ids_np = gold_object_ids.detach().cpu().numpy().reshape(-1) if gold_object_ids is not None else np.full((batch_size,), -1, dtype=np.int32)
			gold_road_ids_np = gold_road_ids.detach().cpu().numpy().reshape(-1) if gold_road_ids is not None else np.full((batch_size,), -1, dtype=np.int32)

			for pred_text, target_text, pred_obj, pred_road, gold_obj, gold_road in zip(
				pred_answers,
				answers,
				pred_object_ids,
				pred_road_ids,
				gold_object_ids_np,
				gold_road_ids_np,
			):
				pred_norm = _normalize_answer(pred_text)
				target_norm = _normalize_answer(target_text)
				is_correct = pred_norm == target_norm
				reason_correct += int(is_correct)
				if target_norm == "success":
					success_total += 1
					success_correct += int(is_correct)
				else:
					failure_total += 1
					failure_reason_correct += int(is_correct)

				if target_norm == "collision" and int(gold_obj) >= 0:
					collision_object_total += 1
					collision_object_correct += int(int(pred_obj) == int(gold_obj))
				elif target_norm == "offroad" and int(gold_road) >= 0:
					offroad_road_total += 1
					offroad_road_correct += int(int(pred_road) == int(gold_road))

			count += batch_size
			if count >= max_samples:
				break

	if was_training:
		model.train()

	return {
		"loss": loss_sum / count if count else 0.0,
		"language_loss": language_loss_sum / count if count else 0.0,
		"object_loss": object_loss_sum / count if count else 0.0,
		"road_segment_loss": road_loss_sum / count if count else 0.0,
		"reason_exact_match": reason_correct / count if count else 0.0,
		"failure_reason_exact_match": failure_reason_correct / failure_total if failure_total else 0.0,
		"success_exact_match": success_correct / success_total if success_total else 0.0,
		"collision_object_id_accuracy": collision_object_correct / collision_object_total if collision_object_total else 0.0,
		"offroad_road_segment_accuracy": offroad_road_correct / offroad_road_total if offroad_road_total else 0.0,
		"object_accuracy": object_accuracy_sum / count if count else 0.0,
		"road_segment_accuracy": road_accuracy_sum / count if count else 0.0,
		"num_samples": float(count),
	}


def _split_paths_by_count(paths: tuple[str, ...], num_val_files: int, shuffle_seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
	ordered = list(paths)
	random.Random(int(shuffle_seed)).shuffle(ordered)
	if num_val_files <= 0:
		return tuple(ordered), ()
	num_val = min(int(num_val_files), len(ordered))
	val_paths = tuple(ordered[:num_val])
	train_paths = tuple(ordered[num_val:])
	return train_paths, val_paths


def _split_train_val(cache_dir: str, file_indices: list[int] | None, num_val_files: int, shuffle_seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
	paths = resolve_cache_paths(cache_dir, file_indices)
	return _split_paths_by_count(paths, num_val_files, shuffle_seed)


def run_training(cfg: FDTrainingConfig) -> None:
	device = resolve_device()
	dtype = resolve_dtype(cfg.dtype)
	torch.backends.cuda.matmul.allow_tf32 = True

	run_name = (cfg.wandb_run_name or "train_fd_vla") + f"_{cfg.model_type}"
	if cfg.tag:
		run_name += f"_{cfg.tag}"
	run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

	output_dir = Path(cfg.output_dir) / run_name
	output_dir.mkdir(parents=True, exist_ok=True)
	save_training_config(cfg, output_dir)

	success_train_paths, success_val_paths = _split_train_val(cfg.success_cache_dir, cfg.success_file_indices, cfg.num_success_val_files, cfg.shuffle_seed)
	failure_train_paths, failure_val_paths = _split_train_val(cfg.failure_cache_dir, cfg.failure_file_indices, cfg.num_failure_val_files, cfg.shuffle_seed + 1)
	print(f"Number of training success cache paths: {len(success_train_paths)}")
	print(f"Number of validation success cache paths: {len(success_val_paths)}")
	print(f"Number of training failure cache paths: {len(failure_train_paths)}")
	print(f"Number of validation failure cache paths: {len(failure_val_paths)}")

	train_loader = build_dataloader(
		success_cache_dir=cfg.success_cache_dir,
		failure_cache_dir=cfg.failure_cache_dir,
		preprocess_cfg=cfg.preprocess_cfg,
		success_cache_paths=success_train_paths,
		failure_cache_paths=failure_train_paths,
		failure_result_root=cfg.failure_result_root,
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
		drop_last=False,
		success_anchor_steps=cfg.success_anchor_steps,
		failure_max_anchor_offset=cfg.failure_max_anchor_offset,
	)
	val_loader = None
	if success_val_paths or failure_val_paths:
		val_loader = build_dataloader(
			success_cache_dir=cfg.success_cache_dir,
			failure_cache_dir=cfg.failure_cache_dir,
			preprocess_cfg=cfg.preprocess_cfg,
			success_cache_paths=success_val_paths,
			failure_cache_paths=failure_val_paths,
			failure_result_root=cfg.failure_result_root,
			batch_size=cfg.batch_size,
			shuffle_seed=cfg.shuffle_seed,
			num_workers=cfg.num_workers,
			pin_memory=cfg.pin_memory,
			drop_last=False,
			success_anchor_steps=cfg.success_anchor_steps,
			failure_max_anchor_offset=cfg.failure_max_anchor_offset,
		)

	model = build_vla_model(cfg)
	if cfg.use_gradient_checkpointing and hasattr(model.llm, "gradient_checkpointing_enable"):
		model.llm.gradient_checkpointing_enable()
	if cfg.freeze_llm and hasattr(model, "freeze_llm"):
		model.freeze_llm()
	model = model.to(device=device, dtype=dtype)
	model.train()

	wandb_run = None
	if cfg.wandb_project:
		try:
			import wandb
		except ImportError as exc:
			raise ImportError("wandb is required when wandb_project is set") from exc
		wandb_run = wandb.init(
			project=cfg.wandb_project,
			entity=cfg.wandb_entity,
			name=run_name,
			mode=cfg.wandb_mode,
			config=asdict(cfg),
		)

	trainable_params = [param for param in model.parameters() if param.requires_grad]
	optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

	scheduler = None
	if cfg.warmup_steps > 0:
		warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=cfg.warmup_steps)
		if cfg.max_steps is not None:
			decay_steps = max(0, cfg.max_steps - cfg.warmup_steps)
			if decay_steps > 0:
				decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps)
				scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, decay_scheduler], milestones=[cfg.warmup_steps])
			else:
				scheduler = warmup_scheduler
		else:
			scheduler = warmup_scheduler

	scaler = torch.cuda.amp.GradScaler() if cfg.dtype == "fp16" and device.type == "cuda" else None

	global_step = 0
	optimizer.zero_grad(set_to_none=True)
	for epoch in range(cfg.num_epochs):
		epoch_total = None if cfg.max_steps is None else max(0, cfg.max_steps - global_step)
		pbar = tqdm(total=epoch_total, desc=f"epoch {epoch + 1}/{cfg.num_epochs}", unit="step")
		for batch in train_loader:
			features, prompts, answers, aux = _flatten_batch(batch)
			if not prompts:
				continue

			prompt_ids, prompt_mask, answer_ids, answer_mask = _build_token_inputs(
				model,
				prompts,
				answers,
				device=device,
				max_prompt_length=cfg.max_prompt_length,
				max_answer_length=cfg.max_answer_length,
				add_eos=cfg.add_eos,
			)
			features = move_features_to_device(features, device, dtype)
			aux_tensors = _move_numeric_dict_to_device(aux, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				outputs = model(
					features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					answer_ids=answer_ids,
					answer_mask=answer_mask,
					critical_object_ids=aux_tensors.get("critical_object_ids"),
					critical_road_segment_ids=aux_tensors.get("critical_road_segment_ids"),
				)
				loss = outputs["loss"] / cfg.grad_accum_steps

			if scaler is not None:
				scaler.scale(loss).backward()
			else:
				loss.backward()

			if (global_step + 1) % cfg.grad_accum_steps == 0:
				if cfg.max_grad_norm > 0:
					if scaler is not None:
						scaler.unscale_(optimizer)
						clip_grad_norm_(trainable_params, cfg.max_grad_norm)
					else:
						clip_grad_norm_(trainable_params, cfg.max_grad_norm)
				if scaler is not None:
					scaler.step(optimizer)
					scaler.update()
				else:
					optimizer.step()
				optimizer.zero_grad(set_to_none=True)
				if scheduler is not None:
					scheduler.step()

			if global_step % cfg.log_every == 0:
				current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
				train_loss = float(outputs["loss"].detach().cpu())
				pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{current_lr:.3e}", batch=len(prompts))
				if wandb_run is not None:
					wandb_run.log(
						{
							"train/step": global_step,
							"train/loss": train_loss,
							"train/language_loss": float(outputs["language_loss"].detach().cpu()),
							"train/object_loss": float(outputs["object_loss"].detach().cpu()),
							"train/road_segment_loss": float(outputs["road_segment_loss"].detach().cpu()),
							"train/object_accuracy": float(outputs["object_accuracy"].detach().cpu()),
							"train/road_segment_accuracy": float(outputs["road_segment_accuracy"].detach().cpu()),
							"train/lr": current_lr,
							"train/batch_size": len(prompts),
						},
						step=global_step,
					)

			if cfg.save_every > 0 and global_step > 0 and global_step % cfg.save_every == 0:
				save_checkpoint(output_dir, f"step_{global_step}", model, optimizer, scheduler, step=global_step, cfg=cfg)

			if global_step % cfg.eval_every == 0 and val_loader is not None:
				val_metrics = evaluate_failure_detection(
					model,
					val_loader,
					device=device,
					dtype=dtype,
					max_samples=cfg.eval_num_samples,
					max_prompt_length=cfg.max_prompt_length,
					max_answer_length=cfg.max_answer_length,
					add_eos=cfg.add_eos,
				)
				if wandb_run is not None:
					wandb_run.log(
						{
							"val/step": global_step,
							"val/loss": val_metrics["loss"],
							"val/language_loss": val_metrics["language_loss"],
							"val/object_loss": val_metrics["object_loss"],
							"val/road_segment_loss": val_metrics["road_segment_loss"],
							"val/reason_exact_match": val_metrics["reason_exact_match"],
							"val/failure_reason_exact_match": val_metrics["failure_reason_exact_match"],
							"val/success_exact_match": val_metrics["success_exact_match"],
							"val/collision_object_id_accuracy": val_metrics["collision_object_id_accuracy"],
							"val/offroad_road_segment_accuracy": val_metrics["offroad_road_segment_accuracy"],
							"val/object_accuracy": val_metrics["object_accuracy"],
							"val/road_segment_accuracy": val_metrics["road_segment_accuracy"],
							"val/num_samples": val_metrics["num_samples"],
						},
						step=global_step,
					)

			global_step += 1
			pbar.update(1)
			if cfg.max_steps is not None and global_step >= cfg.max_steps:
				break
		pbar.close()
		if cfg.max_steps is not None and global_step >= cfg.max_steps:
			break

	save_checkpoint(output_dir, f"step_{global_step}", model, optimizer, scheduler, step=global_step, cfg=cfg)
	if callable(getattr(model.llm, "save_pretrained", None)):
		model.llm.save_pretrained(str(output_dir / "llm"))
	if callable(getattr(model.tokenizer, "save_pretrained", None)):
		model.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
	if wandb_run is not None:
		wandb_run.finish()


def parse_args() -> FDTrainingConfig:
	parser = argparse.ArgumentParser(description="Train failure_detection_vla on success/failure caches.")
	parser.add_argument("--success_cache_dir", type=str, default=DEFAULT_SUCCESS_CACHE_DIR)
	parser.add_argument("--failure_cache_dir", type=str, default=DEFAULT_FAILURE_CACHE_DIR)
	parser.add_argument("--failure_result_root", type=str, default=DEFAULT_FAILURE_RESULT_ROOT)
	parser.add_argument("--success_file_indices", type=str, nargs="*", default=None)
	parser.add_argument("--failure_file_indices", type=str, nargs="*", default=None)
	parser.add_argument("--output_dir", type=str, default="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/train_fd_vla")
	parser.add_argument("--wandb_project", type=str, default="train_fd_vla")
	parser.add_argument("--wandb_run_name", type=str, default="train_fd_vla")
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))
	parser.add_argument("--model_type", type=str, default="failure_detection", choices=("failure_detection",))
	parser.add_argument("--gemma_name", type=str, default="google/gemma-4-E2B-it")
	parser.add_argument("--qwen_name", type=str, default="Qwen/Qwen3-0.6B")
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--learning_rate", type=float, default=5e-5)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--num_epochs", type=int, default=1000)
	parser.add_argument("--max_steps", type=int, default=None)
	parser.add_argument("--grad_accum_steps", type=int, default=1)
	parser.add_argument("--warmup_steps", type=int, default=1000)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--num_success_val_files", type=int, default=1)
	parser.add_argument("--num_failure_val_files", type=int, default=100)
	parser.add_argument("--eval_every", type=int, default=1000)
	parser.add_argument("--eval_num_samples", type=int, default=256)
	parser.add_argument("--log_every", type=int, default=10)
	parser.add_argument("--save_every", type=int, default=1000)
	parser.add_argument("--max_prompt_length", type=int, default=128)
	parser.add_argument("--max_answer_length", type=int, default=16)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--no_pin_memory", action="store_true")
	parser.add_argument("--add_eos", action="store_true", default=True)
	parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16"))
	parser.add_argument("--freeze_llm", action="store_true", default=True)
	parser.add_argument("--no_gradient_checkpointing", action="store_true")
	parser.add_argument("--use_lora", action="store_true", default=True)
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument("--lora_target_modules", type=str, default=None)
	parser.add_argument("--num_scene_tokens", type=int, default=32)
	parser.add_argument("--ego_dim", type=int, default=5)
	parser.add_argument("--goal_dim", type=int, default=3)
	parser.add_argument("--other_dim", type=int, default=15)
	parser.add_argument("--map_dim", type=int, default=25)
	parser.add_argument("--tl_dim", type=int, default=11)
	parser.add_argument("--scene_hidden_dim", type=int, default=512)
	parser.add_argument("--success_anchor_steps", type=str, default=None)
	parser.add_argument("--failure_max_anchor_offset", type=int, default=50)
	parser.add_argument("--tag", type=str, default=None)

	args = parser.parse_args()
	return FDTrainingConfig(
		success_cache_dir=args.success_cache_dir,
		failure_cache_dir=args.failure_cache_dir,
		failure_result_root=args.failure_result_root,
		success_file_indices=_parse_int_list(args.success_file_indices),
		failure_file_indices=_parse_int_list(args.failure_file_indices),
		output_dir=args.output_dir,
		wandb_project=args.wandb_project,
		wandb_run_name=args.wandb_run_name,
		wandb_entity=args.wandb_entity,
		wandb_mode=args.wandb_mode,
		model_type=args.model_type,
		gemma_name=args.gemma_name,
		qwen_name=args.qwen_name,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		num_epochs=args.num_epochs,
		max_steps=args.max_steps,
		grad_accum_steps=args.grad_accum_steps,
		warmup_steps=args.warmup_steps,
		max_grad_norm=args.max_grad_norm,
		shuffle_seed=args.shuffle_seed,
		num_success_val_files=args.num_success_val_files,
		num_failure_val_files=args.num_failure_val_files,
		eval_every=args.eval_every,
		eval_num_samples=args.eval_num_samples,
		log_every=args.log_every,
		save_every=args.save_every,
		max_prompt_length=args.max_prompt_length,
		max_answer_length=args.max_answer_length,
		num_workers=args.num_workers,
		pin_memory=not args.no_pin_memory,
		add_eos=args.add_eos,
		dtype=args.dtype,
		freeze_llm=args.freeze_llm,
		use_gradient_checkpointing=not args.no_gradient_checkpointing,
		use_lora=args.use_lora,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		lora_target_modules=args.lora_target_modules,
		num_scene_tokens=args.num_scene_tokens,
		ego_dim=args.ego_dim,
		goal_dim=args.goal_dim,
		other_dim=args.other_dim,
		map_dim=args.map_dim,
		tl_dim=args.tl_dim,
		scene_hidden_dim=args.scene_hidden_dim,
		success_anchor_steps=_parse_int_tuple(args.success_anchor_steps, DEFAULT_SUCCESS_ANCHOR_STEPS),
		failure_max_anchor_offset=args.failure_max_anchor_offset,
		tag=args.tag,
	)


def main() -> None:
	run_training(parse_args())


if __name__ == "__main__":
	main()