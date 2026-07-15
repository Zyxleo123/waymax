from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data.temporal_cache_loader import build_dataloader
from data.utils import resolve_cache_paths, split_cache_paths
from model.vla.temporal_vla import TemporalGemmaVLA
from train_vla.configs.temporal_vla_pretrain_config import TemporalVLAPretrainConfig, parse_args
from train_vla.utils.utils import (
	move_features_to_device,
	resolve_device,
	resolve_dtype,
	save_checkpoint,
	save_training_config,
	tokenize_text_batch,
)


def _expand_features_for_temporal_qa(
	features: dict[str, torch.Tensor],
	timestep_valid: torch.Tensor,
	prompts: list[list[str]],
	answers: list[list[str]],
	keys: list[list[str]] | None,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str], int]:
	if not prompts or not answers:
		return features, [], [], [], 0

	base_batch = int(features["ego_state"].shape[0])
	seq_count = len(prompts)
	if seq_count % base_batch != 0:
		raise ValueError(
			f"Temporal QA sequence count ({seq_count}) is not divisible by feature batch ({base_batch})."
		)
	qa_per_scenario = seq_count // base_batch

	expanded_features: dict[str, torch.Tensor] = {}
	for key, value in features.items():
		expanded_features[key] = value.repeat_interleave(qa_per_scenario, dim=0)
	expanded_features["history_valid"] = timestep_valid.repeat_interleave(qa_per_scenario, dim=0)

	flat_prompts = [item for sequence in prompts for item in sequence]
	flat_answers = [item for sequence in answers for item in sequence]

	if keys is None:
		flat_keys = ["unknown"] * len(flat_prompts)
	else:
		flat_keys = [item for sequence in keys for item in sequence]
		if len(flat_keys) != len(flat_prompts):
			raise ValueError(
				"Mismatch between flattened QA keys and prompts: "
				f"{len(flat_keys)} != {len(flat_prompts)}"
			)

	return expanded_features, flat_prompts, flat_answers, flat_keys, qa_per_scenario


def build_temporal_vla_model(cfg: TemporalVLAPretrainConfig) -> TemporalGemmaVLA:
	lora_target_modules: tuple[str, ...] | None = None
	if cfg.lora_target_modules is not None and cfg.lora_target_modules.strip():
		lora_target_modules = tuple(
			part.strip() for part in cfg.lora_target_modules.split(",") if part.strip()
		)

	model = TemporalGemmaVLA(
		gemma_name=cfg.gemma_name,
		ego_dim=cfg.ego_dim,
		goal_dim=cfg.goal_dim,
		other_dim=cfg.other_dim,
		map_dim=cfg.map_dim,
		tl_dim=cfg.tl_dim,
		num_scene_tokens_per_step=cfg.num_scene_tokens_per_step,
		num_scene_tokens=cfg.num_scene_tokens,
		num_history_steps=len(cfg.anchor_steps),
		hidden_dim=cfg.scene_hidden_dim,
		num_heads=cfg.num_heads,
		num_attn_layers=cfg.num_attn_layers,
		use_lora=cfg.use_lora,
		lora_r=cfg.lora_r,
		lora_alpha=cfg.lora_alpha,
		lora_dropout=cfg.lora_dropout,
		lora_target_modules=lora_target_modules,
	)

	if model.tokenizer.pad_token is None:
		model.tokenizer.pad_token = model.tokenizer.eos_token
	return model


def evaluate_temporal_answer_accuracy(
	model: TemporalGemmaVLA,
	loader,
	*,
	device: torch.device,
	dtype: torch.dtype,
	max_samples: int,
	max_prompt_length: int,
	max_answer_length: int,
	add_eos: bool = True,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
	if max_samples <= 0:
		return {"accuracy": 0.0, "num_samples": 0.0, "subgoal_mae": 0.0}, {}

	total = 0
	correct = 0
	subgoal_mae_sum = 0.0
	per_question_stats: dict[str, dict[str, int]] = {}

	was_training = model.training
	model.eval()
	with torch.inference_mode():
		for batch in loader:
			features = batch.features
			prompts = batch.prompts
			answers = batch.answers
			keys = batch.keys
			if not prompts or not answers:
				continue

			expanded_features, flat_prompts, flat_answers, flat_keys, qa_per_scenario = _expand_features_for_temporal_qa(
				features,
				batch.timestep_valid,
				prompts,
				answers,
				keys,
			)
			if not flat_prompts:
				continue

			remaining = max_samples - total
			if remaining <= 0:
				break
			if len(flat_prompts) > remaining:
				flat_prompts = flat_prompts[:remaining]
				flat_answers = flat_answers[:remaining]
				flat_keys = flat_keys[:remaining]
				seq_to_keep = max(1, (remaining + len(batch.timestep_valid[0]) - 1) // len(batch.timestep_valid[0]))
				expanded_features = {k: v[:seq_to_keep] for k, v in expanded_features.items()}

			prompt_ids, prompt_mask, _, _ = tokenize_text_batch(
				model.tokenizer,
				flat_prompts,
				flat_answers,
				add_eos=add_eos,
				device=device,
				max_prompt_length=max_prompt_length,
				max_answer_length=max_answer_length,
			)
			expanded_features = move_features_to_device(expanded_features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				pred_answers, pred_subgoals = model.generate_predictions(
					input_features=expanded_features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					max_new_tokens=max_answer_length,
				)

			for pred, target, q_key in zip(pred_answers, flat_answers, flat_keys):
				is_correct = pred.lower().strip() == target.lower().strip()
				if is_correct:
					correct += 1
				if q_key not in per_question_stats:
					per_question_stats[q_key] = {"correct": 0, "total": 0}
				per_question_stats[q_key]["total"] += 1
				if is_correct:
					per_question_stats[q_key]["correct"] += 1

			subgoal_xy = expanded_features.get("subgoal_xy")
			if subgoal_xy is not None:
				subgoal_xy_np = (100.0 * subgoal_xy.float().detach().cpu().numpy()).reshape(-1, subgoal_xy.shape[-1])
				pred_subgoal_np = np.asarray(pred_subgoals).reshape(-1, pred_subgoals.shape[-1])
				pair_count = min(subgoal_xy_np.shape[0], pred_subgoal_np.shape[0])
				if pair_count > 0:
					subgoal_mae_sum += float(
						np.sum(np.abs(subgoal_xy_np[:pair_count] - pred_subgoal_np[:pair_count])) / 2.0
					)

			total += len(flat_answers)
			if total >= max_samples:
				break

	if was_training:
		model.train()

	overall_accuracy = float(correct / total) if total > 0 else 0.0
	overall_metrics = {
		"accuracy": overall_accuracy,
		"num_samples": float(total),
		"subgoal_mae": float(subgoal_mae_sum / total) if total > 0 else 0.0,
	}

	per_question_metrics: dict[str, dict[str, float]] = {}
	for q_key, stats in per_question_stats.items():
		q_total = stats["total"]
		q_acc = float(stats["correct"] / q_total) if q_total > 0 else 0.0
		per_question_metrics[q_key] = {
			"accuracy": q_acc,
			"num_samples": float(q_total),
		}

	return overall_metrics, per_question_metrics


def run_training(cfg: TemporalVLAPretrainConfig) -> None:
	device = resolve_device()
	dtype = resolve_dtype(cfg.dtype)
	torch.backends.cuda.matmul.allow_tf32 = True

	run_name = cfg.wandb_run_name # + f"_{cfg.model_type}"
	if cfg.tag:
		run_name += f"_{cfg.tag}"
	run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

	output_dir = Path(cfg.output_dir) / run_name
	output_dir.mkdir(parents=True, exist_ok=True)
	save_training_config(cfg, output_dir)

	all_cache_paths = resolve_cache_paths(cfg.cache_dir, cfg.file_indices)
	train_cache_paths, val_cache_paths = split_cache_paths(all_cache_paths, cfg.validation_fraction)

	train_loader = build_dataloader(
		cfg.cache_dir,
		preprocess_cfg=cfg.preprocess_cfg,
		annotation_dir=cfg.annotation_dir,
		anchor_steps=cfg.anchor_steps,
		language_label="qa",
		backend="torch",
		include_inst_features=False,
		file_indices=None,
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
		cache_paths=train_cache_paths,
	)

	val_loader = (
		build_dataloader(
			cfg.cache_dir,
			preprocess_cfg=cfg.preprocess_cfg,
			annotation_dir=cfg.annotation_dir,
			anchor_steps=cfg.anchor_steps,
			language_label="qa",
			backend="torch",
			include_inst_features=False,
			file_indices=None,
			batch_size=cfg.batch_size,
			shuffle_seed=cfg.shuffle_seed,
			num_workers=cfg.num_workers,
			pin_memory=cfg.pin_memory,
			cache_paths=val_cache_paths,
		)
		if val_cache_paths
		else None
	)

	model = build_temporal_vla_model(cfg)
	if cfg.use_gradient_checkpointing and hasattr(model.llm, "gradient_checkpointing_enable"):
		model.llm.gradient_checkpointing_enable()
	if cfg.freeze_llm:
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
			config=dataclasses.asdict(cfg),
		)
		wandb_run.define_metric("train/step")
		wandb_run.define_metric("train/*", step_metric="train/step")
		wandb_run.define_metric("eval/*", step_metric="train/step")

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

	scheduler = None
	if cfg.warmup_steps > 0:
		warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor=0.001,
			total_iters=cfg.warmup_steps,
		)
		if cfg.max_steps is not None:
			decay_steps = cfg.max_steps - cfg.warmup_steps
			if decay_steps > 0:
				decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps)
				scheduler = torch.optim.lr_scheduler.SequentialLR(
					optimizer,
					[warmup_scheduler, decay_scheduler],
					milestones=[cfg.warmup_steps],
				)
			else:
				scheduler = warmup_scheduler
		else:
			scheduler = warmup_scheduler

	scaler = torch.cuda.amp.GradScaler() if cfg.dtype == "fp16" else None

	global_step = 0
	optimizer.zero_grad(set_to_none=True)
	for epoch in range(cfg.num_epochs):
		epoch_total = None
		if cfg.max_steps is not None:
			epoch_total = max(0, cfg.max_steps - global_step)
		pbar = tqdm(total=epoch_total, desc=f"epoch {epoch + 1}/{cfg.num_epochs}", unit="step")

		for batch in train_loader:
			features = batch.features
			prompts = batch.prompts
			answers = batch.answers
			if not prompts or not answers:
				continue

			expanded_features, flat_prompts, flat_answers, _, _ = _expand_features_for_temporal_qa(
				features,
				batch.timestep_valid,
				prompts,
				answers,
				batch.keys,
			)
			prompt_ids, prompt_mask, answer_ids, answer_mask = tokenize_text_batch(
				model.tokenizer,
				flat_prompts,
				flat_answers,
				add_eos=cfg.add_eos,
				device=device,
				max_prompt_length=cfg.max_prompt_length,
				max_answer_length=cfg.max_answer_length,
			)
			expanded_features = move_features_to_device(expanded_features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				outputs = model(
					expanded_features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					answer_ids=answer_ids,
					answer_mask=answer_mask,
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
				pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{current_lr:.3e}", batch=len(flat_prompts))
				if wandb_run is not None:
					wandb_run.log(
						{
							"train/step": global_step,
							"train/loss": train_loss,
							"train/subgoal_loss": float(outputs.get("subgoal_loss", torch.tensor(0.0))),
							"train/language_loss": float(outputs.get("language_loss", torch.tensor(0.0))),
							"train/lr": current_lr,
							"train/batch_size": len(flat_prompts),
						},
						step=global_step,
					)

			if cfg.save_every > 0 and global_step % cfg.save_every == 0:
				save_checkpoint(
					output_dir=output_dir, 
					tag=f"step_{(global_step//1000):03d}K", 
					model=model, 
					optimizer=optimizer, 
					scheduler=scheduler, 
					step=global_step, 
					cfg=cfg,
					save_trainable_only=True,
				)

			if (global_step + 1) % 1000 == 0 or global_step == 0:
				save_checkpoint(
					output_dir=output_dir,
					tag="latest",
					model=model,
					optimizer=optimizer,
					scheduler=scheduler,
					step=global_step,
					cfg=cfg,
				)
				val_metrics = {"accuracy": 0.0, "num_samples": 0.0, "subgoal_mae": 0.0}
				val_per_q_metrics: dict[str, dict[str, float]] = {}
				if val_loader is not None:
					val_metrics, val_per_q_metrics = evaluate_temporal_answer_accuracy(
						model,
						val_loader,
						device=device,
						dtype=dtype,
						max_samples=cfg.eval_num_samples,
						max_prompt_length=cfg.max_prompt_length,
						max_answer_length=cfg.max_answer_length,
						add_eos=cfg.add_eos,
					)

					train_metrics, train_per_q_metrics = evaluate_temporal_answer_accuracy(
						model,
						train_loader,
						device=device,
						dtype=dtype,
						max_samples=cfg.eval_num_samples,
						max_prompt_length=cfg.max_prompt_length,
						max_answer_length=cfg.max_answer_length,
						add_eos=cfg.add_eos,
					)

					tqdm.write(
						f"epoch={epoch} "
						f"train_accuracy={train_metrics['accuracy']:.4f} train_samples={int(train_metrics['num_samples'])} "
						f"val_accuracy={val_metrics['accuracy']:.4f} val_samples={int(val_metrics['num_samples'])}"
					)
					if wandb_run is not None:
						wandb_log_dict: dict[str, Any] = {
							"train/step": global_step,
							"overall_acc/train_accuracy": train_metrics["accuracy"],
							"overall_acc/val_accuracy": val_metrics["accuracy"],
							"overall_acc/train_subgoal_mae": train_metrics["subgoal_mae"],
							"overall_acc/val_subgoal_mae": val_metrics["subgoal_mae"],
						}
						for q_key, q_metrics in train_per_q_metrics.items():
							wandb_log_dict[f"train_acc/train_qa_{q_key}_accuracy"] = q_metrics["accuracy"]
						for q_key, q_metrics in val_per_q_metrics.items():
							wandb_log_dict[f"val_acc/val_qa_{q_key}_accuracy"] = q_metrics["accuracy"]
						wandb_run.log(wandb_log_dict, step=global_step)

			global_step += 1
			pbar.update(1)
			if cfg.max_steps is not None and global_step >= cfg.max_steps:
				break

		pbar.close()
		if cfg.max_steps is not None and global_step >= cfg.max_steps:
			break

	save_pretrained = getattr(model.llm, "save_pretrained", None)
	if callable(save_pretrained):
		model.llm.save_pretrained(str(output_dir / "llm"))
	model.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
	save_checkpoint(
		output_dir=output_dir, 
		tag=f"step_{(global_step//1000):03d}K", 
		model=model, 
		optimizer=optimizer, 
		scheduler=scheduler, 
		step=global_step, 
		cfg=cfg,
		save_trainable_only=True,
	)
	if wandb_run is not None:
		wandb_run.finish()


def main() -> None:
	config = parse_args()
	run_training(config)


if __name__ == "__main__":
	main()
