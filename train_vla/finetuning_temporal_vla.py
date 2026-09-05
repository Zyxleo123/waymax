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
from train_vla.configs.temporal_vla_finetuning_config import TemporalVLAFinetuningConfig, parse_args
from train_vla.utils.utils import (
	move_features_to_device,
	resolve_device,
	resolve_dtype,
	save_checkpoint,
	save_training_config,
	tokenize_text_batch,
)


keywords = ["stop", "straight", "left", "right", "turn", "chang", "accel", "slow"]


def compare_keyword(preds: list[str], labels: list[str]) -> list[float]:
	accuracies = []
	for pred, label in zip(preds, labels):
		pred_lower = pred.lower()
		label_lower = label.lower()
		target_keywords = [keyword for keyword in keywords if keyword in label_lower]
		included_keywords = sum(1 for keyword in target_keywords if keyword in pred_lower)
		if len(target_keywords) == 0:
			accuracies.append(1.0 if included_keywords == 0 else 0.0)
		else:
			accuracies.append(included_keywords / len(target_keywords))
	return accuracies

def compare_function(preds: list[str], labels: list[str]) -> list[float]:
	accuracies = []
	for pred, label in zip(preds, labels):
		pred_lower = pred.lower()
		label_lower = label.lower()
		target_functions = [func for func in label_lower.split(",") if func.strip()]
		included_functions = sum(1 for func in target_functions if func in pred_lower)
		if len(target_functions) == 0:
			accuracies.append(1.0 if included_functions == 0 else 0.0)
		else:
			accuracies.append(included_functions / len(target_functions))
	return accuracies

def compare_match(preds: list[str], labels: list[str]) -> list[float]:
	return [1.0 if pred.strip().lower() == label.strip().lower() else 0.0 for pred, label in zip(preds, labels)]


def compare_subgoal(preds: list[str], labels: list[str]) -> tuple[list[float], list[float]]:
	format_accuracies = []
	l2_distances = []
	for pred, label in zip(preds, labels):
		try:
			pred_x, pred_y = pred.split(",")
			pred_x = float(pred_x.strip())
			pred_y = float(pred_y.strip())
		except ValueError:
			format_accuracies.append(0.0)
			continue
		label_x, label_y = label.split(",")
		label_x = float(label_x.strip())
		label_y = float(label_y.strip())
		format_accuracies.append(1.0)
		l2_distance = ((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2) ** 0.5
		l2_distances.append(l2_distance)
	return l2_distances, format_accuracies


def _flatten_temporal_instruction_batch(batch) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
	if batch.prompts is None or batch.answers is None:
		raise ValueError("batch.prompts and batch.answers are required for temporal finetuning")
	features = batch.features
	prompts = [item for scenario_prompts in batch.prompts for item in scenario_prompts]
	answers = [item for scenario_answers in batch.answers for item in scenario_answers]
	return features, prompts, answers


def _resolve_pretrained_checkpoint_path(pretrained_model_path: str | Path) -> Path:
	path = Path(pretrained_model_path)
	if path.is_file():
		return path
	if path.is_dir():
		if path.name == "checkpoints":
			checkpoint_files = sorted(path.glob("step_*.pt"))
			if not checkpoint_files:
				raise FileNotFoundError(f"No checkpoints found in {path}")
			return checkpoint_files[-1]
		checkpoint_dir = path / "checkpoints"
		if checkpoint_dir.exists():
			checkpoint_files = sorted(checkpoint_dir.glob("step_*.pt"))
			if not checkpoint_files:
				raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
			return checkpoint_files[-1]
	raise FileNotFoundError(f"Could not resolve pretrained checkpoint from: {pretrained_model_path}")


def _resolve_resume_run_dir(resume_path: str | Path) -> Path:
	path = Path(resume_path)
	if path.is_file():
		if path.parent.name == "checkpoints":
			return path.parent.parent
		return path.parent
	if path.is_dir():
		if (path / "training_config.json").exists():
			return path
		if path.name == "checkpoints":
			return path.parent
		if (path / "checkpoints").exists():
			return path
	raise FileNotFoundError(f"Could not resolve resume run directory from: {resume_path}")


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
	for state in optimizer.state.values():
		for key, value in state.items():
			if torch.is_tensor(value):
				state[key] = value.to(device=device)


def _build_temporal_vla_model(cfg: TemporalVLAFinetuningConfig) -> TemporalGemmaVLA:
	lora_target_modules: tuple[str, ...] | None = None
	if cfg.lora_target_modules is not None and cfg.lora_target_modules.strip():
		lora_target_modules = tuple(part.strip() for part in cfg.lora_target_modules.split(",") if part.strip())

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
		predict_target_lane=cfg.predict_target_lane,
	)

	if cfg.pretrained_model_path:
		checkpoint_path = _resolve_pretrained_checkpoint_path(cfg.pretrained_model_path)
		checkpoint = torch.load(checkpoint_path, map_location="cpu")
		state_dict = checkpoint.get("model_state_dict", checkpoint)
		if any(key.startswith("module.") for key in state_dict):
			state_dict = {
				key[len("module."):] if key.startswith("module.") else key: value
				for key, value in state_dict.items()
			}
		missing, unexpected = model.load_state_dict(state_dict, strict=False)
		print(f"pretrained checkpoint: {checkpoint_path}")
		print(f"missing keys: {len(missing)}")
		print(f"unexpected keys: {len(unexpected)}")
		if missing:
			print("  missing:", missing[:10])
		if unexpected:
			print("  unexpected:", unexpected[:10])

	if model.tokenizer.pad_token is None:
		model.tokenizer.pad_token = model.tokenizer.eos_token
	return model


def evaluate_accuracy(
	model: TemporalGemmaVLA,
	loader,
	*,
	device: torch.device,
	dtype: torch.dtype,
	max_samples: int,
	max_prompt_length: int,
	max_answer_length: int,
	add_eos: bool,
	compute_subgoal_loss: bool,
) -> dict[str, float]:
	keyword_accuracies, match_accuracies, subgoal_maes = [], [], []
	target_lane_loss_sum = 0.0
	target_lane_accuracy_sum = 0.0
	target_lane_batches = 0
	was_training = model.training
	model.eval()
	with torch.inference_mode():
		for batch in loader:
			features, prompts, answers = _flatten_temporal_instruction_batch(batch)
			if not prompts:
				continue

			steps_per_scenario = len(batch.prompts[0]) if batch.prompts else 0
			remaining = max_samples - len(keyword_accuracies)
			if remaining <= 0:
				break
			if len(prompts) > remaining and steps_per_scenario > 0:
				keep_scenarios = max(1, (remaining + steps_per_scenario - 1) // steps_per_scenario)
				batch.prompts = batch.prompts[:keep_scenarios]
				batch.answers = batch.answers[:keep_scenarios]
				batch.target_lane_ids = batch.target_lane_ids[:keep_scenarios] if batch.target_lane_ids is not None else None
				features = {key: value[:keep_scenarios] for key, value in features.items()}
				features, prompts, answers = _flatten_temporal_instruction_batch(batch)

			prompt_ids, prompt_mask, answer_ids, answer_mask = tokenize_text_batch(
				model.tokenizer,
				prompts,
				answers,
				add_eos=add_eos,
				device=device,
				max_prompt_length=max_prompt_length,
				max_answer_length=max_answer_length,
			)
			features = move_features_to_device(features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				predictions = model.generate_predictions(
					input_features=features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					max_new_tokens=max_answer_length,
				)
				pred_answers = predictions["answer"]
				pred_subgoals = predictions.get("subgoal_preds", None)
				outputs = model(
					features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					answer_ids=answer_ids,
					answer_mask=answer_mask,
					target_lane_ids=batch.target_lane_ids,
					compute_subgoal_loss=compute_subgoal_loss,
				)
			if loader.dataset.cfg.language_label == "function":
				keyword_accuracies.extend(compare_function(pred_answers, answers))
			else:
				keyword_accuracies.extend(compare_keyword(pred_answers, answers))
			match_accuracies.extend(compare_match(pred_answers, answers))

			if pred_subgoals is not None and "subgoal_xy" in features:
				subgoal_xy = 100.0 * features["subgoal_xy"].cpu().float().numpy()
				pred_subgoal_np = np.asarray(pred_subgoals)
				if pred_subgoal_np.ndim == 2 and subgoal_xy.ndim == 3:
					subgoal_xy = subgoal_xy.reshape(-1, subgoal_xy.shape[-1])
				if pred_subgoal_np.shape[0] == subgoal_xy.shape[0]:
					subgoal_mae_ = np.abs(pred_subgoal_np - subgoal_xy).mean(-1).tolist()
					subgoal_maes.extend(subgoal_mae_)
				else:
					subgoal_maes.extend([0.0] * len(pred_answers))
			else:
				subgoal_maes.extend([0.0] * len(pred_answers))

			target_lane_loss_sum += float(outputs.get("target_lane_loss", torch.tensor(0.0)).detach().cpu())
			target_lane_accuracy_sum += float(outputs.get("target_lane_accuracy", torch.tensor(0.0)).detach().cpu())
			target_lane_batches += 1

			if len(keyword_accuracies) >= max_samples:
				break

	if was_training:
		model.train()

	keyword_accuracy = sum(keyword_accuracies) / len(keyword_accuracies) if keyword_accuracies else 0.0
	match_accuracy = sum(match_accuracies) / len(match_accuracies) if match_accuracies else 0.0
	subgoal_mae = sum(subgoal_maes) / len(subgoal_maes) if subgoal_maes else 0.0
	return {
		"keyword_accuracy": keyword_accuracy,
		"match_accuracy": match_accuracy,
		"subgoal_mae": subgoal_mae,
		"target_lane_loss": float(target_lane_loss_sum / target_lane_batches) if target_lane_batches > 0 else 0.0,
		"target_lane_accuracy": float(target_lane_accuracy_sum / target_lane_batches) if target_lane_batches > 0 else 0.0,
	}


def run_training(cfg: TemporalVLAFinetuningConfig) -> None:
	device = resolve_device()
	dtype = resolve_dtype(cfg.dtype)
	torch.backends.cuda.matmul.allow_tf32 = True

	resume_checkpoint_path = _resolve_pretrained_checkpoint_path(cfg.resume_path) if cfg.resume_path else None
	if resume_checkpoint_path is not None:
		output_dir = _resolve_resume_run_dir(cfg.resume_path)
		run_name = output_dir.name
	else:
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
		language_label=cfg.language_label,
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
			language_label=cfg.language_label,
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

	model = _build_temporal_vla_model(cfg)
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

	trainable_params = [param for param in model.parameters() if param.requires_grad]
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
				[warmup_scheduler, decay_scheduler] if decay_steps > 0 else [warmup_scheduler],
				milestones=[cfg.warmup_steps] if decay_steps > 0 else [],
			)
		else:
			scheduler = warmup_scheduler

	scaler = torch.cuda.amp.GradScaler() if cfg.dtype == "fp16" else None

	global_step = 0
	if resume_checkpoint_path is not None:
		checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
		state_dict = checkpoint.get("model_state_dict", checkpoint)
		if any(key.startswith("module.") for key in state_dict):
			state_dict = {
				key[len("module."):] if key.startswith("module.") else key: value
				for key, value in state_dict.items()
			}
		missing, unexpected = model.load_state_dict(state_dict, strict=False)
		print(f"resuming from checkpoint: {resume_checkpoint_path}")
		print(f"missing keys: {len(missing)}")
		print(f"unexpected keys: {len(unexpected)}")
		if missing:
			print("  missing:", missing[:10])
		if unexpected:
			print("  unexpected:", unexpected[:10])
		optimizer_state_dict = checkpoint.get("optimizer_state_dict")
		if optimizer_state_dict is not None:
			optimizer.load_state_dict(optimizer_state_dict)
			_move_optimizer_state_to_device(optimizer, device)
		scheduler_state_dict = checkpoint.get("scheduler_state_dict")
		if scheduler is not None and scheduler_state_dict is not None:
			scheduler.load_state_dict(scheduler_state_dict)
		global_step = int(checkpoint.get("step", -1)) + 1
		if global_step < 0:
			global_step = 0

	optimizer.zero_grad(set_to_none=True)
	for epoch in range(cfg.num_epochs):
		epoch_total = None
		if cfg.max_steps is not None:
			epoch_total = max(0, cfg.max_steps - global_step)
		pbar = tqdm(total=epoch_total, desc=f"epoch {epoch + 1}/{cfg.num_epochs}", unit="step")

		for batch in train_loader:
			features, prompts, answers = _flatten_temporal_instruction_batch(batch)
			if not prompts:
				continue

			prompt_ids, prompt_mask, answer_ids, answer_mask = tokenize_text_batch(
				model.tokenizer,
				prompts,
				answers,
				add_eos=cfg.add_eos,
				device=device,
				max_prompt_length=cfg.max_prompt_length,
				max_answer_length=cfg.max_answer_length,
			)
			features = move_features_to_device(features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				outputs = model(
					features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					answer_ids=answer_ids,
					answer_mask=answer_mask,
					target_lane_ids=batch.target_lane_ids,
					compute_subgoal_loss=cfg.predict_subgoal,
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
				train_target_lane_loss = float(outputs.get("target_lane_loss", torch.tensor(0.0)))
				train_target_lane_accuracy = float(outputs.get("target_lane_accuracy", torch.tensor(0.0)))
				pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{current_lr:.3e}", batch=len(prompts))
				if wandb_run is not None:
					wandb_run.log(
						{
							"train/step": global_step,
							"train/loss": train_loss,
							"train/subgoal_loss": outputs.get("subgoal_loss", 0.0),
							"train/language_loss": outputs.get("language_loss", 0.0),
							"train/target_lane_loss": train_target_lane_loss,
							"train/target_lane_accuracy": train_target_lane_accuracy,
							"train/lr": current_lr,
							"train/batch_size": len(prompts),
						},
						step=global_step,
					)

			if cfg.save_every > 0 and global_step > 0 and global_step % cfg.save_every == 0:
				save_checkpoint(
					output_dir=output_dir, 
					tag=f"step_{(global_step//1000):03d}K", 
					model=model, 
					optimizer=optimizer, 
					scheduler=scheduler, 
					step=global_step, 
					cfg=cfg,
					save_trainable_only=True
				)

			if (global_step + 1) % 1000 == 0 or global_step == 0:
				save_checkpoint(
					output_dir=output_dir, 
					tag=f"latest", 
					model=model, 
					optimizer=optimizer, 
					scheduler=scheduler, 
					step=global_step, 
					cfg=cfg,
					save_trainable_only=True
				)
				val_metrics = {"keyword_accuracy": 0.0, "match_accuracy": 0.0, "subgoal_mae": 0.0, "target_lane_loss": 0.0, "target_lane_accuracy": 0.0}
				if val_loader is not None:
					val_metrics = evaluate_accuracy(
						model,
						val_loader,
						device=device,
						dtype=dtype,
						max_samples=cfg.eval_num_samples,
						max_prompt_length=cfg.max_prompt_length,
						max_answer_length=cfg.max_answer_length,
						add_eos=cfg.add_eos,
						compute_subgoal_loss=cfg.predict_subgoal,
					)
				train_metrics = evaluate_accuracy(
					model,
					train_loader,
					device=device,
					dtype=dtype,
					max_samples=cfg.eval_num_samples,
					max_prompt_length=cfg.max_prompt_length,
					max_answer_length=cfg.max_answer_length,
					add_eos=cfg.add_eos,
					compute_subgoal_loss=cfg.predict_subgoal,
				)
				if wandb_run is not None:
					wandb_log_dict = {
						"val_accuracy/match_accuracy": val_metrics["match_accuracy"],
						"val_accuracy/keyword_accuracy": val_metrics["keyword_accuracy"],
						"val_accuracy/subgoal_mae": val_metrics["subgoal_mae"],
						"val_accuracy/target_lane_loss": val_metrics["target_lane_loss"],
						"val_accuracy/target_lane_accuracy": val_metrics["target_lane_accuracy"],
						"train_accuracy/match_accuracy": train_metrics["match_accuracy"],
						"train_accuracy/keyword_accuracy": train_metrics["keyword_accuracy"],
						"train_accuracy/subgoal_mae": train_metrics["subgoal_mae"],
						"train_accuracy/target_lane_loss": train_metrics["target_lane_loss"],
						"train_accuracy/target_lane_accuracy": train_metrics["target_lane_accuracy"],
					}
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
		save_trainable_only=True
    )
	if wandb_run is not None:
		wandb_run.finish()


def main() -> None:
	config = parse_args()
	run_training(config)


if __name__ == "__main__":
	main()