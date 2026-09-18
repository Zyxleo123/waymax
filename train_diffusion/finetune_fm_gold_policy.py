from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from data.fm_gold_loader import build_dataloader
from data.types import PreprocessConfig
from model.diffusion.fm_policy import (
	FAILURE_REASON_COLLISION,
	FAILURE_REASON_GOAL_NOT_REACHED,
	FAILURE_REASON_OFFROAD,
	FlowMatchingPolicy,
)
from train_diffusion.configs.fm_gold_finetuning_config import (
	FlowMatchingGoldFinetuningConfig,
	config_to_dict,
	parse_args,
)
from train_diffusion.utils.checkpoints import restore_checkpoint, save_checkpoint
from train_diffusion.utils.utils import (
	build_data_parallel_mesh_if_needed,
	clip_grads,
	coerce_overlapping_tree,
	coerce_tree_like,
	configure_jax_compilation_cache,
	ema_update,
	init_wandb_run,
	log_overlapping_restore_report,
	replicate_to_mesh,
	update_tqdm,
	wandb_log_train_step,
)


_TRAINABLE_NAME_TOKENS = (
	"collision_object_encoder",
	"offroad_segment_encoder",
	"goal_encoder",
	"down_film_blocks",
	"mid_film_blocks",
	"up_film_blocks",
)


def _jsonable(value: Any) -> Any:
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	if isinstance(value, dict):
		return {key: _jsonable(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(item) for item in value]
	return str(value)


def _save_training_config(cfg: FlowMatchingGoldFinetuningConfig, output_dir: Path) -> Path:
	config_path = output_dir / "training_config.json"
	config_path.parent.mkdir(parents=True, exist_ok=True)
	with config_path.open("w", encoding="utf-8") as handle:
		json.dump(_jsonable(asdict(cfg)), handle, indent=2, sort_keys=True)
		handle.write("\n")
	return config_path


def _pure_dict_to_state(tree):
	if isinstance(tree, nnx.State):
		return tree
	if hasattr(nnx.State, "from_pure_dict"):
		return nnx.State.from_pure_dict(tree)
	return nnx.State(tree)


def _copy_tree(tree):
	return jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, "copy") else x, tree)


def _path_to_string(path) -> str:
	parts: list[str] = []
	for item in path:
		if hasattr(item, "key"):
			parts.append(str(item.key))
		elif hasattr(item, "name"):
			parts.append(str(item.name))
		elif hasattr(item, "idx"):
			parts.append(str(item.idx))
		else:
			parts.append(str(item))
	return "/".join(parts)


def _is_trainable_path(path) -> bool:
	path_str = _path_to_string(path)
	return any(token in path_str for token in _TRAINABLE_NAME_TOKENS)


def _build_freeze_optimizer(cfg: FlowMatchingGoldFinetuningConfig, params_tree):
	trainable_mask = jax.tree_util.tree_map_with_path(
		lambda path, _leaf: _is_trainable_path(path),
		params_tree,
	)
	param_labels = jax.tree_util.tree_map(
		lambda is_trainable: "train" if is_trainable else "freeze",
		trainable_mask,
	)
	trainable_leaves = sum(int(bool(leaf)) for leaf in jax.tree_util.tree_leaves(trainable_mask))
	total_leaves = len(jax.tree_util.tree_leaves(trainable_mask))
	trainable_modules: set[str] = set()
	frozen_modules: set[str] = set()

	def _collect_modules(path, is_trainable):
		module = _path_to_string(path).split("/")[0]
		if is_trainable:
			trainable_modules.add(module)
		else:
			frozen_modules.add(module)
		return is_trainable

	jax.tree_util.tree_map_with_path(_collect_modules, trainable_mask)
	print(f"[freeze] trainable tensors: {trainable_leaves}/{total_leaves}")
	print("[freeze] trainable modules: " + (", ".join(sorted(trainable_modules)) or "none"))
	print("[freeze] frozen modules: " + ", ".join(sorted(frozen_modules)))
	if trainable_leaves == 0:
		raise RuntimeError(
			"Freeze mask selected no trainable tensors. Expected gold-id encoders and FiLM blocks."
		)

	total_steps = max(1, int(cfg.num_epochs) * int(cfg.steps_per_epoch))
	lr_schedule = optax.warmup_cosine_decay_schedule(
		init_value=0.0,
		peak_value=float(cfg.lr),
		warmup_steps=max(1, int(cfg.warmup_steps)),
		decay_steps=max(int(cfg.warmup_steps) + 1, total_steps),
		end_value=float(cfg.lr) * 0.1,
	)
	tx = optax.multi_transform(
		{
			"train": optax.adamw(learning_rate=lr_schedule, weight_decay=float(cfg.weight_decay)),
			"freeze": optax.set_to_zero(),
		},
		param_labels,
	)
	return tx, lr_schedule, trainable_mask


def _build_preprocess_cfg(cfg: FlowMatchingGoldFinetuningConfig) -> PreprocessConfig:
	return PreprocessConfig(
		model_dt=cfg.model_dt,
		world_dt_fallback=cfg.world_dt_fallback,
		max_range=cfg.max_range,
		ego_range=cfg.ego_range,
		max_velocity=cfg.max_velocity,
		max_width=cfg.max_width,
		max_tl_points=cfg.max_tl_points,
		num_map_type_classes=cfg.num_map_type_classes,
		predict_horizon=cfg.predict_horizon,
		map_unknown_type_index=20,
		max_segments=cfg.max_num_segments,
		max_points_per_segment=cfg.max_points_per_segment,
		num_object_types=8,
		inst_dim=cfg.inst_dim,
	)


def _build_model(cfg: FlowMatchingGoldFinetuningConfig) -> FlowMatchingPolicy:
	return FlowMatchingPolicy(
		rngs=nnx.Rngs(cfg.seed),
		target_dim=cfg.target_dim,
		hidden_dim=cfg.hidden_dim,
		cond_dim=cfg.cond_dim,
		other_dim=cfg.other_dim,
		map_attr_dim=cfg.map_attr_dim,
		tl_attr_dim=cfg.tl_attr_dim,
		inst_attr_dim=cfg.inst_dim,
		predict_horizon=cfg.predict_horizon,
		default_num_steps=cfg.default_num_steps,
	)


def _batch_to_jax(batch_features: dict[str, Any]) -> dict[str, jax.Array]:
	return jax.tree_util.tree_map(lambda value: jnp.asarray(value), batch_features)


def _load_pretrained_overlapping(checkpoint_path: str, template_params, template_nonparam, *, use_ema: bool):
	restored = restore_checkpoint(checkpoint_path)
	if use_ema and "ema_params" in restored:
		raw_params = restored["ema_params"]
		source = "EMA parameters"
	else:
		raw_params = restored["params_state"]
		source = "regular parameters"
	params_state, params_report = coerce_overlapping_tree(template_params, raw_params, name="params_state")
	nonparam_state, nonparam_report = coerce_overlapping_tree(
		template_nonparam,
		restored.get("nonparam_state", template_nonparam),
		name="nonparam_state",
	)
	print(f"[checkpoint] Loaded pretrained {source} from: {checkpoint_path}")
	log_overlapping_restore_report(params_report)
	log_overlapping_restore_report(nonparam_report)
	copied_roots = {path.split("/")[0] for path in params_report["copied"] if path}
	if "scene_tokenizer" not in copied_roots and "flow" not in copied_roots:
		raise RuntimeError(
			"Pretrained checkpoint did not overlap with scene_tokenizer/flow. "
			"Check that --pretrained_ckpt is a FlowMatchingPolicy pretrain run."
		)
	expected_missing = (
		"collision_object_encoder",
		"offroad_segment_encoder",
		"goal_encoder",
	)
	unexpected_missing = [
		path
		for path in params_report["missing"]
		if not any(token in path for token in expected_missing)
	]
	if unexpected_missing:
		shown = unexpected_missing[:12]
		print(
			"[checkpoint] unexpected missing pretrained tensors (kept random init): "
			+ ", ".join(shown)
			+ ("..." if len(unexpected_missing) > 12 else "")
		)
	return params_state, nonparam_state


def _make_train_step(graphdef, nonparam_state, tx, grad_clip_norm: float, ema_decay: float, data_parallel: bool):
	nonparam_state = _pure_dict_to_state(nonparam_state)

	def merge_model(params_tree):
		return nnx.merge(graphdef, _pure_dict_to_state(params_tree), nonparam_state)

	def train_step_impl(params_state, opt_state, ema_params, rng_key, global_step, batch_features, batch_aux):
		rng_key, key_loss = jax.random.split(rng_key)
		batch_features = _batch_to_jax(batch_features)
		batch_aux = _batch_to_jax(batch_aux)

		def loss_fn(p):
			model = merge_model(p)
			features = model._as_features(batch_features)
			cond, failure_cond, failure_mask = model.compute_condition(features)
			target = features.ego_trajectory[:, : model.predict_horizon, :]
			loss = model.loss_from_condition(
				target,
				cond,
				failure_cond,
				failure_mask,
				rng=key_loss,
				data_parallel=data_parallel,
			)
			reason = jnp.asarray(batch_features["failure_reason"]).reshape(-1)
			metrics = {
				"loss": loss,
				"failure_mask_rate": jnp.mean(failure_mask.astype(jnp.float32)),
				"collision_rate": jnp.mean((reason == FAILURE_REASON_COLLISION).astype(jnp.float32)),
				"offroad_rate": jnp.mean((reason == FAILURE_REASON_OFFROAD).astype(jnp.float32)),
				"goal_not_reached_rate": jnp.mean((reason == FAILURE_REASON_GOAL_NOT_REACHED).astype(jnp.float32)),
			}
			return loss, metrics

		(loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_state)
		clipped_grads, grad_norm = clip_grads(grads, grad_clip_norm)
		updates, opt_state_next = tx.update(clipped_grads, opt_state, params_state)
		params_state_next = optax.apply_updates(params_state, updates)
		ema_params_next = ema_update(ema_params, params_state_next, ema_decay)
		metrics = {**metrics, "grad_norm": grad_norm, "loss": loss}
		return params_state_next, opt_state_next, ema_params_next, rng_key, global_step + jnp.int32(1), metrics

	return jax.jit(train_step_impl)


def run_training(cfg: FlowMatchingGoldFinetuningConfig) -> None:
	configure_jax_compilation_cache(cfg.jax_compilation_cache_dir)
	preprocess_cfg = _build_preprocess_cfg(cfg)
	model = _build_model(cfg)

	if cfg.wandb_project:
		run, run_dir_str = init_wandb_run(cfg, config_to_dict(cfg))
		run_dir = Path(run_dir_str).resolve()
	else:
		run = None
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		run_dir = (Path(cfg.output_dir) / f"{cfg.wandb_name}_{stamp}").resolve()
		run_dir.mkdir(parents=True, exist_ok=True)
	_save_training_config(cfg, run_dir)

	graphdef, template_params_state, template_nonparam_state = nnx.split(model, nnx.Param, ...)
	params_state = template_params_state
	nonparam_state = template_nonparam_state
	rng_key = jax.random.PRNGKey(cfg.seed)
	global_step = jnp.array(0, dtype=jnp.int32)
	global_step_host = 0
	start_epoch = 1
	opt_state = None
	ema_params = None
	tx = None
	lr_schedule = None

	if cfg.resume_path:
		tx, lr_schedule, _ = _build_freeze_optimizer(cfg, params_state)
		try:
			(
				params_state,
				nonparam_state,
				opt_state,
				ema_params,
				rng_key,
				start_epoch,
				global_step,
				global_step_host,
			) = restore_checkpoint(
				cfg.resume_path,
				params_state=params_state,
				nonparam_state=nonparam_state,
				tx=tx,
				coerce_tree_like_fn=coerce_tree_like,
			)
			print(f"[checkpoint] Resumed gold-id finetuning from: {cfg.resume_path}")
		except Exception as exc:
			print(
				"[checkpoint] Exact resume failed; falling back to overlapping restore "
				f"and a fresh optimizer. Details: {exc}"
			)
			params_state, nonparam_state = _load_pretrained_overlapping(
				cfg.resume_path,
				template_params_state,
				template_nonparam_state,
				use_ema=cfg.use_ema,
			)
			tx, lr_schedule, _ = _build_freeze_optimizer(cfg, params_state)
			opt_state = tx.init(params_state)
			ema_params = _copy_tree(params_state)
			start_epoch = 1
			global_step = jnp.array(0, dtype=jnp.int32)
			global_step_host = 0
	elif cfg.pretrained_ckpt is not None:
		params_state, nonparam_state = _load_pretrained_overlapping(
			cfg.pretrained_ckpt,
			template_params_state,
			template_nonparam_state,
			use_ema=cfg.use_ema,
		)
		tx, lr_schedule, _ = _build_freeze_optimizer(cfg, params_state)
		opt_state = tx.init(params_state)
		ema_params = _copy_tree(params_state)
		print("[checkpoint] Frozen pretrained weights; training gold-id encoders and FiLM only.")
	else:
		raise RuntimeError("Provide --pretrained_ckpt or --resume_path.")

	if tx is None or lr_schedule is None or opt_state is None or ema_params is None:
		raise RuntimeError("Optimizer state was not initialized.")

	data_mesh = build_data_parallel_mesh_if_needed(cfg)
	data_parallel_enabled = data_mesh is not None
	if data_parallel_enabled and data_mesh is not None:
		with data_mesh:
			params_state = replicate_to_mesh(params_state, data_mesh)
			nonparam_state = replicate_to_mesh(nonparam_state, data_mesh)
			opt_state = replicate_to_mesh(opt_state, data_mesh)
			ema_params = replicate_to_mesh(ema_params, data_mesh)
			rng_key = replicate_to_mesh(rng_key, data_mesh)
			global_step = replicate_to_mesh(global_step, data_mesh)

	train_loader = build_dataloader(
		cache_dir=cfg.cache_dir,
		preprocess_cfg=preprocess_cfg,
		cache_file_indices=cfg.cache_file_indices,
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		anchor_steps=cfg.anchor_steps,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
		backend="jax",
		drop_last=True,
	)
	train_iter = iter(train_loader)
	warmup_batch = next(train_iter)
	_ = _batch_to_jax(warmup_batch.features)

	train_step_fn = _make_train_step(
		graphdef,
		nonparam_state,
		tx,
		cfg.grad_clip_norm,
		cfg.ema_decay,
		data_parallel_enabled,
	)
	ema_loss = None
	last_epoch = max(0, start_epoch - 1)

	for epoch in range(start_epoch, cfg.num_epochs + 1):
		last_epoch = epoch
		pbar = tqdm(total=cfg.steps_per_epoch, desc=f"Epoch {epoch:04d}")
		for _ in range(cfg.steps_per_epoch):
			batch = next(train_iter)
			params_state, opt_state, ema_params, rng_key, global_step, metrics = train_step_fn(
				params_state,
				opt_state,
				ema_params,
				rng_key,
				global_step,
				batch.features,
				batch.aux,
			)
			global_step_host += 1
			pbar.update(1)
			ema_loss = update_tqdm(pbar, global_step_host, max(1, cfg.log_every), metrics, ema_loss)
			if cfg.wandb_project and run is not None and global_step_host % max(1, cfg.log_every) == 0:
				lr = float(lr_schedule(global_step_host - 1))
				ema_loss, _ = wandb_log_train_step(global_step_host, epoch, metrics, ema_loss, lr)
			if cfg.max_steps is not None and global_step_host >= cfg.max_steps:
				break
		pbar.close()
		if cfg.save_every > 0 and epoch % cfg.save_every == 0:
			save_checkpoint(
				str(run_dir),
				config_to_dict(cfg),
				epoch=epoch,
				global_step=global_step,
				params_state=params_state,
				nonparam_state=nonparam_state,
				opt_state=opt_state,
				ema_params=ema_params,
				rng_key=rng_key,
				save_every=cfg.save_every,
			)
		if cfg.max_steps is not None and global_step_host >= cfg.max_steps:
			break

	save_checkpoint(
		str(run_dir),
		config_to_dict(cfg),
		epoch=last_epoch,
		global_step=global_step,
		params_state=params_state,
		nonparam_state=nonparam_state,
		opt_state=opt_state,
		ema_params=ema_params,
		rng_key=rng_key,
		save_every=cfg.save_every,
	)
	if run is not None:
		run.finish()


def main() -> None:
	run_training(parse_args())


if __name__ == "__main__":
	main()
