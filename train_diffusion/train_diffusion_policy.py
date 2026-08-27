from __future__ import annotations
from collections.abc import Mapping
import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from data.cache_loader import build_dataloader
from model.diffusion.diffusion_policy import DiffusionPolicy
try:
	from .configs.diffusion_train_config import DiffusionTrainConfig, config_to_dict, parse_args
except ImportError:
	from train_diffusion.configs.diffusion_train_config import DiffusionTrainConfig, config_to_dict, parse_args
from train_diffusion.utils.checkpoints import restore_checkpoint, save_checkpoint
from train_diffusion.utils.infer import _build_model_from_metadata, _build_preprocess_cfg, _load_checkpoint_metadata
from train_diffusion.utils.utils import (
	clip_grads,
	coerce_tree_like,
	configure_jax_compilation_cache,
	ema_update,
	iter_with_prefetch,
	init_wandb_run,
	wandb_log_train_step,
	log_jsonl,
	update_tqdm,
)


def _jsonable(value: Any) -> Any:
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	if isinstance(value, dict):
		return {k: _jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(v) for v in value]
	return str(value)


def _scalarize_for_logging(metrics: dict[str, Any]) -> dict[str, float]:
	logged: dict[str, float] = {}
	for key, value in metrics.items():
		arr = jnp.asarray(jax.device_get(value))
		logged[key] = float(arr.mean()) if arr.ndim > 0 else float(arr)
	return logged

def _state_to_pure_dict(state):
    """Convert flax.nnx.State to a pure dict. Leave dicts unchanged."""
    if isinstance(state, dict):
        return state

    if hasattr(state, "to_pure_dict"):
        return state.to_pure_dict()

    # flax.nnx.State
    return nnx.to_pure_dict(state)


def _pure_dict_to_state(tree):
    """Convert pure dict to flax.nnx.State. Leave State unchanged."""
    if isinstance(tree, nnx.State):
        return tree

    if hasattr(nnx.State, "from_pure_dict"):
        return nnx.State.from_pure_dict(tree)

    return nnx.State(tree)


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
	return ("instruction_encoder" in path_str) or ("film_blocks" in path_str) or ("subgoal_encoder" in path_str)


def _zero_film_block_params(params_tree):
    def _maybe_zero(path, leaf):
        return jnp.zeros_like(leaf) if "film_blocks" in _path_to_string(path) else leaf

    return jax.tree_util.tree_map_with_path(_maybe_zero, params_tree)


def _load_pretrained_state(
	checkpoint_path: str,
	model: DiffusionPolicy,
	*,
	use_ema: bool,
) -> tuple[Any, Any]:
	metadata = _load_checkpoint_metadata(checkpoint_path, None)
	graphdef, template_params_state, template_nonparam_state = nnx.split(model, nnx.Param, ...)
	restored = restore_checkpoint(checkpoint_path)

	if "params_state" not in restored:
		raise KeyError("Checkpoint is missing 'params_state'.")
	if use_ema and "ema_params" in restored:
		raw_params_state = restored["ema_params"]
	else:
		raw_params_state = restored["params_state"]

	params_state = coerce_tree_like(template_params_state, raw_params_state)
	nonparam_state = coerce_tree_like(template_nonparam_state, restored.get("nonparam_state", template_nonparam_state))
	return graphdef, params_state, nonparam_state


def _build_model_from_checkpoint(checkpoint_path: str, seed: int) -> DiffusionPolicy:
	metadata = _load_checkpoint_metadata(checkpoint_path, None)
	return _build_model_from_metadata(metadata, seed)


def _make_optimizer(args: DiffusionTrainConfig, params_tree) -> tuple[optax.GradientTransformation, Any, Any]:
	# params_tree should be a pure dict pytree, not flax.nnx.State.
	trainable_mask = jax.tree_util.tree_map_with_path(
		lambda path, leaf: _is_trainable_path(path),
		params_tree,
	)

	total_steps = max(1, int(args.epochs) * int(args.steps_per_epoch))
	lr_schedule = optax.warmup_cosine_decay_schedule(
		init_value=0.0,
		peak_value=float(args.lr),
		warmup_steps=max(1, int(args.warmup_steps)),
		decay_steps=max(int(args.warmup_steps) + 1, total_steps),
		end_value=float(args.lr) * 0.1,
	)

	train_tx = optax.adamw(
		learning_rate=lr_schedule,
		weight_decay=float(args.weight_decay),
	)
	tx = optax.masked(train_tx, trainable_mask)
	return tx, lr_schedule, trainable_mask


def _make_train_step(
	graphdef,
	nonparam_state,
	tx,
	trainable_mask,
	grad_clip_norm: float,
	ema_decay: float,
	goal_mask_prob: float,
):
	goal_mask_prob = jnp.asarray(jnp.clip(goal_mask_prob, 0.0, 1.0), dtype=jnp.float32)

	def _mask_goal_features(feats: dict[str, jax.Array], rng: jax.Array) -> dict[str, jax.Array]:
		if "goal_xy" not in feats and "subgoal_valid" not in feats:
			return feats

		rng_goal, rng_subgoal = jax.random.split(rng)
		keep_goal = None
		keep_subgoal = None
		if "goal_xy" in feats:
			keep_goal = jax.random.bernoulli(
				rng_goal,
				p=jnp.asarray(1.0, dtype=jnp.float32) - goal_mask_prob,
				shape=(feats["goal_xy"].shape[0], 1),
			)
		if "subgoal_valid" in feats:
			keep_subgoal = jax.random.bernoulli(
				rng_subgoal,
				p=jnp.asarray(1.0, dtype=jnp.float32) - goal_mask_prob,
				shape=feats["subgoal_valid"].shape,
			)

		masked = dict(feats)
		if keep_goal is not None:
			masked["goal_xy"] = jnp.where(keep_goal, feats["goal_xy"], jnp.zeros_like(feats["goal_xy"]))
			masked["remaining_timesteps"] = jnp.where(
				keep_goal,
				feats["remaining_timesteps"],
				jnp.zeros_like(feats["remaining_timesteps"]),
			)
		if keep_subgoal is not None:
			masked["subgoal_valid"] = jnp.where(
				keep_subgoal,
				feats["subgoal_valid"],
				jnp.zeros_like(feats["subgoal_valid"]),
			)
			masked["subgoal_xy"] = jnp.where(
				keep_subgoal[..., None],
				feats["subgoal_xy"],
				jnp.zeros_like(feats["subgoal_xy"]),
			)
		return masked

	def merge_model(params_tree):
		return nnx.merge(
			graphdef,
			_pure_dict_to_state(params_tree),
			_pure_dict_to_state(nonparam_state),
		)

	def train_step(params_state, opt_state, ema_params, rng_key, batch_features):
		rng_key, loss_key, mask_key = jax.random.split(rng_key, 3)
		batch_features = _mask_goal_features(batch_features, mask_key)

		def loss_fn(p):
			model = merge_model(p)
			loss = model.loss(batch_features, rng=loss_key)
			inst_valid = jnp.asarray(batch_features["inst_valid"], dtype=jnp.float32)
			metrics = {
				"loss": loss,
				"inst_valid_rate": jnp.mean(inst_valid),
			}
			return loss, metrics

		(loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_state)
		clipped_grads, grad_norm = clip_grads(grads, grad_clip_norm)
		updates, opt_state = tx.update(clipped_grads, opt_state, params_state)
		updates = jax.tree_util.tree_map(
			lambda update, mask: update if mask else jnp.zeros_like(update),
			updates,
			trainable_mask,
		)
		params_state = optax.apply_updates(params_state, updates)
		ema_params = ema_update(ema_params, params_state, ema_decay)
		metrics = {**metrics, "grad_norm": grad_norm}
		return params_state, opt_state, ema_params, rng_key, metrics

	return jax.jit(train_step)


def _eval_sets_from_args(args: DiffusionTrainConfig) -> list[dict[str, Any]]:
	"""The named closed-loop eval sets configured for this run.

	Each set shares the rollout settings (start step, replan cadence, num_worlds,
	EMA weights, ...) so the only thing that varies is *which* scenes are driven;
	the success criterion is identical across sets.
	"""
	sets: list[dict[str, Any]] = []
	if args.eval_tfrecord_dir:
		sets.append({
			"tag": str(args.eval_tag),
			"tfrecord_dir": args.eval_tfrecord_dir,
			"lane_graph_dir": args.eval_lane_graph_dir,
			"scenario_indices": args.eval_scenario_indices,
			"num_scenarios": int(args.eval_num_scenarios),
		})
	if args.eval2_tfrecord_dir:
		sets.append({
			"tag": str(args.eval2_tag),
			"tfrecord_dir": args.eval2_tfrecord_dir,
			"lane_graph_dir": args.eval2_lane_graph_dir,
			"scenario_indices": args.eval2_scenario_indices,
			"num_scenarios": int(args.eval2_num_scenarios),
		})
	return sets


def _eval_one_set(
	planner,
	eval_set: dict[str, Any],
	args: DiffusionTrainConfig,
	run_dir: str,
	epoch: int,
	global_step_host: int,
) -> dict[str, float] | None:
	"""Run one closed-loop eval set and log its metrics to wandb.

	Reuses ``simulation.runner.run`` -- the very code that produced the failure
	set -- so the success criterion is identical:
	``success = goal_reached & ~offroad & ~overlap & ~tl_violation``.
	"""
	import wandb
	from types import SimpleNamespace
	from simulation.runner import run as run_closed_loop_eval

	tag = eval_set["tag"]
	eval_args = SimpleNamespace(
		output_dir=os.path.join(run_dir, "eval", tag, f"epoch_{epoch:04d}"),
		tfrecord_dir=eval_set["tfrecord_dir"],
		lane_graph_dir=eval_set["lane_graph_dir"],
		scenario_indices=eval_set["scenario_indices"],
		num_scenarios=int(eval_set["num_scenarios"]),
		num_worlds=int(args.eval_num_worlds),
		max_num_objects=int(args.eval_max_num_objects),
		start_timestep=int(args.eval_start_timestep),
		replan_interval_steps=int(args.eval_replan_interval_steps),
		instruction_interval_steps=int(args.eval_instruction_interval_steps),
		mask_goal=bool(args.eval_mask_goal),
		visualize_mode=args.eval_visualize_mode,
		save_trajectory=False,
		save_instruction=False,
		seed=int(args.eval_seed),
	)

	results = run_closed_loop_eval(eval_args, planner)
	if not results:
		return None

	n = len(results)
	def _rate(key: str) -> float:
		return sum(1 for r in results if bool(r.get(key, False))) / n

	safe = sum(1 for r in results if bool(r.get("success", False)))
	reached = sum(1 for r in results if bool(r.get("goal_reached", False)))
	reached_but_violated = sum(
		1 for r in results if bool(r.get("goal_reached", False)) and not bool(r.get("success", False))
	)
	metrics = {
		f"eval/{tag}/safe_goal_reach_rate": safe / n,
		f"eval/{tag}/goal_reach_rate": reached / n,
		f"eval/{tag}/offroad_rate": _rate("offroad"),
		f"eval/{tag}/overlap_rate": _rate("overlap"),
		f"eval/{tag}/tl_violation_rate": _rate("tl_violation"),
		f"eval/{tag}/reached_but_violated_rate": reached_but_violated / n,
		f"eval/{tag}/n_scenarios": float(n),
		"epoch": int(epoch),
		"step": int(global_step_host),
	}
	wandb.log(metrics, step=int(global_step_host))
	log_jsonl(os.path.join(run_dir, "eval_logs.jsonl"), metrics)
	print(
		f"[eval:{tag}] epoch {epoch}: safe_goal_reach={metrics[f'eval/{tag}/safe_goal_reach_rate']:.4f} "
		f"goal_reach={metrics[f'eval/{tag}/goal_reach_rate']:.4f} (n={n})"
	)
	return metrics


def _params_fingerprint(params) -> str:
	"""Cheap, stable digest of a parameter tree: 8 hex chars off the leaf sums."""
	import hashlib

	sums = [float(jnp.sum(jnp.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(params)]
	return hashlib.sha1(repr(sums).encode()).hexdigest()[:8]


def _run_safe_goal_eval(
	graphdef,
	params_state,
	ema_params,
	nonparam_state,
	eval_preprocess_cfg,
	eval_sets: list[dict[str, Any]],
	args: DiffusionTrainConfig,
	run_dir: str,
	epoch: int,
	global_step_host: int,
) -> None:
	"""Build one planner from the live params and run every configured eval set."""
	from flax import nnx
	from planner.diffusion_planner import DiffusionPlanner

	if not eval_sets:
		return

	# Evaluate the exact parameters closed-loop drives with. The failure-set sim
	# ran on the EMA weights (use_ema defaults True), so mirror that by default.
	eval_params = ema_params if bool(args.eval_use_ema) else params_state
	policy = nnx.merge(
		graphdef,
		_pure_dict_to_state(eval_params),
		_pure_dict_to_state(nonparam_state),
	)
	planner = DiffusionPlanner(
		policy=policy,
		preprocess_cfg=eval_preprocess_cfg,
		num_worlds=int(args.eval_num_worlds),
	)
	# Print a fingerprint of the weights actually being evaluated. Two evals that
	# report the same fingerprint at different epochs mean the eval is pinned to
	# stale parameters, not that the policy stopped changing -- a failure mode
	# that otherwise reads as a flat, plausible-looking metric.
	print(f"[eval] epoch {epoch}: params_fingerprint={_params_fingerprint(eval_params)}")
	for eval_set in eval_sets:
		try:
			_eval_one_set(planner, eval_set, args, run_dir, epoch, global_step_host)
		except Exception as exc:  # one bad set must not sink the others or training
			print(f"[eval:{eval_set.get('tag')}] epoch {epoch} skipped: {type(exc).__name__}: {exc}")


def main() -> None:
	args = parse_args()
	if args.resume_path is None and args.pretrained_ckpt is None:
		raise ValueError("Provide either --resume_path or --pretrained_ckpt.")

	configure_jax_compilation_cache(args.jax_compilation_cache_dir)

	checkpoint_path = args.resume_path or args.pretrained_ckpt
	assert checkpoint_path is not None
	model = _build_model_from_checkpoint(checkpoint_path, args.seed)
	graphdef, params_state, nonparam_state = nnx.split(model, nnx.Param, ...)
	params_state = _state_to_pure_dict(params_state)
	nonparam_state = _state_to_pure_dict(nonparam_state)


	if args.resume_path:
		tx, lr_schedule, trainable_mask = _make_optimizer(args, params_state)
		params_state, nonparam_state, opt_state, ema_params, rng_key, start_epoch, global_step, global_step_host = restore_checkpoint(
			args.resume_path,
			params_state=params_state,
			nonparam_state=nonparam_state,
			tx=tx,
			coerce_tree_like_fn=coerce_tree_like,
		)
	else:
		restored_graphdef, params_state, nonparam_state = _load_pretrained_state(
            args.pretrained_ckpt,
            model,
            use_ema=bool(args.use_ema),
        )
		params_state = _state_to_pure_dict(params_state)
		nonparam_state = _state_to_pure_dict(nonparam_state)
		# params_state = _zero_film_block_params(params_state)
		graphdef = restored_graphdef
		tx, lr_schedule, trainable_mask = _make_optimizer(args, params_state)
		opt_state = tx.init(params_state)
		ema_params = jax.tree_util.tree_map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            params_state,
        )
		rng_key = jax.random.PRNGKey(int(args.seed))
		start_epoch = 1
		global_step = jnp.array(0, dtype=jnp.int32)
		global_step_host = 0

	loader = build_dataloader(
		args.cache_dir,
		preprocess_cfg=args.preprocess_cfg,
		anchor_steps=(0, 10, 20, 30, 40),
		backend='jax',
		include_inst_features=True,
		batch_size=int(args.cache_batch_size),
		shuffle_seed=int(args.shuffle_seed),
		num_workers=int(args.num_workers),
		pin_memory=bool(args.pin_memory),
		drop_last=True,
		risk_table_path=args.risk_table,
		hardness_alpha=float(args.hardness_alpha),
		weight_clip=float(args.weight_clip),
		exclude_dirty=bool(args.exclude_dirty),
	)
	train_step = _make_train_step(
		graphdef,
		nonparam_state,
		tx,
		trainable_mask,
		float(args.grad_clip_norm),
		float(args.ema_decay),
		float(args.goal_mask_prob),
	)

	# The failure set was produced by the closed-loop sim, which reads the
	# preprocess config off the checkpoint metadata. Match it exactly so the eval
	# and training-data pipelines cannot silently diverge.
	eval_sets = _eval_sets_from_args(args) if int(args.eval_every) > 0 else []
	if int(args.eval_every) > 0 and not eval_sets:
		raise ValueError("eval_every>0 requires --eval_tfrecord_dir (and/or --eval2_tfrecord_dir)")
	eval_preprocess_cfg = (
		_build_preprocess_cfg(_load_checkpoint_metadata(checkpoint_path, None))
		if eval_sets
		else None
	)

	config_dict = config_to_dict(args)
	run, run_dir = init_wandb_run(args, config_dict)
	iterator = iter_with_prefetch(iter(loader), 0)
	ema_loss = None

	for epoch in range(int(start_epoch), int(args.epochs) + 1):
		pbar = tqdm(total=int(args.steps_per_epoch), desc=f"Epoch {epoch:04d}")
		for step_in_epoch in range(int(args.steps_per_epoch)):
			try:
				batch = next(iterator)
			except StopIteration:
				iterator = iter_with_prefetch(iter(loader), 0)
				batch = next(iterator)


			params_state, opt_state, ema_params, rng_key, metrics = train_step(
				params_state,
				opt_state,
				ema_params,
				rng_key,
				batch.features,
			)
			# print(f"Step {global_step_host}: loss={metrics['loss']:.6g}, inst_valid_rate={metrics['inst_valid_rate']:.6g}, grad_norm={metrics['grad_norm']:.6g}")
			global_step = global_step + jnp.int32(1)
			global_step_host += 1
			pbar.update(1)
			ema_loss = update_tqdm(pbar, global_step_host, args.pbar_every, metrics, ema_loss)

			if global_step_host % int(args.log_every) == 0:
				metric_values = _scalarize_for_logging(metrics)
				metric_values["epoch"] = int(epoch)
				metric_values["step"] = int(global_step_host)
				metric_values["lr"] = float(lr_schedule(max(global_step_host - 1, 0)))
				# print(json.dumps(metric_values, sort_keys=True))
				ema_loss, payload = wandb_log_train_step(global_step_host, int(epoch), metric_values, ema_loss, float(lr_schedule(max(global_step_host - 1, 0))))
				log_jsonl(os.path.join(run_dir, "train_logs.jsonl"), payload)

		pbar.close()

		if eval_sets and (epoch % int(args.eval_every) == 0 or epoch == int(args.epochs)):
			try:
				_run_safe_goal_eval(
					graphdef,
					params_state,
					ema_params,
					nonparam_state,
					eval_preprocess_cfg,
					eval_sets,
					args,
					run_dir,
					epoch,
					global_step_host,
				)
			except Exception as exc:  # eval must never take training down
				print(f"[eval] epoch {epoch} skipped: {type(exc).__name__}: {exc}")

		if epoch % int(args.save_every) == 0 or epoch == int(args.epochs):
			save_checkpoint(
				run_dir,
				config_dict,
				epoch=epoch,
				global_step=global_step,
				params_state=params_state,
				nonparam_state=nonparam_state,
				opt_state=opt_state,
				ema_params=ema_params,
				rng_key=rng_key,
				save_every=args.save_every,
			)


if __name__ == "__main__":
	main()
