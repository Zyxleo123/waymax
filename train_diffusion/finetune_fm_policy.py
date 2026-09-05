from __future__ import annotations

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

from data.finetuning_loader import build_dataloader
from data.types import PreprocessConfig
from model.diffusion.fm_policy import FlowMatchingPolicy
from train_diffusion.configs.fm_finetuning_config import (
    FlowMatchingFinetuningConfig,
    config_to_dict,
    parse_args,
)
from train_diffusion.utils.checkpoints import (
    restore_checkpoint,
    save_checkpoint,
)
from train_diffusion.utils.utils import (
    build_data_parallel_mesh_if_needed,
    clip_grads,
    coerce_tree_like,
    configure_jax_compilation_cache,
    ema_update,
    init_wandb_run,
    replicate_to_mesh,
    update_tqdm,
    wandb_log_train_step,
)


# -----------------------------------------------------------------------------
# Config / utility
# -----------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        return {
            key: _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item)
            for item in value
        ]

    return str(value)


def _save_training_config(
    cfg: FlowMatchingFinetuningConfig,
    output_dir: Path,
) -> Path:
    config_path = output_dir / "training_config.json"
    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            _jsonable(asdict(cfg)),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return config_path


def _copy_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: x.copy()
        if hasattr(x, "copy")
        else x,
        tree,
    )

def _assert_same_tree_structure(template, value, *, name: str):
    template_structure = jax.tree_util.tree_structure(template)
    value_structure = jax.tree_util.tree_structure(value)

    if template_structure != value_structure:
        template_flat = list(nnx.to_flat_state(template))
        value_flat = list(nnx.to_flat_state(value))

        print(
            f"[tree mismatch] "
            f"template leaves={len(template_flat)}, "
            f"loaded leaves={len(value_flat)}"
        )

        for i, (
            (template_path, template_leaf),
            (value_path, value_leaf),
        ) in enumerate(
            zip(
                template_flat,
                value_flat,
                strict=False,
            )
        ):
            if (
                template_path != value_path
                or type(template_leaf) is not type(value_leaf)
            ):
                print(
                    f"First mismatch at leaf {i}:"
                )
                print(
                    "  template:",
                    template_path,
                    type(template_leaf),
                )
                print(
                    "  loaded:  ",
                    value_path,
                    type(value_leaf),
                )
                break

        raise ValueError(
            f"{name} pytree structure "
            "does not match model template."
        )

# -----------------------------------------------------------------------------
# Preprocess / model
# -----------------------------------------------------------------------------


def _build_preprocess_cfg(
    cfg: FlowMatchingFinetuningConfig,
) -> PreprocessConfig:
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
        max_points_per_segment=(
            cfg.max_points_per_segment
        ),
        num_object_types=8,
        inst_dim=cfg.inst_dim,
    )


def _build_model_from_config(
    cfg: FlowMatchingFinetuningConfig,
) -> FlowMatchingPolicy:
    return FlowMatchingPolicy(
        rngs=nnx.Rngs(cfg.seed),
        target_dim=cfg.target_dim,
        hidden_dim=cfg.hidden_dim,
        cond_dim=cfg.cond_dim,
        map_attr_dim=cfg.map_attr_dim,
        tl_attr_dim=cfg.tl_attr_dim,
        inst_attr_dim=cfg.inst_dim,
        predict_horizon=cfg.predict_horizon,
        subgoal_conditioned=cfg.subgoal_conditioned,
        default_num_steps=cfg.default_num_steps,
    )


def _build_training_components(
    cfg: FlowMatchingFinetuningConfig,
):
    preprocess_cfg = _build_preprocess_cfg(cfg)

    model = _build_model_from_config(cfg)

    total_steps = max(
        1,
        int(cfg.num_epochs)
        * int(cfg.steps_per_epoch),
    )

    lr_schedule = (
        optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(cfg.lr),
            warmup_steps=max(
                1,
                int(cfg.warmup_steps),
            ),
            decay_steps=max(
                int(cfg.warmup_steps) + 1,
                total_steps,
            ),
            end_value=float(cfg.lr) * 0.1,
        )
    )

    tx = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=float(cfg.weight_decay),
    )

    return (
        preprocess_cfg,
        model,
        tx,
        lr_schedule,
    )


# -----------------------------------------------------------------------------
# Feature handling
# -----------------------------------------------------------------------------


def _add_dummy_instruction_features(
    features: dict[str, jax.Array],
    cfg: FlowMatchingFinetuningConfig,
) -> dict[str, jax.Array]:
    batch_size = features["ego_state"].shape[0]

    augmented = dict(features)

    augmented["inst_features"] = jnp.zeros(
        (batch_size, cfg.inst_dim),
        dtype=jnp.float32,
    )

    augmented["inst_valid"] = jnp.zeros(
        (batch_size,),
        dtype=jnp.bool_,
    )

    augmented["subgoal_xy"] = jnp.zeros(
        (batch_size, 2),
        dtype=jnp.float32,
    )

    augmented["subgoal_valid"] = jnp.zeros(
        (batch_size,),
        dtype=jnp.bool_,
    )

    return augmented


def _maybe_mask_goal_xy(
    features: dict[str, jax.Array],
    rng: jax.Array,
    cfg: FlowMatchingFinetuningConfig,
) -> dict[str, jax.Array]:
    if "goal_xy" not in features:
        return features

    goal_mask_prob = jnp.asarray(
        jnp.clip(
            cfg.goal_mask_prob,
            0.0,
            1.0,
        ),
        dtype=jnp.float32,
    )

    keep_goal = jax.random.bernoulli(
        rng,
        p=(
            jnp.asarray(
                1.0,
                dtype=jnp.float32,
            )
            - goal_mask_prob
        ),
        shape=(
            features["goal_xy"].shape[0],
            1,
        ),
    )

    masked = dict(features)

    masked["goal_xy"] = jnp.where(
        keep_goal,
        features["goal_xy"],
        jnp.zeros_like(
            features["goal_xy"]
        ),
    )

    masked["remaining_timesteps"] = (
        jnp.where(
            keep_goal,
            features["remaining_timesteps"],
            jnp.zeros_like(
                features[
                    "remaining_timesteps"
                ]
            ),
        )
    )

    return masked


def _batch_to_jax(
    batch_features: dict[str, Any],
) -> dict[str, jax.Array]:
    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value),
        batch_features,
    )


# -----------------------------------------------------------------------------
# Train step
# -----------------------------------------------------------------------------


def _make_train_step(
    graphdef,
    nonparam_state,
    tx,
    grad_clip_norm: float,
    cfg: FlowMatchingFinetuningConfig,
    data_parallel: bool,
):
    def merge_model(params_tree):
        return nnx.merge(
            graphdef,
            params_tree,
            nonparam_state,
        )

    def train_step_impl(
        params_state,
        opt_state,
        ema_params,
        rng_key,
        global_step,
        batch_features,
        batch_aux,
    ):
        rng_key, key_loss, key_mask = (
            jax.random.split(
                rng_key,
                3,
            )
        )

        batch_features = _batch_to_jax(
            batch_features
        )

        batch_features = _maybe_mask_goal_xy(
            batch_features,
            key_mask,
            cfg,
        )

        batch_features = (
            _add_dummy_instruction_features(
                batch_features,
                cfg,
            )
        )

        batch_aux = _batch_to_jax(
            batch_aux
        )

        def loss_fn(p):
            model = merge_model(p)

            features = model._as_features(
                batch_features
            )

            cond, inst_cond = (
                model.compute_condition(
                    features
                )
            )

            target = (
                features.ego_trajectory[
                    :,
                    : model.predict_horizon,
                    :,
                ]
            )

            inst_cond_mask = jnp.asarray(
                features.inst_valid,
                dtype=bool,
            )

            loss = model.loss_from_condition(
                target,
                cond,
                inst_cond,
                inst_cond_mask,
                rng=key_loss,
                data_parallel=data_parallel,
            )

            failure_source = (
                batch_aux.get(
                    "is_failure",
                    jnp.zeros(
                        (target.shape[0],),
                        dtype=jnp.bool_,
                    ),
                )
            )

            failure_rate = jnp.mean(
                jnp.asarray(
                    failure_source,
                    dtype=jnp.float32,
                )
            )

            return loss, {
                "loss": loss,
                "failure_rate": failure_rate,
            }

        (
            (loss, metrics),
            grads,
        ) = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params_state)

        clipped_grads, grad_norm = (
            clip_grads(
                grads,
                grad_clip_norm,
            )
        )

        updates, opt_state_next = (
            tx.update(
                clipped_grads,
                opt_state,
                params_state,
            )
        )

        params_state_next = (
            optax.apply_updates(
                params_state,
                updates,
            )
        )

        ema_params_next = ema_update(
            ema_params,
            params_state_next,
            cfg.ema_decay,
        )

        metrics = {
            **metrics,
            "grad_norm": grad_norm,
        }

        return (
            params_state_next,
            opt_state_next,
            ema_params_next,
            rng_key,
            global_step + jnp.int32(1),
            metrics,
        )

    return jax.jit(train_step_impl)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def run_training(
    cfg: FlowMatchingFinetuningConfig,
) -> None:
    configure_jax_compilation_cache(
        cfg.jax_compilation_cache_dir
    )

    (
        preprocess_cfg,
        model,
        tx,
        lr_schedule,
    ) = _build_training_components(cfg)

    # -------------------------------------------------------------------------
    # Logging / output directory
    # -------------------------------------------------------------------------

    if cfg.wandb_project:
        run, run_dir_str = init_wandb_run(
            cfg,
            config_to_dict(cfg),
        )
        run_dir = Path(run_dir_str)

    else:
        run = None

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        run_name = (
            f"{cfg.wandb_name}_{stamp}"
        )

        run_dir = (
            Path(cfg.output_dir)
            / run_name
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    _save_training_config(
        cfg,
        run_dir,
    )

    # -------------------------------------------------------------------------
    # Build model state
    # -------------------------------------------------------------------------

    (
        graphdef,
        template_params_state,
        template_nonparam_state,
    ) = nnx.split(
        model,
        nnx.Param,
        ...,
    )

    params_state = template_params_state
    nonparam_state = template_nonparam_state

    rng_key = jax.random.PRNGKey(
        cfg.seed
    )

    global_step = jnp.array(
        0,
        dtype=jnp.int32,
    )

    global_step_host = 0
    start_epoch = 1

    # These are assigned by one of the
    # initialization branches below.
    opt_state = None
    ema_params = None

    # -------------------------------------------------------------------------
    # Case 1:
    # Resume an interrupted FINETUNING run.
    #
    # Restore:
    #   - params
    #   - nonparam state
    #   - optimizer state
    #   - EMA
    #   - RNG
    #   - epoch / global step
    # -------------------------------------------------------------------------

    if cfg.resume_path:
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
            coerce_tree_like_fn=(
                coerce_tree_like
            ),
        )

        _assert_same_tree_structure(
            template_params_state,
            params_state,
            name="resumed params_state",
        )

        _assert_same_tree_structure(
            template_nonparam_state,
            nonparam_state,
            name="resumed nonparam_state",
        )

        print(
            "[checkpoint] Resumed training "
            f"from: {cfg.resume_path}"
        )

        print(
            "[checkpoint] "
            f"start_epoch={start_epoch}, "
            f"global_step={global_step_host}"
        )

    # -------------------------------------------------------------------------
    # Case 2:
    # Start a NEW finetuning run from a PRETRAINED model.
    #
    # Restore only model state.
    #
    # IMPORTANT:
    # optimizer and EMA are initialized AFTER
    # loading the pretrained parameters.
    # -------------------------------------------------------------------------

    elif cfg.pretrained_ckpt is not None:
        restored = restore_checkpoint(
            cfg.pretrained_ckpt
        )

        if (
            cfg.use_ema
            and "ema_params" in restored
        ):
            raw_params_state = (
                restored["ema_params"]
            )
            loaded_param_source = (
                "EMA parameters"
            )

        else:
            raw_params_state = (
                restored["params_state"]
            )
            loaded_param_source = (
                "regular parameters"
            )

        params_state = coerce_tree_like(
            template_params_state,
            raw_params_state,
        )

        raw_nonparam_state = restored.get(
            "nonparam_state",
            template_nonparam_state,
        )

        nonparam_state = coerce_tree_like(
            template_nonparam_state,
            raw_nonparam_state,
        )

        # Make sure coerce_tree_like did not
        # turn nnx.State into an incompatible dict.
        _assert_same_tree_structure(
            template_params_state,
            params_state,
            name="pretrained params_state",
        )

        _assert_same_tree_structure(
            template_nonparam_state,
            nonparam_state,
            name="pretrained nonparam_state",
        )

        # IMPORTANT:
        # Finetuning starts with a fresh optimizer.
        #
        # This must happen AFTER loading params.
        opt_state = tx.init(
            params_state
        )

        # IMPORTANT:
        # EMA starts from the loaded pretrained
        # parameters, not from the random model.
        ema_params = _copy_tree(
            params_state
        )

        # This is a new training run.
        rng_key = jax.random.PRNGKey(
            cfg.seed
        )

        global_step = jnp.array(
            0,
            dtype=jnp.int32,
        )

        global_step_host = 0
        start_epoch = 1

        print(
            "[checkpoint] Loaded pretrained "
            f"{loaded_param_source} from: "
            f"{cfg.pretrained_ckpt}"
        )

        print(
            "[checkpoint] Created a fresh "
            "optimizer state for finetuning."
        )

    # -------------------------------------------------------------------------
    # Case 3:
    # Train completely from scratch.
    # -------------------------------------------------------------------------

    else:
        opt_state = tx.init(
            params_state
        )

        ema_params = _copy_tree(
            params_state
        )

        print(
            "[checkpoint] No checkpoint "
            "provided; training from scratch."
        )

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    if opt_state is None:
        raise RuntimeError(
            "opt_state was not initialized."
        )

    if ema_params is None:
        raise RuntimeError(
            "ema_params was not initialized."
        )

    _assert_same_tree_structure(
        params_state,
        ema_params,
        name="ema_params",
    )

    print(
        "[state] params type:",
        type(params_state),
    )

    print(
        "[state] ema type:",
        type(ema_params),
    )

    # -------------------------------------------------------------------------
    # Data parallel setup
    # -------------------------------------------------------------------------

    data_mesh = (
        build_data_parallel_mesh_if_needed(
            cfg
        )
    )

    data_parallel_enabled = (
        data_mesh is not None
    )

    if (
        data_parallel_enabled
        and data_mesh is not None
    ):
        with data_mesh:
            params_state = replicate_to_mesh(
                params_state,
                data_mesh,
            )

            nonparam_state = (
                replicate_to_mesh(
                    nonparam_state,
                    data_mesh,
                )
            )

            opt_state = replicate_to_mesh(
                opt_state,
                data_mesh,
            )

            ema_params = replicate_to_mesh(
                ema_params,
                data_mesh,
            )

            rng_key = replicate_to_mesh(
                rng_key,
                data_mesh,
            )

            global_step = (
                replicate_to_mesh(
                    global_step,
                    data_mesh,
                )
            )

    # -------------------------------------------------------------------------
    # Dataloader
    # -------------------------------------------------------------------------

    train_loader = build_dataloader(
        source_cache_dir=(
            cfg.source_cache_dir
        ),
        failure_cache_dir=(
            cfg.failure_cache_dir
        ),
        preprocess_cfg=preprocess_cfg,
        source_file_indices=(
            cfg.source_file_indices
        ),
        failure_file_indices=(
            cfg.failure_file_indices
        ),
        batch_size=cfg.batch_size,
        shuffle_seed=cfg.shuffle_seed,
        failure_sampling_prob=(
            cfg.failure_sampling_prob
        ),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        backend="jax",
        include_inst_features=False,
        drop_last=False,
    )

    train_iter = iter(
        train_loader
    )

    # Trigger conversion once before compilation.
    warmup_batch = next(
        train_iter
    )

    _ = _batch_to_jax(
        warmup_batch.features
    )

    # -------------------------------------------------------------------------
    # Train function
    # -------------------------------------------------------------------------

    train_step_fn = _make_train_step(
        graphdef,
        nonparam_state,
        tx,
        cfg.grad_clip_norm,
        cfg,
        data_parallel_enabled,
    )

    ema_loss = None

    last_epoch = max(
        0,
        start_epoch - 1,
    )

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        cfg.num_epochs + 1,
    ):
        last_epoch = epoch

        pbar = tqdm(
            total=cfg.steps_per_epoch,
            desc=f"Epoch {epoch:04d}",
        )

        for _ in range(
            cfg.steps_per_epoch
        ):
            batch = next(
                train_iter
            )
            (
                params_state,
                opt_state,
                ema_params,
                rng_key,
                global_step,
                metrics,
            ) = train_step_fn(
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

            ema_loss = update_tqdm(
                pbar,
                global_step_host,
                max(
                    1,
                    cfg.log_every,
                ),
                metrics,
                ema_loss,
            )

            if (
                cfg.wandb_project
                and run is not None
                and global_step_host
                % max(
                    1,
                    cfg.log_every,
                )
                == 0
            ):
                lr = float(
                    lr_schedule(
                        global_step_host - 1
                    )
                )

                ema_loss, _ = (
                    wandb_log_train_step(
                        global_step_host,
                        epoch,
                        metrics,
                        ema_loss,
                        lr,
                    )
                )

            if (
                cfg.max_steps is not None
                and global_step_host
                >= cfg.max_steps
            ):
                break

        pbar.close()

        # ---------------------------------------------------------------------
        # Periodic checkpoint
        # ---------------------------------------------------------------------

        if (
            cfg.save_every > 0
            and epoch
            % cfg.save_every
            == 0
        ):
            save_checkpoint(
                str(run_dir),
                config_to_dict(cfg),
                epoch=epoch,
                global_step=global_step,
                params_state=params_state,
                nonparam_state=(
                    nonparam_state
                ),
                opt_state=opt_state,
                ema_params=ema_params,
                rng_key=rng_key,
                save_every=(
                    cfg.save_every
                ),
            )

        if (
            cfg.max_steps is not None
            and global_step_host
            >= cfg.max_steps
        ):
            break

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------

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
    run_training(
        parse_args()
    )


if __name__ == "__main__":
    main()