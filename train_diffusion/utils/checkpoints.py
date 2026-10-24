from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from typing import Any

import jax
import orbax.checkpoint as ocp


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def _save_checkpoint_tag(
    save_dir: str,
    train_state: dict[str, Any],
    config_dict: dict[str, Any],
    tag: str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, tag)
    if os.path.exists(ckpt_path):
        if os.path.isdir(ckpt_path):
            for root, dirs, files in os.walk(ckpt_path, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(ckpt_path)
        else:
            os.remove(ckpt_path)

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_path, train_state)

    meta_path = os.path.join(ckpt_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config_dict), f, indent=2, sort_keys=True)

    return ckpt_path


def save_checkpoint(
    save_dir: str,
    config_dict: dict[str, Any],
    *,
    epoch: int,
    global_step,
    params_state,
    nonparam_state,
    opt_state,
    ema_params,
    rng_key,
    save_every: int | None = None,
) -> None:
    train_state = {
        "epoch": jax.numpy.array(epoch, dtype=jax.numpy.int32),
        "global_step": global_step,
        "params_state": params_state,
        "nonparam_state": nonparam_state,
        "opt_state": opt_state,
        "ema_params": ema_params,
        "rng_key": rng_key,
    }
    _save_checkpoint_tag(save_dir, train_state, config_dict, tag="latest")
    if save_every is not None and epoch % save_every == 0:
        _save_checkpoint_tag(save_dir, train_state, config_dict, tag=f"epoch_{epoch:04d}")


def _is_orbax_checkpoint_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    if names & {"checkpoint", "manifest.ocdbt", "_METADATA", "_CHECKPOINT_METADATA"}:
        return True
    return any(
        name.startswith("_CHECKPOINT")
        or name.startswith("ocdbt")
        or name.endswith(".zarray")
        for name in names
    )


def resolve_orbax_checkpoint_path(path: str) -> str:
    """Accept a run directory or an Orbax step directory."""
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if os.path.isfile(resolved):
        resolved = os.path.dirname(resolved)
    if _is_orbax_checkpoint_dir(resolved):
        return resolved

    latest = os.path.join(resolved, "latest")
    if _is_orbax_checkpoint_dir(latest):
        return latest

    epoch_dirs = sorted(
        os.path.join(resolved, name)
        for name in os.listdir(resolved)
        if name.startswith("epoch_") and _is_orbax_checkpoint_dir(os.path.join(resolved, name))
    )
    if epoch_dirs:
        return epoch_dirs[-1]
    return resolved


def _is_nnx_state(value: Any) -> bool:
    return hasattr(value, "_mapping") and not hasattr(value, "shape")


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def coerce_optimizer_state(template: Any, raw: Any, coerce_tree_like_fn) -> Any:
    """Copy a restored optimizer pytree into the live ``tx.init`` structure.

    ``PyTreeCheckpointer`` restores tuples as lists and NamedTuples as dicts,
    and empty states such as ``EmptyState`` come back as ``None``.
    ``coerce_tree_like`` only accepts an NNX ``State``, so passing the optax
    tuple through it raises ``'tuple' object has no attribute '_mapping'``.
    """

    def align(tmpl: Any, src: Any) -> Any:
        if src is None:
            if _is_namedtuple(tmpl) and len(tmpl._fields) == 0:
                return tmpl
            if isinstance(tmpl, tuple) and len(tmpl) == 0:
                return tmpl
            raise TypeError(
                f"Checkpoint optimizer leaf is None, template is {type(tmpl).__name__}"
            )

        if _is_nnx_state(tmpl):
            if coerce_tree_like_fn is None:
                raise TypeError(
                    "NNX state inside the optimizer requires coerce_tree_like_fn"
                )
            return coerce_tree_like_fn(tmpl, src)

        if _is_namedtuple(tmpl):
            if _is_namedtuple(src):
                src_map = src._asdict()
            elif isinstance(src, Mapping):
                src_map = src
            else:
                raise TypeError(
                    f"Expected a mapping for {type(tmpl).__name__}, "
                    f"got {type(src).__name__}"
                )
            values = []
            for field in tmpl._fields:
                if field in src_map:
                    child = src_map[field]
                elif str(field) in src_map:
                    child = src_map[str(field)]
                else:
                    raise KeyError(
                        f"Missing optimizer field {field!r} on {type(tmpl).__name__}"
                    )
                values.append(align(getattr(tmpl, field), child))
            return type(tmpl)(*values)

        if isinstance(tmpl, tuple):
            if not isinstance(src, (list, tuple)):
                raise TypeError(
                    f"Expected a sequence for tuple optimizer state, "
                    f"got {type(src).__name__}"
                )
            if len(tmpl) != len(src):
                raise ValueError(
                    f"Optimizer tuple length {len(tmpl)} != checkpoint length {len(src)}"
                )
            return type(tmpl)(align(child, src[i]) for i, child in enumerate(tmpl))

        if hasattr(tmpl, "shape") and hasattr(src, "shape"):
            if tuple(src.shape) != tuple(tmpl.shape):
                raise ValueError(
                    f"Optimizer leaf shape {tuple(src.shape)} != template {tuple(tmpl.shape)}"
                )
            if getattr(src, "dtype", None) != getattr(tmpl, "dtype", None):
                return jax.numpy.asarray(src, dtype=tmpl.dtype)
            return src

        return src

    return align(template, raw)


def restore_checkpoint(
    path: str,
    *,
    params_state: Any | None = None,
    nonparam_state: Any | None = None,
    tx: Any | None = None,
    coerce_tree_like_fn=None,
) -> Any:
    path = resolve_orbax_checkpoint_path(path)
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(path)

    if params_state is None or tx is None:
        return restored

    loaded_params_state = restored["params_state"]
    loaded_nonparam_state = restored.get("nonparam_state", nonparam_state)

    if coerce_tree_like_fn is not None:
        loaded_params_state = coerce_tree_like_fn(params_state, loaded_params_state)
        if nonparam_state is not None and loaded_nonparam_state is not None:
            loaded_nonparam_state = coerce_tree_like_fn(nonparam_state, loaded_nonparam_state)

    opt_state = restored["opt_state"]

    opt_template = tx.init(loaded_params_state)
    if coerce_tree_like_fn is not None:
        try:
            loaded_opt_state = coerce_optimizer_state(
                opt_template,
                opt_state,
                coerce_tree_like_fn,
            )
        except Exception as exc:
            warnings.warn(
                "Failed to coerce checkpoint opt_state to current optimizer structure; "
                "falling back to freshly initialized optimizer state. "
                f"This can happen after optimizer implementation changes. Details: {exc}",
                RuntimeWarning,
            )
            loaded_opt_state = opt_template
    else:
        loaded_opt_state = restored["opt_state"]
    loaded_global_step_host = int(restored["global_step"])
    loaded_ema_params = restored["ema_params"]
    if coerce_tree_like_fn is not None:
        loaded_ema_params = coerce_tree_like_fn(loaded_params_state, loaded_ema_params)

    return (
        loaded_params_state,
        loaded_nonparam_state,
        loaded_opt_state,
        loaded_ema_params,
        restored["rng_key"],
        int(restored["epoch"]) + 1,
        jax.numpy.array(loaded_global_step_host, dtype=jax.numpy.int32),
        loaded_global_step_host,
    )


def tree_global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    sq = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves if x is not None]
    if not sq:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    return jax.numpy.sqrt(jax.numpy.sum(jax.numpy.stack(sq)))
