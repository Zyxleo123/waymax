"""Turn a non-self-conditioned diffusion checkpoint into a self-conditioned one.

FlowMol3 (arXiv 2508.12629) adds self-conditioning by letting the network see its
own clean prediction from the previous integration step. Here that is a channel
concat at the U-Net stem, so the stem's kernel grows from ``[3, in_ch, base_ch]``
to ``[3, 2*in_ch, base_ch]`` and the checkpoint no longer restores into the model.

Re-pretraining from scratch to get that one tensor is not worth a GPU-week. This
module expands the stem instead, **zero-initializing the new input block**:

    kernel_new[:, :in_ch, :] = kernel_old        kernel_new[:, in_ch:, :] = 0

A zero block contributes exactly nothing to the convolution, so at step 0 the
expanded model computes the *same* function as the base for *any* self-cond
input -- including the zeros the first reverse step supplies. The fine-tune
therefore starts from the base policy's behaviour rather than from a perturbed
one, and whatever the self-cond channels come to mean is learned, not inherited.
``tests/test_self_cond.py`` asserts that bit-identity rather than assuming it.

Every other leaf is copied unchanged, so this is not a re-initialization: the
tokenizer, the instruction encoder and all the U-Net blocks keep their weights.

CLI (writes a new checkpoint directory that ``load_model_for_inference`` and
``rl.diffusion_ft.checkpoint`` load directly, because the expanded
``metadata.json`` carries ``self_cond: true``)::

    python -m rl.diffusion_ft.self_cond_init \\
        --src /.../pretrain_diffusion_without_subgoal_.../latest \\
        --dst /.../base_selfcond/latest
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "STEM_PATH",
    "find_stem_kernel_paths",
    "expand_params_for_self_cond",
    "expand_checkpoint_dir",
]

# Where the stem kernel lives inside a `DiffusionPolicy` parameter tree. Kept as a
# constant so the search below can be asserted against it rather than trusted.
STEM_PATH = ("diffusion", "denoise_fn", "stem", "kernel")


def _get(tree: Any, path: tuple[str, ...]) -> Any:
    for k in path:
        tree = tree[k]
    return tree


def _set(tree: Any, path: tuple[str, ...], value: Any) -> None:
    for k in path[:-1]:
        tree = tree[k]
    tree[path[-1]] = value


def _leaf_value(leaf: Any) -> Any:
    """Unwrap an nnx ``VariableState``-like leaf to its array, if it is one."""
    return leaf.value if hasattr(leaf, "value") else leaf


def find_stem_kernel_paths(params: Any) -> list[tuple[str, ...]]:
    """Every ``(..., 'stem', 'kernel')`` path in a nested-dict parameter tree.

    Returned as a list so a caller can assert there is exactly one: a silent
    second match would mean the tree is not the one this module was written for.
    """
    found: list[tuple[str, ...]] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            p = path + (str(k),)
            if len(p) >= 2 and p[-2] == "stem" and p[-1] == "kernel":
                found.append(p)
            walk(v, p)

    walk(params, ())
    return found


def expand_params_for_self_cond(params: Any, in_ch: int) -> Any:
    """Copy ``params`` with the stem kernel zero-expanded to ``2*in_ch`` fan-in.

    Idempotent in the sense that an already-expanded tree is rejected loudly
    rather than expanded twice.
    """
    paths = find_stem_kernel_paths(params)
    if len(paths) != 1:
        raise ValueError(f"expected exactly one stem kernel, found {len(paths)}: {paths}")
    path = paths[0]
    if path != STEM_PATH:
        raise ValueError(f"stem kernel at unexpected path {path}, expected {STEM_PATH}")

    out = jax.tree_util.tree_map(lambda x: x, params)  # structural copy
    leaf = _get(out, path)
    kernel = np.asarray(_leaf_value(leaf))
    if kernel.ndim != 3:
        raise ValueError(f"stem kernel should be [K, in, out], got shape {kernel.shape}")
    k, fan_in, fan_out = kernel.shape
    if fan_in == 2 * in_ch:
        raise ValueError(
            f"stem kernel fan-in is already {fan_in} == 2*in_ch; this checkpoint "
            "looks self-conditioned already.")
    if fan_in != in_ch:
        raise ValueError(f"stem kernel fan-in {fan_in} != in_ch {in_ch}")

    expanded = np.zeros((k, 2 * in_ch, fan_out), dtype=kernel.dtype)
    expanded[:, :in_ch, :] = kernel
    _set(out, path, jnp.asarray(expanded))
    return out


def expand_checkpoint_dir(src: str, dst: str, *, overwrite: bool = False) -> dict[str, Any]:
    """Write a self-conditioned copy of the checkpoint directory ``src``.

    Expands ``params_state`` and ``ema_params`` (which may be a subset of the
    parameters -- the without-subgoal pretrain did not EMA every module -- so its
    stem is expanded only if it has one), drops ``opt_state`` (an Adam state for
    the old shape is meaningless for the new one, and a fine-tune re-inits it
    anyway) and stamps ``self_cond: true`` into ``metadata.json``.
    """
    import orbax.checkpoint as ocp

    from rl.diffusion_ft.checkpoint import _apply_orbax_sharding_patch

    _apply_orbax_sharding_patch()
    src_p, dst_p = Path(src), Path(dst)
    if dst_p.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_p} exists; pass --overwrite to replace it")
        shutil.rmtree(dst_p)

    meta_path = src_p / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("self_cond"):
        raise ValueError(f"{src} already has self_cond=true in its metadata")
    in_ch = int(metadata["target_dim"])

    restored = ocp.PyTreeCheckpointer().restore(str(src_p))
    restored["params_state"] = expand_params_for_self_cond(restored["params_state"], in_ch)
    ema = restored.get("ema_params")
    if ema is not None and find_stem_kernel_paths(ema):
        restored["ema_params"] = expand_params_for_self_cond(ema, in_ch)
    # An optimizer state shaped for the old stem cannot be applied to the new one.
    restored.pop("opt_state", None)

    # orbax creates the leaf directory itself and objects if it already exists,
    # so only the parent is made here -- the same contract
    # `train_diffusion/utils/checkpoints.py::_save_checkpoint_tag` follows.
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(dst_p), restored)
    dst_p.mkdir(parents=True, exist_ok=True)
    metadata["self_cond"] = True
    metadata["self_cond_expanded_from"] = str(src_p)
    (dst_p / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source checkpoint dir (contains metadata.json)")
    ap.add_argument("--dst", required=True, help="destination checkpoint dir to create")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    meta = expand_checkpoint_dir(a.src, a.dst, overwrite=a.overwrite)
    print(f"wrote self-conditioned checkpoint to {a.dst}")
    print(f"  target_dim={meta['target_dim']}  self_cond={meta['self_cond']}")


if __name__ == "__main__":
    main()
