"""Hardness-weighted Waymax scenario sampling for V-Max SAC fine-tuning.

V-Max feeds training scenarios through waymax's shuffled
``simulator_state_generator``, which yields batched ``SimulatorState``s with no
``(shard, scenario_idx)`` provenance -- so a per-scenario failure risk cannot be
attached to a batch after the fact.  This module instead *draws* ``(shard,
record)`` picks in proportion to a per-scenario risk weight and materialises the
Waymax state for exactly those picks, reusing the O(1) byte-offset reader and the
identical waymax parse the DPPO sampler already uses
(:mod:`rl.diffusion_ft.scene_sampler`).

The per-scenario weight mirrors :mod:`data.hardness_sampler` exactly, but at
scenario (not anchor) granularity -- a SAC episode runs the whole scenario, so
there are no anchor windows::

    w = clip((risk / mean_risk) ** alpha, 1/clip, clip)

``alpha=0`` reproduces uniform sampling (the true control); dirty scenes (the
logged expert's own bounding box leaves the drivable area) are optionally set to
weight 0 -- excluded, never upweighted.

The yielded state carries prefix ``(num_devices, num_envs, num_episode)`` to match
``simulator.make_data_generator(..., distributed=True)``: the SAC trainer pmaps
``run_training`` over a leading device axis (``sac_trainer.py``), so the batch
must be shaped for it.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Iterator

import numpy as np

# Reuse the DPPO sampler's shard index + byte-offset reader (same corpus), but NOT
# its batched parse: `_state_from_bytes` calls `preprocess_serialized_womd_data`
# on a stacked (N,) tensor, where `v[:max_num_objects]` slices the BATCH axis, so
# it can only run with max_num_objects=None (128 objects). V-Max's env is built for
# max_num_objects=64, so scenes must be 64 objects. `_preprocess_single_record`
# parses one record at a time (objects lead), slices objects correctly to 64, then
# we stack -- identical to what V-Max's own make_data_generator produces.
from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, ShardIndex


def _womd_cfg(*, batch_dims, num_paths, num_points_per_path, max_num_objects,
              max_num_rg_points=30000):
    """A WOMD parse ``DatasetConfig`` with V-Max's route/object dims."""
    import dataclasses

    from waymax import config as waymax_config

    return dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING, path="", batch_dims=tuple(batch_dims),
        max_num_objects=int(max_num_objects), max_num_rg_points=int(max_num_rg_points),
        shuffle_seed=0, include_sdc_paths=True,
        num_paths=int(num_paths), num_points_per_path=int(num_points_per_path),
    )


def load_stacked_states(records, cfg):
    """Parse each record singly (correct object slicing) and stack on a new axis 0.

    With ``cfg.batch_dims=()`` each state has prefix ``()``; the result is
    prefix ``(len(records),)``.
    """
    # Import via data.scenario_loader so TF cannot see the GPU (else it preallocates
    # the card and JAX OOMs later inside a waymax metric); mirrors scene_sampler.
    import data.scenario_loader  # noqa: F401
    import jax.tree_util as jtu
    import jax.numpy as jnp

    from viz.render import _preprocess_single_record

    states = [_preprocess_single_record(cfg, r) for r in records]
    if len(states) == 1:
        return states[0]
    return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def glob_from_at_spec(path_dataset: str) -> str:
    """Turn V-Max's ``...training_tfexample.tfrecord@1000`` into a shard glob.

    ``make_data_generator`` accepts the ``@N`` sharded-path spelling; the offset
    index and :class:`ShardIndex` want an actual glob over the shard files.
    """
    if "@" not in path_dataset:
        return path_dataset
    base, count = path_dataset.rsplit("@", 1)
    # TFRecord sharding is the fixed 5-digit `-NNNNN-of-NNNNN` convention (waymax
    # and the WOMD shards both use it), NOT len(count)-wide: `@1000` -> `-of-01000`.
    return f"{base}-*-of-{int(count):05d}"


def build_scenario_weights(
    index: ShardIndex,
    risk_table_path: str,
    *,
    alpha: float = 1.0,
    weight_clip: float = 10.0,
    exclude_dirty: bool = True,
) -> np.ndarray:
    """Per-global-scene sampling weight aligned to ``index`` ids ``[0, total)``.

    Scenes the risk table does not cover keep weight 1.0 (sampled, not
    upweighted); dirty scenes go to 0.0 when ``exclude_dirty``.  ``alpha=0`` makes
    every covered scene weight 1.0, i.e. uniform.
    """
    alpha = float(alpha)
    weight_clip = float(weight_clip)

    with np.load(risk_table_path, allow_pickle=True) as handle:
        shard = handle["shard"].astype(str)
        scenario_idx = handle["scenario_idx"].astype(np.int64)
        scenario_risk = handle["scenario_risk"].astype(np.float64)
        dirty = handle["dirty"].astype(bool)
        mean_risk = float(handle["mean_risk"])

    # (shard basename) -> row indices into the risk table, so each shard is one
    # vectorised scatter rather than a per-scene dict lookup over ~480k rows.
    rows_by_shard: dict[str, list[int]] = {}
    for row, name in enumerate(shard):
        rows_by_shard.setdefault(str(name), []).append(row)

    weights = np.ones(index.total, dtype=np.float64)
    scaled = scenario_risk / max(mean_risk, 1e-9)
    with np.errstate(over="ignore"):
        w_all = np.power(np.maximum(scaled, 1e-9), alpha)
    w_all = np.clip(w_all, 1.0 / weight_clip, weight_clip)
    if exclude_dirty:
        w_all = np.where(dirty, 0.0, w_all)

    covered = 0
    for shard_id, path in enumerate(index.paths):
        base = os.path.basename(path)
        rows = rows_by_shard.get(base)
        if not rows:
            continue
        rows = np.asarray(rows, dtype=np.int64)
        rec = scenario_idx[rows]
        in_range = rec < int(index.counts[shard_id])
        rows, rec = rows[in_range], rec[in_range]
        gid = int(index.cumulative[shard_id]) + rec
        weights[gid] = w_all[rows]
        covered += int(rows.size)

    if covered == 0:
        raise ValueError(
            f"risk table {risk_table_path!r} covered 0 of {index.total} scenes -- "
            f"shard basenames do not match the glob. First index shard: "
            f"{os.path.basename(index.paths[0])!r}."
        )
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("scenario weights are all zero/non-finite after masking.")
    return weights


class HardnessDataGenerator:
    """Iterator of hardness-weighted Waymax batches, a drop-in for the V-Max stream.

    Draws ``num_devices * num_envs * num_episode`` picks per batch in proportion to
    :func:`build_scenario_weights`, loads exactly those scenes, and reshapes the
    flat ``(N, ...)`` state to ``(num_devices, num_envs, num_episode, ...)``.  A
    background thread loads the next batch while the current one trains (parse is
    TF/CPU, training is GPU), so the load is hidden.
    """

    def __init__(
        self,
        shard_glob: str,
        risk_table_path: str,
        *,
        num_devices: int,
        num_envs: int,
        num_episode_per_epoch: int,
        alpha: float = 1.0,
        weight_clip: float = 10.0,
        exclude_dirty: bool = True,
        num_paths: int = 45,
        num_points_per_path: int = 800,
        max_num_objects: int = 64,
        max_num_rg_points: int = 30000,
        seed: int = 0,
        index_dir: str | None = DEFAULT_INDEX_DIR,
        prefetch: bool = True,
    ) -> None:
        import jax

        self.index = ShardIndex(shard_glob, index_dir=index_dir)
        self.weights = build_scenario_weights(
            self.index, risk_table_path, alpha=alpha,
            weight_clip=weight_clip, exclude_dirty=exclude_dirty,
        )
        self._p = self.weights / self.weights.sum()
        self.prefix = (int(num_devices), int(num_envs), int(num_episode_per_epoch))
        self.batch_n = int(np.prod(self.prefix))
        self.rng = np.random.default_rng(int(seed))
        # batch_dims=() -> each state prefix (); stacking N gives (N,), which we
        # reshape to (num_devices, num_envs, num_episode) to match the pmapped
        # trainer's expectation (distributed make_data_generator layout).
        self._cfg = _womd_cfg(
            batch_dims=(), num_paths=int(num_paths),
            num_points_per_path=int(num_points_per_path),
            max_num_objects=int(max_num_objects), max_num_rg_points=int(max_num_rg_points),
        )
        self._jax = jax
        self.n_loaded = 0
        self.load_seconds = 0.0
        self.last_ids = np.zeros(0, dtype=np.int64)

        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        if prefetch:
            self._q = queue.Queue(maxsize=1)
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ #
    def _draw_ids(self) -> np.ndarray:
        # Sampling with replacement: at alpha>0 a hard scene *should* be able to
        # recur within a batch, and 64 draws from ~480k rarely collide anyway.
        return self.rng.choice(self.index.total, size=self.batch_n, replace=True, p=self._p)

    def _load(self, ids: np.ndarray) -> Any:
        t0 = time.time()
        records = [self.index.read(self.index.to_pick(int(i))) for i in ids]
        flat = load_stacked_states(records, self._cfg)  # prefix (batch_n,)
        state = self._jax.tree_util.tree_map(
            lambda a: a.reshape(self.prefix + a.shape[1:]), flat
        )
        self.load_seconds += time.time() - t0
        return state

    def _prefetch_loop(self) -> None:
        while True:
            ids = self._draw_ids()
            try:
                state = self._load(ids)
            except Exception as exc:  # surfaced on the consumer side, not swallowed
                self._q.put(("error", exc, ids))
                return
            self._q.put(("ok", state, ids))

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._q is not None:
            kind, payload, ids = self._q.get()
            if kind == "error":
                raise RuntimeError(f"hardness prefetch failed on ids {ids.tolist()}") from payload
            state, self.last_ids = payload, ids
        else:
            self.last_ids = self._draw_ids()
            state = self._load(self.last_ids)
        self.n_loaded += self.batch_n
        return state

    def describe(self) -> str:
        nz = int((self.weights > 0).sum())
        return (
            f"{len(self.index.paths)} shards, {self.index.total} scenes "
            f"({nz} eligible), batch prefix {self.prefix}"
        )


def make_hardness_data_generator(
    *,
    path_dataset: str,
    risk_table_path: str,
    num_devices: int,
    num_envs: int,
    num_episode_per_epoch: int,
    num_paths: int,
    num_points_per_path: int,
    max_num_objects: int = 64,
    alpha: float = 1.0,
    weight_clip: float = 10.0,
    exclude_dirty: bool = True,
    seed: int = 0,
    index_dir: str | None = None,
) -> HardnessDataGenerator:
    """Build the generator from V-Max's ``env_config`` values (see train.py hook)."""
    gen = HardnessDataGenerator(
        glob_from_at_spec(path_dataset),
        risk_table_path,
        num_devices=num_devices,
        num_envs=num_envs,
        num_episode_per_epoch=num_episode_per_epoch,
        alpha=alpha,
        weight_clip=weight_clip,
        exclude_dirty=exclude_dirty,
        num_paths=num_paths,
        num_points_per_path=num_points_per_path,
        max_num_objects=max_num_objects,
        seed=seed,
        index_dir=index_dir or DEFAULT_INDEX_DIR,
    )
    return gen
