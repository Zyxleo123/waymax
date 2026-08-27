"""Uniform random scene sampling over the WHOLE WOMD training set.

Until now DPPO trained on one fixed batch of scenes loaded once (``--indices``
against a single tfrecord shard), which is an overfit target: the policy sees the
same 8-32 scenarios for every one of hundreds of iterations. This module draws a
*fresh* batch each iteration, uniformly over all 1000 raw-WOMD training shards --
the same corpus the base diffusion checkpoint was pretrained on
(``tfrecord_path=.../training_tfexample.tfrecord@1000`` in its ``metadata.json``),
which is what makes the expert (behavior-cloning) term a coherent anchor rather
than a pull toward a different distribution.

Random access is O(1) via the byte-offset sidecar indexes from
:mod:`rl.tfrecord_fast` (``<shard>.offsets.npy``); a ``TFRecordDataset`` scan to
reach record *i* of a 1.2 GB shard would dominate the ~8 s iteration. Build the
indexes once with ``rl/diffusion_ft/slurm/submit_build_womd_offsets.sh``.

Two knobs matter for correctness rather than speed:

* **Disjoint train/eval scenes.** With the training batch resampled every
  iteration, evaluating on it measures the draw, not the policy. :class:`SceneSampler`
  reserves a fixed, held-out eval set (drawn once, never sampled for training) so
  the eval curve is comparable across iterations *and* is a generalization number.
* **Constant batch shape.** The env/actor are built once, sized to the batch's
  object count, and rebuilding them would reset the trained weights and the Adam
  state. Raw WOMD tf_example is a fixed 128-object feature spec, so every batch
  has the same shape; :meth:`SceneSampler.next_train` asserts it rather than
  trusting it.
"""

from __future__ import annotations

import glob
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rl.tfrecord_fast import default_offset_index_path

Pick = tuple[int, int]  # (shard_id, record_index_within_shard)

# The WOMD shards live in another user's scratch (`eshau`). It is writable, but
# scattering 1000 sidecar files through someone else's data directory is not ours
# to do, so the indexes live under our own scratch and are addressed by basename.
DEFAULT_INDEX_DIR = "/zfsauton/scratch/yixiz/waymax_rs/womd_offsets"


def index_path_for(shard_path: str, index_dir: str | None = None) -> str:
    """Where this shard's ``.offsets.npy`` lives.

    ``index_dir=None`` keeps :mod:`rl.tfrecord_fast`'s sidecar-next-to-the-file
    convention (what the ScenarioMax index already uses); otherwise the index is
    ``<index_dir>/<shard basename>.offsets.npy``.
    """
    if not index_dir:
        return default_offset_index_path(shard_path)
    return str(Path(index_dir) / (Path(shard_path).name + ".offsets.npy"))


# --------------------------------------------------------------------------- #
class ShardIndex:
    """The set of shards backing a dataset, plus their record counts.

    Reads only the sidecar ``.offsets.npy`` files (a few hundred int64 each), so
    constructing this is cheap even across 1000 shards.
    """

    def __init__(self, shard_glob: str, index_dir: str | None = DEFAULT_INDEX_DIR):
        self.paths: list[str] = sorted(glob.glob(shard_glob))
        if not self.paths:
            raise FileNotFoundError(f"No shards matched --shard_glob {shard_glob!r}")
        self.index_dir = index_dir
        self.index_paths = [index_path_for(p, index_dir) for p in self.paths]
        missing = [ip for ip in self.index_paths if not Path(ip).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(self.paths)} shards have no byte-offset index "
                f"(first: {missing[0]}).\nBuild them once (CPU, minutes):\n"
                f"  bash rl/diffusion_ft/slurm/submit_build_womd_offsets.sh"
            )
        self.offsets: list[np.ndarray] = [
            np.load(ip, mmap_mode="r") for ip in self.index_paths
        ]
        self.counts = np.asarray([len(o) for o in self.offsets], dtype=np.int64)
        self.cumulative = np.concatenate([[0], np.cumsum(self.counts)])
        self.total = int(self.counts.sum())

    def __len__(self) -> int:
        return self.total

    def to_pick(self, global_id: int) -> Pick:
        """Map a global scene id in ``[0, total)`` to ``(shard_id, record_index)``."""
        s = int(np.searchsorted(self.cumulative, global_id, side="right") - 1)
        return s, int(global_id - self.cumulative[s])

    def read(self, pick: Pick) -> bytes:
        from rl.tfrecord_fast import read_tfrecord_bytes

        shard_id, rec = pick
        return read_tfrecord_bytes(
            self.paths[shard_id], rec, offsets=self.offsets[shard_id]
        )


# --------------------------------------------------------------------------- #
def _state_from_bytes(records: Sequence[bytes], ds_cfg) -> Any:
    """Batch serialized TFExamples into a Waymax ``SimulatorState``.

    Mirrors the tail of :func:`data.scenario_loader.load_scenario_state_batch_fast`
    (the same waymax calls in the same order), but takes bytes we already fetched
    by byte offset instead of scanning a dataset to find them. ``ds_cfg`` supplies
    only the parse dims (rg points, sdc path shape, aggregation).

    ``ds_cfg.max_num_objects`` must stay ``None`` here: ``preprocess_womd_example``
    slices ``v[:max_num_objects]`` on the *leading* axis, which on a batched parse
    is the batch axis, not objects. Raw WOMD's feature spec is a fixed 128 objects
    regardless, so nothing is lost.
    """
    # Import via data.scenario_loader, not waymax directly: that module hides the
    # GPU from TensorFlow on import, and without it TF preallocates most of the
    # device and JAX later OOMs somewhere unrelated (waymax's offroad metric).
    import data.scenario_loader  # noqa: F401  (side effect: TF sees no GPU)
    import tensorflow as tf
    from waymax import dataloader
    from waymax.dataloader import womd_factories

    serialized = tf.stack([tf.constant(r, dtype=tf.string) for r in records], axis=0)
    processed = dataloader.preprocess_serialized_womd_data(serialized, ds_cfg)
    return womd_factories.simulator_state_from_womd_dict(
        processed, include_sdc_paths=ds_cfg.include_sdc_paths
    )


def build_ds_cfg(*, batch_dims: tuple[int, ...], include_sdc_paths: bool,
                 num_paths: int, num_points_per_path: int,
                 max_num_rg_points: int = 30000, path: str = ""):
    """The parse-side ``DatasetConfig`` used for both sampling and fixed loading."""
    import dataclasses

    from waymax import config as waymax_config

    return dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING, path=str(path), max_num_objects=None,
        max_num_rg_points=int(max_num_rg_points), batch_dims=batch_dims, shuffle_seed=0,
        include_sdc_paths=bool(include_sdc_paths),
        num_paths=int(num_paths), num_points_per_path=int(num_points_per_path),
    )


# --------------------------------------------------------------------------- #
class SceneSampler:
    """Draws fresh training batches (and one fixed eval batch) from the full corpus.

    The eval scenes are drawn first, from the same uniform distribution, and then
    *excluded* from every training draw -- so the eval number is held-out and the
    training stream never memorizes it.

    ``prefetch=True`` loads the next training batch on a background thread while
    the current one trains. Parsing is TF/CPU and training is GPU, so this hides
    essentially all of the load cost; set it off to measure the raw cost.
    """

    def __init__(self, shard_glob: str, *, batch_size: int, eval_size: int = 0,
                 include_sdc_paths: bool = False, num_paths: int = 45,
                 num_points_per_path: int = 800, max_num_rg_points: int = 30000,
                 seed: int = 0, prefetch: bool = True,
                 index_dir: str | None = DEFAULT_INDEX_DIR):
        self.index = ShardIndex(shard_glob, index_dir=index_dir)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(int(seed))
        self._cfg_kwargs = dict(
            include_sdc_paths=include_sdc_paths, num_paths=num_paths,
            num_points_per_path=num_points_per_path, max_num_rg_points=max_num_rg_points,
        )
        self._train_cfg = build_ds_cfg(batch_dims=(self.batch_size,), **self._cfg_kwargs)

        # Held-out eval ids, drawn once and removed from the training pool.
        self.eval_ids: np.ndarray = (
            self.rng.choice(self.index.total, size=int(eval_size), replace=False)
            if eval_size > 0 else np.zeros(0, dtype=np.int64)
        )
        self._eval_set = set(int(i) for i in self.eval_ids.tolist())
        self._eval_cfg = (
            build_ds_cfg(batch_dims=(int(eval_size),), **self._cfg_kwargs)
            if eval_size > 0 else None
        )

        self.last_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.n_loaded = 0
        self.load_seconds = 0.0
        self._n_obj: int | None = None

        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        if prefetch:
            self._q = queue.Queue(maxsize=1)
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ #
    def _draw_ids(self, n: int) -> np.ndarray:
        """``n`` distinct global scene ids, none of them held-out eval scenes."""
        out: list[int] = []
        while len(out) < n:
            cand = self.rng.integers(0, self.index.total, size=2 * (n - len(out)) + 8)
            for c in cand.tolist():
                if c not in self._eval_set and c not in out:
                    out.append(int(c))
                    if len(out) == n:
                        break
        return np.asarray(out, dtype=np.int64)

    def _load_ids(self, ids: Iterable[int], ds_cfg) -> Any:
        t0 = time.time()
        picks = [self.index.to_pick(int(i)) for i in ids]
        records = [self.index.read(p) for p in picks]
        state = _state_from_bytes(records, ds_cfg)
        self.load_seconds += time.time() - t0
        return state

    def _prefetch_loop(self) -> None:
        while True:
            ids = self._draw_ids(self.batch_size)
            try:
                state = self._load_ids(ids, self._train_cfg)
            except Exception as exc:  # surfaced on the consumer side, not swallowed
                self._q.put(("error", exc, ids))
                return
            self._q.put(("ok", state, ids))

    # ------------------------------------------------------------------ #
    def next_train(self) -> Any:
        """A fresh batch of ``batch_size`` scenes as a batched Waymax state."""
        if self._q is not None:
            kind, payload, ids = self._q.get()
            if kind == "error":
                raise RuntimeError(
                    f"scene prefetch failed on ids {ids.tolist()}"
                ) from payload
            state, self.last_ids = payload, ids
        else:
            self.last_ids = self._draw_ids(self.batch_size)
            state = self._load_ids(self.last_ids, self._train_cfg)
        self.n_loaded += self.batch_size
        # A changed object count would silently rebuild the env AND the actor,
        # throwing away the fine-tuned weights and the Adam state mid-run.
        n_obj = int(np.asarray(state.log_trajectory.x).shape[1])
        if self._n_obj is None:
            self._n_obj = n_obj
        elif n_obj != self._n_obj:
            raise RuntimeError(
                f"batch object count changed {self._n_obj} -> {n_obj}; the env/actor "
                f"would be rebuilt and the trained weights lost.")
        return state

    def eval_batch(self) -> Any:
        """The fixed held-out eval batch (loaded once, then cached)."""
        if self._eval_cfg is None:
            raise RuntimeError("SceneSampler was built with eval_size=0")
        if not hasattr(self, "_eval_state"):
            self._eval_state = self._load_ids(self.eval_ids, self._eval_cfg)
        return self._eval_state

    def describe(self) -> str:
        return (f"{len(self.index.paths)} shards, {self.index.total} scenes, "
                f"train batch {self.batch_size}, held-out eval {len(self.eval_ids)}")
