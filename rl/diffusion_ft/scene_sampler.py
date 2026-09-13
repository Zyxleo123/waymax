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
                 index_dir: str | None = DEFAULT_INDEX_DIR,
                 scene_ids: Sequence[int] | None = None,
                 scene_groups: Sequence[Sequence[int]] | None = None,
                 group_weights: Sequence[float] | None = None):
        self.index = ShardIndex(shard_glob, index_dir=index_dir)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(int(seed))

        # Optional POOL restriction: draw only from these global scene ids instead
        # of all `index.total`. This is how a *selected* corpus (e.g. the
        # headroom-scanned hard-WOMD pool from tools/select_hard_womd.py) is
        # trained on without materializing a separate tfrecord -- the ids address
        # the same shards through the same byte-offset index.
        #
        # Uniform sampling over the whole corpus is the default and is unchanged.
        # A pool is a real distribution shift, so it is stated in describe() and
        # must be a deliberate flag, never a silent default.
        # Optional BALANCED sampling: `scene_groups` are the pool's disjoint
        # parts (failures / successes) and `group_weights` the share of every
        # training batch each one gets. Without it a union pool is sampled
        # uniformly, so a 400-scene failure group inside a 23k-scene success pool
        # is 1.7% of the gradient -- present in the pool and absent from the
        # training signal. Weights are a per-batch quota, not a probability: the
        # composition of every batch is then identical, which is what makes the
        # failure/success trade-off in the curve attributable.
        self.groups: list[np.ndarray] | None = None
        self.group_weights: np.ndarray | None = None
        if scene_groups is not None:
            groups = [np.unique(np.asarray(list(g), dtype=np.int64))
                      for g in scene_groups]
            if any(g.size == 0 for g in groups):
                raise ValueError("scene_groups contains an empty group")
            w = (np.ones(len(groups), dtype=np.float64) if group_weights is None
                 else np.asarray(list(group_weights), dtype=np.float64))
            if w.size != len(groups):
                raise ValueError(
                    f"group_weights has {w.size} values for {len(groups)} groups")
            if not np.all(w >= 0) or w.sum() <= 0:
                raise ValueError("group_weights must be non-negative and not all zero")
            self.groups = groups
            self.group_weights = w / w.sum()
            if scene_ids is None:
                scene_ids = np.concatenate(groups)

        self.pool: np.ndarray | None = None
        if scene_ids is not None:
            pool = np.unique(np.asarray(list(scene_ids), dtype=np.int64))
            if pool.size == 0:
                raise ValueError("scene_ids was given but empty")
            if pool.min() < 0 or pool.max() >= self.index.total:
                raise ValueError(
                    f"scene_ids out of range for this shard set: "
                    f"[{int(pool.min())}, {int(pool.max())}] vs total {self.index.total}. "
                    "The ids are positions in THIS glob's concatenated shard order, "
                    "so they only mean anything against the glob they were scanned "
                    "with.")
            need = int(batch_size) + int(eval_size)
            if pool.size < need:
                raise ValueError(
                    f"pool has {pool.size} scenes but batch_size+eval_size = {need}; "
                    "training draws distinct scenes and would loop forever.")
            self.pool = pool
        self._cfg_kwargs = dict(
            include_sdc_paths=include_sdc_paths, num_paths=num_paths,
            num_points_per_path=num_points_per_path, max_num_rg_points=max_num_rg_points,
        )
        self._train_cfg = build_ds_cfg(batch_dims=(self.batch_size,), **self._cfg_kwargs)

        # Held-out eval ids, drawn once and removed from the training pool. Drawn
        # from the SAME distribution the training stream sees -- the pool when one
        # is set -- so the eval number stays a held-out read of the task being
        # trained on rather than of an easier corpus.
        _source = self.pool if self.pool is not None else self.index.total
        self.eval_ids: np.ndarray = (
            self.rng.choice(_source, size=int(eval_size), replace=False)
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
    def _group_quota(self, n: int) -> list[int]:
        """Split a batch of ``n`` across the groups by ``group_weights``.

        Largest-remainder, so the quotas sum to exactly ``n`` and a group whose
        share rounds below 1 still gets a scene whenever it is the largest
        remainder -- rounding each share independently would silently drop the
        minority group at small batch sizes.
        """
        w = self.group_weights
        exact = w * n
        quota = np.floor(exact).astype(np.int64)
        short = int(n - quota.sum())
        if short > 0:
            order = np.argsort(-(exact - quota))
            quota[order[:short]] += 1
        return [int(q) for q in quota]

    def _draw_from(self, source, n: int, taken: set[int]) -> list[int]:
        """``n`` distinct ids from ``source``, skipping ``taken`` and eval ids."""
        out: list[int] = []
        if n <= 0:
            return out
        if isinstance(source, np.ndarray):
            # A group smaller than its own quota makes the draw loop forever
            # (distinct ids, none held out), which on a compute node looks like a
            # hung job with no error. Fail at the first batch instead.
            avail = int(np.isin(source, list(self._eval_set), invert=True).sum())
            if avail < n:
                raise ValueError(
                    f"a scene group has {avail} drawable scenes (after held-out "
                    f"eval ids) but its per-batch quota is {n}; lower "
                    f"--scenes_per_iter, --eval_scenes, or the group's weight")
        while len(out) < n:
            k = 2 * (n - len(out)) + 8
            cand = (self.rng.choice(source, size=k) if isinstance(source, np.ndarray)
                    else self.rng.integers(0, source, size=k))
            for c in cand.tolist():
                c = int(c)
                if c in self._eval_set or c in taken:
                    continue
                taken.add(c)
                out.append(c)
                if len(out) == n:
                    break
        return out

    def _draw_ids(self, n: int) -> np.ndarray:
        """``n`` distinct global scene ids, none of them held-out eval scenes.

        With ``scene_groups`` set, every batch carries each group's quota exactly;
        otherwise the pool (or the whole corpus) is sampled uniformly.
        """
        if self.groups is not None:
            taken: set[int] = set()
            out: list[int] = []
            for g, q in zip(self.groups, self._group_quota(n)):
                out.extend(self._draw_from(g, q, taken))
            self.rng.shuffle(out)
            return np.asarray(out, dtype=np.int64)
        out: list[int] = []
        while len(out) < n:
            if self.pool is not None:
                cand = self.rng.choice(self.pool, size=2 * (n - len(out)) + 8)
            else:
                cand = self.rng.integers(0, self.index.total, size=2 * (n - len(out)) + 8)
            for c in cand.tolist():
                if c not in self._eval_set and c not in out:
                    out.append(int(c))
                    if len(out) == n:
                        break
        return np.asarray(out, dtype=np.int64)

    def group_labels(self, ids: Sequence[int]) -> np.ndarray:
        """Per-id index into ``scene_groups`` (-1 for an id in no group).

        Derived from the ids rather than recorded at draw time on purpose: the
        groups are disjoint id-sets (the trainer refuses overlapping
        ``--train_splits``), so membership is recoverable from the batch alone.
        That keeps ``_draw_ids``'s shuffle and the prefetch queue's (state, ids)
        protocol untouched -- a labels array threaded through the queue would have
        to stay aligned with the shuffle across a thread boundary, which is a
        silent-mislabelling bug waiting to happen (see the ES runner, which
        misnamed scenes for exactly that reason).
        """
        arr = np.asarray(list(ids), dtype=np.int64)
        out = np.full(arr.shape, -1, dtype=np.int64)
        if self.groups is None:
            return out
        for gi, g in enumerate(self.groups):
            out[np.isin(arr, g)] = gi
        return out

    def load_explicit(self, ids: Sequence[int]) -> Any:
        """Load EXACTLY these global scene ids, in order, as one batched state.

        For eval on a fixed, named split (failure / success held-out) rather than
        a random training draw: the ids are positions in this glob's concatenated
        shard order, so they mean the same scenes the scan labelled. Builds a
        one-off ds_cfg sized to the request so any split length is loadable; the
        object/roadgraph dims match the train/eval cfg (same ``_cfg_kwargs``), so
        the env's ``max_num_objects`` still fits.
        """
        ids = [int(i) for i in ids]
        if not ids:
            raise ValueError("load_explicit got an empty id list")
        cfg = build_ds_cfg(batch_dims=(len(ids),), **self._cfg_kwargs)
        return self._load_ids(ids, cfg)

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
        scope = (f"{self.index.total} scenes" if self.pool is None
                 else f"POOL {self.pool.size} of {self.index.total} scenes")
        bal = ""
        if self.groups is not None:
            bal = ("  balanced " + "/".join(
                f"{q}of{g.size}" for g, q in zip(
                    self.groups, self._group_quota(self.batch_size))))
        return (f"{len(self.index.paths)} shards, {scope}, "
                f"train batch {self.batch_size}, held-out eval {len(self.eval_ids)}"
                f"{bal}")
