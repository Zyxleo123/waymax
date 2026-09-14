"""A replay buffer of individual reverse-diffusion transitions.

What this is for
----------------
The analytic safety fine-tune (:mod:`rl.diffusion_ft.train_safety`) has two ways
to produce the trajectory it scores, and both pay for what they get:

* **Free-running** (``--unroll_steps 0``). Denoise from pure noise through the
  whole 100-step chain at the policy's own visited state, score the endpoint.
  Faithful -- the scored object is the deployed sampler's output from the
  deployed sampler's states -- and expensive: 100 network evaluations per
  gradient, 10 of which carry a gradient.
* **Teacher-forced** (``--unroll_steps K``, :mod:`rl.diffusion_ft.unrolled_prediction`).
  Corrupt the human's future to a low-noise timestep, unroll ``K`` steps, score
  the clean prediction. Cheap -- ``K`` evaluations -- and mismatched: the state
  handed to the denoiser is ``q_sample(x0_expert, t)``, a noised *human*
  trajectory, which is not the distribution the sampler's own ``x_t`` is drawn
  from at that timestep. The module docstring there names this "diffusion-state
  mismatch" as the thing teacher forcing does not fix.

This module removes the mismatch without paying the 100 evaluations, by noticing
that the collection rollout **already computes the states teacher forcing is
approximating**. ``sample_with_trace`` records ``(x_t, t)`` for each of the
``k_trainable`` late steps on its way to the trajectory the env executes; the
safety path has been discarding that trace (only DPPO consumed it). Keeping it
costs one device-to-host copy and no extra denoiser work, and each record is a
genuine sampler state at a known low-noise timestep, tagged with the scene and
the replan it came from.

A stored transition is therefore a *complete* training datum::

    (scene_id, t0, x_t, t, cond1, cond2, cond2_mask, aux)

``(cond1, cond2, cond2_mask)`` is the frozen conditioning at that replan, ``aux``
the normalized->world transform of that replan, and ``(scene_id, t0)`` is enough
to recover the geometry: ``candidate_safety`` reads only ``log_trajectory``,
``object_metadata`` and ``roadgraph_points``, all constant for a whole episode,
so the scene can simply be reloaded by id. Nothing in a datum depends on the
weights, which is what makes it replayable after the policy has moved.

What replay costs, stated plainly
---------------------------------
``x_t`` was produced by an **older** policy, so the state distribution is stale
by however long the datum has been in the buffer -- this is off-policy, and the
objective has no importance ratio to correct it (there is nothing to correct: the
gradient is analytic, not a score-function estimate; staleness biases *where* the
loss is evaluated, not the gradient at that point). ``--tb_max_uses`` and the
buffer capacity are the knobs that bound the age. Setting capacity == batch size
degenerates to on-policy.

Storage is host (numpy) memory, never device: a ring of ``capacity`` slots, FIFO
eviction, each slot self-contained. :meth:`TransitionBuffer.nbytes` reports the
true footprint, which is dominated by ``cond1`` (``[32, 256]`` scene tokens =
32 KB) rather than by ``x_t``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TransitionBuffer"]

# Fields every datum carries besides the `aux` pytree leaves. `scene_id`, `t0`
# and `t` are bookkeeping the loss needs (which scene, which replan, which noise
# level); the rest are the denoiser's inputs.
_CORE_FIELDS = ("scene_id", "t0", "t", "x_t", "cond1", "cond2", "cond2_mask")


class TransitionBuffer:
    """Fixed-capacity FIFO ring of per-row diffusion transitions.

    One slot is ONE transition of ONE scene -- not a batched replan context. Rows
    of the same collection batch are stored independently, so a replayed
    minibatch mixes scenes, replan boundaries (``t0``) and noise levels (``t``),
    which is what the per-row ``t0`` path in
    :func:`rl.diffusion_ft.safety_objective.candidate_safety` exists to serve.

    Args:
      capacity: slots. Eviction is FIFO on the write head.
      max_uses: draws a slot may serve before it is dropped (0 = unlimited). A
        low value bounds how often one stale state can steer the update; it does
        not bound the age of what remains, which capacity does.
      seed: the sampling stream. Kept inside the buffer so a resumed run's draws
        do not depend on how much training happened before the resume.
    """

    def __init__(self, capacity: int, *, max_uses: int = 0, seed: int = 0):
        if int(capacity) < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self.max_uses = int(max_uses)
        self._rng = np.random.default_rng(int(seed))
        self._store: dict[str, np.ndarray] = {}
        self._aux_keys: tuple[str, ...] = ()
        self._valid = np.zeros(self.capacity, bool)
        self._uses = np.zeros(self.capacity, np.int32)
        self._born = np.zeros(self.capacity, np.int64)   # iteration written
        self._head = 0
        self.n_added = 0
        self.n_evicted = 0
        self.n_drawn = 0

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self._valid.sum())

    @property
    def ready(self) -> bool:
        return len(self) > 0

    @property
    def nbytes(self) -> int:
        """Host bytes held, including the unused tail of the ring."""
        return int(sum(a.nbytes for a in self._store.values()))

    @property
    def bytes_per_slot(self) -> float:
        return self.nbytes / self.capacity if self._store else 0.0

    # ------------------------------------------------------------------ #
    def _allocate(self, rows: dict[str, np.ndarray]) -> None:
        """Size the ring from the first batch's shapes (one slot per row)."""
        self._aux_keys = tuple(sorted(k for k in rows if k.startswith("aux/")))
        for k, v in rows.items():
            a = np.asarray(v)
            self._store[k] = np.zeros((self.capacity,) + a.shape[1:], dtype=a.dtype)

    def add(self, rows: dict[str, np.ndarray], *, iteration: int = 0) -> int:
        """Append every row of ``rows`` (leading axis = rows) to the ring.

        Each value must have the same leading length. Returns rows written. A
        batch larger than the whole ring keeps its LAST ``capacity`` rows, since
        those are the ones that would have survived the overwrites anyway.
        """
        if not rows:
            return 0
        for f in _CORE_FIELDS:
            if f not in rows:
                raise KeyError(f"transition rows are missing '{f}'")
        n = int(np.asarray(rows["t"]).shape[0])
        if n == 0:
            return 0
        if any(int(np.asarray(v).shape[0]) != n for v in rows.values()):
            bad = {k: np.asarray(v).shape for k, v in rows.items()}
            raise ValueError(f"ragged transition batch: {bad}")
        if not self._store:
            self._allocate(rows)
        elif set(rows) != set(self._store):
            raise KeyError(
                f"transition fields changed: had {sorted(self._store)}, got {sorted(rows)}")

        if n > self.capacity:
            rows = {k: np.asarray(v)[-self.capacity:] for k, v in rows.items()}
            n = self.capacity

        idx = (self._head + np.arange(n)) % self.capacity
        self.n_evicted += int(self._valid[idx].sum())
        for k, dst in self._store.items():
            dst[idx] = np.asarray(rows[k], dtype=dst.dtype)
        self._valid[idx] = True
        self._uses[idx] = 0
        self._born[idx] = int(iteration)
        self._head = int((self._head + n) % self.capacity)
        self.n_added += n
        return n

    # ------------------------------------------------------------------ #
    def sample(self, n: int, *, iteration: int = 0) -> dict[str, np.ndarray] | None:
        """Draw ``n`` transitions without replacement, or None if too few exist.

        Charges a use against each drawn slot and evicts those past
        ``max_uses``. Also returns ``age`` (iterations since the slot was
        written), which is the number to watch: it is how stale the states the
        gradient is being taken at have become.
        """
        n = int(n)
        live = np.flatnonzero(self._valid)
        if live.size < n:
            return None
        pick = self._rng.choice(live, size=n, replace=False)
        out = {k: np.array(v[pick]) for k, v in self._store.items()}
        out["age"] = (int(iteration) - self._born[pick]).astype(np.int64)
        self._uses[pick] += 1
        self.n_drawn += n
        if self.max_uses > 0:
            spent = pick[self._uses[pick] >= self.max_uses]
            self._valid[spent] = False
            self.n_evicted += int(spent.size)
        return out

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, float]:
        live = np.flatnonzero(self._valid)
        if live.size == 0 or not self._store:
            return {"tb_size": 0.0, "tb_mb": 0.0, "tb_added": float(self.n_added),
                    "tb_evicted": float(self.n_evicted), "tb_drawn": float(self.n_drawn)}
        return {
            "tb_size": float(live.size),
            "tb_mb": self.nbytes / 1e6,
            "tb_added": float(self.n_added),
            "tb_evicted": float(self.n_evicted),
            "tb_drawn": float(self.n_drawn),
            "tb_mean_t": float(np.mean(self._store["t"][live])),
            "tb_mean_uses": float(np.mean(self._uses[live])),
            "tb_scenes": float(np.unique(self._store["scene_id"][live]).size),
        }
