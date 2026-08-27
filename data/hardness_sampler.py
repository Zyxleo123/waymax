"""Risk-weighted example sampling for diffusion training.

Turns the per-scenario failure risk from ``train_diffusion/hardness/risk_model.py``
into sampling weights with a single knob:

    w(x) = clip( (risk(x) / mean_risk) ** alpha, 1/clip, clip )

``alpha=0`` reproduces the unweighted run exactly -- not approximately: the
sampler takes a plain permutation in that case rather than a uniform draw with
replacement, so the alpha=0 arm of a sweep is a true control.  ``alpha=1``
samples in proportion to predicted failure risk.

Resampling rather than loss weighting is deliberate.  A shard is fully
preprocessed in memory before any batch is emitted, so drawing indices by weight
is free, whereas a loss weight leaves the same easy examples occupying the batch
and just shrinks the effective batch size.

Scenes flagged ``dirty`` (the logged expert's own bounding box leaves the
drivable area) get weight 0 -- excluded, never upweighted.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np


def shard_name_from_path(cache_path: str) -> str:
    return os.path.basename(str(cache_path)).removesuffix(".sim_state_cache.npz")


class RiskTable:
    """Per-(scenario, anchor) sampling weights, loaded once per process."""

    def __init__(self, path: str, alpha: float = 0.0, weight_clip: float = 10.0,
                 exclude_dirty: bool = True):
        self.alpha = float(alpha)
        self.weight_clip = float(weight_clip)
        self.exclude_dirty = bool(exclude_dirty)

        with np.load(path, allow_pickle=True) as handle:
            shards = handle["shard"].astype(str)
            scenario_idx = handle["scenario_idx"].astype(np.int64)
            anchor_risk = handle["anchor_risk"].astype(np.float64)
            self.anchors = handle["anchors"].astype(np.int32).tolist()
            dirty = handle["dirty"].astype(bool)
            self.mean_risk = float(handle["mean_risk"])
            # Anchor windows cover 5s of a ~8s episode, so their risk runs about
            # half the scenario-level value. Normalising anchor weights by the
            # scenario-level mean cancels out of the sampling probability but
            # NOT out of the clip, and would pin ~33% of examples to the floor at
            # alpha=1 instead of ~23%. Older tables lack the key.
            self.mean_anchor_risk = (
                float(handle["mean_anchor_risk"]) if "mean_anchor_risk" in handle.files
                else self.mean_risk
            )

        # shard -> (scenario_idx -> row), so a block lookup is one dict hit plus
        # a vectorised gather rather than a scan over 2.3M rows.
        self._rows: dict[str, dict[int, int]] = {}
        for row, (shard, scenario) in enumerate(zip(shards, scenario_idx)):
            self._rows.setdefault(str(shard), {})[int(scenario)] = row
        self._anchor_risk = anchor_risk
        self._dirty = dirty

    def anchor_slot(self, anchor_step: int) -> int | None:
        try:
            return self.anchors.index(int(anchor_step))
        except ValueError:
            return None

    def weights(self, cache_path: str, scenario_indices: np.ndarray, anchor_step: int) -> np.ndarray:
        """Sampling weight per example; 1.0 where the table has no entry."""
        shard = shard_name_from_path(cache_path)
        lookup = self._rows.get(shard)
        n = len(scenario_indices)
        weights = np.ones(n, dtype=np.float64)
        if lookup is None:
            return weights
        slot = self.anchor_slot(anchor_step)
        if slot is None:
            return weights

        rows = np.array([lookup.get(int(s), -1) for s in scenario_indices], dtype=np.int64)
        known = rows >= 0
        if not known.any():
            return weights

        risk = self._anchor_risk[rows[known], slot]
        scaled = risk / max(self.mean_anchor_risk, 1e-9)
        with np.errstate(over="ignore"):
            w = np.power(np.maximum(scaled, 1e-9), self.alpha)
        weights[known] = np.clip(w, 1.0 / self.weight_clip, self.weight_clip)
        if self.exclude_dirty:
            weights[known] = np.where(self._dirty[rows[known]], 0.0, weights[known])
        return weights


@lru_cache(maxsize=4)
def load_risk_table(path: str, alpha: float, weight_clip: float, exclude_dirty: bool) -> RiskTable:
    """Cached so each dataloader worker pays the load once, not once per shard."""
    return RiskTable(path, alpha=alpha, weight_clip=weight_clip, exclude_dirty=exclude_dirty)


def draw_order(weights: np.ndarray, rng: np.random.Generator, alpha: float) -> np.ndarray:
    """Example order for one block.

    Exclusion (weight 0) and reweighting (alpha) are independent: dropping dirty
    scenes has to work at alpha=0 too, since "clean the data, change nothing
    else" is its own arm of the sweep -- and given that 22.6% of policy failures
    sit on scenes the logged expert also fails, arguably the important one.

    With nothing excluded, alpha=0 is a plain permutation, so that control arm is
    exactly the unweighted run rather than a same-in-expectation approximation.
    """
    n = weights.size
    eligible = np.flatnonzero(weights > 0.0)
    if eligible.size == 0:
        return rng.permutation(n)
    if alpha == 0.0:
        return eligible[rng.permutation(eligible.size)]
    kept = weights[eligible]
    total = kept.sum()
    if not np.isfinite(total) or total <= 0.0:
        return eligible[rng.permutation(eligible.size)]
    return eligible[rng.choice(eligible.size, size=eligible.size, replace=True, p=kept / total)]
