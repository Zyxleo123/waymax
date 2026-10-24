"""Planning-facing multimodal packing and independent-agent collision risk.

Training assigns best-mode independently per agent (winner-take-all ADE), so
mode index k is **not** a scene-level joint future. Collision risk therefore
treats modes as agent-wise mixtures:

  p_collision_i = sum_k p_ik * 1{ego collides with agent i under mode k}
  collision_score = prod_i (1 - p_collision_i)   # independence across agents

Planning modes
--------------
  top1                     — argmax mode per agent; hard SAT (0/1)
  expected_independent     — probability-weighted risk (above)
  worst_case_independent   — unsafe if any mode of any agent collides

Legacy aliases ``expected`` / ``worst_case`` map to the independent variants
(do **not** build fake joint worlds by taking the same k across agents).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import PLANNING_MODES, canonicalize_planning_mode

# Reuse production SAT; multifuture is always launched with waymax_rs on PYTHONPATH.
from scores.polygon import pairwise_intersections, polygon_corners


def pack_collision_worlds(
    primary_traj: np.ndarray,
    primary_mask: np.ndarray,
    traj_k: np.ndarray,
    mask_k: np.ndarray | None,
    *,
    planning_mode: str = "top1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack primary trajectories for the ES interface.

    Independent-agent risk scoring does not use joint mode worlds. For all
    supported planning modes the collision tensors match the primary (top-1)
    forecast; soft / worst-case risk is computed in
    ``score_independent_agent_collision``.
    """
    mode = canonicalize_planning_mode(planning_mode)
    if mode not in PLANNING_MODES:
        raise ValueError(f"planning_mode must be one of {PLANNING_MODES}, got {planning_mode!r}")
    # Keep signature / traj_k validation for callers that still pass K futures.
    if traj_k.ndim != 4:
        raise ValueError(f"traj_k must be [N, K, T, 5], got {traj_k.shape}")
    return primary_traj, primary_mask, primary_traj, primary_mask


def planning_outputs_from_model(
    traj_k: np.ndarray,
    mask: np.ndarray,
    mode_probs: np.ndarray,
    *,
    planning_mode: str = "top1",
) -> dict[str, Any]:
    """Bundle multimodal outputs for planning / debugging."""
    mode = canonicalize_planning_mode(planning_mode)
    probs = _normalize_mode_probs(np.asarray(mode_probs, dtype=np.float32))
    top1 = np.argmax(probs, axis=-1)
    n = traj_k.shape[0]
    primary = traj_k[np.arange(n), top1]
    primary_traj, primary_mask, col_traj, col_mask = pack_collision_worlds(
        primary,
        mask,
        traj_k,
        None,
        planning_mode=mode,
    )
    return {
        "planning_mode": mode,
        "primary_traj": primary_traj,
        "primary_mask": primary_mask,
        "collision_traj": col_traj,
        "collision_mask": col_mask,
        "traj_k": traj_k,
        "mode_probs": probs,
        "top1_mode": top1,
    }


def _normalize_mode_probs(mode_probs: np.ndarray) -> np.ndarray:
    """Ensure per-agent probs sum to 1 when the agent is active.

    Accepts softmax probs or raw logits (detected when any row has negatives or
    fails to sum ~1).
    """
    p = np.asarray(mode_probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"mode_probs must be [N, K], got {p.shape}")
    row_sum = p.sum(axis=-1, keepdims=True)
    looks_like_logits = bool(np.any(p < -1e-6)) or bool(
        np.any((row_sum > 1e-8) & (np.abs(row_sum - 1.0) > 0.05))
    )
    if looks_like_logits:
        # Stable softmax
        z = p - p.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True).clip(min=1e-12)
        row_sum = p.sum(axis=-1, keepdims=True)
    out = np.where(row_sum > 1e-8, p / row_sum.clip(min=1e-12), 0.0)
    return out.astype(np.float32)


def ego_agent_mode_collisions(
    ego_trajectories: np.ndarray | jnp.ndarray,
    traj_k: np.ndarray | jnp.ndarray,
    agent_masks: np.ndarray | jnp.ndarray,
) -> np.ndarray:
    """Hard SAT collisions for every (ego, agent, mode).

    Parameters
    ----------
    ego_trajectories: [E, T, 5]
    traj_k: [N, K, T, 5]
    agent_masks: [N, T] (True = agent present); broadcast across modes.

    Returns
    -------
    collision_eik: bool array [E, N, K] — True if any-timestep box overlap.
    """
    ego = jnp.asarray(ego_trajectories, dtype=jnp.float32)
    traj = jnp.asarray(traj_k, dtype=jnp.float32)
    masks = jnp.asarray(agent_masks).astype(jnp.bool_)
    if ego.ndim != 3 or ego.shape[-1] != 5:
        raise ValueError(f"ego_trajectories must be [E, T, 5], got {ego.shape}")
    if traj.ndim != 4 or traj.shape[-1] != 5:
        raise ValueError(f"traj_k must be [N, K, T, 5], got {traj.shape}")
    n, k, t, _ = traj.shape
    if ego.shape[1] != t:
        raise ValueError(f"ego T={ego.shape[1]} != traj_k T={t}")
    if masks.shape != (n, t):
        raise ValueError(f"agent_masks must be [N, T]=[{n}, {t}], got {masks.shape}")

    # Flatten modes on object axis (agent-major); reshape back after SAT.
    obj = traj.reshape((n * k, t, 5))
    ego_poly = polygon_corners(ego)
    obj_poly = polygon_corners(obj)

    def _time_pairwise(ego_t: jax.Array, obj_t: jax.Array) -> jax.Array:
        return pairwise_intersections(ego_t, obj_t)

    # [T, E, N*K]
    hits_t = jax.vmap(_time_pairwise, in_axes=(1, 1), out_axes=0)(ego_poly, obj_poly)
    valid_t = jnp.swapaxes(masks, 0, 1)  # [T, N]
    valid_t = jnp.repeat(valid_t, k, axis=1)  # [T, N*K]
    hits_t = hits_t & valid_t[:, None, :]
    hits = jnp.any(hits_t, axis=0)  # [E, N*K]
    return np.asarray(hits.reshape((-1, n, k)), dtype=bool)


def combine_independent_agent_collision(
    collision_eik: np.ndarray,
    mode_probs: np.ndarray,
    *,
    planning_mode: str = "expected_independent",
    agent_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine per-(ego,agent,mode) hard collisions into a soft/hard score.

    Unit-testable without SAT: pass explicit collision_eik and probs.
    """
    mode = canonicalize_planning_mode(planning_mode)
    coll = np.asarray(collision_eik, dtype=bool)
    if coll.ndim != 3:
        raise ValueError(f"collision_eik must be [E, N, K], got {coll.shape}")
    e, n, k = coll.shape
    probs = _normalize_mode_probs(np.asarray(mode_probs, dtype=np.float32))
    if probs.shape != (n, k):
        raise ValueError(f"mode_probs must be [N, K]=[{n}, {k}], got {probs.shape}")

    if agent_valid is None:
        agent_valid = np.ones((n,), dtype=bool)
    else:
        agent_valid = np.asarray(agent_valid, dtype=bool)
        if agent_valid.shape != (n,):
            raise ValueError(f"agent_valid must be [N]={n}, got {agent_valid.shape}")

    # Inactive agents contribute factor 1 (no risk).
    active = agent_valid.astype(np.float32)  # [N]

    if mode == "top1":
        top1 = np.argmax(probs, axis=-1)  # [N] independent per agent
        coll_top1 = coll[:, np.arange(n), top1]  # [E, N]
        coll_top1 = coll_top1 & agent_valid[None, :]
        unsafe = coll_top1.any(axis=-1)
        score = np.where(unsafe, 0.0, 1.0).astype(np.float32)
        p_collision_i = coll_top1.astype(np.float32)
        debug = {
            "planning_mode": mode,
            "p_collision_i": p_collision_i,
            "max_agent_collision_prob": p_collision_i.max(axis=-1) if n else np.zeros((e,), np.float32),
            "argmax_risk_agent": np.argmax(p_collision_i, axis=-1) if n else np.zeros((e,), np.int32),
            "collision_score": score,
            "top1_collision": unsafe,
            "top1_mode": top1,
            "num_colliding_modes": coll.astype(np.int32).sum(axis=-1).sum(axis=-1),
        }
        return score, debug

    if mode == "worst_case_independent":
        # Any mode of any active agent.
        coll_active = coll & agent_valid[None, :, None]
        unsafe = coll_active.any(axis=(1, 2))
        score = np.where(unsafe, 0.0, 1.0).astype(np.float32)
        # Diagnostic: per-agent max over modes (0/1)
        p_collision_i = coll_active.any(axis=-1).astype(np.float32)
        debug = {
            "planning_mode": mode,
            "p_collision_i": p_collision_i,
            "max_agent_collision_prob": p_collision_i.max(axis=-1) if n else np.zeros((e,), np.float32),
            "argmax_risk_agent": np.argmax(p_collision_i, axis=-1) if n else np.zeros((e,), np.int32),
            "collision_score": score,
            "top1_collision": coll[:, np.arange(n), np.argmax(probs, axis=-1)].any(axis=-1)
            if n
            else np.zeros((e,), dtype=bool),
            "num_colliding_modes": coll_active.astype(np.int32).sum(axis=-1).sum(axis=-1),
        }
        return score, debug

    if mode == "expected_independent":
        # p_collision_i[e, i] = sum_k p_ik * collision_eik
        p_collision_i = (coll.astype(np.float32) * probs[None, :, :]).sum(axis=-1)  # [E, N]
        p_collision_i = p_collision_i * active[None, :]
        # p_safe = prod_i (1 - p_i); empty product = 1
        score = np.prod(1.0 - p_collision_i, axis=-1).astype(np.float32)
        top1 = np.argmax(probs, axis=-1)
        top1_hit = coll[:, np.arange(n), top1]
        top1_hit = (top1_hit & agent_valid[None, :]).any(axis=-1)
        debug = {
            "planning_mode": mode,
            "p_collision_i": p_collision_i.astype(np.float32),
            "max_agent_collision_prob": p_collision_i.max(axis=-1).astype(np.float32)
            if n
            else np.zeros((e,), np.float32),
            "argmax_risk_agent": np.argmax(p_collision_i, axis=-1).astype(np.int32)
            if n
            else np.zeros((e,), np.int32),
            "collision_score": score,
            "top1_collision": top1_hit,
            "num_colliding_modes": coll.astype(np.int32).sum(axis=-1).sum(axis=-1),
        }
        return score, debug

    raise ValueError(f"Unsupported planning_mode {mode!r}")


def score_independent_agent_collision(
    ego_trajectories: np.ndarray | jnp.ndarray,
    traj_k: np.ndarray | jnp.ndarray,
    agent_masks: np.ndarray | jnp.ndarray,
    mode_probs: np.ndarray,
    *,
    planning_mode: str = "expected_independent",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full independent-agent multimodal collision score for ES ego candidates."""
    coll = ego_agent_mode_collisions(ego_trajectories, traj_k, agent_masks)
    agent_valid = np.asarray(agent_masks).astype(bool).any(axis=-1)
    return combine_independent_agent_collision(
        coll,
        mode_probs,
        planning_mode=planning_mode,
        agent_valid=agent_valid,
    )


# ---------------------------------------------------------------------------
# MF collision-method ablation (opt-in via mf_collision_method; the planning
# modes above are untouched).
#
#   P(C_i | tau) = sum_k w_ik * c_ik(tau),   S(tau) = prod_i (1 - P(C_i | tau))
#
#   mean_uniform      c_ik = SAT hit on the mode mean,           w_ik = 1/K
#   mean_learned      c_ik = SAT hit on the mode mean,           w_ik = p_ik
#                     (== expected_independent)
#   gaussian_uniform  c_ik = fraction of M samples x ~ N(mu_ik, diag sigma_ik^2)
#                     that hit (same SAT),                       w_ik = 1/K
#   gaussian_learned  same c_ik as gaussian_uniform,             w_ik = p_ik
# ---------------------------------------------------------------------------

MF_COLLISION_METHODS = ("mean_uniform", "mean_learned", "gaussian_uniform", "gaussian_learned")
_MF_COLLISION_ALIASES = {"expected_independent": "mean_learned"}


def canonicalize_mf_collision_method(method: str) -> str:
    m = _MF_COLLISION_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    if m not in MF_COLLISION_METHODS:
        raise ValueError(
            f"mf_collision_method must be one of {MF_COLLISION_METHODS} "
            f"(alias: expected_independent), got {method!r}"
        )
    return m


def mf_uses_sampling(method: str) -> bool:
    return canonicalize_mf_collision_method(method).startswith("gaussian_")


def mf_mode_weights(method: str, mode_probs: np.ndarray) -> np.ndarray:
    """[N, K] weights: learned p_ik (normalized as in expected_independent) or exactly 1/K."""
    m = canonicalize_mf_collision_method(method)
    probs = _normalize_mode_probs(np.asarray(mode_probs, dtype=np.float32))
    if m.endswith("_learned"):
        return probs
    return np.full(probs.shape, 1.0 / probs.shape[-1], dtype=np.float32)


def combine_mf_collision(
    collision_eik: np.ndarray,
    weights: np.ndarray,
    agent_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """S = prod_i (1 - sum_k w_ik c_eik) over active agents.

    ``collision_eik`` may be hard (bool) or sampled (float in [0, 1]). Same
    arithmetic as the expected_independent branch, so mean_learned matches it
    bit-for-bit.
    """
    c = np.asarray(collision_eik).astype(np.float32)
    active = np.asarray(agent_valid, dtype=bool).astype(np.float32)
    p_collision_i = (c * np.asarray(weights, dtype=np.float32)[None, :, :]).sum(axis=-1)
    p_collision_i = p_collision_i * active[None, :]
    score = np.prod(1.0 - p_collision_i, axis=-1).astype(np.float32)
    return score, p_collision_i.astype(np.float32)


@jax.jit
def _any_timestep_hits(ego_poly: jax.Array, obj: jax.Array, valid_t: jax.Array) -> jax.Array:
    """Same SAT reduction as ego_agent_mode_collisions, jitted for fixed chunk shapes.

    ego_poly [E, T, 4, 2], obj [O, T, 5], valid_t [T, O] -> [E, O] any-timestep hit.
    """
    obj_poly = polygon_corners(obj)
    hits_t = jax.vmap(pairwise_intersections, in_axes=(1, 1), out_axes=0)(ego_poly, obj_poly)
    return jnp.any(hits_t & valid_t[:, None, :], axis=0)


def sampled_mode_collision_prob(
    ego_trajectories: np.ndarray,
    traj_k: np.ndarray,
    sample_noise_k: np.ndarray,
    agent_masks: np.ndarray,
    mean_collision_eik: np.ndarray,
    *,
    max_pairs_per_call: int = 2_500_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """c_hat[e, i, k] = mean over M samples of the SAT hit with sampled agent boxes.

    ego_trajectories [E, T, 5]; traj_k [N, K, T, 5] mode means; sample_noise_k
    [N, K, M, T, 2] xy offsets (world frame) added to the means; agent_masks [N, T].
    Agents with all-zero noise (static / unmodeled / invalid) have every sample
    equal to the mean, so they reuse ``mean_collision_eik`` instead of M SAT calls.

    Returns (c_hat [E, N, K] float32, sampled-agent mask [N], number of
    (ego, agent-sample) trajectory checks run).
    """
    ego = np.asarray(ego_trajectories, dtype=np.float32)
    traj = np.asarray(traj_k, dtype=np.float32)
    noise = np.asarray(sample_noise_k, dtype=np.float32)
    masks = np.asarray(agent_masks, dtype=bool)
    e, t = ego.shape[:2]
    n, k = traj.shape[:2]
    m = noise.shape[2]
    if noise.shape != (n, k, m, t, 2):
        raise ValueError(f"sample_noise_k must be [N, K, M, T, 2]=[{n}, {k}, M, {t}, 2], got {noise.shape}")

    c_hat = np.asarray(mean_collision_eik, dtype=np.float32).copy()
    sampled = masks.any(axis=-1) & np.any(noise != 0.0, axis=(1, 2, 3, 4))
    idx = np.nonzero(sampled)[0]
    if idx.size == 0:
        return c_hat, sampled, 0

    # Fixed agent-chunk size keeps jitted shapes constant across calls.
    chunk = max(1, int(max_pairs_per_call) // max(1, e * k * m * t))
    ego_poly = polygon_corners(jnp.asarray(ego))
    for start in range(0, idx.size, chunk):
        sel = idx[start : start + chunk]
        pad = np.concatenate([sel, np.full(chunk - sel.size, sel[0])]) if sel.size < chunk else sel
        obj = np.repeat(traj[pad][:, :, None], m, axis=2)  # [C, K, M, T, 5]
        obj[..., :2] += noise[pad]
        valid = np.repeat(masks[pad][:, None, None, :], k * m, axis=1).reshape(-1, t)  # [C*K*M, T]
        valid[sel.size * k * m :] = False
        hits = _any_timestep_hits(ego_poly, jnp.asarray(obj.reshape(-1, t, 5)), jnp.asarray(valid.T))
        hits = np.asarray(hits).reshape(e, chunk, k, m)[:, : sel.size]
        c_hat[:, sel] = hits.mean(axis=-1, dtype=np.float32)
    return c_hat, sampled, int(e * idx.size * k * m)


class MFCollisionDiagnostics:
    """Running aggregates for the gaussian_* methods (sampled agents only).

    c_hat takes values j/M, so a histogram over j gives exact quantiles.
    """

    def __init__(self, method: str, num_samples: int, num_modes: int) -> None:
        self.method = method
        self.m = int(num_samples)
        self.k = int(num_modes)
        self.n_predictions = 0
        self.n_score_calls = 0
        self.n_traj_checks = 0
        self.n_box_checks = 0
        self.hist = np.zeros(self.m + 1, dtype=np.int64)
        self.mode_sum = np.zeros(self.k, dtype=np.float64)
        self.mode_cnt = np.zeros(self.k, dtype=np.int64)
        self.p_agent_sum = 0.0
        self.p_agent_cnt = 0
        self.mean0 = self.mean0_hat_pos = 0
        self.mean1 = self.mean1_hat_lt1 = 0
        self.score_absdiff_sum = 0.0
        self.score_changed = 0
        self.n_candidates = 0

    def update(
        self,
        c_hat: np.ndarray,
        c_mean: np.ndarray,
        sampled: np.ndarray,
        p_collision_i: np.ndarray,
        score: np.ndarray,
        score_mean_method: np.ndarray,
        n_traj_checks: int,
        horizon: int,
    ) -> None:
        self.n_score_calls += 1
        self.n_traj_checks += int(n_traj_checks)
        self.n_box_checks += int(n_traj_checks) * int(horizon)
        self.n_candidates += int(score.shape[0])
        d = np.abs(np.asarray(score, np.float64) - np.asarray(score_mean_method, np.float64))
        self.score_absdiff_sum += float(d.sum())
        self.score_changed += int((d > 1e-6).sum())
        if not np.any(sampled):
            return
        ch = c_hat[:, sampled]  # [E, Ns, K]
        cm = c_mean[:, sampled]
        self.hist += np.bincount(np.rint(ch * self.m).astype(np.int64).ravel(), minlength=self.m + 1)[: self.m + 1]
        self.mode_sum += ch.sum(axis=(0, 1))
        self.mode_cnt += ch.shape[0] * ch.shape[1]
        p = p_collision_i[:, sampled]
        self.p_agent_sum += float(p.sum())
        self.p_agent_cnt += int(p.size)
        self.mean0 += int((~cm).sum())
        self.mean0_hat_pos += int(((~cm) & (ch > 0)).sum())
        self.mean1 += int(cm.sum())
        self.mean1_hat_lt1 += int((cm & (ch < 1)).sum())

    def _quantile(self, q: float) -> float | None:
        total = int(self.hist.sum())
        if total == 0:
            return None
        j = int(np.searchsorted(np.cumsum(self.hist), q * total, side="left"))
        return j / self.m

    def to_dict(self) -> dict[str, Any]:
        total = int(self.hist.sum())
        pct = lambda a, b: (100.0 * a / b) if b else None  # noqa: E731
        return {
            "mf_collision_method": self.method,
            "gaussian_num_samples": self.m,
            "num_modes": self.k,
            "n_predictions_sampled": self.n_predictions,
            "n_score_calls": self.n_score_calls,
            "total_sat_trajectory_checks": self.n_traj_checks,
            "total_sat_box_checks": self.n_box_checks,
            "scope": "agent-mode pairs of valid, modeled dynamic agents (the only ones that receive noise)",
            "n_agent_mode_pairs": total,
            "c_hat_mean": float((self.hist * np.arange(self.m + 1)).sum() / (self.m * total)) if total else None,
            "c_hat_per_mode_mean": (self.mode_sum / np.maximum(self.mode_cnt, 1)).tolist(),
            "agent_collision_prob_mean": self.p_agent_sum / self.p_agent_cnt if self.p_agent_cnt else None,
            "pct_pairs_mean0_sampled_gt0": pct(self.mean0_hat_pos, total),
            "pct_pairs_mean1_sampled_lt1": pct(self.mean1_hat_lt1, total),
            "pct_of_mean0_pairs_with_sampled_gt0": pct(self.mean0_hat_pos, self.mean0),
            "pct_of_mean1_pairs_with_sampled_lt1": pct(self.mean1_hat_lt1, self.mean1),
            "c_hat_summary": {
                "mean": float((self.hist * np.arange(self.m + 1)).sum() / (self.m * total)) if total else None,
                "median": self._quantile(0.5),
                "p25": self._quantile(0.25),
                "p75": self._quantile(0.75),
                "frac_eq_0": self.hist[0] / total if total else None,
                "frac_eq_1": self.hist[-1] / total if total else None,
                "frac_between": self.hist[1:-1].sum() / total if total else None,
                "hist_j_over_M": self.hist.tolist(),
            },
            "score_vs_mean_method": {
                "mean_abs_diff": self.score_absdiff_sum / self.n_candidates if self.n_candidates else None,
                "frac_candidates_changed": self.score_changed / self.n_candidates if self.n_candidates else None,
            },
        }
