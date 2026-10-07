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
