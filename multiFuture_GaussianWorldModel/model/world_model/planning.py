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
  gaussian_independent     — integrates each mode's Gaussian over the
                             heading-aware collision box (see
                             ``score_gaussian_independent_collision``)

Legacy aliases ``expected`` / ``worst_case`` map to the independent variants
(do **not** build fake joint worlds by taking the same k across agents).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
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
# gaussian_independent: closed-form Gaussian collision probability
# ---------------------------------------------------------------------------
#
# For agent i, mode k, timestep t the agent center c ~ N(mu, diag(sigma_u^2,
# sigma_v^2)) in the sigma frame (angle theta_i, fixed over the horizon). Ego and
# agent boxes overlap iff c lies in the Minkowski octagon around the ego; we
# replace it by its bounding box in the sigma frame (conservative, exact when
# both boxes are axis-aligned with that frame):
#
#   h_u = La|cos a_a| + Wa|sin a_a| + Le|cos a_e| + We|sin a_e|
#   h_v = La|sin a_a| + Wa|cos a_a| + Le|sin a_e| + We|cos a_e|
#   P_ikt = [Phi((d_u+h_u)/s_u) - Phi((d_u-h_u)/s_u)] * [same for v]
#
# with (L, W) half-sizes, a_* = yaw_* - theta_i, d = R(-theta_i)(ego - mu).
#
#   P_it  = sum_k pi_ik P_ikt
#   P_i   = max_t w_t P_it,  w_t = 1 (t <= t_commit) else max(w_min, gamma^steps)
#   P_i   = 0 if P_i < eps
#   score = prod_i (1 - P_i)        (agent_reduce="product")
#         = 1 - max_i P_i           (agent_reduce="max")


@dataclass(frozen=True)
class GaussianRiskParams:
    t_commit_s: float = 1.0  # full weight up to this time
    gamma: float = 0.9  # per-dt decay after t_commit
    w_min: float = 0.3  # floor on the time weight
    eps: float = 0.01  # per-agent risk below this is dropped
    sigma_floor_m: float = 0.05  # avoids 0/0 at t=0 and for static objects
    radius_m: float | None = None  # ignore agents farther than this at t=0
    agent_reduce: str = "product"  # product | max
    dt: float = 0.1

    def __post_init__(self) -> None:
        if self.agent_reduce not in ("product", "max"):
            raise ValueError(f"agent_reduce must be 'product' or 'max', got {self.agent_reduce!r}")


def time_weights(num_steps: int, params: GaussianRiskParams) -> np.ndarray:
    """w_t for t = 0..T-1 (timestep t is at t * dt seconds)."""
    t_s = np.arange(num_steps, dtype=np.float32) * float(params.dt)
    steps_after = np.maximum(t_s - float(params.t_commit_s), 0.0) / float(params.dt)
    w = np.maximum(float(params.w_min), float(params.gamma) ** steps_after)
    return np.where(t_s <= float(params.t_commit_s), 1.0, w).astype(np.float32)


def gaussian_box_collision_probs(
    ego_trajectories: np.ndarray | jnp.ndarray,
    traj_k: np.ndarray | jnp.ndarray,
    sigma_k: np.ndarray | jnp.ndarray,
    sigma_frame_yaw: np.ndarray | jnp.ndarray,
    *,
    sigma_floor_m: float = 0.05,
) -> jax.Array:
    """P(collision) for every (ego, agent, mode, timestep).

    Parameters
    ----------
    ego_trajectories: [E, T, 5] world (x, y, yaw, length, width)
    traj_k:           [N, K, T, 5] world mean boxes per mode
    sigma_k:          [N, K, T, 2] std (m) along the sigma-frame axes
    sigma_frame_yaw:  [N] world angle of the sigma frame

    Returns
    -------
    probs: [E, N, K, T]
    """
    ego = jnp.asarray(ego_trajectories, dtype=jnp.float32)[:, None, None]  # [E,1,1,T,5]
    agt = jnp.asarray(traj_k, dtype=jnp.float32)[None]  # [1,N,K,T,5]
    sig = jnp.maximum(jnp.asarray(sigma_k, dtype=jnp.float32), float(sigma_floor_m))[None]
    theta = jnp.asarray(sigma_frame_yaw, dtype=jnp.float32)[None, :, None, None]  # [1,N,1,1]
    if ego.shape[3] != agt.shape[3]:
        raise ValueError(f"ego T={ego.shape[3]} != traj_k T={agt.shape[3]}")

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    dx = ego[..., 0] - agt[..., 0]
    dy = ego[..., 1] - agt[..., 1]
    d_u = cos_t * dx + sin_t * dy
    d_v = -sin_t * dx + cos_t * dy

    a_a = agt[..., 2] - theta
    a_e = ego[..., 2] - theta
    la, wa = 0.5 * agt[..., 3], 0.5 * agt[..., 4]
    le, we = 0.5 * ego[..., 3], 0.5 * ego[..., 4]
    ca, sa = jnp.abs(jnp.cos(a_a)), jnp.abs(jnp.sin(a_a))
    ce, se = jnp.abs(jnp.cos(a_e)), jnp.abs(jnp.sin(a_e))
    h_u = la * ca + wa * sa + le * ce + we * se
    h_v = la * sa + wa * ca + le * se + we * ce

    s_u = sig[..., 0]
    s_v = sig[..., 1]
    p_u = jsp_special.ndtr((d_u + h_u) / s_u) - jsp_special.ndtr((d_u - h_u) / s_u)
    p_v = jsp_special.ndtr((d_v + h_v) / s_v) - jsp_special.ndtr((d_v - h_v) / s_v)
    return jnp.clip(p_u * p_v, 0.0, 1.0)


def combine_gaussian_independent_collision(
    probs_eikt: np.ndarray | jnp.ndarray,
    mode_probs: np.ndarray,
    agent_masks: np.ndarray,
    params: GaussianRiskParams,
    *,
    agent_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce [E, N, K, T] probabilities to a per-ego safety score in [0, 1].

    Unit-testable without the box geometry: pass explicit probs_eikt.
    """
    p = jnp.asarray(probs_eikt, dtype=jnp.float32)
    if p.ndim != 4:
        raise ValueError(f"probs_eikt must be [E, N, K, T], got {p.shape}")
    e, n, k, t = p.shape
    probs = jnp.asarray(_normalize_mode_probs(np.asarray(mode_probs, dtype=np.float32)))
    if probs.shape != (n, k):
        raise ValueError(f"mode_probs must be [N, K]=[{n}, {k}], got {probs.shape}")
    masks = jnp.asarray(agent_masks).astype(jnp.float32)
    if masks.shape != (n, t):
        raise ValueError(f"agent_masks must be [N, T]=[{n}, {t}], got {masks.shape}")
    if agent_valid is None:
        agent_valid = np.ones((n,), dtype=bool)

    p_it = jnp.einsum("enkt,nk->ent", p, probs) * masks[None]  # [E, N, T]
    w = jnp.asarray(time_weights(t, params))
    weighted = p_it * w[None, None, :]
    p_i = weighted.max(axis=-1)  # [E, N]
    worst_t = weighted.argmax(axis=-1)
    p_i = jnp.where(p_i < float(params.eps), 0.0, p_i)
    p_i = p_i * jnp.asarray(agent_valid, dtype=jnp.float32)[None, :]

    if params.agent_reduce == "max":
        score = 1.0 - (p_i.max(axis=-1) if n else jnp.zeros((e,)))
    else:
        score = jnp.prod(1.0 - p_i, axis=-1)

    p_i_np = np.asarray(p_i, dtype=np.float32)
    score_np = np.asarray(score, dtype=np.float32)
    debug = {
        "planning_mode": "gaussian_independent",
        "p_collision_i": p_i_np,
        "max_agent_collision_prob": p_i_np.max(axis=-1) if n else np.zeros((e,), np.float32),
        "argmax_risk_agent": np.argmax(p_i_np, axis=-1).astype(np.int32)
        if n
        else np.zeros((e,), np.int32),
        "worst_timestep_i": np.asarray(worst_t, dtype=np.int32),
        "collision_score": score_np,
        "params": params,
    }
    return score_np, debug


def score_gaussian_independent_collision(
    ego_trajectories: np.ndarray | jnp.ndarray,
    traj_k: np.ndarray,
    sigma_k: np.ndarray,
    sigma_frame_yaw: np.ndarray,
    agent_masks: np.ndarray,
    mode_probs: np.ndarray,
    params: GaussianRiskParams | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full gaussian_independent collision score for ES ego candidates."""
    params = params or GaussianRiskParams()
    agent_masks = np.asarray(agent_masks).astype(bool)
    agent_valid = agent_masks.any(axis=-1)
    if params.radius_m is not None:
        ego0 = np.asarray(ego_trajectories, dtype=np.float32)[0, 0, :2]
        agent0 = np.asarray(traj_k, dtype=np.float32)[:, 0, 0, :2]
        agent_valid &= np.linalg.norm(agent0 - ego0[None], axis=-1) <= float(params.radius_m)

    probs_eikt = gaussian_box_collision_probs(
        ego_trajectories,
        traj_k,
        sigma_k,
        sigma_frame_yaw,
        sigma_floor_m=params.sigma_floor_m,
    )
    return combine_gaussian_independent_collision(
        probs_eikt,
        mode_probs,
        agent_masks,
        params,
        agent_valid=agent_valid,
    )
