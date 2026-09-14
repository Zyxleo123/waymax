"""Counterfactual branching: snapshot a closed-loop episode at a replan, try
something different there, and continue with the base policy.

Three consumers share this engine, and each needs the same guarantee -- a
branch that changes nothing reproduces the original episode exactly:

* **repair search** (:mod:`rl.diffusion_ft.counterfactual` ``--mode repair``):
  rewind to a replan before the failure, execute a perturbed candidate, resume
  the base, keep the repair only if the whole continuation succeeds;
* **selector labels** (``--mode labels``): switch to the fine-tune's candidate at
  exactly one replan, resume the base, and record whether that one switch turned
  the episode's outcome +1 / 0 / -1 -- the per-decision label a learned selector
  needs, which the episode-level rows of ``proposal_select`` cannot provide;
* **look-ahead selection** (``proposal_select`` selector ``look``): execute each
  candidate plus a few base replans, then judge.

WHY IT CAN BE EXACT. Every piece of mutable env state is an immutable JAX pytree
held by reference (``_state``, ``_reward_state``, ``_rng``, goals, the last
preprocess batch), so a snapshot is a dict of references and costs nothing. The
sampler keys are drawn with the same ``key, rk = split(key)`` chain as
``SafetyFineTuner.rollout``, precomputed per replan, so a continuation from a
snapshot hands every replan the key the original episode used. Rows whose
override is off therefore replay the original bit for bit: same state, same
batch, same noise tensor. (Noise depends on the whole batch, so the batch must
be the original batch -- never a subset.)

Also here, as pure functions so they are CPU-testable:

* the candidate PERTURBATIONS (time warp along the path, a smooth lateral shift,
  finite-difference velocities), all of which keep the anchor pose and are
  continuous in position, heading and speed at the join;
* :func:`world_to_norm`, the inverse of ``postprocess_predictions``, which is
  what turns a repaired world trajectory back into a denoiser training target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

_ENV_FIELDS = ("_state", "_reward_state", "_rng", "_goal_xy_b2", "_goal_t_b",
               "_init_goal_dist_b", "_batch_size", "_last_pre_batch")


# --------------------------------------------------------------------------- #
# Perturbation families (world [B, W, 5] = x, y, yaw, vx, vy; index 0 = anchor)
# --------------------------------------------------------------------------- #
def _unwrap(yaw_bw):
    d = jnp.diff(yaw_bw, axis=1)
    d = jnp.arctan2(jnp.sin(d), jnp.cos(d))
    return jnp.concatenate([yaw_bw[:, :1], yaw_bw[:, :1] + jnp.cumsum(d, axis=1)], axis=1)


def _wrap(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def fd_velocity(world_bw5, dt: float):
    """Replace vx, vy by finite differences of x, y (central inside, one-sided at
    the far end). Index 0 keeps its anchor velocity: it is the current state."""
    p = world_bw5[..., :2]
    v = jnp.zeros_like(p)
    v = v.at[:, 1:-1].set((p[:, 2:] - p[:, :-2]) / (2.0 * dt))
    v = v.at[:, -1].set((p[:, -1] - p[:, -2]) / dt)
    v = v.at[:, 0].set(world_bw5[:, 0, 3:5])
    return world_bw5.at[..., 3:5].set(v)


def time_warp(world_bw5, alpha: float, *, ramp_steps: int, dt: float):
    """Drive the same path at ``alpha`` x the speed, easing in over ``ramp_steps``.

    Sample time is ``tau_i = sum_{j<=i} (1 + (alpha-1) * min(1, j/ramp))``, so the
    speed factor starts at exactly 1 (no jump at the anchor) and reaches alpha.
    Positions and unwrapped heading are linearly interpolated along the original
    index; past the candidate's end the car holds the last pose (``alpha > 1``
    runs out of path, which on a 5 s candidate executed for 1 s never binds).
    ``alpha < 1`` is yield / arrive later, ``> 1`` is go earlier.
    """
    B, W, _ = world_bw5.shape
    j = jnp.arange(W, dtype=jnp.float32)
    ramp = jnp.minimum(1.0, j / max(int(ramp_steps), 1))
    rate = 1.0 + (float(alpha) - 1.0) * ramp
    tau = jnp.concatenate([jnp.zeros(1), jnp.cumsum(rate[1:])])
    tau = jnp.clip(tau, 0.0, W - 1.0)
    lo = jnp.floor(tau).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, W - 1)
    f = (tau - lo)[None, :, None]
    yaw_u = _unwrap(world_bw5[..., 2])
    feat = jnp.concatenate([world_bw5[..., :2], yaw_u[..., None]], axis=-1)
    out = feat[:, lo] * (1.0 - f) + feat[:, hi] * f
    w = world_bw5.at[..., :2].set(out[..., :2]).at[..., 2].set(_wrap(out[..., 2]))
    return fd_velocity(w, dt)


def lateral_shift(world_bw5, amp_m: float, *, ramp_steps: int, dt: float):
    """Shift the path ``amp_m`` metres along its left normal (negative = right).

    The offset follows a smoothstep from 0 at the anchor to ``amp_m`` at
    ``ramp_steps`` and then holds, so position, heading and lateral velocity are
    all continuous at the join. Heading gains ``atan(d offset / d arclength)``,
    the slope of the offset path; the arclength is floored so a near-stationary
    car does not get a 90-degree yaw from a tiny step.
    """
    B, W, _ = world_bw5.shape
    u = jnp.clip(jnp.arange(W, dtype=jnp.float32) / max(int(ramp_steps), 1), 0.0, 1.0)
    d = float(amp_m) * (3.0 * u ** 2 - 2.0 * u ** 3)                      # [W]
    yaw = world_bw5[..., 2]
    n = jnp.stack([-jnp.sin(yaw), jnp.cos(yaw)], axis=-1)                # [B, W, 2]
    p = world_bw5[..., :2] + d[None, :, None] * n
    ds = jnp.linalg.norm(jnp.diff(world_bw5[..., :2], axis=1), axis=-1)  # [B, W-1]
    dd = jnp.diff(d)[None, :]
    slope = jnp.arctan2(dd, jnp.maximum(ds, 0.05))
    dyaw = jnp.concatenate([jnp.zeros((B, 1)), slope], axis=1)
    w = world_bw5.at[..., :2].set(p).at[..., 2].set(_wrap(yaw + dyaw))
    return fd_velocity(w, dt)


@dataclass(frozen=True)
class Candidate:
    """One repair candidate. ``family``: id | tw (time warp) | lat | rn (re-noise)."""
    name: str
    family: str
    param: float = 0.0
    hold: int = 1           # consecutive replans the perturbation is applied on
    seed: int = 0           # rn only
    magnitude: float = 0.0  # ordering key: smaller = closer to the base

    @staticmethod
    def parse(tok: str) -> "Candidate":
        """``id`` | ``tw0.7`` | ``tw0.7h2`` | ``lat-0.5`` | ``rn30s1`` (t=30, seed 1)."""
        t = tok.strip()
        hold = 1
        if "h" in t and t != "id":
            t, h = t.rsplit("h", 1)
            hold = int(h)
        if t == "id":
            return Candidate(tok, "id")
        if t.startswith("tw"):
            a = float(t[2:])
            if a <= 0:
                raise ValueError(f"time warp factor must be > 0: {tok}")
            return Candidate(tok, "tw", a, hold, magnitude=abs(np.log(a)) * 2.0 + 0.1 * hold)
        if t.startswith("lat"):
            a = float(t[3:])
            return Candidate(tok, "lat", a, hold, magnitude=abs(a) + 0.1 * hold)
        if t.startswith("rn"):
            body = t[2:]
            seed = 0
            if "s" in body:
                body, s = body.split("s", 1)
                seed = int(s)
            ts = int(body)
            # Re-noising picks a different manoeuvre: always "bigger" than a
            # local edit, ordered by how much of the chain it re-draws.
            return Candidate(tok, "rn", float(ts), hold, seed, magnitude=10.0 + ts / 100.0)
        raise ValueError(f"unknown candidate '{tok}'")


def parse_candidates(spec: str) -> list[Candidate]:
    cands = [Candidate.parse(t) for t in spec.split(",") if t.strip()]
    names = [c.name for c in cands]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate candidates in {names}")
    return cands


# --------------------------------------------------------------------------- #
def model_step_valid(aux, horizon: int, model_dt: float):
    """[B, T] mask: model steps the WORLD time base actually covers.

    ``preprocess`` clamps ``world_t_seconds`` past the scenario's end (every
    sample beyond it repeats the last valid time), so for a scene with less than
    ``horizon * model_dt`` left the world trajectory holds a frozen pose there.
    The forward map can produce those samples; the inverse CANNOT recover the
    model steps behind them -- the information is gone. Measured on the first
    repair smoke, that showed up as a 0.256 normalized (25.6 m) round-trip error
    on the tail. Anything reading ``world_to_norm`` must mask with this: an
    unmasked repair target teaches the denoiser a frozen tail it never drove.
    """
    valid = jnp.asarray(aux["world_t_valid"])
    t_end = jnp.max(jnp.where(valid, jnp.asarray(aux["world_t_seconds"]), 0.0), axis=1)
    steps = (jnp.arange(1, horizon + 1, dtype=jnp.float32) * float(model_dt))
    return steps[None, :] <= t_end[:, None] + 1e-6


def world_to_norm(world_bw5, aux, cfg, *, resample_fn=None):
    """Inverse of ``postprocess_predictions``: world @ world_dt -> normalized [B, T, 5].

    Resample onto the model time base (whose index 0 is the anchor), drop the
    anchor, move into the anchor's ego frame, and normalize with the same
    ``ego_range`` / ``max_velocity`` / pi the forward map divides out. Layout of the
    result is the denoiser's ``[x, y, vx, vy, yaw]``.
    """
    from data.preprocess import _rotate_xy, _wrap_to_pi, resample_trajectory_timebase_jax

    rs = resample_fn or resample_trajectory_timebase_jax
    model = rs(traj_btd=world_bw5, src_t_s=aux["world_t_seconds"],
               dst_t_s=aux["model_t_seconds"], yaw_index=2)[:, 1:, :]
    yaw0 = aux["anchor_yaw"][:, None]
    pos = _rotate_xy(model[..., :2] - aux["origin_xy"][:, None, :], yaw0)
    vel = _rotate_xy(model[..., 3:5], yaw0)
    yaw = _wrap_to_pi(model[..., 2] - yaw0)
    return jnp.concatenate([pos / cfg.ego_range, vel / cfg.max_velocity,
                            (yaw / jnp.pi)[..., None]], axis=-1).astype(jnp.float32)


# --------------------------------------------------------------------------- #
@dataclass
class Snapshot:
    env: dict
    features: dict
    carry: dict          # alive, offroad, overlap, reached (host/device arrays)
    replan: int


@dataclass
class Episode:
    """Outcome of one (possibly branched) closed-loop episode, per scene [B]."""
    solved: np.ndarray
    offroad: np.ndarray
    overlap: np.ndarray
    reached: np.ndarray
    clean: np.ndarray
    fail_replan: np.ndarray        # first replan whose executed steps violated; -1 none
    n_replans: int
    snapshots: dict = field(default_factory=dict)   # replan -> Snapshot (record=True)
    stash: list = field(default_factory=list)       # override hook's records
    final_goal_dist: np.ndarray | None = None


class Brancher:
    """Closed-loop driver over a :class:`SafetyFineTuner`'s env + actor.

    ``override(r, ctx) -> world [B, W, 5] | None`` is called at every replan with
    the base proposal already in ``ctx``; returning None executes the base
    proposal. ``ctx`` carries ``world_p``, ``x_norm`` (the proposal in the
    denoiser's [B, C, T] layout), ``x_prefix``/``key_late``, ``cond1``/``cond2``/
    ``mask``/``aux``, ``t0``, ``alive`` and ``rk``.
    """

    def __init__(self, tuner, params):
        self.t = tuner
        self.env = tuner.env
        self.actor = tuner.actor
        self.params = params
        self.dt = float(tuner.ckpt.preprocess_cfg.world_dt_fallback)

    # -- snapshots ------------------------------------------------------ #
    def capture(self, features, carry, replan) -> Snapshot:
        return Snapshot({k: getattr(self.env, k) for k in _ENV_FIELDS}, features,
                        {k: v for k, v in carry.items()}, replan)

    def restore(self, snap: Snapshot):
        for k, v in snap.env.items():
            setattr(self.env, k, v)
        return snap.features, {k: v for k, v in snap.carry.items()}

    def keys_for(self, rng):
        """The per-replan sampler keys ``SafetyFineTuner.rollout`` would draw."""
        keys, key = [], rng
        for _ in range(int(self.t.args.max_replans)):
            key, rk = jax.random.split(key)
            keys.append(rk)
        return keys

    # -- one episode ----------------------------------------------------- #
    def run(self, state=None, *, keys, snapshot: Snapshot | None = None,
            override: Callable | None = None, record: bool = False,
            stop_after: int | None = None) -> Episode:
        """Run from reset (``state``) or from ``snapshot`` to the episode end, or
        until replan index ``stop_after`` (exclusive) for a look-ahead."""
        env, actor, args, t = self.env, self.actor, self.t.args, self.t
        actor.num_sample_steps = t._steps_eval
        if snapshot is None:
            features = env.reset(state, seed=args.seed)
            clean = np.asarray(t._gate_expert_clean(env.sim_state)) > 0.5
            B = env.batch_size
            carry = {"alive": clean.astype(np.float32), "offroad": np.zeros(B, bool),
                     "overlap": np.zeros(B, bool), "reached": np.zeros(B, bool),
                     "clean": clean, "fail_replan": np.full(B, -1, np.int32)}
            r0 = 0
        else:
            features, carry = self.restore(snapshot)
            r0 = snapshot.replan
        ep_snaps, stash = {}, []
        r = r0
        last = int(args.max_replans) if stop_after is None else min(int(stop_after), int(args.max_replans))
        while r < last:
            alive = carry["alive"]
            if float(np.sum(alive)) <= 0.0:
                break
            t0 = env.timestep
            horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
            if min(args.replan_interval_steps, horizon - 1 - t0) <= 0:
                break
            if record:
                ep_snaps[r] = self.capture(features, carry, r)
            cond1, cond2 = env.compute_condition(features)
            mask = features["inst_valid"]
            aux = env.last_aux
            x_norm, (x_prefix, key_late) = actor.sample_world_with_prefix(
                self.params, cond1, cond2, mask, keys[r])
            world_p = env.norm_to_world_with(jnp.transpose(x_norm, (0, 2, 1)), aux)
            world = world_p
            if override is not None:
                ctx = {"world_p": world_p, "x_norm": x_norm, "x_prefix": x_prefix,
                       "key_late": key_late, "cond1": cond1, "cond2": cond2, "mask": mask,
                       "aux": aux, "t0": t0, "alive": alive, "rk": keys[r], "replan": r,
                       "stash": stash}
                o = override(r, ctx)
                if o is not None:
                    world = o
            out = env.step(trajectory_world_bt5=world)
            comp = out.components
            off = np.asarray(comp["offroad_flag"], np.float32) > 0
            ovl = np.asarray(comp["collision_flag"], np.float32) > 0
            rea = np.asarray(comp["reached_flag"], np.float32) > 0
            newly = (off | ovl) & (carry["fail_replan"] < 0) & (alive > 0.5)
            carry["fail_replan"] = np.where(newly, r, carry["fail_replan"]).astype(np.int32)
            carry["offroad"] = carry["offroad"] | off
            carry["overlap"] = carry["overlap"] | ovl
            carry["reached"] = carry["reached"] | rea
            done = np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
            carry["alive"] = (alive * (1.0 - done.astype(np.float32))).astype(np.float32)
            features = out.features
            r += 1
        from rl.diffusion_ft.env import _ego_pose_now_b
        gd = np.asarray(jnp.linalg.norm(_ego_pose_now_b(env.sim_state) - env._goal_xy_b2, axis=-1))
        solved = carry["reached"] & ~carry["offroad"] & ~carry["overlap"]
        return Episode(solved=solved, offroad=carry["offroad"], overlap=carry["overlap"],
                       reached=carry["reached"], clean=carry["clean"],
                       fail_replan=carry["fail_replan"], n_replans=r, snapshots=ep_snaps,
                       stash=stash, final_goal_dist=gd)
