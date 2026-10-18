"""Loss-optimized lateral repair: the ramp-20 bank with its profile SOLVED FOR.

WHY. The ship bank (``l24`` at ramp 20, cubic) offers 24 rigid lateral shifts
of the base proposal, each ``A * smoothstep(t / 20)``, and executes the smallest
one that the exact clearance test certifies. Its residual is a shape limit, not
a count limit: at coverage-limited replans the clearest candidate still falls
~0.7 m short, and that shortfall is FLAT in K (4 -> 24 candidates), because a
rigid translation cannot bend around a hazard that occupies only part of the
window, or pull away from an edge on one side without walking into one on the
other. Every candidate is the same shape, scaled.

WHAT CHANGES. Only the profile. Everything else is the bank's: the offset is
applied along the path's left normal by :func:`branching.lateral_offset` (the
same function ``lateral_shift`` now calls), the candidate is scored by the same
exact scorer, and the ``cons`` rule decides. The offset ``d_1..d_n`` over the
selector window is found per replan by minimizing::

    L(d) = safety(world(d))                     # the training barrier (safety_objective)
         + (1/n) [ w_acc  * sum_t (a_t / a_ref)^2    # lateral accel OF THE EDIT
                 + w_jerk * sum_t (j_t / j_ref)^2    # lateral jerk  OF THE EDIT
                 + w_prox * sum_t (d_t / d_ref)^2 ]

with the car on the base's line before the anchor (``d_-2 = d_-1 = d_0 = 0``:
zero offset, offset velocity and offset acceleration at the join) and the offset
HELD after the window, as the ramp holds its amplitude. ``a``/``j`` are second
and third differences of that padded profile, so the join and the hold are both
charged.

THE RAMP IS THE SPECIAL CASE. Minimizing ``sum a^2`` alone subject to reaching
an offset ``A`` and holding it is the discrete cubic smoothstep -- exactly the
ramp-20 profile (``tests/test_lateral_opt.py`` checks this). So where a hazard
demands a constant offset at the end of the window, this reproduces the bank's
shape at the amplitude the barrier asks for; it departs from it only where the
hazard asks for something else (a bump that comes back, a bend that grows).
The jerk term turns the cubic's acceleration step at the join into a quintic-
like ease-in.

THE BARRIER WINS. The barrier's slope saturates at ~1 per corner-step once
inside, while the comfort terms' slope grows with the edit, so comfort must stay
small enough never to buy back a violation: at the defaults a rigid 1.5 m edit
costs ~0.05 of comfort per metre, against >= 0.1 per metre of barrier for any
violation longer than one step. What the comfort terms set is the SHAPE the
optimizer converges to among edits that clear the margin -- the smallest,
smoothest one -- which is the continuous version of "the smallest eligible
candidate".

THE OPTIMIZER is preconditioned (covariant) gradient descent, as in CHOMP
(Ratliff et al. 2009): ``step = H^-1 grad L``, with ``H`` the Hessian of the
comfort quadratic -- a constant ``n x n`` matrix. A barrier gradient is LOCAL
(it pushes the few steps whose corners touch the hazard); plain gradient descent
turns it into a spike that the comfort term irons out over hundreds of steps,
while ``H^-1`` spreads it into a smooth, boundary-respecting profile in one. A
per-row trust region caps how far any ``d_t`` moves per iteration, decaying
geometrically so the barrier's knee is settled rather than oscillated across.
The iterate with the lowest loss seen is returned, so the result is never worse
than its initialization by the loss.

LONGITUDINAL (``lon=1``, 2026-09-30). The o_0 residual is half collisions, and
half of THOSE are rear-ends: the ego lags its own log by ~3 m and a log-replayed
follower at ~6.7 m/s runs into it -- no lateral edit clears a car in the same
lane. ``lon`` appends a second profile ``s_1..s_n``: metres moved ALONG the path
(:func:`branching.path_retime`, applied before the lateral offset), i.e. an
optimized per-step time warp. Its comfort block is the same accel + jerk
quadratic (own weights ``w_lon_*``), pinned the same way at the join (no speed
or acceleration step), but padded after the window with its VELOCITY held rather
than its value: a speed change may persist, a lateral velocity may not. The
edit may not fall more than ``lon_back_m`` behind the base plan, so a braking
edit stays inside ``cons``'s progress tolerance (1 m) instead of being written
and then rejected by it; speeding up is unbounded but pays comfort. With
``lon=0`` the parameter vector, the Hessian and the edited world are exactly
the lateral ones.

It is offered as an EXTRA candidate (init tokens ``lon0`` / ``lonprev``), not
as a replacement for the lateral edit. Measured in the real-scene gate
(2026-09-30, 19 triggered rows): run alone, lon repairs LESS of the lateral
distortions than lateral-only (0.58 vs 0.74) and sits at the ``lon_back_m``
bound -- the soft barrier sums penetration over the window, so arriving at a
hazard later lowers the loss without clearing it. Ranked behind the lateral
edit, it can only add coverage: wherever o_0's edit is clean, o_0's edit runs.

WHAT IT IS NOT: the certificate. The loss is a soft barrier with margins, under
``edge_geometry="segment"`` by default (same zero level set as waymax's offroad
-- the sign logic is untouched -- but a gradient that survives the crossing,
where ``point`` measured 0.054 against 1; see ``SafetyLossConfig``). Whether a
switch happens is still decided by ``proposal_select.make_scorer``'s exact hard
clearance under ``cons``, so the do-no-harm property is unchanged: a base
proposal predicted clean is never touched.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

__all__ = ["LateralOptConfig", "comfort_operators", "comfort_hessian", "edit_hessian",
           "edit_world", "expand_offset", "expand_retime", "init_kind", "init_variant",
           "project_onto_path",
           "variant_cfg",
           "make_lateral_optimizer", "parse_inits"]


@dataclass(frozen=True)
class LateralOptConfig:
    """Loss weights and optimizer schedule. Parsed from ``k=v,k=v`` by :meth:`parse`."""

    iters: int = 32
    # Trust region: max |change| of any d_t per iteration, metres, decaying by
    # `decay` each iteration. The sum of the schedule is the optimizer's REACH
    # from its init: 0.25 * (1 - 0.92^32) / 0.08 = 2.9 m, past l24's +/-1.6 plus
    # the ~0.7 m shortfall measured at its coverage failures. The last step is
    # 0.017 m, fine enough to settle the barrier's 0.15 m knee.
    step_m: float = 0.25
    decay: float = 0.92
    # Fraction of the preconditioned step. 1.0 would jump straight to the
    # minimizer of the comfort quadratic plus the LINEARIZED barrier, which
    # overshoots a barrier's knee; the trust region bounds the damage either way.
    damping: float = 0.5
    w_acc: float = 0.02
    w_jerk: float = 0.02
    w_prox: float = 0.0
    a_ref: float = 1.0          # m/s^2
    j_ref: float = 10.0         # m/s^3 (a 0.1 s sim: the r20 cubic's join step is ~15A)
    d_ref: float = 1.0          # m
    # Barrier shape for the LOSS only. hinge2 has compact support: exactly zero
    # value and gradient beyond the margin, so a clear corner or a far car exerts
    # no push and the comfort terms can shrink the edit back to the smallest one
    # that clears. softplus would keep pushing everything away from everything.
    barrier_shape: str = "hinge2"
    edge_geometry: str = "segment"
    # Where the LOSS starts to push, metres. None = the run's own calibrated
    # margins (SafetyLossConfig: 0.3 edge, 0.75 agent). Setting them near 0 aims
    # the optimizer at "just clean", like the bank's smallest clean candidate --
    # the margin half of the shape-vs-margin attribution. Not 0 exactly: under
    # hinge2 the comfort pull settles the edit ~tau * (comfort slope) INSIDE the
    # margin, so a 0 margin would land slightly violating and fail the certificate.
    edge_margin_m: float | None = None
    agent_margin_m: float | None = None
    # Longitudinal profile (see LONGITUDINAL above). 0 = lateral only, the o_0 edit.
    lon: int = 0
    w_lon_acc: float = 0.02
    w_lon_jerk: float = 0.02
    # How far (metres of arclength) the edit may fall behind the base plan. Kept
    # under `cons`'s tol_m = 1.0 so a braking edit is not one it would reject.
    lon_back_m: float = 0.9
    # Quadratic cost on the along-path offset itself, (1/n) sum (s_t / d_ref)^2:
    # the "stay on the policy's timing" term. With it, braking or speeding up
    # happens only where the barrier outbids it -- the argmin arms use it IN PLACE
    # of the lon_back_m cap and cons's progress tolerance. 0 = off (unchanged lon).
    w_lon_prox: float = 0.0

    @classmethod
    def parse(cls, spec: str) -> "LateralOptConfig":
        """``"iters=24,w_acc=0.05"`` -> config; unknown keys are an error. A key
        repeated later wins, so ``global + "," + per_bank`` composes overrides."""
        fields = {f.name: f.type for f in dataclasses.fields(cls)}
        kw = {}
        for tok in [t.strip() for t in str(spec).split(",") if t.strip()]:
            if "=" not in tok:
                raise ValueError(f"lateral opt config: '{tok}' is not key=value")
            k, v = (s.strip() for s in tok.split("=", 1))
            if k not in fields:
                raise ValueError(f"lateral opt config: unknown key '{k}' (have {sorted(fields)})")
            kw[k] = int(v) if fields[k] in ("int", int) else (
                v if fields[k] in ("str", str) else float(v))
        cfg = cls(**kw)
        if cfg.w_acc <= 0 and cfg.w_jerk <= 0:
            raise ValueError("lateral opt config: need w_acc > 0 or w_jerk > 0 "
                             "(the comfort Hessian is the preconditioner)")
        if cfg.barrier_shape not in ("softplus", "hinge2"):
            raise ValueError(f"barrier_shape {cfg.barrier_shape!r}: softplus | hinge2")
        if cfg.edge_geometry not in ("point", "segment"):
            raise ValueError(f"edge_geometry {cfg.edge_geometry!r}: point | segment")
        if cfg.lon not in (0, 1):
            raise ValueError(f"lon={cfg.lon}: 0 | 1")
        if cfg.lon and cfg.w_lon_acc <= 0 and cfg.w_lon_jerk <= 0:
            raise ValueError("lateral opt config: lon=1 needs w_lon_acc > 0 or w_lon_jerk > 0")
        if cfg.lon_back_m < 0:
            raise ValueError(f"lon_back_m={cfg.lon_back_m} must be >= 0")
        return cfg


# Init token -> (edits the previous plan?, along-path profile on?). ``latA`` tokens
# are the lateral fresh kind, like ``0`` and ``best``.
INIT_TIERS = {"0": (False, False), "best": (False, False), "prev": (True, False),
              "lon0": (False, True), "lonprev": (True, True),
              "emerg": (False, True), "brake": (False, True),
              # argmin arms only: STARTS of one optimization over the fresh plan
              # (zero, or the previous plan's tail projected onto it).
              "init0": (False, True), "initprev": (True, True)}

# EMERGENCY tiers (2026-09-30), for rows where every gentle candidate is predicted
# to violate. Measured on o_lpA3's residual: at its edge-only coverage failures the
# optimizer KNOWS it has not cleared (loss 0.63 -> 0.23, never < 0.01) and stops at
# a peak offset of 0.69 m (p90 2.9 m, the trust region's whole reach) because the
# comfort terms outbid the saturating barrier; and 90% of its rear-collision lon
# edits sit at the 0.9 m braking cap. `emerg` = the lon optimizer with the comfort
# weights scaled by EMERG_COMFORT_SCALE and a longer reach; `brake` = the same,
# allowed BRAKE_BACK_M behind the plan (its progress tolerance is widened to match
# in proposal_select, and it runs only where the violation is a car's and the
# road edge is clean, so braking cannot "clear" a static edge by pushing it past
# the scored window).
EMERG_COMFORT_SCALE = 0.1
EMERG_STEP_M = 0.5
EMERG_ITERS = 48
BRAKE_BACK_M = 4.0


def init_kind(tok: str) -> tuple[bool, bool]:
    """``(uses_prev, lon)`` for an init token (see :func:`parse_inits`)."""
    return INIT_TIERS.get(tok, (False, False))


def init_variant(tok: str) -> str:
    """``""`` for the ordinary tokens, else the emergency variant name."""
    return tok if tok in ("emerg", "brake") else ""


def variant_cfg(cfg: "LateralOptConfig", variant: str, lon: bool) -> "LateralOptConfig":
    """The optimizer config a token runs with: ``lon`` switched on if asked, and the
    emergency variants' comfort scale / reach / braking allowance applied."""
    out = dataclasses.replace(cfg, lon=1) if lon else cfg
    if variant in ("emerg", "brake"):
        k = EMERG_COMFORT_SCALE
        out = dataclasses.replace(out, w_acc=cfg.w_acc * k, w_jerk=cfg.w_jerk * k,
                                  w_lon_acc=cfg.w_lon_acc * k, w_lon_jerk=cfg.w_lon_jerk * k,
                                  step_m=EMERG_STEP_M, iters=EMERG_ITERS)
    if variant == "brake":
        out = dataclasses.replace(out, lon_back_m=BRAKE_BACK_M)
    return out


def parse_inits(spec: str) -> list[str]:
    """``"0,best,lat1.0,lat-1.0"`` -> validated init tokens.

    * ``0``     -- start on the base's own line (the minimal-edit start);
    * ``latA``  -- start from the bank's own ``latA`` profile (its ramp and blend);
    * ``best``  -- start, per row, from the bank's CLEAREST ``lat`` candidate:
      at a coverage failure that is the one that fell shortest, and the loss
      then bends it the rest of the way.
    * ``prev``  -- edit a DIFFERENT plan: the remaining tail of the plan the car
      executed at the previous replan (from 0). At a replan whose fresh sample
      leaves the road within a step or two, no smooth edit of that sample can get
      back in time, but the previous plan was certified clean over steps the car
      has not driven yet. Ranked below every edit of the fresh sample.
    * ``lon0`` / ``lonprev`` -- ``0`` / ``prev`` with the along-path profile
      switched on for that candidate alone (``lon=1``), ranked below the
      lateral-only edit of the same plan (see LONGITUDINAL).
    * ``emerg`` / ``brake`` -- emergency fresh edits (see EMERG_COMFORT_SCALE),
      ranked below everything else and run only on predicted violations.

    The ranking (``proposal_select.OPT_TIER_M``): fresh lateral < fresh lon <
    prev lateral < prev lon, so each later kind runs only where every earlier
    kind is predicted to violate.
    """
    out = []
    for tok in [t.strip() for t in str(spec).split(",") if t.strip()]:
        if tok not in INIT_TIERS:
            if not tok.startswith("lat"):
                raise ValueError(f"lateral opt init '{tok}': 0 | best | prev | lon0 | lonprev "
                                 f"| emerg | brake | init0 | initprev | lat<metres>")
            float(tok[3:])
        out.append(tok)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate lateral opt inits in {out}")
    return out


# --------------------------------------------------------------------------- #
# The comfort quadratic: linear operators on d_1..d_n, and their Hessian
# --------------------------------------------------------------------------- #
def comfort_operators(n: int, dt: float, hold: str = "value"):
    """``(Ka, Kj)``: the edit's lateral acceleration and jerk as linear maps of
    ``d_1..d_n``, in m/s^2 and m/s^3.

    The profile they act on is ``d`` padded with the boundary conditions: three
    zeros before (steps -2, -1, 0: on the base's line, not drifting, not
    accelerating) and two steps after. ``hold="value"`` holds the last value (the
    lateral hold: arrive parallel); ``hold="velocity"`` continues its last slope
    (the longitudinal hold: a speed change may persist past the window). Pure
    numpy; ``n`` and ``dt`` are static per compiled call.
    """
    n = int(n)
    E = np.zeros((n + 5, n))                    # steps -2 .. n+2
    E[3:3 + n] = np.eye(n)
    if hold == "value":
        E[3 + n:, n - 1] = 1.0
    elif hold == "velocity":
        # d_{n+k} = d_n + k (d_n - d_{n-1}); d_0 = 0 when n == 1.
        for k in (1, 2):
            E[2 + n + k, n - 1] += 1.0 + k
            if n >= 2:
                E[2 + n + k, n - 2] -= k
    else:
        raise ValueError(f"hold {hold!r}: value | velocity")
    D2 = np.zeros((n + 3, n + 5))               # accel at steps -1 .. n+1
    for i in range(n + 3):
        D2[i, i:i + 3] = (1.0, -2.0, 1.0)
    Ka = D2 @ E / dt ** 2
    D1 = np.zeros((n + 2, n + 3))               # jerk between consecutive accels
    for i in range(n + 2):
        D1[i, i:i + 2] = (-1.0, 1.0)
    Kj = D1 @ Ka / dt
    return Ka, Kj


def comfort_hessian(n: int, dt: float, cfg: LateralOptConfig) -> np.ndarray:
    """Hessian of the comfort quadratic ``Q(d) = 0.5 d^T H d`` (per row), ``[n, n]``."""
    Ka, Kj = comfort_operators(n, dt)
    H = (cfg.w_acc / cfg.a_ref ** 2) * Ka.T @ Ka + (cfg.w_jerk / cfg.j_ref ** 2) * Kj.T @ Kj
    H = H + (cfg.w_prox / cfg.d_ref ** 2) * np.eye(n)
    return (2.0 / n) * H


def edit_hessian(n: int, dt: float, cfg: LateralOptConfig) -> np.ndarray:
    """Hessian of the whole edit's comfort quadratic: the lateral block alone
    (``[n, n]``, exactly :func:`comfort_hessian`) or, under ``lon``, block-diagonal
    with the longitudinal one (``[2n, 2n]``; the two profiles are charged
    independently, so the preconditioner never mixes them)."""
    H = comfort_hessian(n, dt, cfg)
    if not cfg.lon:
        return H
    Ka, Kj = comfort_operators(n, dt, hold="velocity")
    Hs = (2.0 / n) * ((cfg.w_lon_acc / cfg.a_ref ** 2) * Ka.T @ Ka
                      + (cfg.w_lon_jerk / cfg.j_ref ** 2) * Kj.T @ Kj
                      + (cfg.w_lon_prox / cfg.d_ref ** 2) * np.eye(n))
    out = np.zeros((2 * n, 2 * n))
    out[:n, :n], out[n:, n:] = H, Hs
    return out


def expand_retime(s_bn, W: int):
    """``[B, n]`` along-path offsets -> ``[B, W]``: 0 at the anchor, the window,
    then CONTINUED at its last slope (the velocity hold of the lon block)."""
    import jax.numpy as jnp

    B, n = s_bn.shape
    full = jnp.concatenate([jnp.zeros((B, 1), s_bn.dtype), s_bn], axis=1)     # 0..n
    k = jnp.arange(1, W - n, dtype=s_bn.dtype)[None, :]
    tail = full[:, -1:] + k * (full[:, -1:] - full[:, -2:-1])
    return jnp.concatenate([full, tail], axis=1)


def project_onto_path(world_p, world_q, n: int):
    """``[B, 2n]`` (lateral d, along-path s) offsets of ``world_q``'s steps 1..n
    relative to ``world_p``'s path -- the coordinates :func:`edit_world` uses.

    For each pose ``q_i`` the nearest point on ``world_p``'s polyline gives its
    arclength sigma and its signed offset along that point's left normal
    (heading-based, as :func:`branching.lateral_offset` uses); then
    ``s_i = sigma - S_i`` (S_i = arclength of ``world_p``'s own step i) and
    ``d_i`` = the offset. Used to START the argmin optimizer from another plan
    (the previous plan's tail) inside the SAME objective, so starts are compared
    on one loss. Approximate where the path turns sharply; the optimizer refines
    from there. Pure numpy, ``[B, W, 5]`` inputs on the same time base.
    """
    P = np.asarray(world_p, np.float64)[..., :2]                # [B, W, 2]
    Q = np.asarray(world_q, np.float64)[:, 1:n + 1, :2]         # [B, n, 2]
    yaw = np.asarray(world_p, np.float64)[..., 2]
    seg = P[:, 1:] - P[:, :-1]                                  # [B, W-1, 2]
    L = np.linalg.norm(seg, axis=-1)                            # [B, W-1]
    S = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(L, axis=1)], axis=1)
    rel = Q[:, :, None, :] - P[:, None, :-1, :]                 # [B, n, W-1, 2]
    t = np.clip(np.sum(rel * seg[:, None], -1) / np.maximum(L[:, None] ** 2, 1e-12), 0.0, 1.0)
    foot = P[:, None, :-1, :] + t[..., None] * seg[:, None]
    dist = np.linalg.norm(Q[:, :, None, :] - foot, axis=-1)     # [B, n, W-1]
    j = np.argmin(dist, axis=-1)                                # [B, n]
    bi = np.arange(P.shape[0])[:, None]
    tj = t[bi, np.arange(Q.shape[1])[None, :], j]
    sigma = S[bi, j] + tj * L[bi, j]
    yj = yaw[bi, j] + tj * np.angle(np.exp(1j * (yaw[bi, j + 1] - yaw[bi, j])))
    nrm = np.stack([-np.sin(yj), np.cos(yj)], -1)
    d = np.sum((Q - foot[bi, np.arange(Q.shape[1])[None, :], j]) * nrm, -1)
    s = sigma - S[:, 1:n + 1]
    return np.concatenate([d, s], axis=1).astype(np.float32)


def edit_world(world_p, x, *, n: int, dt: float, lon: int = 0):
    """The edited plan for parameters ``x``: ``[B, n]`` (lateral) or ``[B, 2n]``
    (lateral then along-path, under ``lon``). The retiming is applied first, so the
    lateral offset follows the retimed path's own normals. With ``lon=0`` this is
    exactly ``lateral_offset(world_p, expand_offset(x, W))``, the o_0 edit."""
    from rl.diffusion_ft.branching import lateral_offset, path_retime

    W = world_p.shape[1]
    if lon:
        world_p = path_retime(world_p, expand_retime(x[:, n:], W), dt=dt)
        x = x[:, :n]
    return lateral_offset(world_p, expand_offset(x, W), dt=dt)


def expand_offset(x_bn, W: int):
    """``[B, n]`` window offsets -> ``[B, W]`` full profile: 0 at the anchor, held after."""
    import jax.numpy as jnp

    B, n = x_bn.shape
    tail = jnp.broadcast_to(x_bn[:, -1:], (B, W - 1 - n))
    return jnp.concatenate([jnp.zeros((B, 1), x_bn.dtype), x_bn, tail], axis=1)


# --------------------------------------------------------------------------- #
def make_lateral_optimizer(objcfg, cfg: LateralOptConfig, *, dt: float,
                           neighbor_pred: str = "log", world_dt_method: str = "median",
                           forecaster=None):
    """Jitted ``opt(world_p, x0, state, t0, active_b, n) -> (x, loss_best, loss_init)``.

    ``world_p`` ``[B, W, 5]`` is the base proposal; ``x0`` ``[B, n]`` the init
    (``[B, 2n]`` under ``cfg.lon``: lateral, then along-path);
    ``active_b`` ``[B]`` masks the barrier to the rows that need a repair (the
    others only feel the comfort pull toward 0, and their result is not used).
    ``n`` (static) is the selector's window. ``dt`` is the world step the bank's
    edits already assume (``preprocess_cfg.world_dt_fallback``). The neighbour
    source is the arm's own, so an honest arm's loss never reads the future
    either. A ``wm_*`` source runs its world model once per replan, eagerly,
    and hands the boxes to the jitted program (``forecaster`` defaults to the
    process-wide one, :func:`world_model_forecast.get_forecaster`).
    """
    import jax
    import jax.numpy as jnp

    from data.preprocess import compute_world_dt_seconds
    from rl.diffusion_ft.safety_objective import candidate_safety

    margins = {k: v for k, v in (("edge_margin_m", cfg.edge_margin_m),
                                 ("agent_margin_m", cfg.agent_margin_m)) if v is not None}
    loss_cfg = dataclasses.replace(
        objcfg, safety=dataclasses.replace(objcfg.safety, barrier_shape=cfg.barrier_shape,
                                           edge_geometry=cfg.edge_geometry, **margins))

    def project(x, n):
        # The along-path half may not fall more than lon_back_m behind the base.
        if not cfg.lon:
            return x
        return x.at[:, n:].set(jnp.maximum(x[:, n:], -cfg.lon_back_m))

    from rl.diffusion_ft.safety_objective import WM_PREDICTIONS

    def opt(world_p, x0, state, t0, active_b, n, nb_boxes=None):
        B, W, _ = world_p.shape
        H = edit_hessian(n, dt, cfg)
        Hj = jnp.asarray(H, jnp.float32)
        Hinv = jnp.asarray(np.linalg.inv(H), jnp.float32)
        dt_b = (compute_world_dt_seconds(state, 0.1, world_dt_method)
                if neighbor_pred in ("cv", "none") else None)

        def loss(x):
            w = edit_world(world_p, x, n=n, dt=dt, lon=cfg.lon)
            terms = candidate_safety(loss_cfg, w, state=state, t0=t0, n_steps=n,
                                     alive_b=active_b, neighbor_pred=neighbor_pred,
                                     neighbor_dt_b=dt_b, neighbor_boxes=nb_boxes)
            per_b = terms["loss_b"] / n + 0.5 * jnp.einsum("bi,ij,bj->b", x, Hj, x)
            return jnp.sum(per_b), per_b

        vg = jax.value_and_grad(loss, has_aux=True)

        def body(k, carry):
            x, best_x, best_l = carry
            (_, l_b), g = vg(x)
            better = l_b < best_l
            best_x = jnp.where(better[:, None], x, best_x)
            best_l = jnp.where(better, l_b, best_l)
            step = cfg.damping * (g @ Hinv)             # H is symmetric
            cap = cfg.step_m * jnp.power(cfg.decay, k.astype(jnp.float32))
            big = jnp.max(jnp.abs(step), axis=1, keepdims=True)
            step = step * jnp.minimum(1.0, cap / jnp.maximum(big, 1e-9))
            return project(x - step, n), best_x, best_l

        x0 = project(x0, n)
        l0 = loss(x0)[1]
        x, best_x, best_l = jax.lax.fori_loop(0, int(cfg.iters), body, (x0, x0, l0))
        l_last = loss(x)[1]                             # the final iterate is a candidate too
        better = l_last < best_l
        return (jnp.where(better[:, None], x, best_x), jnp.where(better, l_last, best_l), l0)

    jopt = jax.jit(opt, static_argnums=(5,))
    if neighbor_pred not in WM_PREDICTIONS:
        return jopt
    from rl.diffusion_ft.world_model_forecast import get_forecaster
    fc = forecaster or get_forecaster(neighbor_pred)

    def wm_opt(world_p, x0, state, t0, active_b, n):
        dt_b = compute_world_dt_seconds(state, 0.1, world_dt_method)
        return jopt(world_p, x0, state, t0, active_b, n, fc(state, t0, n, dt_b))

    return wm_opt
