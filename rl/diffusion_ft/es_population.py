"""Population search at a replan: sample, PROJECT with the repair bank, smooth by
re-noise/denoise, hand the fittest feasible member to ``cons``.

WHY. Diffusion-ES (``es_baseline/``) searches by mutating a sampled population
with truncated re-noise/denoise and keeping elites under a safety score, so its
mutation has to FIND a clean trajectory by chance. The repair bank finds one
geometrically: l24 lateral shifts at ramp 20 with constant-velocity neighbours
(``b_cv``, 2026-09-25) takes failure_heldout 0.288 -> 0.717 at m2 +0.0091 with
nothing from the recorded future. So project every population member with that
bank FIRST -- the population is then feasible right off the bat wherever a
lateral shift can make it so -- and use re-noise/denoise only to pull the
geometric (C1, cubic-ramp) edit back onto the policy's own trajectory manifold.

THE CATCH, computed before writing this. This diffusion's LOWEST noise level is
``sqrt(1 - abar_0) = 0.025`` normalized, i.e. 2.5 m per waypoint (ego_range
100 m); the Diffusion-ES default mutation (resample_timesteps 3, noise_scale 2)
starts at t=2 with 2 x 5.8 = 11.6 m. A 0.1-1.6 m lateral repair sits far below
either, so a denoiser that trusts its prior can simply undo it. ``t:1,scale:0``
is the gentlest setting the model supports -- one deterministic Tweedie step at
t=0, ``x0_hat(sqrt(abar_0) x)``, no noise injected -- and even that step
believes its input carries 2.5 m of noise. Whether a repair survives the
smoothing is therefore the first thing this MEASURES (``survival`` below), not
an assumption.

ONE REPLAN (only where the base proposal is predicted to violate -- the ``cons``
trigger; everywhere else the base is executed untouched and nothing runs):

1. population: member 0 IS the base proposal (so every arm stays paired draw for
   draw); members 1..P-1 are fresh base samples from ``fold_in(rk, salt)``,
   which is the SAME key in every ES arm, so ES arms with equal ``pop`` start
   from identical populations and differ only in their operators;
2. project: a member predicted to violate takes the smallest-magnitude bank edit
   that is clean and keeps its own progress within ``tol`` (``bank_choice``'s
   ``cons`` rule with the member as its own primary). If no edit is clean it
   takes its clearest edit when that is clearer than itself -- a better start
   for the smoothing and the next generation, never a certificate;
3. smooth: every member the projection touched, and every member still
   infeasible, is re-noised to ``t-1`` with ``scale`` x the forward noise and
   denoised ``t`` steps by the frozen base (``GaussianDiffusion.resample``, the
   ES's own mutation). A clean raw sample is already a draw from the policy and
   is left exactly as drawn;
4. generations 1..iters: the top ``elite`` members are replicated back to P,
   projected (a no-op on feasible ones) and smoothed again; the pool keeps the
   elites (Diffusion-ES's elitism);
5. the fittest member goes to ``select_switch("cons")``: executed only if it is
   clean over the window and keeps the BASE proposal's progress within ``tol``.

Feasible = ``clear >= 0`` over the window and ``prog >= prog_base - tol``.
Fitness ranks every feasible member above every infeasible one; among feasible
members ``rank:dev`` (default) prefers the one closest to the base proposal --
the bank's minimal-intervention ordering carried over -- and ``rank:prog`` the
one that gains most ground; among infeasible ones it is the total constraint
shortfall in metres, so the elites move toward feasibility.

With ``pop:1,iters:0,t:0`` and the same bank, the arm is ``bank@<bank>:cons``
exactly (member 0 is the base proposal, the choice rule is ``bank_choice``'s,
and the final check is ``cons``) -- a free plumbing check to run alongside.

Config grammar (``--select_es``), ``name=key:value,...;name=...``:
``pop`` population P (16), ``iters`` generations after the first (2), ``elite``
(4), ``bank`` named bank to project with ("" = no projection), ``t`` re-noise /
denoise length (1; 0 = no smoothing), ``scale`` noise scale (0.0), ``nb``
neighbour source for the trigger, fitness and projection, log|cv|none (cv),
``rank`` dev|prog (dev), ``opt`` 1 = project with the loss-optimized lateral
repair (``lateral_opt``, the ``o_0`` arm's optimizer) instead of a bank (0),
``order`` ps|sp (ps). An arm references a config as ``es@<name>``.

MEASURED 2026-09-29 (smoke, jobs 68920-68924): the smoothing reading fails. One
deterministic t=0 step kept 12-30% of each lateral repair (``survival`` p50) and
0-11% of the projected-feasible members stayed feasible, so ``order:ps`` with any
``t > 0`` undoes the projection. ``order:sp`` is the fix: re-noise/denoise only
MUTATES elites (exploration, generations >= 1) and the projection -- which, with
``opt:1``, already penalizes lateral accel and jerk -- is the last operator, so
every executed trajectory is a raw model sample or a projection.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

# Per-replan diagnostics, one row per scene; NaN where the ES did not run. The
# counts are over population members of generation 0 (the ``n_*`` columns) or
# the final pool (``n_final_feas``).
ES_DIAG = (
    "trig",           # base proposal predicted to violate, scene alive: the ES ran
    "n_raw_feas",     # raw samples already feasible (best-of-P with no search)
    "n_proj_feas",    # feasible after the projection ("correct right off the bat")
    "n_gen0_feas",    # feasible after projection + smoothing
    "n_edit",         # members the projection edited
    "n_edit_feas",    # ... of which the edit made feasible
    "n_edit_kept",    # ... and still feasible after smoothing (retention numerator)
    "survival",       # mean over edited members: fraction of the edit the smoothing
                      # kept, <smooth-raw, proj-raw> / |proj-raw|^2 over the window
                      # (1 = kept, 0 = snapped back to the raw sample)
    "smooth_disp_m",  # mean over edited members of max |smooth - proj| over the window
    "n_final_feas",   # feasible members in the final pool
    "best_feas",      # the member handed to cons is feasible
    "best_gen",       # generation that produced it
    "best_edited",    # its lineage was projected at least once
    "best_line0",     # it descends from member 0, the base proposal itself
    "best_dev_m",     # its mean distance from the base proposal over the window
    "wall_s",         # seconds this replan's search took (whole batch)
)
SCORE_KEYS = ("clear", "edge", "agent", "prog")
FEAS_BONUS = 1.0e4
_POP_SALT = 0xE5      # fold_in constant for the population key; +1+g for smoothing


@dataclass(frozen=True)
class ESConfig:
    name: str
    pop: int = 16
    iters: int = 2
    elite: int = 4
    bank: str = ""
    t: int = 1
    scale: float = 0.0
    nb: str = "cv"
    rank: str = "dev"
    # 1 = project with the loss-optimized lateral repair (rl.diffusion_ft.lateral_opt,
    # init 0, the `o_0` arm's optimizer and --select_opt_cfg) instead of a bank.
    opt: int = 0
    # "ps": project, then re-noise/denoise (the smoothing reading, measured
    # 2026-09-29 to UNDO the repair). "sp": re-noise/denoise first -- exploration
    # only, from generation 1 -- then project, so every executed trajectory is a
    # raw model sample or a projection, never a smoothed-over repair.
    order: str = "ps"


_ES_KEYS = {"pop": int, "iters": int, "elite": int, "bank": str, "t": int,
            "scale": float, "nb": str, "rank": str, "opt": int, "order": str}


def parse_es_specs(spec: str) -> dict[str, ESConfig]:
    """``'proj=pop:16,bank:l24cv,t:1;raw=pop:16,t:3,scale:2'`` -> configs."""
    out: dict[str, ESConfig] = {}
    for tok in [t.strip() for t in str(spec or "").split(";") if t.strip()]:
        if "=" not in tok:
            raise SystemExit(f"--select_es: '{tok}' is not name=key:value,...")
        name, body = (s.strip() for s in tok.split("=", 1))
        if not name:
            raise SystemExit(f"--select_es: '{tok}' has an empty name")
        if name in out:
            raise SystemExit(f"--select_es: duplicate config name '{name}'")
        kw = {}
        for kv in [s.strip() for s in body.split(",") if s.strip()]:
            if ":" not in kv:
                raise SystemExit(f"--select_es {name}: '{kv}' is not key:value")
            k, v = (s.strip() for s in kv.split(":", 1))
            if k not in _ES_KEYS:
                raise SystemExit(f"--select_es {name}: unknown key '{k}' (have {sorted(_ES_KEYS)})")
            kw[k] = _ES_KEYS[k](v)
        # An unset elite count follows a small population down (pop:1 -> elite 1).
        kw.setdefault("elite", min(ESConfig.elite, kw.get("pop", ESConfig.pop)))
        c = ESConfig(name, **kw)
        if c.pop < 1 or c.iters < 0 or not 1 <= c.elite <= c.pop:
            raise SystemExit(f"--select_es {name}: need pop >= 1, iters >= 0, 1 <= elite <= pop")
        if c.t < 0 or c.scale < 0:
            raise SystemExit(f"--select_es {name}: need t >= 0 and scale >= 0")
        if c.nb not in ("log", "cv", "none"):
            raise SystemExit(f"--select_es {name}: nb '{c.nb}' not in log|cv|none")
        if c.rank not in ("dev", "prog"):
            raise SystemExit(f"--select_es {name}: rank '{c.rank}' not in dev|prog")
        if c.opt not in (0, 1):
            raise SystemExit(f"--select_es {name}: opt must be 0 or 1")
        if c.opt and c.bank:
            raise SystemExit(f"--select_es {name}: opt:1 and bank:{c.bank} are two "
                             f"projections; pick one")
        if c.order not in ("ps", "sp"):
            raise SystemExit(f"--select_es {name}: order '{c.order}' not in ps|sp")
        out[name] = c
    return out


# --------------------------------------------------------------------------- #
# Pure numpy pieces (CPU-testable: tests/test_es_population.py)
# --------------------------------------------------------------------------- #
def feasible(clear, prog, prog_ref, tol_m: float):
    """[M, B] bool: clean over the window and within ``tol`` of the reference progress."""
    return (np.asarray(clear) >= 0.0) & (np.asarray(prog) >= np.asarray(prog_ref) - float(tol_m))


def project_choice(clear_m, prog_m, clear_km, prog_km, magnitude, tol_m: float):
    """[M, B] int32: bank index applied to each member, ``-1`` = left as is.

    ``clear_m``/``prog_m`` ``[M, B]``, ``clear_km``/``prog_km`` ``[K, M, B]``,
    ``magnitude`` ``[K]``. A member predicted clean is never edited. One predicted
    to violate takes the smallest-magnitude edit that is clean and keeps the
    member's OWN progress within ``tol`` -- ``bank_choice``'s ``cons`` rule with
    the member as its own primary, ties to the earlier bank entry. Failing that,
    its clearest edit if that is clearer than the member (a starting point, not a
    certificate: feasibility is re-checked on the final pool)."""
    clear_m = np.asarray(clear_m, np.float64)
    prog_m = np.asarray(prog_m, np.float64)
    clear_km = np.asarray(clear_km, np.float64)
    prog_km = np.asarray(prog_km, np.float64)
    K = clear_km.shape[0]
    if len(magnitude) != K:
        raise ValueError(f"magnitude has {len(magnitude)} entries for {K} edits")
    need = clear_m < 0.0
    elig = (clear_km >= 0.0) & (prog_km >= prog_m[None] - float(tol_m)) & need[None]
    cost = np.where(elig, np.asarray(magnitude, np.float64)[:, None, None], np.inf)
    best = np.argmin(cost, axis=0)
    mi, bi = np.indices(clear_m.shape)
    has = np.isfinite(cost[best, mi, bi])
    clr = np.argmax(clear_km, axis=0)
    better = need & ~has & (clear_km[clr, mi, bi] > clear_m)
    return np.where(has, best, np.where(better, clr, -1)).astype(np.int32)


def es_fitness(clear, prog, dev, *, prog_ref, tol_m: float, rank: str = "dev"):
    """[M, B]: feasible members above infeasible ones; see the module docstring."""
    clear, prog, dev = (np.asarray(a, np.float64) for a in (clear, prog, dev))
    feas = feasible(clear, prog, prog_ref, tol_m)
    sec = -dev if rank == "dev" else prog
    short = (np.minimum(clear, 0.0)
             + np.minimum(prog - (np.asarray(prog_ref, np.float64) - float(tol_m)), 0.0))
    return np.where(feas, FEAS_BONUS + sec, short)


def top_k(fit_mb, k: int):
    """[k, B] member indices, best first; ties go to the lower member index, so
    member 0 (the base proposal) wins any exact tie."""
    return np.argsort(-np.asarray(fit_mb), axis=0, kind="stable")[:k]


def replicate(elite: int, pop: int):
    """[pop] elite slot per child: Diffusion-ES's ``_replicate_elites`` layout."""
    base, rem = divmod(int(pop), int(elite))
    return np.concatenate([np.full(base + (1 if e < rem else 0), e, np.int64)
                           for e in range(int(elite))])


def survival_ratio(raw, proj, smooth, n: int):
    """[...]: fraction of the projection's edit the smoothing kept, over steps 1..n.

    ``<smooth - raw, proj - raw> / |proj - raw|^2`` on positions: 1 = the edit is
    fully kept, 0 = the smoothing snapped back to the raw sample, > 1 overshoot.
    NaN where the projection moved nothing."""
    raw, proj, smooth = (np.asarray(a, np.float64)[..., 1:n + 1, :2] for a in (raw, proj, smooth))
    a, b = proj - raw, smooth - raw
    den = (a * a).sum((-1, -2))
    return np.where(den > 1e-8, (a * b).sum((-1, -2)) / np.maximum(den, 1e-12), np.nan)


def window_dev(worlds_mbw5, world_bw5, n: int):
    """[M, B] mean distance (m) from ``world_bw5`` over steps 1..n."""
    w = np.asarray(worlds_mbw5, np.float64)[..., 1:n + 1, :2]
    r = np.asarray(world_bw5, np.float64)[None, :, 1:n + 1, :2]
    return np.linalg.norm(w - r, axis=-1).mean(-1)


# --------------------------------------------------------------------------- #
class PopulationES:
    """One named ES config, bound to a runner's policy, scorer and bank.

    Called once per replan by ``ProposalSelectEval.rollout``; returns the
    fittest member's world trajectory ``[B, W, 5]``, its scores (``SCORE_KEYS``
    -> ``[B]``, or None when nothing triggered) and the ``ES_DIAG`` row ``[B, D]``.
    Pool arrays are member-major: ``[M, B, ...]``, flattened ``m * B + b``."""

    def __init__(self, tuner, cfg: ESConfig, *, params, score_fn, cands, ramp_steps: int,
                 blend: str, tol_m: float, map_batch: int = 4, optimizer=None):
        import jax
        import jax.numpy as jnp

        from rl.diffusion_ft.branching import (lateral_shift, model_step_valid,
                                               time_warp, world_to_norm)

        self.jax, self.jnp = jax, jnp
        self.t, self.cfg, self.params = tuner, cfg, params
        self.tol_m = float(tol_m)
        self.P, self.E = int(cfg.pop), min(int(cfg.elite), int(cfg.pop))
        pcfg = tuner.ckpt.preprocess_cfg
        diffusion = tuner.actor.diffusion
        horizon = int(diffusion.denoise_fn.horizon)
        self._ego_range = float(pcfg.ego_range)
        self._w2n = jax.jit(lambda w, aux: world_to_norm(w, aux, pcfg))
        self._valid = jax.jit(lambda aux: model_step_valid(aux, horizon, float(pcfg.model_dt)))
        # No bank in the config means no projection, whatever the caller passed.
        self.cands = list(cands) if cfg.bank else []
        self._mag = np.asarray([c.magnitude for c in self.cands], np.float64)
        self._rt_checked = False
        # The bank's own edits, with the runner's world dt (the fallback, exactly
        # as ProposalSelectEval builds them) so `pop:1,iters:0,t:0` reproduces the
        # bank arm edit for edit.
        dt = float(pcfg.world_dt_fallback)
        # opt:1 -- lateral_opt's jitted optimizer (the runner's own when it has one
        # for this neighbour source), applied at the same dt as `o_0`.
        self._opt = optimizer if cfg.opt else None
        if cfg.opt and optimizer is None:
            raise ValueError(f"es '{cfg.name}': opt:1 needs an optimizer")
        self._opt_dt = dt
        self._score_fn = score_fn
        self._rows = None                  # this replan's triggered rows (set per call)
        fns = [(lambda w, _f=(time_warp if c.family == "tw" else lateral_shift),
                _a=float(c.param), _kw=({} if c.family == "tw" else {"blend": blend}):
                _f(w, _a, ramp_steps=int(ramp_steps), dt=dt, **_kw))
               for c in self.cands]
        mb = int(map_batch)

        def score_many(worlds, state, t0, goal, n):              # [M, B, W, 5]
            return jax.lax.map(lambda w: score_fn(w, state, t0, goal, n), worlds,
                               batch_size=mb)

        def edit_score(worlds, state, t0, goal, n):              # -> [K, M, B, ...]
            M, B = worlds.shape[:2]
            flat = worlds.reshape((M * B,) + worlds.shape[2:])
            ek = jnp.stack([f(flat) for f in fns]).reshape((len(fns), M, B) + worlds.shape[2:])
            sc = score_many(ek.reshape((len(fns) * M, B) + worlds.shape[2:]),
                            state, t0, goal, n)
            return ek, tuple(s.reshape(len(fns), M, B) for s in sc)

        self._score_many = jax.jit(score_many, static_argnums=(4,))
        self._edit_score = jax.jit(edit_score, static_argnums=(4,)) if fns else None
        self._smooth = None
        if cfg.t > 0:
            ts, sc = int(cfg.t), float(cfg.scale)
            if ts > int(diffusion.timesteps):
                raise SystemExit(f"es '{cfg.name}': t={ts} exceeds the chain ({diffusion.timesteps})")
            self._smooth = jax.jit(lambda x, c1, c2, m, key: diffusion.resample(
                x, c1, c2, m, ts, rng=key, noise_scale=sc))

    # -- array plumbing -------------------------------------------------- #
    def _tile(self, tree, k: int):
        jnp = self.jnp
        return self.jax.tree_util.tree_map(
            lambda a: jnp.tile(jnp.asarray(a), (k,) + (1,) * (jnp.ndim(a) - 1)), tree)

    def _to_world(self, x_mbct, aux_m):
        M, B = x_mbct.shape[:2]
        flat = x_mbct.reshape((M * B,) + x_mbct.shape[2:])
        w = self.t.env.norm_to_world_with(self.jnp.transpose(flat, (0, 2, 1)), aux_m)
        return w.reshape((M, B) + w.shape[1:])

    def _to_norm(self, w_mbw5, x_fallback, aux_m):
        """world -> normalized ``[M, B, C, T]``; model steps past the scenario end
        cannot be inverted (``branching.model_step_valid``) and keep the member's
        previous normalized values, as counterfactual repair targets do."""
        jnp = self.jnp
        M, B = w_mbw5.shape[:2]
        back = self._w2n(w_mbw5.reshape((M * B,) + w_mbw5.shape[2:]), aux_m)   # [MB, T, 5]
        fb = jnp.transpose(x_fallback.reshape((M * B,) + x_fallback.shape[2:]), (0, 2, 1))
        out = jnp.where(self._valid(aux_m)[..., None], back, fb)
        return jnp.transpose(out, (0, 2, 1)).reshape(x_fallback.shape)

    def _scores(self, worlds, state, t0j, goal, n):
        # np.array, not asarray: a view of a device buffer is read-only, and the
        # caller overwrites member 0's row.
        return {k: np.array(v) for k, v in
                zip(SCORE_KEYS, self._score_many(worlds, state, t0j, goal, n))}

    def _roundtrip_check(self, x_p, world_p, aux):
        """Once: the smoothing path goes world -> norm -> world, so the inverse must
        return the base's own sample on covered steps (~2 cm interpolation residual
        expected; a metre would mean the smoother sees a different trajectory)."""
        if self._rt_checked:
            return
        jnp = self.jnp
        err = jnp.abs(self._w2n(world_p, aux) - jnp.transpose(x_p, (0, 2, 1)))[..., :2]
        err = jnp.where(self._valid(aux)[..., None], err, 0.0)
        worst = float(jnp.max(err)) * self._ego_range
        print(f"[es:{self.cfg.name}] world<->norm round trip on covered steps: max "
              f"{worst:.3f} m", flush=True)
        if worst > 1.0:
            raise SystemExit(f"[es:{self.cfg.name}] round trip is lossy ({worst:.2f} m): "
                             f"the smoother would not see the projected trajectory")
        self._rt_checked = True

    # -- operators -------------------------------------------------------- #
    def _project_opt(self, w, x, s, aux_m, state, t0j, goal, n):
        """Loss-optimized lateral repair of each violating member, from init 0 --
        the `o_0` arm's call, member by member at the runner's own batch shape, so
        member 0 goes through exactly o_0's computation. Rows are independent in the
        optimizer; the mask only decides which rows feel the barrier. A member takes
        its optimized edit when that is clean and keeps its own progress within
        ``tol`` (o_0's cons rule), else when it is at least clearer (a start for the
        next generation, never a certificate). ``ch`` is 0 where edited, -1 not."""
        jnp = self.jnp
        from rl.diffusion_ft.branching import lateral_offset
        from rl.diffusion_ft.lateral_opt import expand_offset

        M, B = s["clear"].shape
        W = int(w.shape[2])
        rows = np.ones(B, bool) if self._rows is None else self._rows
        need = (s["clear"] < 0.0) & rows[None]
        ch = np.full((M, B), -1, np.int32)
        if not need.any():
            return w, x, ch, s
        s_new = {k: np.array(v) for k, v in s.items()}
        x0 = jnp.zeros((B, n), jnp.float32)
        ws = []
        for m in range(M):
            if not need[m].any():
                ws.append(w[m])
                continue
            d, _, _ = self._opt(w[m], x0, state, t0j, jnp.asarray(need[m], jnp.float32), n)
            wo = lateral_offset(w[m], expand_offset(d, W), dt=self._opt_dt)
            so = [np.asarray(v) for v in self._score_fn(wo, state, t0j, goal, n)]
            elig = (so[0] >= 0.0) & (so[3] >= s["prog"][m] - self.tol_m)
            take = need[m] & (elig | (so[0] > s["clear"][m]))
            ch[m] = np.where(take, 0, -1)
            ws.append(jnp.where(jnp.asarray(take)[:, None, None], wo, w[m]))
            for j, k in enumerate(SCORE_KEYS):
                s_new[k][m] = np.where(take, so[j], s[k][m])
        w_new = jnp.stack(ws, 0)
        selj = jnp.asarray(ch >= 0)[..., None, None]
        return w_new, jnp.where(selj, self._to_norm(w_new, x, aux_m), x), ch, s_new

    def _project(self, w, x, s, aux_m, state, t0j, goal, n):
        if self._opt is not None:
            return self._project_opt(w, x, s, aux_m, state, t0j, goal, n)
        jnp = self.jnp
        M, B = s["clear"].shape
        ch = np.full((M, B), -1, np.int32)
        if self._edit_score is None or not (s["clear"] < 0.0).any():
            return w, x, ch, s
        ek, sk = self._edit_score(w, state, t0j, goal, n)
        sk = [np.asarray(v) for v in sk]
        ch = project_choice(s["clear"], s["prog"], sk[0], sk[3], self._mag, self.tol_m)
        sel = ch >= 0
        if not sel.any():
            return w, x, ch, s
        mi, bi = np.indices((M, B))
        chc = np.maximum(ch, 0)
        selj = jnp.asarray(sel)[..., None, None]
        w_new = jnp.where(selj, ek[chc, mi, bi], w)
        x_new = jnp.where(selj, self._to_norm(w_new, x, aux_m), x)
        s_new = {k: np.where(sel, sk[j][chc, mi, bi], s[k]) for j, k in enumerate(SCORE_KEYS)}
        return w_new, x_new, ch, s_new

    def _smooth_members(self, w, x, s, touch, conds, aux_m, state, t0j, goal, n, key):
        """Re-noise/denoise the ``touch`` members; everything else is returned bit
        for bit, so an untouched member keeps its exact trajectory and scores."""
        jnp = self.jnp
        if self._smooth is None or not touch.any():
            return w, x, s
        M, B = touch.shape
        c1, c2, m = conds
        y = self._smooth(x.reshape((M * B,) + x.shape[2:]), c1, c2, m, key).reshape(x.shape)
        tj = jnp.asarray(touch)[..., None, None]
        y = jnp.where(tj, y, x)
        wy = jnp.where(tj, self._to_world(y, aux_m), w)
        sy = self._scores(wy, state, t0j, goal, n)
        return wy, y, {k: np.where(touch, sy[k], s[k]) for k in SCORE_KEYS}

    def _fitness(self, pool, world_p, pp, n):
        dev = window_dev(pool["w"], world_p, n)
        return es_fitness(pool["s"]["clear"], pool["s"]["prog"], dev, prog_ref=pp,
                          tol_m=self.tol_m, rank=self.cfg.rank), dev

    @staticmethod
    def _gather(pool, idx):
        """``idx`` [k, B] member indices -> a pool of k members."""
        B = idx.shape[1]
        bi = np.arange(B)[None, :]
        return {"w": pool["w"][idx, bi], "x": pool["x"][idx, bi],
                "s": {k: v[idx, bi] for k, v in pool["s"].items()},
                **{k: pool[k][idx, bi] for k in ("gen", "edit", "line")}}

    @staticmethod
    def _concat(a, b, jnp):
        return {"w": jnp.concatenate([a["w"], b["w"]], 0),
                "x": jnp.concatenate([a["x"], b["x"]], 0),
                "s": {k: np.concatenate([a["s"][k], b["s"][k]], 0) for k in SCORE_KEYS},
                **{k: np.concatenate([a[k], b[k]], 0) for k in ("gen", "edit", "line")}}

    # -- one replan ------------------------------------------------------- #
    def __call__(self, *, x_p, world_p, cp, pp, cond1, cond2, mask, aux, state, t0, goal,
                 n: int, rk, alive):
        jax, jnp = self.jax, self.jnp
        B = int(world_p.shape[0])
        P, E, tol = self.P, self.E, self.tol_m
        cp, pp = np.asarray(cp), np.asarray(pp)
        trig = (np.asarray(alive) > 0.5) & (cp < 0.0)
        diag = np.full((B, len(ES_DIAG)), np.nan, np.float32)
        diag[:, 0] = trig
        if not trig.any():
            return world_p, None, diag
        tic = time.time()
        self._rows = trig
        ps = self.cfg.order == "ps"
        self._roundtrip_check(x_p, world_p, aux)
        t0j = jnp.asarray(t0, jnp.int32)
        aux_p = self._tile(aux, P)
        conds = tuple(self._tile(a, P) for a in (cond1, cond2, mask))

        # generation 0: member 0 is the base proposal itself
        if P > 1:
            c1, c2, m = (self._tile(a, P - 1) for a in (cond1, cond2, mask))
            x_new, _ = self.t.actor.sample_world_with_prefix(
                self.params, c1, c2, m, jax.random.fold_in(rk, _POP_SALT))
            x_pop = jnp.concatenate([x_p[None], x_new.reshape((P - 1, B) + x_new.shape[1:])], 0)
        else:
            x_pop = x_p[None]
        w_raw = self._to_world(x_pop, aux_p).at[0].set(world_p)
        s_raw = self._scores(w_raw, state, t0j, goal, n)
        # Member 0's decision-relevant scores are the runner's own, bit for bit, so
        # a float-level difference at clear ~ 0 cannot split es_p1 from the bank.
        s_raw["clear"][0], s_raw["prog"][0] = cp, pp
        w_proj, x_proj, ch, s_proj = self._project(w_raw, x_pop, s_raw, aux_p, state, t0j, goal, n)
        edited = ch >= 0
        f_raw = feasible(s_raw["clear"], s_raw["prog"], pp, tol)
        f_proj = feasible(s_proj["clear"], s_proj["prog"], pp, tol)
        if ps:
            # the smoothing reading: re-noise/denoise what the projection touched
            touch = edited | ~f_proj
            w_y, x_y, s_y = self._smooth_members(w_proj, x_proj, s_proj, touch, conds, aux_p,
                                                 state, t0j, goal, n,
                                                 jax.random.fold_in(rk, _POP_SALT + 1))
        else:
            # "sp": fresh samples need no mutation; the projection is the last word
            w_y, x_y, s_y = w_proj, x_proj, s_proj
        f_y = feasible(s_y["clear"], s_y["prog"], pp, tol)

        d = {"n_raw_feas": f_raw.sum(0), "n_proj_feas": f_proj.sum(0),
             "n_gen0_feas": f_y.sum(0), "n_edit": edited.sum(0),
             "n_edit_feas": (edited & f_proj).sum(0),
             "n_edit_kept": (edited & f_proj & f_y).sum(0)}
        if ps and self._smooth is not None and edited.any():
            wr, wp, wy = np.asarray(w_raw), np.asarray(w_proj), np.asarray(w_y)
            sr = np.where(edited, survival_ratio(wr, wp, wy, n), np.nan)
            disp = np.linalg.norm(wy[..., 1:n + 1, :2] - wp[..., 1:n + 1, :2], axis=-1).max(-1)
            with warnings.catch_warnings():       # all-NaN columns = no edit in that scene
                warnings.simplefilter("ignore", RuntimeWarning)
                d["survival"] = np.nanmean(sr, 0)
                d["smooth_disp_m"] = np.nanmean(np.where(edited, disp, np.nan), 0)

        pool = {"w": w_y, "x": x_y, "s": s_y, "gen": np.zeros((P, B), np.int32),
                "edit": edited, "line": np.broadcast_to(np.arange(P)[:, None], (P, B)).copy()}
        rep = replicate(E, P)
        for g in range(1, int(self.cfg.iters) + 1):
            fit, _ = self._fitness(pool, world_p, pp, n)
            el = self._gather(pool, top_k(fit, E))
            kid = self._gather(el, np.broadcast_to(rep[:, None], (P, B)))
            key = jax.random.fold_in(rk, _POP_SALT + 1 + g)
            every = np.ones((P, B), bool)
            if ps:      # project (a no-op on feasible elites), then re-noise/denoise
                w_k, x_k, ch_k, s_k = self._project(kid["w"], kid["x"], kid["s"], aux_p,
                                                    state, t0j, goal, n)
                w_n, x_n, s_n = self._smooth_members(w_k, x_k, s_k, every, conds, aux_p,
                                                     state, t0j, goal, n, key)
            else:       # mutate (exploration), then project the mutants
                w_k, x_k, s_k = self._smooth_members(kid["w"], kid["x"], kid["s"], every,
                                                     conds, aux_p, state, t0j, goal, n, key)
                w_n, x_n, ch_k, s_n = self._project(w_k, x_k, s_k, aux_p, state, t0j, goal, n)
            new = {"w": w_n, "x": x_n, "s": s_n, "gen": np.full((P, B), g, np.int32),
                   "edit": kid["edit"] | (ch_k >= 0), "line": kid["line"]}
            pool = self._concat(el, new, jnp)

        fit, dev = self._fitness(pool, world_p, pp, n)
        best = top_k(fit, 1)[0]                                              # [B]
        bi = np.arange(B)
        s_best = {k: pool["s"][k][best, bi] for k in SCORE_KEYS}
        f_pool = feasible(pool["s"]["clear"], pool["s"]["prog"], pp, tol)
        d.update({"n_final_feas": f_pool.sum(0), "best_feas": f_pool[best, bi],
                  "best_gen": pool["gen"][best, bi], "best_edited": pool["edit"][best, bi],
                  "best_line0": pool["line"][best, bi] == 0, "best_dev_m": dev[best, bi]})
        d["wall_s"] = np.full(B, time.time() - tic)
        for j, col in enumerate(ES_DIAG):
            if col in d:
                diag[trig, j] = np.asarray(d[col], np.float32)[trig]
        return pool["w"][best, bi], s_best, diag


def build_es(runner, spec: str, arms, *, bank_spec: str, banks_spec: str, ramp_steps: int,
             bank_ramps: dict | None, bank_blends: dict | None,
             bank_neighbors: dict | None):
    """``(configs, {name: PopulationES})`` for the configs the arms reference."""
    from rl.diffusion_ft.proposal_select import make_banks, make_scorer

    cfgs = parse_es_specs(spec)
    want = {a.es for a in arms if a.alt == "es"}
    missing = want - set(cfgs)
    if missing:
        raise SystemExit(f"--arms reference ES config(s) {sorted(missing)} that --select_es "
                         f"does not define (have {sorted(cfgs)})")
    cfgs = {k: v for k, v in cfgs.items() if k in want}
    banks = (make_banks(bank_spec, banks_spec, frozenset(getattr(runner, "bank_opts", {}) or {}))
             if any(c.bank for c in cfgs.values()) else {})
    t = runner.t
    out = {}
    for name, c in cfgs.items():
        cands, ramp, blend = [], int(ramp_steps), "cubic"
        if c.bank:
            if c.bank not in banks:
                raise SystemExit(f"es '{name}': bank '{c.bank}' is not defined by "
                                 f"--select_bank/--select_banks (have {sorted(banks)})")
            cands = list(banks[c.bank])
            if not cands:
                raise SystemExit(f"es '{name}': bank '{c.bank}' has no fixed candidates")
            ramp = int((bank_ramps or {}).get(c.bank, ramp_steps))
            blend = str((bank_blends or {}).get(c.bank, "cubic"))
            # One predictor per decision: a projection certified against the
            # recorded future and ranked against constant velocity would be
            # neither honest nor the oracle.
            bnb = (bank_neighbors or {}).get(c.bank)
            if bnb is not None and bnb != c.nb:
                raise SystemExit(f"es '{name}' scores with nb={c.nb} but its bank "
                                 f"'{c.bank}' is set to {bnb} (--select_bank_neighbors)")
        if c.nb not in runner._scorers:
            runner._scorers[c.nb] = make_scorer(t.objcfg, c.nb, t.env.world_dt_method)
        optimizer = None
        if c.opt:
            # The runner's own optimizer for this neighbour source when an `o_*` bank
            # arm built one (one compiled program for both), else the same factory
            # with the same --select_opt_cfg.
            opts = runner.__dict__.setdefault("_optimizers", {})
            if getattr(getattr(runner, "opt_cfg", None), "lon", 0):
                # The projection below builds [B, n] lateral-only parameters.
                raise SystemExit("es opt:1 projects laterally only; set lon=1 per bank "
                                 "(--select_bank_opt_cfgs), not in --select_opt_cfg")
            if c.nb not in opts:
                from rl.diffusion_ft.lateral_opt import LateralOptConfig, make_lateral_optimizer
                opts[c.nb] = make_lateral_optimizer(
                    t.objcfg, getattr(runner, "opt_cfg", None) or LateralOptConfig(),
                    dt=float(t.ckpt.preprocess_cfg.world_dt_fallback), neighbor_pred=c.nb,
                    world_dt_method=t.env.world_dt_method)
            optimizer = opts[c.nb]
        out[name] = PopulationES(t, c, params=runner.base_params, score_fn=runner._scorers[c.nb],
                                 cands=cands, ramp_steps=ramp, blend=blend, tol_m=runner.tol_m,
                                 optimizer=optimizer)
    return cfgs, out
