"""Pure-logic tests for the loss-optimized lateral repair (``lateral_opt.py``).

No model, no corpus, no GPU. What each block pins, and how it would lie if wrong:

1. the ``lateral_shift`` -> ``lateral_offset`` refactor is BIT-identical to the
   pre-refactor code (copied here verbatim) -- otherwise every ramp-20 number on
   record stops being reproducible;
2. the comfort Hessian is the Hessian of the quadratic the docstring states,
   boundary conditions included (checked against an independent jnp version
   differentiated by jax) -- a wrong pad would silently let the edit jump at the
   join or drift after the window;
3. the ramp IS the special case: minimizing sum a^2 subject to holding an offset
   A gives the cubic smoothstep, i.e. the ramp-20 profile -- the claim the module
   docstring rests on;
4. config / init parsing, and ``expand_offset``'s anchor + hold;
5. ``bank_choice`` with a per-row ``[K, B]`` magnitude picks the smallest eligible
   edit PER ROW, and a ``[K]`` magnitude still behaves exactly as before;
7. the longitudinal half (``lon``, 2026-09-30): ``path_retime`` is the identity at
   s = 0, moves poses exactly s along the path, never reverses, holds past the
   end and is differentiable; its comfort block is the Hessian of the stated
   velocity-hold quadratic; with ``lon=0`` the edit, the Hessian and the
   parameter vector are BIT-identical to the lateral-only o_0 ones; and the lon
   gradient stays finite through stationary steps and a held tail.

    python -m rl.diffusion_ft.tests.test_lateral_opt
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.branching import (_wrap, fd_velocity, lateral_offset, lateral_shift,
                                      path_retime)
from rl.diffusion_ft.lateral_opt import (LateralOptConfig, comfort_hessian, comfort_operators,
                                         edit_hessian, edit_world, expand_offset, expand_retime,
                                         parse_inits)  # noqa: F401
from rl.diffusion_ft.proposal_select import bank_choice

FAILED = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def raises(fn):
    try:
        fn()
    except (ValueError, SystemExit):
        return True
    return False


def _arc(B=3, W=52, v=8.0, dt=0.1, curv=0.03):
    t = np.arange(W) * dt
    s = v * t
    yaw = curv * s
    x = np.cumsum(np.concatenate([[0.0], v * dt * np.cos(yaw[:-1])]))
    y = np.cumsum(np.concatenate([[0.0], v * dt * np.sin(yaw[:-1])]))
    w = np.stack([x, y, yaw, v * np.cos(yaw), v * np.sin(yaw)], axis=-1)
    w = np.stack([w + np.array([10.0 * b, -3.0 * b, 0.4 * b, 0, 0]) for b in range(B)])
    return jnp.asarray(w, jnp.float32)


def _lateral_shift_pre_refactor(world_bw5, amp_m, *, ramp_steps, dt, blend="cubic"):
    """branching.lateral_shift as of f9831ae, verbatim."""
    B, W, _ = world_bw5.shape
    u = jnp.clip(jnp.arange(W, dtype=jnp.float32) / max(int(ramp_steps), 1), 0.0, 1.0)
    if blend == "quintic":
        ease = u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)
    elif blend == "cubic":
        ease = 3.0 * u ** 2 - 2.0 * u ** 3
    else:
        raise ValueError(f"unknown blend {blend!r} (cubic | quintic)")
    d = float(amp_m) * ease
    yaw = world_bw5[..., 2]
    n = jnp.stack([-jnp.sin(yaw), jnp.cos(yaw)], axis=-1)
    p = world_bw5[..., :2] + d[None, :, None] * n
    ds = jnp.linalg.norm(jnp.diff(world_bw5[..., :2], axis=1), axis=-1)
    dd = jnp.diff(d)[None, :]
    slope = jnp.arctan2(dd, jnp.maximum(ds, 0.05))
    dyaw = jnp.concatenate([jnp.zeros((B, 1)), slope], axis=1)
    w = world_bw5.at[..., :2].set(p).at[..., 2].set(_wrap(yaw + dyaw))
    return fd_velocity(w, dt)


def t_refactor():
    w = _arc()
    worst = 0.0
    for amp in (-1.6, -0.25, 0.0, 0.1, 0.75, 1.35):
        for ramp in (10, 20, 30):
            for blend in ("cubic", "quintic"):
                for jit in (False, True):
                    f_new = lambda x: lateral_shift(x, amp, ramp_steps=ramp, dt=0.1, blend=blend)
                    f_old = lambda x: _lateral_shift_pre_refactor(x, amp, ramp_steps=ramp, dt=0.1,
                                                                  blend=blend)
                    if jit:
                        f_new, f_old = jax.jit(f_new), jax.jit(f_old)
                    worst = max(worst, float(jnp.max(jnp.abs(f_new(w) - f_old(w)))))
    check(worst == 0.0, f"1 lateral_shift is bit-identical to the pre-refactor code "
                        f"(72 cases, eager+jit, max diff {worst:.1e})")
    d = jnp.asarray(np.random.default_rng(0).normal(size=(3, 52)).cumsum(1) * 0.05, jnp.float32)
    d = d.at[:, 0].set(0.0)
    out = lateral_offset(w, d, dt=0.1)
    check(bool(jnp.all(out[:, 0, :3] == w[:, 0, :3])), "1 lateral_offset keeps the anchor pose")
    off = np.asarray(out[..., :2] - w[..., :2])
    nrm = np.stack([-np.sin(np.asarray(w[..., 2])), np.cos(np.asarray(w[..., 2]))], -1)
    along = np.sum(off * nrm, -1)
    check(np.allclose(along, np.asarray(d), atol=1e-5) and
          np.allclose(np.linalg.norm(off, axis=-1), np.abs(np.asarray(d)), atol=1e-5),
          "1 lateral_offset moves each pose exactly d along its left normal")
    g = jax.grad(lambda z: jnp.sum(lateral_offset(w, z, dt=0.1)[..., 1]))(d)
    check(bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.1,
          "1 lateral_offset is differentiable in d")


def _quad_independent(x, n, dt, cfg):
    """The comfort quadratic written out directly, not through the operators."""
    B = x.shape[0]
    pad0 = jnp.zeros((B, 3))
    hold = jnp.repeat(x[:, -1:], 2, axis=1)
    D = jnp.concatenate([pad0, x, hold], axis=1)          # steps -2 .. n+2
    a = (D[:, 2:] - 2.0 * D[:, 1:-1] + D[:, :-2]) / dt ** 2  # steps -1 .. n+1
    j = (a[:, 1:] - a[:, :-1]) / dt
    q = (cfg.w_acc * jnp.sum((a / cfg.a_ref) ** 2, 1) + cfg.w_jerk * jnp.sum((j / cfg.j_ref) ** 2, 1)
         + cfg.w_prox * jnp.sum((x / cfg.d_ref) ** 2, 1)) / n
    return jnp.sum(q)


def t_hessian():
    for n in (20, 7):
        cfg = LateralOptConfig(w_acc=0.03, w_jerk=0.01, w_prox=0.002)
        H = comfort_hessian(n, 0.1, cfg)
        Hj = np.asarray(jax.hessian(lambda z: _quad_independent(z[None], n, 0.1, cfg))(
            jnp.zeros(n, jnp.float32)))
        rel = float(np.max(np.abs(H - Hj)) / np.max(np.abs(H)))
        check(rel < 1e-4, f"2 comfort_hessian (n={n}) is the Hessian of the stated quadratic "
                          f"(rel err {rel:.1e})")
        check(bool(np.all(np.linalg.eigvalsh(H) > 0)), f"2 H (n={n}) is positive definite")
    Ka, Kj = comfort_operators(20, 0.1)
    x = np.full(20, 1.0)
    check(np.abs(Ka @ x).max() > 0 and np.all(np.isfinite(Ka @ x)),
          "2 a rigid step from 0 at the anchor is CHARGED (the join is in the operator)")
    x = np.linspace(0.05, 1.0, 20) ** 2
    a = Ka @ x                                   # rows: steps -1 .. n+1
    check(abs(a[-2]) > 0 and a[-1] == 0,
          "2 ...and arriving with nonzero slope into the hold is charged too (step n)")


def t_cubic():
    """min x^T H x with x_n = A, accel only -> the smoothstep ramp."""
    for n in (20, 40):
        cfg = LateralOptConfig(w_acc=1.0, w_jerk=0.0)
        H = comfort_hessian(n, 0.1, cfg)
        A = 1.3
        y = -np.linalg.solve(H[:-1, :-1], H[:-1, -1] * A)
        x = np.concatenate([y, [A]])
        u = np.arange(1, n + 1) / n
        cubic = A * (3 * u ** 2 - 2 * u ** 3)
        err = float(np.max(np.abs(x - cubic)) / A)
        check(err < 0.06, f"3 accel-only optimum holding A is the cubic smoothstep, n={n} "
                          f"(max err {100 * err:.1f}% of A)")
    cfg = LateralOptConfig(w_acc=0.0, w_jerk=1.0)
    n = 40
    H = comfort_hessian(n, 0.1, cfg)
    y = -np.linalg.solve(H[:-1, :-1], H[:-1, -1])
    x = np.concatenate([y, [1.0]])
    u = np.arange(1, n + 1) / n
    quint = u ** 3 * (10 - 15 * u + 6 * u ** 2)
    err = float(np.max(np.abs(x - quint)))
    check(err < 0.08, f"3 jerk-only optimum is the quintic smootherstep (max err {100 * err:.1f}%)")


def t_parse():
    c = LateralOptConfig.parse("iters=12, w_acc=0.5,barrier_shape=softplus")
    check(c.iters == 12 and isinstance(c.iters, int) and c.w_acc == 0.5
          and c.barrier_shape == "softplus" and c.w_jerk == LateralOptConfig().w_jerk,
          "4 LateralOptConfig.parse sets typed fields and keeps defaults")
    check(LateralOptConfig.parse("") == LateralOptConfig(), "4 empty spec = defaults")
    check(raises(lambda: LateralOptConfig.parse("itres=3")), "4 unknown key is an error")
    check(raises(lambda: LateralOptConfig.parse("w_acc=0,w_jerk=0")),
          "4 no comfort weight (no preconditioner) is an error")
    check(parse_inits("0, best,lat1.0,lat-1.0") == ["0", "best", "lat1.0", "lat-1.0"],
          "4 parse_inits")
    check(raises(lambda: parse_inits("0,0")) and raises(lambda: parse_inits("tw0.7"))
          and raises(lambda: parse_inits("latx")), "4 bad / duplicate inits are errors")
    x = jnp.asarray([[0.1, 0.2, 0.3], [1.0, -1.0, 0.5]], jnp.float32)
    e = np.asarray(expand_offset(x, 8))
    check(e.shape == (2, 8) and np.all(e[:, 0] == 0) and np.allclose(e[:, 1:4], x)
          and np.allclose(e[:, 4:], np.asarray(x[:, -1:])),
          "4 expand_offset: 0 at the anchor, the window, then held")
    check(np.asarray(expand_offset(jnp.ones((1, 7)), 8)).shape == (1, 8),
          "4 expand_offset with no tail")


def t_choice():
    clear_p = np.array([-1.0, -1.0, 0.5])
    prog_p = np.zeros(3)
    clear_k = np.array([[0.2, -0.1, 0.3], [0.1, 0.4, 0.3], [0.5, 0.5, 0.5]])
    prog_k = np.zeros((3, 3))
    kw = dict(tol_m=1.0, margin_m=0.3, delta_m=0.2)
    old = bank_choice("cons", clear_p=clear_p, prog_p=prog_p, clear_k=clear_k, prog_k=prog_k,
                      magnitude=[0.3, 0.5, 0.9], **kw)
    check(old.tolist() == [0, 1, -1], f"5 [K] magnitude unchanged ({old.tolist()})")
    mag = np.array([[0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.2, 0.9, 0.1]])
    new = bank_choice("cons", clear_p=clear_p, prog_p=prog_p, clear_k=clear_k, prog_k=prog_k,
                      magnitude=mag, **kw)
    check(new.tolist() == [2, 1, -1], f"5 [K,B] magnitude: smallest eligible PER ROW, "
                                      f"clean base untouched ({new.tolist()})")
    check(raises(lambda: bank_choice("cons", clear_p=clear_p, prog_p=prog_p, clear_k=clear_k,
                                     prog_k=prog_k, magnitude=np.ones((3, 2)), **kw)),
          "5 a magnitude of the wrong row count is an error")
    # Margin preference, prefer_k[k, b]. Row 0: clean k=0 (small) and k=2 (large);
    # only k=2 is preferred, so it wins. Row 1: the only preferred candidate (k=0)
    # is NOT clean, so the smallest clean one (k=1) still fires -- the fallback
    # tier. Row 2: the base is clean, so nothing fires whatever is preferred.
    pref = np.array([[False, True, True], [False, False, True], [True, False, True]])
    got = bank_choice("cons", clear_p=clear_p, prog_p=prog_p, clear_k=clear_k, prog_k=prog_k,
                      magnitude=[0.3, 0.5, 0.9], prefer_k=pref, **kw)
    check(got.tolist() == [2, 1, -1],
          f"5 prefer_k: preferred clean wins, else the smallest clean fires, clean base "
          f"untouched ({got.tolist()})")
    none_pref = bank_choice("cons", clear_p=clear_p, prog_p=prog_p, clear_k=clear_k,
                            prog_k=prog_k, magnitude=[0.3, 0.5, 0.9],
                            prefer_k=np.zeros((3, 3), bool), **kw)
    check(none_pref.tolist() == old.tolist(),
          "5 prefer_k all False is exactly the plain rule (same coverage, same choice)")
    fires = lambda c: (c >= 0).tolist()
    check(fires(got) == fires(old), "5 prefer_k never changes WHETHER a row fires")


def t_overrides():
    g = "w_acc=0.05,iters=20"
    c = LateralOptConfig.parse(f"{g},edge_margin_m=0.05,agent_margin_m=0.05,iters=12")
    check(c.iters == 12 and c.w_acc == 0.05 and c.edge_margin_m == 0.05
          and c.agent_margin_m == 0.05,
          "6 per-bank overrides compose onto the global spec (later key wins)")
    check(LateralOptConfig.parse(f"{g},") == LateralOptConfig.parse(g),
          "6 an empty per-bank override is the global config")
    d = LateralOptConfig()
    check(d.edge_margin_m is None and d.agent_margin_m is None,
          "6 default loss margins inherit the run's calibrated ones (None)")
    from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig
    import dataclasses as dc
    obj = SafetyObjectiveConfig()
    m = {k: v for k, v in (("edge_margin_m", c.edge_margin_m),
                           ("agent_margin_m", c.agent_margin_m)) if v is not None}
    s2 = dc.replace(obj.safety, **m)
    check(s2.edge_margin_m == 0.05 and s2.agent_margin_m == 0.05
          and obj.safety.edge_margin_m == 0.3 and obj.safety.agent_margin_m == 0.75,
          "6 the override reaches SafetyLossConfig and leaves the calibrated one alone")


def _quad_velocity_hold(x, n, dt, w_acc, w_jerk):
    """The longitudinal comfort quadratic written out directly: zeros before, the
    last SLOPE continued for two steps after."""
    B = x.shape[0]
    full = jnp.concatenate([jnp.zeros((B, 3)), x], axis=1)            # steps -2 .. n
    slope = full[:, -1:] - full[:, -2:-1]
    D = jnp.concatenate([full, full[:, -1:] + slope, full[:, -1:] + 2.0 * slope], axis=1)
    a = (D[:, 2:] - 2.0 * D[:, 1:-1] + D[:, :-2]) / dt ** 2
    j = (a[:, 1:] - a[:, :-1]) / dt
    return jnp.sum((w_acc * jnp.sum(a ** 2, 1) + w_jerk * jnp.sum((j / 10.0) ** 2, 1)) / n)


def t_lon():
    w = _arc(B=2, W=51, v=8.0, curv=0.0)                   # straight, 0.8 m per step
    z = jnp.zeros((2, 51), jnp.float32)
    r0 = path_retime(w, z, dt=0.1)
    check(float(jnp.max(jnp.abs(r0[..., :3] - w[..., :3]))) < 1e-4,
          "7 path_retime at s = 0 is the identity on pose")
    s = jnp.full((2, 51), 0.3, jnp.float32).at[:, 0].set(0.0)
    r = path_retime(w, s, dt=0.1)
    # Along the path's own direction (from its positions): _arc offsets row b's
    # heading by 0.4 b without rotating its positions, so `yaw` is not the tangent.
    tan = np.diff(np.asarray(w[..., :2]), axis=1)[:, 1:45]
    tan /= np.linalg.norm(tan, axis=-1, keepdims=True)
    dxy = np.asarray(r[:, 1:45, :2] - w[:, 1:45, :2])
    fwd, side = np.sum(dxy * tan, -1), np.abs(tan[..., 0] * dxy[..., 1] - tan[..., 1] * dxy[..., 0])
    check(np.allclose(fwd, 0.3, atol=2e-3) and side.max() < 1e-4
          and bool(jnp.all(r[:, 0, :2] == w[:, 0, :2])),
          f"7 a constant s moves every pose s metres along the path, anchor kept "
          f"(along {fwd.min():.4f}..{fwd.max():.4f}, off-path {side.max():.1e})")
    back = jnp.asarray(-np.arange(51) * 1.5, jnp.float32)[None].repeat(2, 0)  # faster than v
    rb = np.asarray(path_retime(w, back, dt=0.1))
    prog = np.diff(rb[..., 0], axis=1)
    check(bool(np.all(prog >= -1e-6)), "7 path_retime never reverses (stops at worst)")
    far = jnp.full((2, 51), 1e3, jnp.float32).at[:, 0].set(0.0)
    rf = np.asarray(path_retime(w, far, dt=0.1))
    check(np.allclose(rf[:, 1:, :2], np.asarray(w[:, -1:, :2]), atol=1e-3),
          "7 running past the end holds the last pose")
    wc = _arc(B=2, W=51, v=8.0, curv=0.05)
    g = jax.grad(lambda q: jnp.sum(path_retime(wc, q, dt=0.1)[..., 1]))(
        jnp.full((2, 51), 0.2, jnp.float32).at[:, 0].set(0.0))
    check(bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.1,
          "7 path_retime is differentiable in s")
    stop = w.at[:, 10:, :2].set(w[:, 10:11, :2]).at[:, 10:, 2].add(
        jnp.asarray(np.random.default_rng(1).normal(size=41) * 0.05, jnp.float32))
    gs = jax.grad(lambda q: jnp.sum(path_retime(stop, q, dt=0.1)[..., :3]))(
        jnp.full((2, 51), 0.2, jnp.float32).at[:, 0].set(0.0))
    check(bool(jnp.all(jnp.isfinite(gs))) and float(jnp.max(jnp.abs(gs))) < 10.0,
          f"7 ...and stays finite and bounded through a stop with jittering heading "
          f"(max |grad| {float(jnp.max(jnp.abs(gs))):.2f})")

    for n in (20, 7):
        cfg = LateralOptConfig(lon=1, w_lon_acc=0.03, w_lon_jerk=0.01)
        H = edit_hessian(n, 0.1, cfg)
        Hs = np.asarray(jax.hessian(lambda q: _quad_velocity_hold(q[None], n, 0.1, 0.03, 0.01))(
            jnp.zeros(n, jnp.float32)))
        rel = float(np.max(np.abs(H[n:, n:] - Hs)) / np.max(np.abs(Hs)))
        check(H.shape == (2 * n, 2 * n) and rel < 1e-4 and np.all(H[:n, n:] == 0)
              and np.array_equal(H[:n, :n], comfort_hessian(n, 0.1, cfg)),
              f"7 edit_hessian (n={n}, lon): lateral block + the velocity-hold block, "
              f"no coupling (rel err {rel:.1e})")
        check(bool(np.all(np.linalg.eigvalsh(H) > 0)), f"7 lon H (n={n}) is positive definite")
    Ka, _ = comfort_operators(20, 0.1, hold="velocity")
    ramp = np.arange(1, 21) * 0.1
    check(abs((Ka @ ramp)[-1]) < 1e-9 and abs((Ka @ ramp)[-2]) < 1e-9,
          "7 a speed change carried through the window end is NOT charged there")
    cfg0 = LateralOptConfig()
    check(np.array_equal(edit_hessian(20, 0.1, cfg0), comfort_hessian(20, 0.1, cfg0)),
          "7 lon=0: edit_hessian IS comfort_hessian")
    e = np.asarray(expand_retime(jnp.asarray([[0.1, 0.3, 0.6]], jnp.float32), 7))
    check(np.allclose(e, [[0.0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]]),
          "7 expand_retime: 0 at the anchor, the window, then the last slope continued")

    d = jnp.asarray(np.random.default_rng(2).normal(size=(3, 20)).cumsum(1) * 0.05, jnp.float32)
    wa = _arc()
    old_edit = lateral_offset(wa, expand_offset(d, wa.shape[1]), dt=0.1)
    check(bool(jnp.all(edit_world(wa, d, n=20, dt=0.1, lon=0) == old_edit)),
          "7 lon=0: edit_world is bit-identical to o_0's lateral_offset(expand_offset(x))")
    both = edit_world(wa, jnp.concatenate([d, jnp.zeros_like(d)], 1), n=20, dt=0.1, lon=1)
    check(float(jnp.max(jnp.abs(both[..., :3] - old_edit[..., :3]))) < 2e-3,
          "7 lon=1 with s = 0 is the lateral edit (to interpolation precision)")
    # The rollout's plans stop (a queue, a light) and a `prev` tail is padded by
    # holding its last pose: consecutive poses coincide. The plain norm inside
    # lateral_offset then made the WHOLE row's lon gradient NaN (09-30, the lon
    # optimizer froze on 56% of rows) -- even at steps outside the loss.
    held = w.at[:, 30:, :2].set(w[:, 30:31, :2]).at[:, 12:16, :2].set(w[:, 12:13, :2])
    x = jnp.concatenate([jnp.full((2, 20), 0.1), jnp.full((2, 20), 0.2)], 1).astype(jnp.float32)
    g = jax.grad(lambda q: jnp.sum(edit_world(held, q, n=20, dt=0.1, lon=1)[:, 1:21, :2] ** 2))(x)
    check(bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g[:, 20:]))) > 0.0,
          "7 lon gradient is finite (and nonzero along-path) through a stop and a held tail")
    far = x.at[:, 20:].set(50.0)
    g2 = jax.grad(lambda q: jnp.sum(edit_world(held, q, n=20, dt=0.1, lon=1)[:, 1:21, :2] ** 2))(far)
    check(bool(jnp.all(jnp.isfinite(g2))), "7 ...and when the retime runs past the path end")
    g0 = jax.grad(lambda q: jnp.sum(edit_world(held, q, n=20, dt=0.1, lon=0)[:, 1:21, :2] ** 2))(x[:, :20])
    check(bool(jnp.all(jnp.isfinite(g0))), "7 lateral-only gradient finite on the same plan")
    c = LateralOptConfig.parse("lon=1,lon_back_m=0.5")
    check(c.lon == 1 and isinstance(c.lon, int) and c.lon_back_m == 0.5,
          "7 lon / lon_back_m parse")
    check(raises(lambda: LateralOptConfig.parse("lon=2"))
          and raises(lambda: LateralOptConfig.parse("lon=1,w_lon_acc=0,w_lon_jerk=0"))
          and raises(lambda: LateralOptConfig.parse("lon_back_m=-1")),
          "7 bad lon settings are errors")
    check(parse_inits("0,prev,lon0,lonprev") == ["0", "prev", "lon0", "lonprev"],
          "7 parse_inits accepts prev / lon0 / lonprev")
    from rl.diffusion_ft.lateral_opt import init_kind
    from rl.diffusion_ft.proposal_select import OPT_TIER_M
    kinds = [init_kind(t) for t in ("0", "best", "lat0.5", "lon0", "prev", "lonprev")]
    tiers = [OPT_TIER_M[k] for k in kinds]
    check(tiers[:3] == [0.0] * 3 and tiers[3] < tiers[4] < tiers[5] and tiers[3] >= 50.0,
          f"7 tiers: fresh lateral < fresh lon < prev lateral < prev lon, gaps >> an edit "
          f"({tiers})")


def t_proj():
    """`proj` = cons on violating rows, plus a strictly-clearer clean edit on clean
    rows; an unmoved edit carries the base's own scores and must never fire."""
    from rl.diffusion_ft.proposal_select import PROJ_MIN_GAIN_M, parse_arms, select_switch
    kw = dict(tol_m=1.0, margin_m=0.3, delta_m=0.2)
    #          viol+clean  unmoved  clearer  barely   a violates  prog lost  viol+still
    cp = np.array([-0.5,    0.40,    0.40,    0.40,    0.40,       0.40,      -0.5])
    ca = np.array([0.10,    0.40,    0.70,    0.40 + PROJ_MIN_GAIN_M / 2, -0.1, 0.70, -0.1])
    pp = np.zeros(7)
    pa = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0])
    got = select_switch("proj", clear_p=cp, clear_a=ca, prog_p=pp, prog_a=pa, **kw)
    check(got.tolist() == [True, False, True, False, False, False, False],
          f"8 proj: cons where violating; on clean rows only a clean, prog-ok edit at "
          f"least PROJ_MIN_GAIN_M clearer; unmoved never ({got.tolist()})")
    cons = select_switch("cons", clear_p=cp, clear_a=ca, prog_p=pp, prog_a=pa, **kw)
    check(bool(np.all(got[cp < 0] == cons[cp < 0])), "8 proj == cons on violating rows")
    a = parse_arms("x=bank@o:proj:20,y=bank@o:proj:20:0.3")
    check(a[0].theta is None and a[1].theta == 0.3 and a[1].selector == "proj",
          "8 proj arms parse, with an optional trigger theta")
    check(raises(lambda: parse_arms("z=ft:proj:20")) and raises(lambda: parse_arms("z=bank@o:cons:20:0.3")),
          "8 proj needs a bank; theta stays illegal on cons")


def t_emerg():
    """Emergency tiers: token parsing, their configs, their rank, and `brake`'s
    widened progress tolerance reaching bank_choice."""
    from rl.diffusion_ft.lateral_opt import (BRAKE_BACK_M, EMERG_COMFORT_SCALE, init_kind,
                                             init_variant, variant_cfg)
    from rl.diffusion_ft.proposal_select import EMERG_TIER_M, OPT_TIER_M
    check(parse_inits("0,lon0,prev,lonprev,emerg,brake")[-2:] == ["emerg", "brake"]
          and init_variant("emerg") == "emerg" and init_variant("lon0") == ""
          and init_kind("brake") == (False, True), "9 emerg / brake parse as fresh lon tokens")
    c = LateralOptConfig()
    e, b, l = variant_cfg(c, "emerg", True), variant_cfg(c, "brake", True), variant_cfg(c, "", True)
    k = EMERG_COMFORT_SCALE
    check(e.lon == 1 and abs(e.w_acc - c.w_acc * k) < 1e-12 and abs(e.w_lon_jerk - c.w_lon_jerk * k) < 1e-12
          and e.lon_back_m == c.lon_back_m and b.lon_back_m == BRAKE_BACK_M
          and b.w_acc == e.w_acc and l == dataclasses.replace(c, lon=1) and variant_cfg(c, "", False) == c,
          "9 variant_cfg: comfort x scale for both, braking allowance only for brake; ordinary untouched")
    check(min(EMERG_TIER_M.values()) > max(OPT_TIER_M.values()) and EMERG_TIER_M["emerg"] < EMERG_TIER_M["brake"],
          f"9 emergency tiers rank below every ordinary kind, emerg before brake ({EMERG_TIER_M})")
    kw = dict(margin_m=0.3, delta_m=0.2)
    clear_p, prog_p = np.array([-1.0]), np.array([5.0])
    ck, pk = np.array([[0.5], [0.5]]), np.array([[5.0 - 3.0], [5.0 - 3.0]])   # both lose 3 m
    got = bank_choice("cons", clear_p=clear_p, prog_p=prog_p, clear_k=ck, prog_k=pk,
                      magnitude=[0.1, 500.0], tol_m=1.0, tol_k=[1.0, 1.0 + BRAKE_BACK_M], **kw)
    base = bank_choice("cons", clear_p=clear_p, prog_p=prog_p, clear_k=ck, prog_k=pk,
                       magnitude=[0.1, 500.0], tol_m=1.0, **kw)
    check(got.tolist() == [1] and base.tolist() == [-1],
          "9 tol_k: only the widened-tolerance candidate may give up 3 m of progress")


def t_argmin():
    """The argmin arm's pieces: prev-tail projection round-trips through edit_world,
    w_lon_prox lands in the Hessian, and the arm/bank grammar."""
    from rl.diffusion_ft.lateral_opt import project_onto_path
    from rl.diffusion_ft.proposal_select import ARGMIN_SELECTORS, parse_arms
    n = 20
    for curv in (0.0, 0.04):
        # One row per profile: _arc offsets row b's heading by 0.4 b WITHOUT
        # rotating its positions, and edit_world offsets along the heading normal,
        # so only b = 0 has heading == path direction (as real plans do).
        w0 = _arc(B=1, W=51, v=7.0, curv=curv)
        w = jnp.concatenate([w0, w0], axis=0)
        u = np.arange(1, n + 1) / n
        d = np.stack([0.8 * (3 * u ** 2 - 2 * u ** 3), -0.5 * np.sin(np.pi * u)])
        sv = np.stack([1.2 * u ** 2, -0.6 * u])
        x = jnp.asarray(np.concatenate([d, sv], 1), jnp.float32)
        q = edit_world(w, x, n=n, dt=0.1, lon=1)
        xh = project_onto_path(w, q, n)
        err_d = float(np.abs(xh[:, :n] - np.asarray(x[:, :n])).max())
        err_s = float(np.abs(xh[:, n:] - np.asarray(x[:, n:])).max())
        check(err_d < 0.05 and err_s < 0.05,   # an init: cm-level is plenty
              f"10 project_onto_path inverts edit_world (curv {curv}): max err d {err_d:.3f} m, s {err_s:.3f} m")
    cfg = LateralOptConfig(lon=1, w_lon_acc=0.03, w_lon_jerk=0.01, w_lon_prox=0.05)
    H = edit_hessian(n, 0.1, cfg)
    H0 = edit_hessian(n, 0.1, dataclasses.replace(cfg, w_lon_prox=0.0))
    check(np.allclose(H[n:, n:] - H0[n:, n:], (2.0 / n) * 0.05 * np.eye(n)) and np.array_equal(H[:n, :n], H0[:n, :n]),
          "10 w_lon_prox adds (2/n) w I to the lon block only")
    a = parse_arms("x=bank@am:argmin:20,y=bank@am:argmin_c:20")
    check([q.selector for q in a] == list(ARGMIN_SELECTORS), "10 argmin arms parse")
    check(raises(lambda: parse_arms("z=ft:argmin:20")), "10 argmin needs a bank")
    check(parse_inits("init0,initprev") == ["init0", "initprev"], "10 argmin start tokens parse")


def t_amfix():
    """11. o_am failure-anatomy fixes (2026-10-11), all default off.

    join: the history pad reduces to the old zero pad (comfort_join at h = 0 is
    0.5 x^T H x), the boundary columns are the padded profile's own differences,
    join_history is zero on the fresh plan's own backward line, and the claim
    the fix rests on -- with the car's real history, continuing the previous plan
    is CHEAPER than the fresh plan's jump, where the zero pad billed it the jump.
    curvature: zero at x = 0, negligible for a lane change at speed, huge for a
    stationary car's 2 cm/step slide, finite gradient through a stop.
    starts: shapes, signs, and a brake that never reverses and comes to rest.
    """
    from rl.diffusion_ft.lateral_opt import (START_SWERVE_M, comfort_boundary, comfort_join,
                                             edit_curvature, join_history, project_onto_path,
                                             start_profile)
    n, dt = 20, 0.1
    c0 = LateralOptConfig()
    check((c0.join, c0.w_contact_edge, c0.w_contact_agent, c0.w_curv) == (0, 0.0, 0.0, 0.0),
          "11 the new terms default off")
    c = LateralOptConfig.parse("lon=1,w_lon_prox=0.05,join=1,w_contact_edge=1,"
                               "w_contact_agent=1,w_curv=0.02")
    check(c.join == 1 and c.w_curv == 0.02 and c.w_contact_agent == 1.0, "11 new keys parse")
    check(all(raises(lambda b=b: LateralOptConfig.parse(b))
              for b in ("join=2", "w_curv=-1", "contact_eps_m=0", "k_ref=0", "w_contact_edge=-1")),
          "11 bad new keys raise")
    check(parse_inits("init0,initprev,initbrake,initleft,initright")
          == ["init0", "initprev", "initbrake", "initleft", "initright"], "11 new start tokens parse")

    # -- boundary columns = differences of the padded profile ----------------- #
    rng = np.random.default_rng(3)
    Ka, Kj = comfort_operators(n, dt)
    Ba, Bj = comfort_boundary(n, dt)
    d = rng.normal(size=n) * 0.2
    h = rng.normal(size=2) * 0.1
    pad = np.concatenate([h, [0.0], d, [d[-1], d[-1]]])           # steps -2 .. n+2, value hold
    a_ref = np.diff(pad, 2) / dt ** 2
    j_ref = np.diff(a_ref) / dt
    check(np.allclose(Ka @ d + Ba @ h, a_ref, atol=1e-6) and np.allclose(Kj @ d + Bj @ h, j_ref, atol=1e-4),
          "11 comfort_boundary: Ka d + Ba h is the padded profile's accel (and jerk)")

    # -- zero history is the old quadratic ----------------------------------- #
    x = jnp.asarray(rng.normal(size=(3, 2 * n)) * 0.3, jnp.float32)
    H = edit_hessian(n, dt, c)
    old = 0.5 * np.einsum("bi,ij,bj->b", np.asarray(x, np.float64), H, np.asarray(x, np.float64))
    new = np.asarray(comfort_join(x, jnp.zeros((3, 4)), n=n, dt=dt, cfg=c))
    check(np.allclose(new, old, rtol=1e-4), f"11 comfort_join at h = 0 == 0.5 x^T H x (max rel "
          f"{np.max(np.abs(new - old) / np.abs(old)):.1e})")

    # -- join_history ---------------------------------------------------------- #
    W = 51
    P = np.asarray(_arc(B=1, W=W + 2, v=8.0, curv=0.0))           # straight, 8 m/s
    F = jnp.asarray(P[:, 2:])                                     # fresh plan, anchor = step 2
    hist = P[:, :2]                                               # the car's last two poses
    hz = join_history(F, hist)
    check(np.abs(hz).max() < 1e-4, f"11 join_history: on the plan's own line -> 0 ({np.abs(hz).max():.1e})")
    sh = hist.copy()
    yaw0 = float(P[0, 2, 2])
    sh[..., 0] -= 0.2 * np.sin(yaw0)
    sh[..., 1] += 0.2 * np.cos(yaw0)
    hl = join_history(F, sh)
    check(np.allclose(hl[0, :2], 0.2, atol=1e-4) and np.abs(hl[0, 2:]).max() < 1e-4,
          "11 join_history: 0.2 m to the left -> d = 0.2, s = 0")

    # -- the claim: continuing the previous plan beats the fresh plan's jump -- #
    k = np.arange(W)
    jump = 0.3 * (1.0 - np.exp(-k / 2.0))                         # fresh plan veers off at once
    Fj = lateral_offset(F, jnp.asarray(jump[None], jnp.float32), dt=dt)
    prev = F                                                      # what the car was executing
    xp = jnp.asarray(project_onto_path(Fj, prev, n))
    x0 = jnp.zeros_like(xp)
    hj = jnp.asarray(join_history(Fj, hist))
    zero_pad = [float(comfort_join(v, jnp.zeros((1, 4)), n=n, dt=dt, cfg=c)[0]) for v in (x0, xp)]
    join_pad = [float(comfort_join(v, hj, n=n, dt=dt, cfg=c)[0]) for v in (x0, xp)]
    check(zero_pad[1] > zero_pad[0], f"11 zero pad bills continuing ({zero_pad[1]:.3f}) over the jump "
          f"({zero_pad[0]:.3f}) -- the bug the fix removes")
    check(join_pad[1] < 0.25 * join_pad[0], f"11 join pad: continuing ({join_pad[1]:.3f}) is far cheaper "
          f"than the fresh plan's jump ({join_pad[0]:.3f})")

    # -- curvature ----------------------------------------------------------- #
    fast = jnp.asarray(_arc(B=1, W=W, v=10.0, curv=0.0))
    u = np.arange(1, n + 1) / n
    lane = jnp.asarray(np.concatenate([1.5 * (3 * u ** 2 - 2 * u ** 3), np.zeros(n)])[None], jnp.float32)
    k0 = np.asarray(edit_curvature(fast, jnp.zeros((1, 2 * n)), n=n, dt=dt, lon=1))
    kl = np.asarray(edit_curvature(fast, lane, n=n, dt=dt, lon=1))
    stop = jnp.asarray(np.repeat(np.asarray(fast)[:, :1], W, axis=1))     # a parked car
    crab = jnp.asarray(np.concatenate([0.02 * np.arange(1, n + 1), np.zeros(n)])[None], jnp.float32)
    ks = np.asarray(edit_curvature(stop, crab, n=n, dt=dt, lon=1))
    check(np.abs(k0).max() == 0.0, "11 curvature: exactly 0 at x = 0")
    check(np.abs(kl).max() < 0.05, f"11 curvature: 1.5 m lane change at 10 m/s peaks {np.abs(kl).max():.3f} /m")
    check(np.abs(ks).max() > 1.0, f"11 curvature: a parked car sliding 2 cm/step peaks {np.abs(ks).max():.1f} /m")
    g = jax.grad(lambda v: jnp.sum(edit_curvature(stop, v, n=n, dt=dt, lon=1) ** 2))(crab + 0.01)
    check(bool(np.isfinite(np.asarray(g)).all()), "11 curvature gradient finite through a stop (lon on)")

    # -- starts -------------------------------------------------------------- #
    slow = jnp.asarray(_arc(B=2, W=W, v=6.0, curv=0.02))
    s0 = start_profile("init0", slow, n, dt)
    sl, sr = start_profile("initleft", slow, n, dt), start_profile("initright", slow, n, dt)
    sb = start_profile("initbrake", slow, n, dt)
    check(s0.shape == (2, 2 * n) and not s0.any(), "11 init0 start is zero")
    check(np.allclose(sl, -sr) and np.isclose(sl[0, n - 1], START_SWERVE_M) and not sl[:, n:].any(),
          "11 swerve starts: mirror images, reach START_SWERVE_M, no retime")
    seg = np.linalg.norm(np.diff(np.asarray(slow)[:, :n + 1, :2], axis=1), axis=-1)
    q = np.cumsum(seg, 1) + sb[:, n:]
    check(not sb[:, :n].any() and (sb[:, n:] <= 0).all() and (np.diff(q, axis=1) >= -1e-6).all()
          and np.abs(np.diff(q, axis=1)[:, -4:]).max() < 1e-3,
          "11 brake start: behind the plan, never reverses, at rest by 1.5 s")
    check(raises(lambda: start_profile("initprev", slow, n, dt)), "11 start_profile refuses initprev")


def main():
    for f in (t_refactor, t_hessian, t_cubic, t_parse, t_choice, t_overrides, t_lon, t_proj,
              t_emerg, t_argmin, t_amfix):
        f()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED")
        raise SystemExit(1)
    print("all lateral_opt tests passed")


if __name__ == "__main__":
    main()
