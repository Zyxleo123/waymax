"""Pure-logic tests for branching, repair candidates, and the learned selector.

No model, no corpus, no GPU. What each block pins, and how it would lie if wrong:

1. perturbations keep the ANCHOR (index 0 is the current state -- moving it
   teleports the car), are identities at alpha=1 / amp=0, and do what they say
   (a time warp keeps the path, a lateral shift moves it by the amplitude);
2. the candidate grammar and its magnitude order (the kept repair is the
   smallest one that works, so the order is part of the result);
3. world_to_norm inverts the REAL postprocess_predictions -- otherwise a repair
   target is not the trajectory that was verified -- including the truncated-horizon
   case, where the inverse is LOSSY and the mask is what keeps it honest;
4. selector weights reproduce corpus proportions (sum to 1, each stratum its
   weight, each scene equal within a stratum), and the fit recovers a planted rule;
5. the repair summary's counts.

    python -m rl.diffusion_ft.tests.test_branching
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.branching import (Candidate, lateral_shift, parse_candidates,
                                       time_warp, world_to_norm)
from rl.diffusion_ft.tools.fit_selector import (featurize, fit_logreg, learned_switch,
                                                row_weights)
from rl.diffusion_ft.tools.repair_summary import summarize

FAILED = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _arc(B=2, W=52, v=8.0, dt=0.1, curv=0.02):
    """A constant-speed arc in world coordinates, [B, W, 5]."""
    t = np.arange(W) * dt
    s = v * t
    yaw = curv * s
    x = np.cumsum(np.concatenate([[0.0], v * dt * np.cos(yaw[:-1])]))
    y = np.cumsum(np.concatenate([[0.0], v * dt * np.sin(yaw[:-1])]))
    w = np.stack([x, y, yaw, v * np.cos(yaw), v * np.sin(yaw)], axis=-1)
    w = np.stack([w + np.array([10.0 * b, -3.0 * b, 0, 0, 0]) for b in range(B)])
    return jnp.asarray(w, jnp.float32)


def t_perturb():
    w = _arc()
    dt = 0.1
    for name, out in (("tw0.7", time_warp(w, 0.7, ramp_steps=10, dt=dt)),
                      ("tw1.3", time_warp(w, 1.3, ramp_steps=10, dt=dt)),
                      ("lat0.5", lateral_shift(w, 0.5, ramp_steps=10, dt=dt))):
        check(np.allclose(np.asarray(out[:, 0]), np.asarray(w[:, 0]), atol=1e-5),
              f"1 {name} keeps the anchor state")
        jump = np.linalg.norm(np.asarray(out[:, 1, :2] - out[:, 0, :2]), axis=-1)
        check(np.all(jump < 8.0 * dt * 1.5), f"1 {name} no position jump at the join ({jump.max():.3f} m)")
    same = time_warp(w, 1.0, ramp_steps=10, dt=dt)
    check(np.allclose(np.asarray(same[..., :3]), np.asarray(w[..., :3]), atol=1e-4),
          "1 time warp at alpha=1 is the identity on pose")
    same = lateral_shift(w, 0.0, ramp_steps=10, dt=dt)
    check(np.allclose(np.asarray(same[..., :3]), np.asarray(w[..., :3]), atol=1e-5),
          "1 lateral shift of 0 m is the identity on pose")
    slow = np.asarray(time_warp(w, 0.5, ramp_steps=1, dt=dt))
    d_slow = np.linalg.norm(slow[:, 20, :2] - slow[:, 0, :2], axis=-1)
    d_base = np.linalg.norm(np.asarray(w)[:, 20, :2] - np.asarray(w)[:, 0, :2], axis=-1)
    check(np.all(np.abs(d_slow / d_base - 0.5) < 0.05), "1 alpha=0.5 covers half the distance")
    sp = np.linalg.norm(slow[:, 25, 3:5], axis=-1)
    check(np.all(np.abs(sp - 4.0) < 0.2), f"1 velocities recomputed from the warped path ({sp})")
    lat = np.asarray(lateral_shift(w, 0.5, ramp_steps=10, dt=dt))
    wn = np.asarray(w)
    n = np.stack([-np.sin(wn[..., 2]), np.cos(wn[..., 2])], -1)
    off = np.sum((lat[..., :2] - wn[..., :2]) * n, -1)
    check(np.allclose(off[:, 15:], 0.5, atol=1e-4), "1 lateral shift holds +0.5 m left after the ramp")


def t_cands():
    c = parse_candidates("id,tw0.7,tw0.7h2,lat-0.5,rn30s1")
    check([x.family for x in c] == ["id", "tw", "tw", "lat", "rn"], "2 families")
    check(c[2].hold == 2 and c[1].hold == 1, "2 hold suffix")
    check(c[4].param == 30.0 and c[4].seed == 1, "2 re-noise step and seed")
    check(c[3].param == -0.5, "2 negative lateral amplitude")
    mags = {x.name: x.magnitude for x in parse_candidates("tw0.85,tw0.7,lat0.25,lat0.5,rn20s0,rn50s0")}
    check(mags["tw0.85"] < mags["tw0.7"] and mags["lat0.25"] < mags["lat0.5"],
          "2 bigger edits have bigger magnitude")
    check(max(mags["tw0.7"], mags["lat0.5"]) < mags["rn20s0"] < mags["rn50s0"],
          "2 re-noising ranks after every local edit")
    try:
        parse_candidates("tw0.7,tw0.7"); ok = False
    except ValueError:
        ok = True
    check(ok, "2 duplicates refused")


def t_roundtrip():
    from types import SimpleNamespace

    from data.postprocess import postprocess_predictions
    rng = np.random.default_rng(0)
    B, T = 3, 25
    cfg = SimpleNamespace(ego_range=100.0, max_velocity=30.0)
    # A smooth normalized trajectory (well inside [-1, 1]).
    tt = np.linspace(0, 1, T)[None, :, None]
    base = np.concatenate([0.3 * tt, 0.05 * tt ** 2, 0.2 + 0 * tt, 0.02 * tt, 0.05 * tt], -1)
    pred = jnp.asarray(base + 0.001 * rng.normal(size=(B, T, 5)), jnp.float32)
    aux = {"origin_xy": jnp.asarray(rng.normal(size=(B, 2)) * 50, jnp.float32),
           "anchor_yaw": jnp.asarray(rng.uniform(-3, 3, B), jnp.float32),
           "model_t_seconds": jnp.asarray(np.arange(T + 1) * 0.2, jnp.float32),
           "world_t_seconds": jnp.asarray(np.arange(2 * T + 1) * 0.1, jnp.float32),
           "world_t_valid": jnp.ones(2 * T + 1, bool)}
    # Anchor = the model trajectory extrapolated back to t=0 is not required; any
    # state works as long as both directions use the same aux.
    aux["anchor_world_state"] = jnp.concatenate(
        [aux["origin_xy"], jnp.zeros((B, 2)), aux["anchor_yaw"][:, None]], -1)
    world = postprocess_predictions(pred, aux, cfg)["trajectory_world_world_dt"]
    back = world_to_norm(world, aux, cfg)
    err = np.abs(np.asarray(back) - np.asarray(pred))
    check(err[..., :2].max() < 1e-4, f"3 position round trip ({err[..., :2].max():.2e})")
    check(err[..., 2:4].max() < 1e-3, f"3 velocity round trip ({err[..., 2:4].max():.2e})")
    check(err[..., 4].max() < 1e-4, f"3 heading round trip ({err[..., 4].max():.2e})")


def t_truncated_horizon():
    """A scene with less than the full horizon left: `world_t_seconds` is clamped
    past the end, so the inverse CANNOT recover those model steps. The mask must
    mark exactly the covered ones, and the covered ones must still round-trip.
    Measured on the first repair smoke as a 0.256 normalized (25.6 m) tail error."""
    from types import SimpleNamespace

    from data.postprocess import postprocess_predictions
    from rl.diffusion_ft.branching import model_step_valid
    B, T, model_dt, world_dt = 2, 25, 0.2, 0.1
    cfg = SimpleNamespace(ego_range=100.0, max_velocity=30.0)
    tt = np.linspace(0, 1, T)[None, :, None]
    pred = jnp.asarray(np.tile(np.concatenate(
        [0.3 * tt, 0.05 * tt ** 2, 0.2 + 0 * tt, 0.02 * tt, 0.05 * tt], -1), (B, 1, 1)), jnp.float32)
    # Scene 0 has the whole 5 s; scene 1 only 2 s, so steps past 2 s are clamped.
    W = 2 * T + 2
    idx = np.arange(W, dtype=np.float32)[None, :]
    ends = np.array([5.0, 2.0])[:, None]
    valid = idx <= np.floor(ends / world_dt)
    world_t = np.where(valid, idx * world_dt, np.broadcast_to(ends, (B, W)))
    aux = {"origin_xy": jnp.zeros((B, 2)), "anchor_yaw": jnp.zeros(B),
           "model_t_seconds": jnp.asarray(np.tile(np.arange(T + 1) * model_dt, (B, 1)), jnp.float32),
           "world_t_seconds": jnp.asarray(world_t, jnp.float32),
           "world_t_valid": jnp.asarray(valid),
           "anchor_world_state": jnp.zeros((B, 5))}
    m = np.asarray(model_step_valid(aux, T, model_dt))
    check(m[0].all(), "6 a full-horizon scene covers every model step")
    check(m[1].sum() == 10 and m[1][:10].all() and not m[1][10:].any(),
          f"6 a 2 s scene covers exactly its first 10 model steps ({m[1].sum()})")
    world = postprocess_predictions(pred, aux, cfg)["trajectory_world_world_dt"]
    err = np.abs(np.asarray(world_to_norm(world, aux, cfg)) - np.asarray(pred))
    check(err[m][..., :2].max() < 1e-4, f"6 covered steps round-trip ({err[m][..., :2].max():.2e})")
    check(err[1][~m[1]][:, :2].max() > 1e-2,
          "6 the uncovered tail does NOT round-trip (which is why it is masked)")


def t_selector():
    scenes = np.array([1, 1, 1, 2, 3, 3, 10, 11, 11, 11])
    strata = np.array(["f"] * 6 + ["s"] * 4)
    w = row_weights(scenes, strata, {"f": 0.02, "s": 0.98})
    check(abs(w.sum() - 1.0) < 1e-12, "4 weights sum to 1")
    check(abs(w[strata == "f"].sum() - 0.02) < 1e-12, "4 stratum mass is its weight")
    check(abs(w[scenes == 1].sum() - w[scenes == 2].sum()) < 1e-12, "4 scenes equal within a stratum")
    rng = np.random.default_rng(1)
    n = 3000
    F = rng.normal(size=(n, 8)); F[:, 2] = np.abs(F[:, 2]) * 5; F[:, 6] = np.abs(F[:, 6]) * 5
    # planted rule: switching helps iff base violates (clear_p<0) and alt is clean
    y = np.where((F[:, 0] < 0) & (F[:, 4] > 0), 1, np.where((F[:, 0] > 0) & (F[:, 4] < 0), -1, 0))
    X = featurize(F, np.zeros(n))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    W, b = fit_logreg((X - mu) / sd, y + 1, np.ones(n), l2=1e-4, iters=2000)
    model = {"mean": mu, "std": sd, "W": W, "b": b, "theta": 0.0, "max_replans": 8}
    sw = learned_switch(model, F, 0)
    check(sw.shape == (n,), "4 learned_switch returns [B]")
    from rl.diffusion_ft.tools.fit_selector import predict_score
    sc = predict_score(model, X)
    check(sc[y > 0].mean() > sc[y == 0].mean() > sc[y < 0].mean(),
          "4 score is ordered by label (+1 > 0 > -1)")
    nz = y != 0
    agree = np.mean(sw[nz] == (y[nz] > 0))
    check(agree > 0.95, f"4 switches where it helps, not where it harms ({agree:.3f})")


def t_summary():
    rows = [{"todo": True, "base_solved": False, "id_consistent": True, "repaired": True,
             "fail_type": "overlap", "cand": "tw0.7h2", "rewind": 0, "n_targets": 2,
             "success_by_cand": {"tw0.7h2": 1, "tw0.6h2": 1}},
            {"todo": True, "base_solved": False, "id_consistent": True, "repaired": False,
             "fail_type": "offroad", "success_by_cand": {}},
            {"todo": False, "base_solved": True, "id_consistent": True, "repaired": False,
             "fail_type": None, "success_by_cand": {}}]
    s = summarize(rows)
    check(s["failures_searched"] == 2 and s["repaired"] == 1, "5 counts")
    check(s["by_fail_type"]["overlap"]["rate"] == 1.0 and s["by_fail_type"]["offroad"]["rate"] == 0.0,
          "5 by failure type")
    check(s["chosen_family"] == {"tw": 1} and s["targets"] == 2, "5 family and targets")


def main():
    for fn in (t_perturb, t_cands, t_roundtrip, t_truncated_horizon, t_selector,
               t_summary):
        fn()
    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
