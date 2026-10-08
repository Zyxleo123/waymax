"""Gates for the population search's pure pieces (``rl.diffusion_ft.es_population``).

WHY. The ES arm is only interpretable against ``b_cv`` if, with one member and
no smoothing, it IS ``b_cv``: same trigger, same edit, same certificate. The
closed-loop run checks that end to end (the ``es_p1`` arm); these check the
rules it rests on, on CPU, before a GPU is spent.

* **E1 projection == bank_choice** -- for a single member, ``project_choice``
  picks exactly the edit ``bank_choice("cons")`` picks wherever the bank fires
  (ties in magnitude included), and never a clean one where the bank does not.
* **E2 clean members are never edited.**
* **E3 fitness order** -- every feasible member outranks every infeasible one;
  ``dev`` prefers the member closest to the base, ``prog`` the one furthest
  along; infeasible members order by total shortfall; exact ties go to the lower
  member index (member 0, the base proposal).
* **E4 replication** matches Diffusion-ES's ``_replicate_elites`` layout.
* **E5 survival ratio** is 1 for a kept edit, 0 for one snapped back, 0.5 halfway,
  NaN where nothing was edited.
* **E6 config grammar** -- defaults, overrides, and loud failures.

Pure numpy, seconds. Submitted, not run inline::

    bash rl/diffusion_ft/slurm/submit_es_population.sh     (its first link)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

FAILS: list[str] = []
NUMS: dict = {}


def check(name, ok, detail=""):
    print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def _l24():
    from rl.diffusion_ft.branching import parse_candidates
    return parse_candidates(
        "lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,"
        "lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,"
        "lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6")


def e7_end_to_end(mag):
    """**E7** the JAX path end to end on CPU, with a fake policy on a synthetic road.

    Everything geometric is REAL -- ``postprocess_predictions`` (norm -> world),
    ``world_to_norm``, ``lateral_shift`` and ``PopulationES`` itself -- so shape
    and layout bugs surface here instead of after a GPU queue. Faked: the sampler
    (straight 10 m/s lines with a random lateral drift rate), the smoother (halves
    the lateral position, i.e. pulls toward the lane centre) and the scorer (lane
    half-width 1 m, goal far down +x). The base proposal drifts 0.6 m/s sideways,
    so it leaves the lane at 1.67 s, inside the 2 s window; scene 2 drifts 0.2 m/s
    and stays clean.

    * E7a pop 1 / no smoothing picks bank_choice's edit and returns exactly
      ``lateral_shift`` of the base proposal by that amount (the es_p1 == b_cv
      identity, on the real edit code);
    * E7b the full search runs, touches only triggered scenes, and its winner is
      feasible there;
    * E7c the diagnostics are populated where it ran and NaN where it did not.
    """
    import types
    from functools import partial

    import jax
    import jax.numpy as jnp

    from data.postprocess import postprocess_predictions
    from data.types import PreprocessConfig
    from rl.diffusion_ft.branching import lateral_shift
    from rl.diffusion_ft.es_population import ES_DIAG, ESConfig, PopulationES
    from rl.diffusion_ft.proposal_select import bank_choice

    B, T, W, n = 3, 25, 51, 20
    pcfg = PreprocessConfig()
    tile = lambda v: jnp.tile(jnp.asarray(v, jnp.float32)[None], (B,) + (1,) * jnp.ndim(v))
    aux = {"origin_xy": jnp.zeros((B, 2)), "anchor_yaw": jnp.zeros(B),
           "anchor_world_state": tile([0.0, 0.0, 10.0, 0.0, 0.0]),
           "model_t_seconds": tile(np.arange(T + 1) * 0.2),
           "world_t_seconds": tile(np.arange(W) * 0.1),
           "world_t_valid": jnp.ones((B, W), bool), "world_dt_seconds": jnp.full(B, 0.1)}

    def traj(rate):                                   # lateral m/s [N] -> norm [N, 5, T]
        t = (jnp.arange(T) + 1) * 0.2
        N = rate.shape[0]
        return jnp.stack([jnp.broadcast_to(10.0 * t / 100.0, (N, T)), rate[:, None] * t / 100.0,
                          jnp.full((N, T), 10.0 / 25.0), jnp.broadcast_to(rate[:, None] / 25.0, (N, T)),
                          jnp.zeros((N, T))], axis=1)

    def sample(params, c1, c2, m, key):
        return traj(0.3 + 0.4 * jax.random.normal(key, (c1.shape[0],))), (None, None)

    class Diff:
        timesteps = 100
        denoise_fn = types.SimpleNamespace(horizon=T)

        def resample(self, x, c1, c2, m, ts, *, rng, noise_scale=1.0):
            return x.at[:, 1, :].multiply(0.5)

    env = types.SimpleNamespace(norm_to_world_with=lambda x, a: postprocess_predictions(
        x, a, pcfg)["trajectory_world_world_dt"])
    tuner = types.SimpleNamespace(ckpt=types.SimpleNamespace(preprocess_cfg=pcfg), env=env,
                                  actor=types.SimpleNamespace(diffusion=Diff(),
                                                              sample_world_with_prefix=sample))
    goal = jnp.tile(jnp.asarray([1000.0, 0.0])[None], (B, 1))

    @partial(jax.jit, static_argnums=(4,))
    def score(world, state, t0, goal_xy, n):
        edge = jnp.min(1.0 - jnp.abs(world[:, 1:n + 1, 1]), axis=1)
        agent = jnp.full(edge.shape, 1.0e3)
        prog = (jnp.linalg.norm(world[:, 0, :2] - goal_xy, axis=-1)
                - jnp.linalg.norm(world[:, n, :2] - goal_xy, axis=-1))
        return jnp.minimum(edge, agent), edge, agent, prog

    state, t0, rk = jnp.zeros(1), 0, jax.random.PRNGKey(3)
    x_p = traj(jnp.asarray([0.6, 0.6, 0.2]))
    world_p = env.norm_to_world_with(jnp.transpose(x_p, (0, 2, 1)), aux)
    cp, _, _, pp = (np.asarray(v) for v in score(world_p, state, t0, goal, n))
    cands = _l24()
    kw = dict(x_p=x_p, world_p=world_p, cp=cp, pp=pp, cond1=jnp.zeros((B, 4)),
              cond2=jnp.zeros((B, 4)), mask=jnp.zeros(B), aux=aux, state=state, t0=t0,
              goal=goal, n=n, rk=rk, alive=jnp.ones(B))
    def fake_opt(world, x0, st, t0_, active, n_):    # lateral_opt's contract, fixed answer
        d = -0.4 * jnp.asarray(active, jnp.float32)[:, None] * jnp.ones((world.shape[0], n_))
        return d, jnp.zeros(world.shape[0]), jnp.ones(world.shape[0])

    mk = lambda c: PopulationES(tuner, c, params=None, score_fn=score, cands=cands,
                                ramp_steps=20, blend="cubic", tol_m=1.0,
                                optimizer=fake_opt if c.opt else None)
    ci = {c: i for i, c in enumerate(ES_DIAG)}

    # E7a: the bank's own path, then the ES with one member
    edits = [lateral_shift(world_p, float(c.param), ramp_steps=20, dt=0.1) for c in cands]
    sc = [tuple(np.asarray(v) for v in score(w, state, t0, goal, n)) for w in edits]
    ch = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=np.stack([s[0] for s in sc]),
                     prog_k=np.stack([s[3] for s in sc]), magnitude=mag, tol_m=1.0,
                     margin_m=0.3, delta_m=0.2)
    w1, s1, d1 = mk(ESConfig("p1", pop=1, iters=0, elite=1, bank="l24", t=0))(**kw)
    fired = ch >= 0
    same = all(np.allclose(np.asarray(w1[b]), np.asarray(edits[ch[b]][b]), atol=1e-5)
               for b in np.nonzero(fired)[0])
    feas1 = (s1["clear"] >= 0) & (s1["prog"] >= pp - 1.0)
    NUMS["e7_bank_choice"] = ch.tolist()
    check("E7a pop 1 returns the bank's edit of the base proposal",
          list(fired) == [True, True, False] and same and bool(np.all(feas1[fired])),
          f"bank choice {ch.tolist()} ({[cands[i].name if i >= 0 else '-' for i in ch]})")

    # E7b / E7c: the full search, with smoothing and generations
    w2, s2, d2 = mk(ESConfig("proj", pop=6, iters=2, elite=2, bank="l24", t=1))(**kw)
    trig = d2[:, ci["trig"]] > 0.5
    feas2 = (s2["clear"] >= 0) & (s2["prog"] >= pp - 1.0)
    check("E7b full search runs and its winner is feasible where it ran",
          np.asarray(w2).shape == (B, W, 5) and list(trig) == [True, True, False]
          and bool(np.all(feas2[trig])), f"clear {np.round(s2['clear'], 3)}")
    ran = d2[trig]
    check("E7c diagnostics populated where it ran, NaN elsewhere",
          bool(np.all(np.isfinite(ran[:, [ci[c] for c in ("n_raw_feas", "n_proj_feas",
                                                          "n_final_feas", "best_gen",
                                                          "wall_s")]])))
          and bool(np.all(ran[:, ci["n_proj_feas"]] >= ran[:, ci["n_raw_feas"]]))
          and bool(np.all(np.isnan(d2[~trig][:, 1:]))),
          f"raw/proj/gen0 feasible {ran[:, [ci['n_raw_feas'], ci['n_proj_feas'], ci['n_gen0_feas']]].tolist()}"
          f" survival {np.round(ran[:, ci['survival']], 3).tolist()}")
    w3, s3, d3 = mk(ESConfig("raw", pop=6, iters=2, elite=2, bank="", t=3, scale=2.0))(**kw)
    # E7d / E7e: the lateral_opt projection (a fake optimizer with its contract)
    from rl.diffusion_ft.lateral_opt import expand_offset
    from rl.diffusion_ft.branching import lateral_offset
    w4, s4, d4 = mk(ESConfig("o1", pop=1, iters=0, elite=1, opt=1, t=0))(**kw)
    want = lateral_offset(world_p, expand_offset(fake_opt(world_p, None, None, None,
                                                          jnp.ones(B), n)[0], W), dt=0.1)
    trig4 = d4[:, ci["trig"]] > 0.5
    feas4 = (s4["clear"] >= 0) & (s4["prog"] >= pp - 1.0)
    check("E7d pop 1 with opt:1 returns the optimizer's edit where it fires (o_0)",
          list(trig4) == [True, True, False] and bool(np.all(feas4[trig4]))
          and np.allclose(np.asarray(w4)[trig4], np.asarray(want)[trig4], atol=1e-5),
          f"clear {np.round(s4['clear'], 3)}")
    w5, s5, d5 = mk(ESConfig("osp", pop=6, iters=2, elite=2, opt=1, order="sp", t=3,
                             scale=2.0))(**kw)
    feas5 = (s5["clear"] >= 0) & (s5["prog"] >= pp - 1.0)
    check("E7e mutate-then-project search runs; winner feasible where it ran",
          np.asarray(w5).shape == (B, W, 5) and bool(np.all(feas5[trig4]))
          and bool(np.all(np.isnan(d5[trig4][:, ci["survival"]]))),
          f"n_edit {d5[:, ci['n_edit']].tolist()} best_gen {d5[:, ci['best_gen']].tolist()}")

    check("E7b the unprojected search runs too", np.asarray(w3).shape == (B, W, 5)
          and int(np.nansum(d3[:, ci["n_edit"]])) == 0,
          f"shape {np.asarray(w3).shape} n_edit {d3[:, ci['n_edit']].tolist()} "
          f"trig {d3[:, ci['trig']].tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", default="")
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()

    from rl.diffusion_ft.es_population import (FEAS_BONUS, es_fitness, parse_es_specs,
                                               project_choice, replicate, survival_ratio,
                                               top_k, window_dev)
    from rl.diffusion_ft.proposal_select import bank_choice

    rng = np.random.default_rng(0)
    mag = np.asarray([c.magnitude for c in _l24()])
    K, B, tol = len(mag), 32, 1.0

    # ---- E1 / E2 ---------------------------------------------------------------- #
    agree = fired = mism = clean_edit = 0
    for _ in range(args.trials):
        cp = rng.normal(-0.1, 0.4, B)
        pp = rng.normal(5.0, 2.0, B)
        ck = cp[None] + rng.normal(0.0, 0.5, (K, B))
        pk = pp[None] + rng.normal(-0.5, 1.0, (K, B))
        want = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=ck, prog_k=pk,
                           magnitude=mag, tol_m=tol, margin_m=0.3, delta_m=0.2)
        got = project_choice(cp[None], pp[None], ck[:, None], pk[:, None], mag, tol)[0]
        fire = want >= 0
        fired += int(fire.sum())
        agree += int((got[fire] == want[fire]).sum())
        # where the bank keeps the base, any edit ES takes must be one the bank
        # would NOT certify (the "clearest edit" fallback), never a clean one
        g = got[~fire]
        idx = np.nonzero(~fire)[0][g >= 0]
        if len(idx):
            gi = got[idx]
            ok = (ck[gi, idx] < 0.0) | (pk[gi, idx] < pp[idx] - tol) | (cp[idx] >= 0.0)
            mism += int((~ok).sum())
        clean_edit += int(((got >= 0) & (cp >= 0.0)).sum())
    NUMS["e1_fired"] = fired
    check("E1 projection picks bank_choice's edit wherever the bank fires",
          fired > 0 and agree == fired, f"{agree}/{fired}")
    check("E1 where the bank keeps the base, no certified edit is taken", mism == 0,
          f"{mism} violations")
    check("E2 clean members are never edited", clean_edit == 0, f"{clean_edit} edited")

    # exact magnitude ties: lat0.5 / lat-0.5 both clean -> the earlier (lat0.5)
    cp = np.array([-0.2]); pp = np.array([3.0])
    ck = np.full((K, 1), -1.0); pk = np.full((K, 1), 3.0)
    i05, im05 = 10, 11
    ck[i05] = ck[im05] = 0.4
    got = int(project_choice(cp[None], pp[None], ck[:, None], pk[:, None], mag, tol)[0, 0])
    check("E1 magnitude ties go to the earlier bank entry", got == i05, f"got index {got}")

    # ---- E3 --------------------------------------------------------------------- #
    clear = np.array([[0.5], [-0.1], [0.2], [1.0], [-2.5]])
    prog = np.array([[4.0], [9.0], [1.0], [6.0], [9.0]])
    dev = np.array([[0.3], [0.0], [0.1], [1.2], [0.0]])
    ref = np.array([4.0])
    fd = es_fitness(clear, prog, dev, prog_ref=ref, tol_m=tol, rank="dev")[:, 0]
    fp = es_fitness(clear, prog, dev, prog_ref=ref, tol_m=tol, rank="prog")[:, 0]
    feas = np.array([True, False, False, True, False])   # #2 fails progress (1 < 3)
    check("E3 every feasible member outranks every infeasible one",
          fd[feas].min() > fd[~feas].max() and fp[feas].min() > fp[~feas].max(),
          f"dev {np.round(fd - FEAS_BONUS * feas, 2)}")
    check("E3 rank:dev prefers the member closest to the base", int(np.argmax(fd)) == 0)
    check("E3 rank:prog prefers the member furthest along", int(np.argmax(fp)) == 3)
    check("E3 infeasible members order by total shortfall",
          fd[1] > fd[4] and fd[2] > fd[4], f"{fd[1]:.2f} {fd[2]:.2f} {fd[4]:.2f}")
    tie = np.array([[1.0, 2.0], [1.0, 2.0], [0.5, 2.0]])
    check("E3 exact ties go to the lower member index", list(top_k(tie, 1)[0]) == [0, 0])

    # ---- E4 --------------------------------------------------------------------- #
    def ref_rep(e, p):     # es_baseline DiffusionESPlanner._replicate_elites
        base, rem = p // e, p % e
        return np.concatenate([np.full((base + (1 if i < rem else 0),), i) for i in range(e)])
    ok = all(np.array_equal(replicate(e, p), ref_rep(e, p))
             for e, p in ((4, 64), (4, 16), (3, 16), (1, 1), (5, 7)))
    check("E4 replication matches Diffusion-ES", ok)

    # ---- E5 --------------------------------------------------------------------- #
    W, n = 51, 20
    raw = rng.normal(0, 1, (3, W, 5))
    proj = raw.copy(); proj[..., 1:, 1] += 0.8
    s = survival_ratio(np.stack([raw] * 4), np.stack([proj, proj, proj, raw]),
                       np.stack([proj, raw, 0.5 * (raw + proj), raw]), n)
    check("E5 survival 1 / 0 / 0.5 / NaN", np.allclose(s[:3], [[1] * 3, [0] * 3, [0.5] * 3])
          and np.all(np.isnan(s[3])), f"{np.round(s[:, 0], 3)}")
    check("E5 window_dev is 0 against itself", float(np.abs(window_dev(raw[None], raw, n)).max()) == 0.0)

    # ---- E6 --------------------------------------------------------------------- #
    c = parse_es_specs("a=pop:16,bank:l24cv;b=pop:1,iters:0,t:0,bank:l24cv;c=t:3,scale:2,rank:prog")
    check("E6 defaults and overrides parse",
          c["a"].pop == 16 and c["a"].t == 1 and c["a"].scale == 0.0 and c["a"].nb == "cv"
          and c["b"].pop == 1 and c["b"].iters == 0 and c["b"].t == 0
          and c["c"].bank == "" and c["c"].scale == 2.0 and c["c"].rank == "prog")
    bad = 0
    for spec in ("x=pop:0", "x=elite:5,pop:4", "x=nb:oracle", "x=foo:1", "x", "x=pop:2;x=pop:3",
                 "x=rank:best", "x=t:-1", "x=opt:1,bank:l24", "x=order:zz", "x=opt:2"):
        try:
            parse_es_specs(spec)
        except SystemExit:
            bad += 1
    check("E6 malformed configs fail loudly", bad == 11, f"{bad}/11")
    c = parse_es_specs("o=pop:16,opt:1,order:sp,t:3,scale:2")
    check("E6 opt/order parse", c["o"].opt == 1 and c["o"].order == "sp" and c["o"].bank == ""
          and parse_es_specs("d=pop:2")["d"].order == "ps")

    e7_end_to_end(mag)

    print()
    head = "PASS" if not FAILS else "FAIL: " + "; ".join(FAILS)
    if args.verdict:
        Path(args.verdict).parent.mkdir(parents=True, exist_ok=True)
        Path(args.verdict).write_text(head + "\n" + json.dumps(NUMS) + "\n")
        print(f"[gate] verdict -> {args.verdict}")
    if FAILS:
        print(f"[gate] {len(FAILS)} FAILED")
        raise SystemExit(1)
    print("[gate] all ES population gates passed")


if __name__ == "__main__":
    main()
