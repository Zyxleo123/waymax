"""GPU check + A/B: ``--reuse_prefix`` and the traced-``t0`` gradient.

After ``jit_replan`` (bench_env_jit, job 51380) a training iteration was 3.5 s
collection + 0.6 s BC + 3.8 s gradients, and the gradient pass re-ran the whole
100-step chain per context although only the last ``k_trainable`` steps carry a
gradient. The first 90 steps run the FROZEN base denoiser, so their output is a
function of the noise key alone; ``--reuse_prefix`` stores it at collection and
the gradient replays only the late steps.

What must hold, checked here:

1. **Sampler.** For one key: the full-chain sample, the with-prefix sample and the
   late-only replay from its stored prefix are the same trajectory.
2. **Gradient.** On real collected contexts, the reuse gradient equals the
   full-chain gradient taken with the SAME key (loss and every grad leaf).
   The only behavioural change of the flag is WHICH key the gradient sample
   uses -- the executed one instead of a fresh draw -- never the math.
3. **One compile.** ``t0`` is now traced: the jitted gradient's cache must hold
   one entry per (has_prefix) variant after all 8 replans, not 8.
4. **Speed.** Interleaved training iterations, reuse off vs on, min kept.

Run via SLURM (a6000/a5000): slurm/bench_reuse_prefix.sbatch. Exit 1 on any FAIL.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.train_safety import ReplanContext, SafetyFineTuner, build_argparser


def _maxdiff(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
               for x, y in zip(la, lb))


def _relnorm(a, b):
    num = sum(float(jnp.sum((x - y) ** 2)) for x, y in
              zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    den = sum(float(jnp.sum(x ** 2)) for x in jax.tree_util.tree_leaves(a))
    return (num / max(den, 1e-30)) ** 0.5


def _copy_ctx(c):
    n = ReplanContext(c.t0, c.cond1, c.cond2, c.mask, c.aux, c.alive, c.n_steps, c.n_exec)
    n.expert_target = c.expert_target
    return n


def _leaf_sig(x):
    aval = getattr(x, "aval", None)
    return (type(x).__name__, tuple(np.shape(x)), str(getattr(x, "dtype", "?")),
            bool(getattr(aval, "weak_type", False)),
            str(getattr(x, "sharding", "-")))


def _signature_diff(contexts, bc):
    """Print every raw context leaf whose abstract signature differs between
    replan 0 and replan 1 -- the inputs the gradient is traced on."""
    def fields(c):
        return {"cond1": c.cond1, "cond2": c.cond2, "mask": c.mask, "aux": c.aux,
                "alive": c.alive, "expert_target": c.expert_target,
                "x_prefix": c.x_prefix, "key_late": c.key_late}
    print("\n[reuse] SIGNATURE replan 0 vs 1 (raw context leaves)")
    n = 0
    for tag, seq in (("policy", contexts), ("bc", bc)):
        if len(seq) < 2:
            continue
        a = jax.tree_util.tree_flatten_with_path(fields(seq[0]))[0]
        b = jax.tree_util.tree_flatten_with_path(fields(seq[1]))[0]
        for (pa, xa), (pb, xb) in zip(a, b):
            sa, sb = _leaf_sig(xa), _leaf_sig(xb)
            if sa != sb or pa != pb:
                n += 1
                print(f"  {tag}{jax.tree_util.keystr(pa)}: {sa}  vs  {sb}", flush=True)
    print(f"  {n} differing leaves", flush=True)


def _train_iter(t, state):
    t0 = time.perf_counter()
    contexts, _ = t.rollout(state, train=True)
    t1 = time.perf_counter()
    bc = t.collect_bc_contexts(state) if float(t.args.bc_weight) > 0 else []
    t2 = time.perf_counter()
    for j, ctx in enumerate(contexts):
        if float(jnp.sum(ctx.alive)) <= 0.0:
            continue
        t.rng, rk = jax.random.split(t.rng)
        grads, loss, _ = t._grads_for(state, ctx, rk, bc[j % len(bc)] if bc else None)
        jax.block_until_ready((loss, grads))
    t3 = time.perf_counter()
    return {"collect": t1 - t0, "bc": t2 - t1, "grads": t3 - t2, "iter": t3 - t0}


def main() -> int:
    p = build_argparser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--tol_traj", type=float, default=1e-4)
    p.add_argument("--tol_grad", type=float, default=1e-3,
                   help="relative L2 of (reuse - full) gradient")
    args = p.parse_args()
    args.reuse_prefix = True

    t = SafetyFineTuner(args)
    state = t._next_batch()
    ok = True
    print(f"[reuse] devices={jax.devices()} B={args.scenes_per_iter} "
          f"k_trainable={args.k_trainable} bc_weight={args.bc_weight} "
          f"reuse_prefix={t.reuse_prefix}", flush=True)
    if not t.reuse_prefix:
        raise SystemExit("[reuse] reuse_prefix did not engage (unroll or log states?)")

    # ---- 1. sampler ---------------------------------------------------------- #
    t.actor.num_sample_steps = t._steps_train
    feats = t.env.reset(state, seed=int(args.seed))
    c1, c2 = t.env.compute_condition(feats)
    m = feats["inst_valid"]
    params = t.actor._tparams
    key = jax.random.PRNGKey(777)
    full = t.actor.sample_world_for(params, c1, c2, m, key)
    wp, (xpre, klate) = t.actor.sample_world_with_prefix(params, c1, c2, m, key)
    late = t.actor.sample_late_for(params, c1, c2, m, xpre, klate)
    d1, d2 = _maxdiff(full, wp), _maxdiff(wp, late)
    s_ok = d1 <= args.tol_traj and d2 <= args.tol_traj
    ok &= s_ok
    print(f"\n[reuse] SAMPLER full vs with_prefix {d1:.3e}  with_prefix vs late_only "
          f"{d2:.3e}  {'OK' if s_ok else 'FAIL'}", flush=True)

    # ---- 2. gradient on real contexts ---------------------------------------- #
    contexts, _ = t.rollout(state, train=True)
    bc = t.collect_bc_contexts(state) if float(args.bc_weight) > 0 else []
    _signature_diff(contexts, bc)
    # Names the argument behind any retrace of the jitted gradient from here on.
    jax.config.update("jax_explain_cache_misses", True)
    print(f"\n[reuse] GRADIENT ({len(contexts)} contexts, same key both ways)", flush=True)
    for j, ctx in enumerate(contexts):
        if float(jnp.sum(ctx.alive)) <= 0.0:
            continue
        k = jax.random.fold_in(jax.random.PRNGKey(99), j)
        _, (ctx.x_prefix, ctx.key_late) = t.actor.sample_world_with_prefix(
            params, ctx.cond1, ctx.cond2, ctx.mask, k)
        rk = jax.random.fold_in(jax.random.PRNGKey(5), j)
        bcx = bc[j % len(bc)] if bc else None
        g_r, l_r, _ = t._grads_for(state, ctx, rk, bcx)
        g_f, l_f, _ = t._grads_for(state, _copy_ctx(ctx), rk, bcx, propose_key=k)
        rel = _relnorm(g_f, g_r)
        dl = abs(float(l_r) - float(l_f))
        g_ok = rel <= args.tol_grad and dl <= args.tol_grad * max(abs(float(l_f)), 1.0)
        ok &= g_ok
        print(f"  ctx {j} t0={ctx.t0:>3} loss full {float(l_f):.6f} reuse "
              f"{float(l_r):.6f}  grad rel-L2 {rel:.3e}  {'OK' if g_ok else 'FAIL'}",
              flush=True)

    # ---- 3. compile count ---------------------------------------------------- #
    try:
        n_cache = t._free_grad_fn()._cache_size()
        c_ok = n_cache <= 2       # has_prefix True + False (the full-chain check)
        ok &= c_ok
        print(f"\n[reuse] COMPILE jitted-gradient cache entries after all replans: "
              f"{n_cache} (expect <=2: reuse + full)  {'OK' if c_ok else 'FAIL'}")
    except AttributeError:
        print("\n[reuse] COMPILE cache size unavailable in this jax; skipped")

    # ---- 4. speed ------------------------------------------------------------ #
    print("\n[reuse] SPEED (round 0 = warm-up)")
    timings = {False: [], True: []}
    for r in range(int(args.rounds) + 1):
        for mode in (False, True):
            t.reuse_prefix = mode
            ti = _train_iter(t, state)
            if r:
                timings[mode].append(ti)
            print(f"  round {r} reuse={int(mode)} "
                  + " ".join(f"{k}={v:.2f}" for k, v in ti.items()), flush=True)

    print("\n=========== reuse_prefix A/B ===========")
    print(f"  {'phase':<9} {'full_s':>8} {'reuse_s':>8} {'speedup':>8}")
    for key_ in ("collect", "bc", "grads", "iter"):
        f = min(x[key_] for x in timings[False])
        rr = min(x[key_] for x in timings[True])
        print(f"  {key_:<9} {f:>8.2f} {rr:>8.2f} {f / max(rr, 1e-9):>7.2f}x")
    print(f"\n  equivalence: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
