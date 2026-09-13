"""GPU A/B: eager vs jitted replan in ``DiffusionWaymaxEnv`` (``jit_replan``).

Profiling the 09-12 stopping sweep (bc300_s1) put a 17 s training iteration at
7.5 s policy collection + 4.2 s BC collection + ~5.4 s gradients, and the BC pass
does NO sampling -- it is 8 replans of pure env stepping, i.e. ~0.5 s per replan of
host-dispatched ops (the eager preprocessor's vmapped lexsort/unique/cond roadgraph
build, the per-step reward). The DDIM bench agreed: cutting the chain 100 -> 15
steps bought only 1.38x. Eval (1173 scenes every 10 iters) was 61% of wall clock
and pays the same per-replan cost.

This bench decides whether ``jit_replan=True`` is safe to adopt and what it buys:

1. **Equivalence, open loop (strict).** Both modes drive the SAME log trajectory
   (no sampler), so any difference is the env alone. Compares every feature,
   aux, per-step reward component and the simulated trajectory, per replan.
   Flags must match exactly; floats within ``--tol``.
2. **Equivalence, closed loop.** One fixed-seed policy rollout per mode: the
   solved/offroad/overlap flags per scene and the driven trajectory.
3. **Speed.** Interleaved rounds, min kept, compilation excluded: BC pass (log
   replay), policy collection, one full training iteration (collection + BC +
   every gradient), and one 32-scene eval chunk.

Run via SLURM on a6000/a5000 only: slurm/bench_env_jit.sbatch.
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

from rl.diffusion_ft.train_safety import SafetyFineTuner, build_argparser


def _np_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _log_replay(env, state, seed):
    """Drive the log for the whole episode; record everything the env emits."""
    feats = env.reset(state, seed=seed)
    log_world = env.ego_log_world_trajectory()
    horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
    rec = []
    for _ in range(8):
        t0 = env.timestep
        if horizon - 1 - t0 <= 0:
            break
        out = env.step(trajectory_world_bt5=log_world[:, t0:t0 + env._replan + 1, :])
        rec.append({"features": _np_tree(out.features),
                    "aux": _np_tree(env.last_aux),
                    "comps": _np_tree(out.per_step_components),
                    "reward": np.asarray(out.per_step_reward_bk),
                    "term": np.asarray(out.terminated_b),
                    "trunc": np.asarray(out.truncated_b)})
    rec.append({"features0": _np_tree(feats),
                "sim": np.asarray(env.ego_sim_world_trajectory())})
    return rec


def _compare(a, b, tol):
    """Max abs diff per leaf path; exact-match check for bool/int leaves."""
    worst_f, bad_exact = 0.0, []
    la = jax.tree_util.tree_flatten_with_path(a)[0]
    lb = jax.tree_util.tree_flatten_with_path(b)[0]
    if [p for p, _ in la] != [p for p, _ in lb]:
        return float("inf"), ["<tree structure differs>"], ""
    where = ""
    for (path, x), (_, y) in zip(la, lb):
        name = jax.tree_util.keystr(path)
        if x.shape != y.shape:
            bad_exact.append(f"{name} shape {x.shape} vs {y.shape}")
            continue
        if x.dtype.kind in "biu":
            if not np.array_equal(x, y):
                bad_exact.append(f"{name} ({int(np.sum(x != y))} mismatches)")
        else:
            d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) \
                if x.size else 0.0
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                bad_exact.append(f"{name} NaN pattern")
            if d > worst_f:
                worst_f, where = d, name
    return worst_f, bad_exact, where


def _train_iter(t, state):
    """One training iteration as `SafetyFineTuner.train` runs it, minus the step."""
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
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max abs float diff accepted between eager and jit")
    args = p.parse_args()

    t = SafetyFineTuner(args)
    env = t.env
    state = t._next_batch()
    print(f"[envjit] devices={jax.devices()} B={args.scenes_per_iter} "
          f"bc_weight={args.bc_weight}", flush=True)

    # ---- 1. open-loop equivalence ------------------------------------------ #
    runs = {}
    for mode in (False, True):
        env._jit_replan_enabled = mode
        runs[mode] = _log_replay(env, state, seed=int(args.seed))
    ok = len(runs[False]) == len(runs[True])
    print(f"\n[envjit] OPEN LOOP (log replay, {len(runs[False]) - 1} replans)")
    worst = 0.0
    for i, (ra, rb) in enumerate(zip(runs[False], runs[True])):
        d, bad, where = _compare(ra, rb, args.tol)
        worst = max(worst, d)
        flag = "OK" if (d <= args.tol and not bad) else "DIFF"
        ok &= flag == "OK"
        print(f"  replan {i}: max|d| {d:.3e} at {where or '-'}  exact-mismatch "
              f"{bad or 'none'}  {flag}", flush=True)

    # ---- 2. closed-loop equivalence ---------------------------------------- #
    print("\n[envjit] CLOSED LOOP (policy rollout, fixed seed)")
    cl = {}
    for mode in (False, True):
        env._jit_replan_enabled = mode
        t.rollout(state, train=False, rng=jax.random.PRNGKey(int(args.eval_seed)))
        cl[mode] = ({k: np.asarray(v) for k, v in t._last_scene.items()},
                    np.asarray(env.ego_sim_world_trajectory()))
    for k in ("solved", "offroad", "overlap", "reached"):
        n = int(np.sum(cl[False][0][k] != cl[True][0][k]))
        print(f"  {k:<8} eager {cl[False][0][k].mean():.3f}  jit "
              f"{cl[True][0][k].mean():.3f}  scenes differing {n}", flush=True)
    traj_d = np.abs(cl[False][1][..., :2] - cl[True][1][..., :2])
    print(f"  driven xy max|d| {traj_d.max():.3e} m  mean {traj_d.mean():.3e} m")

    # ---- 3. speed ---------------------------------------------------------- #
    # Warm both modes once (compiles jit programs, and the shared gradient/sampler
    # executables), then interleave timed rounds and keep the min.
    timings = {m: [] for m in (False, True)}
    evals = {m: [] for m in (False, True)}
    for r in range(int(args.rounds) + 1):
        for mode in (False, True):
            env._jit_replan_enabled = mode
            ti = _train_iter(t, state)
            e0 = time.perf_counter()
            t.evaluate(state, rng=jax.random.PRNGKey(int(args.eval_seed)))
            te = time.perf_counter() - e0
            if r:
                timings[mode].append(ti)
                evals[mode].append(te)
            print(f"  round {r} {'jit  ' if mode else 'eager'} "
                  + " ".join(f"{k}={v:.2f}" for k, v in ti.items())
                  + f" eval_chunk={te:.2f}" + ("  (warm-up)" if not r else ""),
                  flush=True)

    def best(mode, key):
        return min(x[key] for x in timings[mode])

    print("\n=========== env jit_replan A/B ===========")
    print(f"  {'phase':<12} {'eager_s':>9} {'jit_s':>9} {'speedup':>8}")
    for key in ("collect", "bc", "grads", "iter"):
        e, j = best(False, key), best(True, key)
        print(f"  {key:<12} {e:>9.2f} {j:>9.2f} {e / max(j, 1e-9):>7.2f}x")
    e, j = min(evals[False]), min(evals[True])
    print(f"  {'eval_chunk':<12} {e:>9.2f} {j:>9.2f} {e / max(j, 1e-9):>7.2f}x")
    print(f"\n  open-loop equivalence: {'PASS' if ok else 'FAIL'} "
          f"(worst float diff {worst:.3e}, tol {args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
