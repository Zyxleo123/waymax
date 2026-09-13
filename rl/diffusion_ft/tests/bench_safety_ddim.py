"""GPU A/B: DDIM frozen-prefix striding for the SAFETY fine-tune.

The free-running proposal costs ``timesteps`` (100) denoiser evals, and each
replan pays it TWICE -- once in the collection rollout, once inside the gradient
pass. Only the last ``k_trainable`` steps carry a gradient, so the other 90 are
frozen forward compute that both passes redo identically. ``--collect_steps S``
replaces that prefix with ``S - k_trainable`` strided DDIM steps.

This is an approximation, not a refactor: a strided prefix hands the trainable
chain a slightly different ``x_{k-1}``, so it is a different policy. This bench
is what decides whether that difference is small enough to buy the speed. For
each S it reports:

* **train_iter_s** -- one real training iteration (collection rollout + the
  gradient over every replan context), the number that actually sets the run's
  throughput. Timed after a warm-up, so it excludes compilation.
* **quality** -- ``solved`` / ``offroad`` / ``drift_m`` of the BASE policy rolled
  out at that step count on a fixed scene set and a fixed seed. No training runs,
  so any movement here is purely the sampler.

Adopt the smallest S whose quality matches the full chain within noise. Note the
comparison is paired (same scenes, same seeds), so the bar is a matched
difference, not the split's absolute binomial floor.

WHY THIS FILE AND NOT ``bench_ddim_ab.py``: that one benches DPPO and scores by
``return``, which is not what steers this run -- and until the 2026-09-09 cache
fix in ``DiffusionRLActor._sampler_for`` it silently measured nothing at all,
because assigning ``actor.num_sample_steps`` could not change an already-compiled
sampler. Re-read its numbers with that in mind.

Run via SLURM on a GPU node (a6000/a5000 only), see slurm/bench_safety_ddim.sbatch.
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


def _one_train_iter(t, state) -> float:
    """Time one full training iteration: collection rollout + every gradient.

    Mirrors `SafetyFineTuner.train`'s inner body (minus the optimizer step and
    logging), so the number is the per-iteration cost the run actually pays.
    """
    t0 = time.perf_counter()
    contexts, _ = t.rollout(state, train=True)
    for ctx in contexts:
        if float(jnp.sum(ctx.alive)) <= 0.0:
            continue
        t.rng, rk = jax.random.split(t.rng)
        grads, loss, _ = t._grads_for(state, ctx, rk)
        # Block until the device is done, or we would time dispatch, not compute.
        jax.block_until_ready((loss, grads))
    return time.perf_counter() - t0


def main() -> int:
    p = build_argparser()
    p.add_argument("--ab_steps", default="0,50,30,20,15",
                   help="comma list of --collect_steps values; 0 = full chain")
    p.add_argument("--ab_rounds", type=int, default=5,
                   help="timed rounds per setting, interleaved; the MIN is kept")
    args = p.parse_args()

    steps_list = [int(x) for x in args.ab_steps.split(",") if x.strip()]
    t = SafetyFineTuner(args)
    n_t = int(t.actor.diffusion.timesteps)
    k = int(args.k_trainable)

    bad = [S for S in steps_list if S != 0 and not (k <= S <= n_t)]
    if bad:
        raise SystemExit(f"[ab] step counts {bad} outside [k_trainable={k}, {n_t}]")

    state = t._next_batch()
    settings = [(f"full({n_t})" if S == 0 else str(S), None if S == 0 else S)
                for S in steps_list]
    print(f"[ab] devices={jax.devices()}", flush=True)
    print(f"[ab] timesteps={n_t} k_trainable={k} scenes/iter={args.scenes_per_iter} "
          f"max_replans={args.max_replans}", flush=True)
    print(f"[ab] sweeping collect_steps in {steps_list} "
          f"({args.ab_rounds} interleaved rounds)", flush=True)

    # ---- quality: the BASE policy at each step count, fixed scenes+seed ------ #
    # Both knobs are set so the eval rollout itself runs at S -- this measures the
    # sampler, so the sampler under test has to be the one producing the states.
    quality = {}
    for label, S in settings:
        t._steps_train = t._steps_eval = S
        m = t.evaluate(state, rng=jax.random.PRNGKey(int(args.eval_seed)))
        quality[label] = (m["solved_rate"], m["offroad_rate"], m["drift_m"])
        print(f"[ab] quality steps={label:<10} solved={m['solved_rate']:.3f} "
              f"offroad={m['offroad_rate']:.3f} drift={m['drift_m']:.2f}m", flush=True)

    # ---- speed: one real training iteration at each step count -------------- #
    # `_jit_free_grad` bakes num_sample_steps in at trace time, so it must be
    # dropped per setting or every S after the first would reuse the first's
    # compiled gradient program -- the same staleness that made the DPPO bench
    # measure nothing. Eval stays on the full chain here: only the training
    # passes are being timed.
    t._steps_eval = None
    reps: dict[str, list[float]] = {label: [] for label, _ in settings}
    warmed = set()
    for r in range(int(args.ab_rounds) + 1):     # round 0 is warm-up, discarded
        for label, S in settings:
            t._steps_train = S
            if label not in warmed:
                t._jit_free_grad = None
                warmed.add(label)
            dt = _one_train_iter(t, state)
            if r:
                reps[label].append(dt)
        if r == 0:
            print("[ab] warm-up round done (compilation excluded from timings)",
                  flush=True)

    rows = []
    for label, _ in settings:
        rows.append((label, min(reps[label]), *quality[label]))
        print(f"  steps={label:<10} reps={[round(x, 2) for x in reps[label]]}",
              flush=True)

    full_t = rows[0][1] if rows else 1.0
    f_sol, f_off, f_dr = rows[0][2], rows[0][3], rows[0][4]
    print("\n=========== DDIM frozen-prefix A/B (safety, base policy) ===========")
    print(f"  {'steps':<10} {'train_iter_s':>13} {'speedup':>8} {'solved':>8} "
          f"{'d_solved':>9} {'offroad':>8} {'d_offroad':>10} {'drift_m':>8}")
    for label, ts, sol, off, dr in rows:
        print(f"  {label:<10} {ts:>13.2f} {full_t / ts:>7.2f}x {sol:>8.3f} "
              f"{sol - f_sol:>+9.3f} {off:>8.3f} {off - f_off:>+10.3f} {dr:>8.2f}",
              flush=True)
    print(f"\n  Full chain reference: solved={f_sol:.3f} offroad={f_off:.3f} "
          f"drift={f_dr:.2f}m over {int(args.scenes_per_iter)} scenes.")
    print("  Adopt the smallest S whose d_solved / d_offroad are within noise of 0.")
    print("  Then set DFT_COLLECT_STEPS=S and leave DFT_EVAL_STEPS=0.")

    # Self-check. A strided prefix runs strictly fewer denoiser evals, so the
    # smallest S must be measurably faster than the full chain. If every row
    # reads ~1.00x the knob is not reaching the sampler -- which is exactly how
    # the older DPPO bench failed (jitted sampler cached on rl_mode alone, so
    # assigning num_sample_steps changed nothing). Say so loudly rather than
    # letting it be read as "striding does not help".
    strided = [(lab, ts) for (lab, ts, *_), (_, S) in zip(rows, settings) if S]
    if strided:
        best = max(full_t / ts for _, ts in strided)
        if best < 1.05:
            print(f"\n  *** SUSPECT: best speedup is only {best:.2f}x. A strided "
                  f"prefix does strictly fewer denoiser evals, so this should "
                  f"not happen. Check that DiffusionRLActor._sampler_for still "
                  f"keys its cache on (rl_mode, num_sample_steps) and that "
                  f"_jit_free_grad is reset per setting -- a stale compile makes "
                  f"every row identical. Do NOT read this table as evidence "
                  f"against striding. ***", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
