"""GPU A/B: DDIM frozen-prefix subsampling -- speed vs rollout quality.

For the *base* (untrained) diffusion policy on the configured scenes, sweeps
``num_sample_steps`` and reports, at each setting:

* collect wall time (the thing we are trying to shave), and its speedup vs full;
* deterministic rollout return (fixed seed) -- the deployment-trajectory quality;
* stochastic rollout return (the distribution PPO actually collects from).

If the returns hold roughly flat as the step count drops while collect time
falls, the subsampling preserves the policy and is safe to adopt for training.
A return that collapses at low step counts is the signal to keep more steps.
Base policy only -- no training -- so any change is purely the sampler.

Run via SLURM on a GPU node (bench_ddim_ab.sbatch).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax

from rl.diffusion_ft.train_dppo import DPPOTrainer, build_argparser, load_batch


def main() -> int:
    args = build_argparser().parse_args()
    # Comma list of step counts; 0 = full chain (timesteps). Default sweep.
    ab_steps = getattr(args, "ab_steps", None) or "0,40,30,25,20"
    steps_list = [int(x) for x in ab_steps.split(",") if x.strip()]

    print(f"[ab] devices={jax.devices()}  indices={args.indices}  k_trainable={args.k_trainable}")
    print(f"[ab] sweeping num_sample_steps in {steps_list}", flush=True)

    trainer = DPPOTrainer(args)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    state = load_batch(args.tfrecord, indices)
    trainer._build_for(state)
    timesteps = int(trainer.actor.diffusion.timesteps)

    # Recompile guard: the late chain is a lax.scan that captures the trainable
    # params. If a param update forced a recompile, real training would pay it
    # every collect. Warm once, then time a collect before and after mutating the
    # params -- the two should match (params flow in as operands, not constants).
    import jax as _jax
    trainer.collect(state)                       # warm compile
    t0 = time.perf_counter(); trainer.collect(state); warm = time.perf_counter() - t0
    trainer.actor._tparams = _jax.tree_util.tree_map(lambda a: a + 1e-4, trainer.actor._tparams)
    t0 = time.perf_counter(); trainer.collect(state); after = time.perf_counter() - t0
    print(f"[ab] recompile guard: collect warm={warm:.2f}s  post-param-update={after:.2f}s  "
          f"ratio={after/warm:.2f} (want ~1.0; >>1 means a recompile per update)", flush=True)

    settings = [(f"full({timesteps})" if S == 0 else str(S), None if S == 0 else S)
                for S in steps_list]

    # Quality (deterministic + stochastic return) and warm-up, per setting. Warm
    # generously: shapes vary across the 8 replans so several collects are needed
    # before every variant is compiled.
    quality = {}
    for label, S in settings:
        trainer.actor.num_sample_steps = S
        det0 = trainer.evaluate(state, deterministic=True, seed_offset=0)["return"]
        sto0 = trainer.evaluate(state, deterministic=False, seed_offset=1)["return"]
        quality[label] = (det0, sto0)
        for _ in range(5):
            trainer.collect(state)

    # Timed collect, INTERLEAVED across settings: on a shared node a burst of
    # contention lands on whole consecutive windows, so timing one setting's reps
    # back-to-back can bias it. Round-robin instead and take the min per setting,
    # so each setting's number comes from its cleanest (least-contended) round.
    n_rounds = 8
    reps = {label: [] for label, _ in settings}
    for r in range(n_rounds):
        for label, S in settings:
            trainer.actor.num_sample_steps = S
            t0 = time.perf_counter()
            trainer.collect(state)
            reps[label].append(time.perf_counter() - t0)

    rows = []
    for label, _ in settings:
        det0, sto0 = quality[label]
        collect_s = min(reps[label])
        rows.append((label, collect_s, det0, sto0))
        print(f"  steps={label:<10} collect_min={collect_s:6.2f}s  "
              f"reps={[round(x,1) for x in reps[label]]}", flush=True)

    full_t = rows[0][1] if rows else 1.0
    print("\n=========== DDIM frozen-prefix A/B (base policy) ===========")
    print(f"  {'steps':<10} {'collect_s':>10} {'speedup':>8} {'det_return':>12} {'stoch_return':>13}")
    for label, cs, det, sto in rows:
        print(f"  {label:<10} {cs:>10.2f} {full_t/cs:>7.2f}x {det:>+12.4f} {sto:>+13.4f}", flush=True)
    print("\n  Adopt the smallest step count whose returns match full within noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
