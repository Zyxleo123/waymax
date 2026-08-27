"""GPU profiler: where does one `collect` spend its time?

Reproduces the trainer's collect loop but times each part per replan with
`block_until_ready`, so we can see the split between:

* compute_condition   (frozen scene tokenizer)
* diffusion sample    (frozen early fori_loop + eager late k-step loop)
* env.step            (waymax sim rollout)
* norm/value/misc

and, within the diffusion sample, isolates the eager late-step cost by also
timing an all-frozen (k=0) sample of the same shape. This decides whether jitting
the late sampling loop is the right lever before we change any code.

Run via SLURM on a GPU node (bench_collect_profile.sbatch).
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.train_dppo import DPPOTrainer, build_argparser, load_batch


def _sync(x):
    jax.block_until_ready(x)
    return x


def main() -> int:
    args = build_argparser().parse_args()
    print(f"[prof] devices={jax.devices()}  indices={args.indices} "
          f"k_trainable={args.k_trainable}", flush=True)

    trainer = DPPOTrainer(args)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    state = load_batch(args.tfrecord, indices)
    trainer._build_for(state)
    env, actor, critic = trainer.env, trainer.actor, trainer.critic

    # Warm up compiles (one full untimed collect), then time a fresh one.
    trainer.collect(state)

    features = env.reset(state)
    B = env.batch_size
    horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
    n_steps = args.max_replans or ((horizon - env.timestep) // args.replan_interval_steps)
    shape = (B, actor.diffusion.denoise_fn.in_ch, actor.diffusion.denoise_fn.horizon)

    t = defaultdict(float)
    prev_done = np.zeros(B, dtype=bool)
    rng = jax.random.PRNGKey(123)

    t_all0 = time.perf_counter()
    for step in range(int(n_steps)):
        t0 = time.perf_counter()
        cond1, cond2 = env.compute_condition(features)
        _sync((cond1, cond2)); t["compute_condition"] += time.perf_counter() - t0
        mask = features["inst_valid"]

        rng, krng = jax.random.split(rng)
        t0 = time.perf_counter()
        traj_ncT, trace = actor.collect(cond1, cond2, mask, rng=krng)
        _sync((traj_ncT, trace["log_prob"])); t["diffusion_sample(k)"] += time.perf_counter() - t0

        # Isolate the eager late-step cost: an all-frozen sample of the same shape
        # (k_trainable=0 -> only the compiled fori_loop, no unrolled late steps).
        rng, k2 = jax.random.split(rng)
        t0 = time.perf_counter()
        xf, _ = actor.diffusion.sample_with_trace(
            shape, cond1, cond2, mask, rng=k2, k_trainable=0,
            sigma_sample_floor=actor.sigma_sample_floor,
            sigma_logprob_floor=actor.sigma_logprob_floor, prefix_len=actor.prefix_len)
        _sync(xf); t["diffusion_sample(frozen_only)"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
        value = np.asarray(critic.value(cond1))
        _sync(world); t["norm+value"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        out = env.step(trajectory_world_bt5=world)
        _sync(out.reward_b); t["env.step"] += time.perf_counter() - t0

        done = np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
        prev_done = prev_done | done
        features = out.features
        if prev_done.all():
            break
    total = time.perf_counter() - t_all0
    n_done = step + 1

    print(f"\n=========== collect profile ({n_done} replans, B={B}) ===========")
    for k in ("compute_condition", "diffusion_sample(k)", "diffusion_sample(frozen_only)",
              "norm+value", "env.step"):
        print(f"  {k:<32} {t[k]*1e3:9.1f} ms   ({t[k]/total*100:4.1f}% of loop)", flush=True)
    late = t["diffusion_sample(k)"] - t["diffusion_sample(frozen_only)"]
    print(f"  {'-> eager late-step overhead':<32} {late*1e3:9.1f} ms   "
          f"(diffusion_sample(k) - frozen_only)")
    print(f"\n  loop total (excl. frozen_only probe): "
          f"{(total - t['diffusion_sample(frozen_only)'])*1e3:.1f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
