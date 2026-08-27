"""GPU benchmark: new jitted K-batched update vs the old Python-loop reference.

Builds the *real* diffusion actor via :class:`DPPOTrainer`, runs one real
`collect` to get a real trace, then times -- on whatever device JAX picks up --
the three hot pieces of the PPO update, first with the new (jitted, K-batched)
implementation and then with the old per-step Python loop monkeypatched back in:

* `log_probs`            (forward only; used by the PG loss)
* `_posterior_means_std` (forward only; used by the KL gate)
* `ppo_update`           (full step: PG fwd+bwd + candidate + gate)

Each is warmed up (JIT compile / first-touch excluded) then timed over N reps
with `block_until_ready`. Prints per-call ms and the new/old speedup.

Run via SLURM on a GPU node (see bench_update_speed.sbatch). Configure the batch
with the same DFT_* / argv the trainer uses; defaults match the 8-scene run.
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

from rl.diffusion_ft.train_dppo import DPPOTrainer, build_argparser, load_batch


# ---- old reference implementations (the pre-optimization Python loops) ----- #
def ref_log_probs(actor, params, c1, c2, m, trace):
    fn = actor._trainable_fn(params)
    k = int(trace["timestep"].shape[0])
    lps = []
    for j in range(k):
        lps.append(actor.diffusion.transition_log_prob(
            trace["x_t"][j], trace["x_prev"][j], trace["timestep"][j], c1, c2, m,
            sigma_floor=actor.sigma_logprob_floor, prefix_len=actor.prefix_len,
            denoise_fn=fn))
    return jnp.stack(lps, axis=0)


def ref_means_std(actor, params, c1, c2, m, trace):
    fn = actor._trainable_fn(params)
    k = int(trace["timestep"].shape[0])
    means, stds = [], []
    for j in range(k):
        mean, std, _ = actor.diffusion._posterior_mean_std(
            trace["x_t"][j], trace["timestep"][j], c1, c2, m, denoise_fn=fn)
        means.append(mean); stds.append(std)
    return jnp.stack(means, axis=0), jnp.stack(stds, axis=0)


def _time(fn, *, warmup=2, reps=10):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / reps * 1e3  # ms/call


def main() -> int:
    # Reuse the trainer's argparser so the model/batch match a real run.
    args = build_argparser().parse_args()
    print(f"[bench] devices={jax.devices()}")
    print(f"[bench] ckpt={args.ckpt}")
    print(f"[bench] tfrecord={args.tfrecord}")
    print(f"[bench] indices={args.indices} k_trainable={args.k_trainable} "
          f"ppo_epochs={args.ppo_epochs}", flush=True)

    trainer = DPPOTrainer(args)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    state = load_batch(args.tfrecord, indices)
    trainer._build_for(state)

    buf, adv, returns, alive, rewards = trainer.collect(state)
    merged = trainer._merge(buf, adv, returns, alive)
    cond1, cond2, mask, trace, old_lp, adv_f, ret_f, expert_target, alive_f = merged
    K, NB = trace["x_t"].shape[0], trace["x_t"].shape[1]
    print(f"[bench] trace x_t shape={tuple(trace['x_t'].shape)}  (K={K}, n*B={NB})", flush=True)

    actor = trainer.actor
    p = actor._tparams

    # --- forward pieces --------------------------------------------------- #
    lp_new = _time(lambda: actor.log_probs(p, cond1, cond2, mask, trace))
    lp_old = _time(lambda: ref_log_probs(actor, p, cond1, cond2, mask, trace))
    ms_new = _time(lambda: actor._posterior_means_std(p, cond1, cond2, mask, trace))
    ms_old = _time(lambda: ref_means_std(actor, p, cond1, cond2, mask, trace))

    # --- full ppo_update -------------------------------------------------- #
    # New: as shipped (jitted + behavior-cache). Reset the cache each timed call
    # so we measure a representative first-epoch step, not a cache hit.
    def new_update():
        actor._behavior_musd = None
        return actor.ppo_update(cond1, cond2, mask, trace, old_lp, adv_f, alive_f,
                                target_kl=args.target_kl)

    upd_new = _time(new_update, warmup=2, reps=6)

    # Old: monkeypatch the per-step Python loops back in and disable the cache.
    actor.log_probs = lambda params, c1, c2, mm, tr: ref_log_probs(actor, params, c1, c2, mm, tr)
    actor._posterior_means_std = lambda params, c1, c2, mm, tr: ref_means_std(actor, params, c1, c2, mm, tr)

    def old_update():
        actor._behavior_musd = None
        return actor.ppo_update(cond1, cond2, mask, trace, old_lp, adv_f, alive_f,
                                target_kl=args.target_kl)

    upd_old = _time(old_update, warmup=2, reps=6)

    def row(name, new, old):
        print(f"  {name:<22}  old={old:8.2f} ms   new={new:8.2f} ms   "
              f"speedup={old/new:5.2f}x", flush=True)

    print("\n=========== update-path timing (per call) ===========")
    row("log_probs (fwd)", lp_new, lp_old)
    row("posterior_means_std", ms_new, ms_old)
    row("ppo_update (full)", upd_new, upd_old)
    print(f"\nppo_epochs={args.ppo_epochs} -> est. per-iter update: "
          f"old~{upd_old*args.ppo_epochs:.0f} ms  new~{upd_new*args.ppo_epochs:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
