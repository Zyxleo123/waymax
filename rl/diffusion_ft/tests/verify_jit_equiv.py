"""Verify the jitted K-batched update path matches the reference Python loop.

The speedup (rl_actor: fold the per-step `for j in range(k)` denoiser loops into a
single fused, jitted K*B-batch call, and cache the behavior posterior across
epochs) must not change the numbers PPO acts on. This reconstructs the old
Python-loop implementation inline and compares, on a tiny synthetic actor:

* per-transition log-probs  [K, B]
* posterior means / stds     [K, B, C, T]
* the exact Gaussian KL the trust-region gate reads (`kl_gauss`)

Prints max abs differences; exits non-zero if any exceeds tolerance.

Run (tiny, CPU, seconds):  python -m rl.diffusion_ft.tests.verify_jit_equiv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model.diffusion.diffusion import GaussianDiffusion, UNet1DConditioned
from rl.diffusion_ft.rl_actor import DiffusionRLActor


def _tiny_actor(**kw):
    C, T, cond_dim, hidden = 5, 8, 16, 32
    rngs = nnx.Rngs(0)
    unet = UNet1DConditioned(
        in_ch=C, cond_dim=cond_dim, cond2_dim=cond_dim, time_emb_dim=cond_dim,
        horizon=T, hidden_dim=hidden, base_ch=16, rngs=rngs,
    )
    diff = GaussianDiffusion(unet, timesteps=10, predict_type="v")
    B = 3
    cond1 = jax.random.normal(jax.random.PRNGKey(1), (B, 4, cond_dim))
    cond2 = jax.random.normal(jax.random.PRNGKey(2), (B, cond_dim))
    mask = jnp.ones((B,), dtype=jnp.float32)
    actor = DiffusionRLActor(diff, k_trainable=4, prefix_len=3, seed=0, **kw)
    return actor, cond1, cond2, mask


# ---- reference implementations (the pre-optimization Python loops) --------- #
def ref_log_probs(actor, params, c1, c2, m, trace):
    fn = actor._trainable_fn(params)
    k = int(trace["timestep"].shape[0])
    lps = []
    for j in range(k):
        lp = actor.diffusion.transition_log_prob(
            trace["x_t"][j], trace["x_prev"][j], trace["timestep"][j], c1, c2, m,
            sigma_floor=actor.sigma_logprob_floor, prefix_len=actor.prefix_len,
            denoise_fn=fn,
        )
        lps.append(lp)
    return jnp.stack(lps, axis=0)


def ref_means_std(actor, params, c1, c2, m, trace):
    fn = actor._trainable_fn(params)
    k = int(trace["timestep"].shape[0])
    means, stds = [], []
    for j in range(k):
        mean, std, _ = actor.diffusion._posterior_mean_std(
            trace["x_t"][j], trace["timestep"][j], c1, c2, m, denoise_fn=fn)
        means.append(mean)
        stds.append(std)
    return jnp.stack(means, axis=0), jnp.stack(stds, axis=0)


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> int:
    actor, c1, c2, m = _tiny_actor(actor_lr=1e-3, ppo_clip=0.5)
    traj, trace = actor.collect(c1, c2, m, rng=jax.random.PRNGKey(100))

    # Two distinct param sets so we exercise more than the identity case.
    p0 = actor._tparams
    p1 = jax.tree_util.tree_map(lambda x: x + 0.01, actor._tparams)

    tol = 1e-4
    worst = 0.0
    for name, p in (("behavior", p0), ("perturbed", p1)):
        lp_new = actor.log_probs(p, c1, c2, m, trace)
        lp_ref = ref_log_probs(actor, p, c1, c2, m, trace)
        mu_new, sd_new = actor._posterior_means_std(p, c1, c2, m, trace)
        mu_ref, sd_ref = ref_means_std(actor, p, c1, c2, m, trace)
        d_lp = _maxabs(lp_new, lp_ref)
        d_mu = _maxabs(mu_new, mu_ref)
        d_sd = _maxabs(sd_new, sd_ref)
        worst = max(worst, d_lp, d_mu, d_sd)
        print(f"[{name}]  d(log_probs)={d_lp:.2e}  d(mean)={d_mu:.2e}  d(std)={d_sd:.2e}")

    # Gate KL: candidate vs behavior, both via the jitted path, compared to the
    # same quantity built from the reference loops.
    alive = jnp.ones((c1.shape[0],))
    mu_old_new, std_new = actor._posterior_means_std(p0, c1, c2, m, trace)
    mu_cand_new, _ = actor._posterior_means_std(p1, c1, c2, m, trace)
    kl_new = actor._transition_kl(mu_old_new, mu_cand_new, std_new, alive)

    mu_old_ref, std_ref = ref_means_std(actor, p0, c1, c2, m, trace)
    mu_cand_ref, _ = ref_means_std(actor, p1, c1, c2, m, trace)
    kl_ref = actor._transition_kl(mu_old_ref, mu_cand_ref, std_ref, alive)
    d_kl = abs(float(kl_new) - float(kl_ref))
    worst = max(worst, d_kl)
    print(f"[gate]   kl_new={float(kl_new):.6e}  kl_ref={float(kl_ref):.6e}  d(kl)={d_kl:.2e}")

    ok = worst <= tol
    print(f"\nworst abs diff = {worst:.2e}  (tol {tol:.0e})  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
