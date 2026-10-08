"""DDIM frozen-prefix subsampling: correctness + PPO-invariance checks.

The ``num_sample_steps`` knob subsamples only the *frozen* early portion of the
reverse chain. This asserts:

* the generalized strided step reduces to the unit-step DDIM update for a unit
  step (so ``num_sample_steps=timesteps`` is the old sampler);
* a subsampled sample runs and its recorded *late* transitions still live at
  timesteps ``[k-1 .. 0]`` -- i.e. the trainable/PPO region is byte-for-byte the
  same schedule, only its input ``x_{k-1}`` changes;
* the default (``num_sample_steps=None``) trajectory is bit-identical to passing
  ``num_sample_steps=timesteps`` (both take the untouched full chain).

Tiny synthetic model, CPU, seconds.
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


def _tiny():
    C, T, cond_dim = 5, 8, 16
    rngs = nnx.Rngs(0)
    unet = UNet1DConditioned(in_ch=C, cond_dim=cond_dim, cond2_dim=cond_dim,
                             time_emb_dim=cond_dim, horizon=T, hidden_dim=32, base_ch=16, rngs=rngs)
    diff = GaussianDiffusion(unet, timesteps=20, predict_type="v")
    B = 3
    c1 = jax.random.normal(jax.random.PRNGKey(1), (B, 4, cond_dim))
    c2 = jax.random.normal(jax.random.PRNGKey(2), (B, cond_dim))
    m = jnp.ones((B,), jnp.float32)
    return diff, c1, c2, m, (B, C, T)


def unit_step_equivalence() -> bool:
    diff, c1, c2, m, shape = _tiny()
    x = jax.random.normal(jax.random.PRNGKey(3), shape)
    noise = jax.random.normal(jax.random.PRNGKey(4), shape)
    worst = 0.0
    for tv in (17, 11, 5, 1):
        t = jnp.full((shape[0],), tv, jnp.int32)
        _, _, x0 = diff.p_mean_variance(x, t, c1, c2, m)
        a = diff._ddim_update(x, t, x0, noise, eta=1.0)
        # `_ddim_step_strided` returns (x_next, x0_pred) since the self-cond
        # port: the sampler needs the step's clean prediction to carry forward.
        b, _ = diff._ddim_step_strided(x, t, t - 1, c1, c2, m, noise, eta=1.0)
        worst = max(worst, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    print(f"  unit-step |ddim_strided - ddim_update| max = {worst:.2e}")
    return worst < 1e-5


def late_region_schedule_unchanged() -> bool:
    diff, c1, c2, m, shape = _tiny()
    k = 6
    ok = True
    for S in (20, 12, 8):
        _, tr = diff.sample_with_trace(shape, c1, c2, m, rng=jax.random.PRNGKey(7),
                                       k_trainable=k, rl_mode=True,
                                       sigma_sample_floor=0.1, sigma_logprob_floor=0.1,
                                       num_sample_steps=(None if S == 20 else S))
        ts = np.asarray(tr["timestep"])  # [k, B]
        want = np.array([k - 1 - j for j in range(k)])  # [k-1 .. 0]
        same = ts.shape[0] == k and np.array_equal(ts[:, 0], want)
        print(f"  S={S:>2}: late timesteps={ts[:,0].tolist()} (want {want.tolist()})  ok={same}")
        ok = ok and same
    return ok


def scan_late_matches_unrolled() -> bool:
    """The compiled rl_mode late chain (scan) == the unrolled late loop, same rng.

    Checks every recorded trace field and the final trajectory, with and without
    frozen-prefix subsampling, so collect stays numerically the sampler it was.
    """
    diff, c1, c2, m, shape = _tiny()
    k = 6
    worst = 0.0
    for S in (None, 12):
        kw = dict(k_trainable=k, rl_mode=True, sigma_sample_floor=0.1,
                  sigma_logprob_floor=0.1, prefix_len=3, num_sample_steps=S)
        xs, trs = diff.sample_with_trace(shape, c1, c2, m, rng=jax.random.PRNGKey(5),
                                         scan_late=True, **kw)
        xu, tru = diff.sample_with_trace(shape, c1, c2, m, rng=jax.random.PRNGKey(5),
                                         scan_late=False, **kw)
        d = float(np.max(np.abs(np.asarray(xs) - np.asarray(xu))))
        for key in ("x_t", "x_prev", "timestep", "mean", "std", "log_prob"):
            d = max(d, float(np.max(np.abs(np.asarray(trs[key]) - np.asarray(tru[key])))))
        print(f"  S={S}: max|scan - unrolled| over traj+trace = {d:.2e}")
        worst = max(worst, d)
    return worst < 1e-5


def default_matches_full() -> bool:
    diff, c1, c2, m, shape = _tiny()
    a, _ = diff.sample_with_trace(shape, c1, c2, m, rng=jax.random.PRNGKey(9),
                                  k_trainable=6, rl_mode=False, num_sample_steps=None)
    b, _ = diff.sample_with_trace(shape, c1, c2, m, rng=jax.random.PRNGKey(9),
                                  k_trainable=6, rl_mode=False, num_sample_steps=20)  # == timesteps
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    print(f"  |default - num_sample_steps=timesteps| max = {d:.2e}")
    return d == 0.0


def main() -> int:
    print(f"[ddim_stride] devices={jax.devices()}")
    tests = [("unit-step equivalence", unit_step_equivalence),
             ("late region schedule unchanged", late_region_schedule_unchanged),
             ("scan late == unrolled late", scan_late_matches_unrolled),
             ("default == full chain", default_matches_full)]
    results = {}
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            results[name] = fn()
        except Exception:
            import traceback; traceback.print_exc(); results[name] = False
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    allp = all(results.values())
    print(f"\nDDIM-stride overall: {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
