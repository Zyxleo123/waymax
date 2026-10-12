"""Measure the inference-time distribution drift FlowMol3 is aimed at.

THE CLAIM UNDER TEST. FlowMol3 (arXiv 2508.12629) says transport models "produce
suboptimal geometries during inference despite never observing suboptimal
geometries at training time": the reverse chain's own ``x_t`` is not a sample
from the forward path the network was trained on, and the gap compounds. This
repo already states the same thing in
``rl/diffusion_ft/unrolled_prediction.py`` ("Diffusion-state mismatch") but has
never put a number on it. Without that number the distortion scale would be a
guess transplanted from a molecule.

WHAT IS MEASURED, per reverse timestep t, on real scene conditions:

  path_std      per-element std of ``r_t = (x_t - sqrt(abar_t) x0_final) / sqrt(1-abar_t)``.
                If the chain were a forward path from its OWN final sample,
                ``r_t`` would be a standard normal and this would be 1.0.
                Departure from 1.0 is drift, in units of the forward noise.

  d_free_m      || x0_hat(x_t, t) - x0_final ||, positions, metres. How far the
                network's clean prediction at step t is from where the chain
                ends up. This is the revision still outstanding at t.

  d_tf_m        the same, but evaluated at a TEACHER-FORCED state
                ``q_sample(x0_final, t)`` -- an exact point of the forward path
                carrying the same amount of noise. Same t, same target, the only
                difference is which x_t the network is asked about.

  gap_m         d_free_m - d_tf_m. THIS IS THE DRIFT PENALTY. It is what the
                network loses by being asked about its own trajectory instead of
                a training-distribution one. A gap near zero at low t means the
                free chain is as in-distribution as a noised ground truth and
                there is nothing here for distortion to fix -- in which case do
                not spend a fine-tune on it.

  rev_m         || x0_hat(x_t, t) - x0_hat(x_{t+1}, t+1) ||, metres: how much the
                network revises its answer per reverse step. Its late-chain value
                is the natural SCALE for the distortion: a kick far below it is
                invisible against the chain's own motion, one far above it is
                just noise the network cannot be expected to undo.

WHAT TO DO WITH THE OUTPUT. The printed summary ends with a suggested
``--distort_sigma``. In ``relative`` mode the suggestion is the median of
``rev_m`` over the distorted window divided by the forward noise there, i.e. "a
kick the size of one step's own revision". Treat it as a centre for a sweep, not
a setting: the training arms sweep it either way.

Everything is written to ``drift.npz`` + ``summary.json`` so the figure is
redrawable on a CPU node without re-rolling anything.

GPU job. Submit it::

    bash rl/diffusion_ft/slurm/submit_probe_drift.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.train_dppo import load_batch


def _pos_rmse_m(a_bct: jnp.ndarray, b_bct: jnp.ndarray, ego_range: float) -> jnp.ndarray:
    """Per-scene RMS position error in metres, over the horizon. ``[B]``.

    Channels 0/1 are x/y normalized by ``ego_range``; the rest (velocity, yaw)
    live on their own scales and are deliberately excluded -- FlowMol3's
    distortion is a GEOMETRY distortion and the quantity that sets its size must
    be a length.
    """
    d = (a_bct[:, :2, :] - b_bct[:, :2, :]) * ego_range
    return jnp.sqrt(jnp.mean(jnp.sum(d ** 2, axis=1), axis=-1))


def probe_chunk(diffusion, cond1, cond2, mask, *, rng, shape, ego_range):
    """Run one full reverse chain, recording the per-step diagnostics.

    The chain is re-run here rather than via ``sample_with_trace`` because we need
    a denoiser evaluation at a SECOND, teacher-forced state at every t, and the
    sampler must not be perturbed by it. The free-running half reproduces
    ``sample()`` exactly (same rng, same unit-step eta-DDIM updates).
    """
    steps = int(diffusion.timesteps)
    key_x, key_steps = jax.random.split(rng)
    x = jax.random.normal(key_x, shape, dtype=jnp.float32)

    xs, x0s, ts = [], [], []
    key = key_steps
    sc = jnp.zeros(shape, jnp.float32) if diffusion.self_cond else None
    for i in range(steps):
        t = jnp.full((shape[0],), steps - 1 - i, dtype=jnp.int32)
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, shape, dtype=jnp.float32)
        _, _, x0p = diffusion.p_mean_variance(x, t, cond1, cond2, mask, x_sc=sc)
        xs.append(x); x0s.append(x0p); ts.append(int(steps - 1 - i))
        x = diffusion._ddim_update(x, t, x0p, noise, eta=1.0, sampling_temp=1.0)
        sc = diffusion._next_sc(x0p)
    x0_final = x  # what the chain actually emitted

    sqrt_ab = np.asarray(diffusion.sqrt_alphas_cumprod.value)
    sqrt_1mab = np.asarray(diffusion.sqrt_one_minus_alphas_cumprod.value)

    out = {k: [] for k in ("t", "path_std", "d_free_m", "d_tf_m", "gap_m", "rev_m")}
    key_tf = jax.random.fold_in(rng, 999)
    for j, t_scalar in enumerate(ts):
        t = jnp.full((shape[0],), t_scalar, dtype=jnp.int32)
        x_t, x0_free = xs[j], x0s[j]

        # (1) is x_t on the forward path of the chain's OWN output?
        r = (x_t - sqrt_ab[t_scalar] * x0_final) / sqrt_1mab[t_scalar]
        out["path_std"].append(float(jnp.std(r)))

        # (2) the same network, the same t, at an exact forward-path point.
        key_tf, k = jax.random.split(key_tf)
        tf_noise = jax.random.normal(k, shape, dtype=jnp.float32)
        x_t_tf = diffusion.q_sample(x0_final, t, noise=tf_noise)
        # The teacher-forced state has no previous prediction, which is also the
        # convention a training step uses for the unconditioned share.
        _, _, x0_tf = diffusion.p_mean_variance(
            x_t_tf, t, cond1, cond2, mask,
            x_sc=jnp.zeros_like(x_t_tf) if diffusion.self_cond else None)

        d_free = _pos_rmse_m(x0_free, x0_final, ego_range)
        d_tf = _pos_rmse_m(x0_tf, x0_final, ego_range)
        out["t"].append(t_scalar)
        out["d_free_m"].append(np.asarray(d_free))
        out["d_tf_m"].append(np.asarray(d_tf))
        out["gap_m"].append(np.asarray(d_free - d_tf))
        rev = (_pos_rmse_m(x0s[j], x0s[j - 1], ego_range) if j > 0
               else jnp.zeros((shape[0],)))
        out["rev_m"].append(np.asarray(rev))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--tfrecord", default="/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--n_scenes", type=int, default=64)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_diffusion_checkpoint(a.ckpt)
    env = DiffusionWaymaxEnv(ckpt, use_ema=bool(a.use_ema), seed=a.seed, max_replans=1)
    policy = env.policy
    diffusion = policy.diffusion
    ego_range = float(ckpt.preprocess_cfg.ego_range)
    print(f"[drift] ckpt={a.ckpt}")
    print(f"[drift] timesteps={diffusion.timesteps} self_cond={diffusion.self_cond} "
          f"ego_range={ego_range}", flush=True)

    idx_all = list(range(a.start, a.start + a.n_scenes))
    per_t = None
    n_done = 0
    for c0 in range(0, len(idx_all), a.chunk):
        idx = idx_all[c0:c0 + a.chunk]
        state = load_batch(a.tfrecord, idx)
        feats = env.reset(state, seed=a.seed)
        cond1, cond2 = env.compute_condition(feats)
        # The float mask the env itself passes into the FiLM blocks -- casting
        # it to bool here would be a second, subtly different conditioning.
        mask = feats["inst_valid"]
        shape = (len(idx), int(policy.target_dim), int(policy.predict_horizon))
        res = probe_chunk(diffusion, cond1, cond2, mask,
                          rng=jax.random.PRNGKey(a.seed + c0), shape=shape,
                          ego_range=ego_range)
        if per_t is None:
            per_t = {k: ([] if k != "t" else res["t"]) for k in res}
        for k in ("d_free_m", "d_tf_m", "gap_m", "rev_m"):
            per_t[k].append(np.stack(res[k], axis=0))          # [steps, B]
        per_t.setdefault("_path_std", []).append(np.asarray(res["path_std"]))
        n_done += len(idx)
        print(f"[drift] {n_done}/{len(idx_all)} scenes", flush=True)

    ts = np.asarray(per_t["t"], dtype=np.int32)                 # high t -> 0
    arrs = {k: np.concatenate(per_t[k], axis=1) for k in ("d_free_m", "d_tf_m", "gap_m", "rev_m")}
    path_std = np.mean(np.stack(per_t["_path_std"], axis=0), axis=0)

    sqrt_1mab = np.asarray(diffusion.sqrt_one_minus_alphas_cumprod.value)
    np.savez_compressed(out_dir / "drift.npz", t=ts, path_std=path_std,
                        sqrt_1mab=sqrt_1mab, **arrs)

    print("\n    t   path_std   d_tf_m  d_free_m    gap_m    rev_m")
    for i, t in enumerate(ts):
        if t % 10 == 0 or t < 5:
            print(f" {int(t):4d}   {path_std[i]:7.3f}  {np.median(arrs['d_tf_m'][i]):7.3f} "
                  f"{np.median(arrs['d_free_m'][i]):9.3f} {np.median(arrs['gap_m'][i]):8.3f} "
                  f"{np.median(arrs['rev_m'][i]):8.3f}")

    t_max = diffusion.timesteps // 2 - 1
    late = ts <= t_max
    rev_late = float(np.median(arrs["rev_m"][late]))
    gap_late = float(np.median(arrs["gap_m"][late]))
    # Suggested relative sigma: one step's own revision, expressed in units of the
    # forward noise present over the distorted window.
    fwd_late_m = float(np.median(sqrt_1mab[ts[late]])) * ego_range
    sigma_rel = rev_late / max(fwd_late_m, 1e-6)
    summary = {
        "ckpt": a.ckpt, "tfrecord": a.tfrecord, "n_scenes": n_done,
        "timesteps": int(diffusion.timesteps), "ego_range": ego_range,
        "distort_t_max": int(t_max),
        "path_std_late_median": float(np.median(path_std[late])),
        "gap_m_late_median": gap_late,
        "rev_m_late_median": rev_late,
        "forward_noise_late_median_m": fwd_late_m,
        "suggested_distort_sigma_relative": float(sigma_rel),
        "suggested_distort_sigma_absolute": float(rev_late / ego_range),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[drift] late window t<={t_max}: path_std={summary['path_std_late_median']:.3f} "
          f"(1.0 == on the forward path)")
    print(f"[drift] drift penalty gap={gap_late:+.3f} m; per-step revision={rev_late:.3f} m")
    print(f"[drift] suggested --distort_mode relative --distort_sigma {sigma_rel:.3f}")
    print(f"[drift]        (or --distort_mode absolute --distort_sigma "
          f"{rev_late / ego_range:.5f}, i.e. {rev_late:.2f} m)")
    print(f"[drift] wrote {out_dir}/drift.npz and summary.json")


if __name__ == "__main__":
    main()
