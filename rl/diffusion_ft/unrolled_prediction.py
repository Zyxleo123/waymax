"""Teacher-forced, unrolled clean prediction for the analytic safety objective.

Ported from the ``bondunroll`` mechanic in ``~/Proteina-Complexa``
(``src/proteinfoundation/training/unrolled_prediction.py``). The protein version
penalises the endpoint of a short ODE unroll instead of the network's one-step
clean prediction; this is the same idea on a discrete-time DDPM/DDIM chain, in
JAX, with the closing-bond distance replaced by :mod:`rl.diffusion_ft.safety_loss`.

Why unroll at all
-----------------
An auxiliary loss that reads a *decoded trajectory* during training is normally
applied to the network's one-step clean prediction
``x0_hat(x_t, t) = E[x0 | x_t]`` -- the conditional **mean**. Every quantity the
safety loss measures (a distance to a road edge, a box-box gap) is a nonlinear
functional of the coordinates, so

    clearance( E[x0 | x_t] )  !=  E[ clearance(x0) | x_t ].

For clearance the inequality runs the wrong way. Averaging over the posterior of
plausible futures at a given noise level smooths the trajectory toward the middle
of the drivable corridor, so the mean can sit comfortably inside the margin while
a fair share of the trajectories the sampler actually emits clip the edge. That
is a Jensen gap, not estimation error: more data will not close it, and more loss
weight only controls the mean harder while leaving the spread -- the thing you
sample from -- untouched.

``k_steps`` reverse steps from ``(x_t, t)`` down to a low-noise timestep move the
penalised object from "the posterior mean" toward "a sample". ``k_steps=1``
reproduces the one-step behaviour exactly, so this is a strict generalisation and
the natural ablation control.

Teacher forcing
---------------
``x_t`` here is a *corrupted ground-truth* trajectory, ``q_sample(x0_expert, t)``,
not the state a free-running sampler would have reached from pure noise. That is
the teacher-forcing half, and it is what makes this cheap: the corruption replaces
the ``timesteps - t`` frozen denoiser evaluations that a free-running sample would
have spent getting to the same noise level. At the shipped defaults one gradient
step costs ``k_steps`` network evaluations instead of the 100 that
``sample_with_trace`` runs.

What it does NOT fix
--------------------
Exposure bias, in two distinct places, both worth stating because the cheapness
above is bought with exactly these:

* **Diffusion-state mismatch.** ``q_sample(x0_expert, t)`` is a noised human
  trajectory; the sampler's ``x_t`` is whatever the frozen high-noise steps
  produced. They are different distributions at the same ``t``.
* **Environment-state mismatch**, if collection is switched to log states
  (``--anchor_states log``): the policy is then only ever taught to be safe from
  states a human visited. The free-running collection in
  :mod:`rl.diffusion_ft.train_safety` is the arm that does not have this, and it
  is the default.

This is also a plain eta-DDIM unroll: with the default ``eta=0`` it injects no
noise, which deliberately does not reproduce the deployed stochastic sampler
(``eta=1``). Read this module as the cheap variance-aware correction and
``train_safety``'s free-running path as the faithful one.

Where the unroll may start
--------------------------
Only inside the actor's trainable window. ``DiffusionRLActor`` hands the
trainable denoiser copy the last ``k_trainable`` reverse steps and leaves the rest
frozen, so a gradient taken at ``t >= k_trainable`` trains the copy at timesteps
it is never invoked at during sampling -- it reaches the deployed behaviour only
through weight sharing, which is not what anyone is asking for.
:func:`unroll_timestep_grid` does not enforce this; the trainer does.

Residual forward noise on the position channels at each timestep, in metres
(``ego_range=100``), for choosing ``t_start``/``t_end`` on the checkpoint's
100-step cosine schedule::

    t      0     1     2     3     5     9    19    49    99
    m    2.5   4.2   5.8   7.4  10.5  16.7  31.8  71.1  100.0
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "unroll_timestep_grid",
    "unrolled_clean_prediction",
    "teacher_forced_unroll",
]


def unroll_timestep_grid(t_start_b: jax.Array, k_steps: int, t_end: int) -> jax.Array:
    """``[K, B]`` descending per-sample timestep grid from ``t_start`` to ``t_end``.

    ``t_start`` is per-sample (a batch draws a different corruption level for
    every scene), so the grid is per-sample too: a scene starting at ``t=9`` takes
    large strides and one starting at ``t=2`` takes small ones, both landing on
    ``t_end`` after ``k_steps`` network evaluations.

    Rounding is monotone, so the grid is non-increasing by construction. A tied
    pair ``from_t == to_t`` (unavoidable when ``k_steps`` exceeds the number of
    timesteps spanned) is an exact no-op rather than an error: the strided DDIM
    sigma vanishes and the direction coefficient collapses to ``sqrt(1-abar)``,
    which reconstructs ``x_t`` identically.

    Row ``K-1`` is set to ``t_end`` exactly rather than left to float arithmetic,
    because the caller reads it back as "the noise level the loss is evaluated at".
    """
    t_start = jnp.asarray(t_start_b, jnp.float32)
    k = int(k_steps)
    if k < 1:
        raise ValueError(f"k_steps must be >= 1, got {k}")
    if k == 1:
        return jnp.round(t_start)[None, :].astype(jnp.int32)
    frac = jnp.arange(k, dtype=jnp.float32) / float(k - 1)          # 0 .. 1
    grid = t_start[None, :] + (float(t_end) - t_start[None, :]) * frac[:, None]
    grid = jnp.round(grid).astype(jnp.int32)
    return grid.at[k - 1].set(jnp.full(grid.shape[1:], int(t_end), jnp.int32))


def unrolled_clean_prediction(
    diffusion,
    merge_fn: Callable,
    params,
    *,
    x_t: jax.Array,
    t_grid_kb: jax.Array,
    cond1: jax.Array,
    cond2: jax.Array,
    cond2_mask: jax.Array,
    grad_steps: int = 1,
    eta: float = 0.0,
    rng: jax.Array,
) -> jax.Array:
    """Clean prediction at the end of a ``K``-step reverse unroll from ``x_t``.

    Runs ``K-1`` eta-DDIM transitions along ``t_grid_kb`` and returns the clean
    prediction of the ``K``-th (final, lowest-noise) network evaluation. With
    ``t_end=0`` that endpoint is *exactly* what the sampler emits: at ``t=0`` the
    DDIM update has ``abar_prev=1``, a zero direction coefficient and its noise
    masked off, so ``x_out == x0_pred`` identically.

    Args:
      diffusion: the :class:`GaussianDiffusion`.
      merge_fn: ``params -> denoiser module``, i.e. ``DiffusionRLActor._trainable_fn``.
      params: the trainable denoiser's parameter pytree -- the thing being
        differentiated. Passed through ``merge_fn`` rather than as a module so the
        caller can trace it, which is what lets ``jax.grad`` reach the weights and
        what keeps the jit cache from growing an entry per iteration.
      x_t: ``[B, C, T]`` noised trajectory at ``t_grid_kb[0]``.
      t_grid_kb: ``[K, B]`` from :func:`unroll_timestep_grid`.
      grad_steps: how many of the FINAL evaluations keep a gradient. Earlier steps
        run against ``stop_gradient(params)``, so no graph is built through them --
        this is DRaFT-K's truncated backprop, and it is what keeps activation
        memory flat as ``k_steps`` grows. Clamped into ``[1, K]``.
      eta: DDIM noise scale. 0 (default) is the deterministic ODE unroll, matching
        the mechanic this is ported from; 1.0 is the ancestral chain the deployed
        sampler runs. Noise is masked at ``to_t == 0`` either way, as
        ``_ddim_update`` does.
      rng: key for the per-step DDIM noise. Unused when ``eta == 0``, still split
        so the step count does not silently change the stream.

    Returns:
      ``[B, C, T]`` normalized clean-trajectory prediction.
    """
    k = int(t_grid_kb.shape[0])
    grad_steps = max(1, min(int(grad_steps), k))
    first_grad_step = k - grad_steps

    # Two merges of the same graphdef: one traced, one cut off from the gradient.
    # Building them once outside the loop keeps the per-step cost to a dict lookup.
    fn_grad = merge_fn(params)
    fn_nograd = merge_fn(jax.lax.stop_gradient(params)) if first_grad_step > 0 else fn_grad

    x = x_t
    # Self-conditioning carry. `None` (an empty pytree) off that path, so the
    # unroll below is unchanged for the default architecture. The unroll ENTERS at
    # a corrupted expert state with no previous prediction, which is the same
    # zeros convention the first reverse step uses.
    sc = jnp.zeros_like(x) if getattr(diffusion, "self_cond", False) else None
    for step in range(k - 1):
        from_t, to_t = t_grid_kb[step], t_grid_kb[step + 1]
        rng, step_key = jax.random.split(rng)
        if eta == 0.0:
            noise = jnp.zeros_like(x)
        else:
            # `_ddim_step_strided` always adds `sigma * noise`; the unit-step
            # sampler suppresses it at t=0. Zero the noise there instead of
            # special-casing the landing, so the two agree at `t_end=0`.
            noise = jax.random.normal(step_key, x.shape, dtype=x.dtype)
            noise = noise * (to_t != 0).astype(x.dtype)[:, None, None]
        x, x0_step = diffusion._ddim_step_strided(
            x, from_t, to_t, cond1, cond2, cond2_mask, noise,
            eta=float(eta), denoise_fn=(fn_grad if step >= first_grad_step else fn_nograd),
            x_sc=sc,
        )
        sc = diffusion._next_sc(x0_step)

    _, _, x0_pred = diffusion.p_mean_variance(
        x, t_grid_kb[k - 1], cond1, cond2, cond2_mask, denoise_fn=fn_grad, x_sc=sc)
    return x0_pred


def teacher_forced_unroll(
    diffusion,
    merge_fn: Callable,
    params,
    *,
    x0_bct: jax.Array,
    t_grid_kb: jax.Array,
    cond1: jax.Array,
    cond2: jax.Array,
    cond2_mask: jax.Array,
    grad_steps: int = 1,
    eta: float = 0.0,
    rng: jax.Array,
) -> jax.Array:
    """Corrupt an expert trajectory to ``t_grid_kb[0]``, then unroll it back down.

    The teacher-forcing entry point: ``x0_bct`` is the normalized expert (log)
    future in ``[B, C, T]`` -- the same target
    :meth:`DiffusionRLActor._expert_loss` trains the anchor on, transposed -- and
    the returned trajectory is what the safety loss scores.

    One rng draws both the corruption noise and the DDIM noise, so a caller only
    has to thread one key per gradient step.
    """
    key_corrupt, key_steps = jax.random.split(rng)
    noise = jax.random.normal(key_corrupt, x0_bct.shape, dtype=x0_bct.dtype)
    x_t = diffusion.q_sample(x0_bct, t_grid_kb[0], noise=noise)
    return unrolled_clean_prediction(
        diffusion, merge_fn, params,
        x_t=x_t, t_grid_kb=t_grid_kb, cond1=cond1, cond2=cond2,
        cond2_mask=cond2_mask, grad_steps=grad_steps, eta=eta, rng=key_steps,
    )
