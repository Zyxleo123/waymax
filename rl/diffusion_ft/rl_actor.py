"""DPPO actor: frozen base denoiser + trainable late-step denoiser copy.

Phase 2, item 4. The pretrained diffusion policy is split so that only the last
``k_trainable`` denoising steps are learnable:

* the **scene tokenizer and all instruction/subgoal modules are frozen** -- their
  output (the condition tensors ``cond1``/``cond2``/``cond2_mask``) is computed
  once by the frozen policy and passed in as arrays, so no gradient can reach
  them;
* the **frozen base denoiser** drives the early (high-noise) denoising steps and
  never updates;
* a **trainable copy** of the denoiser drives the final ``k_trainable`` steps and
  is the only thing PPO optimizes.

The actor exposes:

* :meth:`collect` -- run :meth:`GaussianDiffusion.sample_with_trace` (frozen early,
  trainable late, RL sampling mode) to produce a trajectory and the recorded
  transitions with old-policy log-probs;
* :meth:`ppo_update` -- one clipped-PPO gradient step on the trainable denoiser
  only, from stored transitions + advantages;
* :meth:`state_dict` / :meth:`load_state_dict` -- checkpoint just the trainable
  params, optimizer, and rng, so a run resumes bit-for-bit.

Gradient isolation is structural: the loss is differentiated w.r.t. the trainable
denoiser's ``nnx.Param`` state alone; everything else is captured as a constant.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rl.diffusion_ft.unrolled_prediction import (
    teacher_forced_unroll,
    unrolled_clean_prediction,
)


def _clone_module(module: nnx.Module) -> nnx.Module:
    """Deep-copy an nnx module (independent parameter arrays)."""
    gdef, state = nnx.split(module)
    state_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
    return nnx.merge(gdef, state_copy)


class DiffusionRLActor:
    """Wraps a :class:`GaussianDiffusion` for DPPO fine-tuning of late steps."""

    def __init__(
        self,
        diffusion,
        *,
        k_trainable: int = 10,
        prefix_len: int | None = None,
        sigma_sample_floor: float = 0.1,
        sigma_logprob_floor: float = 0.1,
        actor_lr: float = 3e-5,
        grad_clip_norm: float = 1.0,
        ppo_clip: float = 0.1,
        pg_scale: float | None = None,
        num_sample_steps: int | None = None,
        expert_lr: float | None = None,
        seed: int = 0,
    ):
        self.diffusion = diffusion
        self.k_trainable = int(k_trainable)
        self.prefix_len = prefix_len
        # None = full unit-step reverse chain (unchanged). An int < timesteps
        # DDIM-subsamples the frozen prefix to speed up collection; the trainable
        # late steps are unaffected (see GaussianDiffusion.sample_with_trace).
        self.num_sample_steps = None if num_sample_steps is None else int(num_sample_steps)
        self.sigma_sample_floor = float(sigma_sample_floor)
        self.sigma_logprob_floor = float(sigma_logprob_floor)
        # PPO requires the behavior policy (what `sample_with_trace` draws from)
        # and the density the ratio is computed under to be the *same* Gaussian.
        # A smaller sample floor than log-prob floor makes every recorded action
        # off-policy w.r.t. the density being optimized, biasing the gradient with
        # no importance correction. Enforce equality rather than silently mixing.
        if abs(self.sigma_sample_floor - self.sigma_logprob_floor) > 1e-9:
            raise ValueError(
                "sigma_sample_floor and sigma_logprob_floor must be equal "
                f"(got {self.sigma_sample_floor} vs {self.sigma_logprob_floor}); "
                "sampling and log-prob must describe the same policy."
            )
        self.ppo_clip = float(ppo_clip)
        # `None` = auto: undo the 1/(channels*prefix_len) averaging that
        # `_reduce_logdensity` applies, so the policy-gradient term is comparable
        # in magnitude to the (unscaled) expert diffusion loss. Resolved on the
        # first `ppo_update` once the trace shape is known.
        self.pg_scale = None if pg_scale is None else float(pg_scale)
        # The number of scalar log-densities `_reduce_logdensity` averages over
        # (channels * executed prefix). Recorded so the reported per-transition
        # KL/ratio can be recovered from the averaged quantity that PPO clips on.
        # Resolved alongside `pg_scale` on the first `ppo_update`.
        self._reduce_n: float | None = None
        self._rng = jax.random.PRNGKey(int(seed))

        # Frozen base denoiser = the diffusion's own denoise_fn (never updated).
        self.frozen_denoise = diffusion.denoise_fn
        # Trainable copy drives the last k_trainable steps.
        self.trainable_denoise = _clone_module(self.frozen_denoise)

        # Split the trainable denoiser into (graphdef, params, non-params). Only
        # `params` is optimized; everything else is a constant to the loss.
        self._tgdef, self._tparams, self._tnonparam = nnx.split(
            self.trainable_denoise, nnx.Param, ...
        )

        self.tx = optax.chain(
            optax.clip_by_global_norm(float(grad_clip_norm)),
            optax.adam(float(actor_lr)),
        )
        self._opt_state = self.tx.init(self._tparams)
        # The expert (behavior-cloning) term gets its OWN Adam, not a share of the
        # PPO one. Both objectives write to the same parameters, but their gradient
        # scales differ by orders of magnitude (the PG term is multiplied by
        # `pg_scale` = channels*prefix_len to undo the log-density averaging, the
        # diffusion loss is unscaled). Adam normalizes by a running RMS, so pooling
        # them into one moment buffer makes each objective's step size depend on how
        # often the *other* one ran -- which would confound exactly the expert:DPPO
        # step-ratio sweep this exists for. Separate state keeps the ratio the only
        # variable. `expert_lr=None` reuses the actor lr.
        self.tx_expert = optax.chain(
            optax.clip_by_global_norm(float(grad_clip_norm)),
            optax.adam(float(actor_lr if expert_lr is None else expert_lr)),
        )
        self._opt_state_expert = self.tx_expert.init(self._tparams)
        self._jit_expert_grad = None
        # (cond, trace) that `expert_update` measures its KL-vs-behavior on; set
        # per iteration by the trainer via `set_expert_kl_probe`.
        self._expert_kl_cond = None
        self._expert_kl_trace = None
        # Behavior params = the params `collect` last sampled under. The KL gate
        # measures the candidate policy against *this* snapshot (the policy that
        # generated the data), not against the drifting current params. Refreshed
        # in `collect`; seeded here so a `ppo_update` before any `collect` is safe.
        self._behavior_tparams = self._tparams
        # Cached (mu_old, std) of the behavior policy for the current data batch.
        # Behavior params, cond, and trace are all frozen across the PPO epochs of
        # one iteration, so the gate's `mu_old`/`std` are recomputed identically
        # every epoch -- pure redundancy. Filled on the first `ppo_update` of an
        # iteration and invalidated in `collect` when new data arrives.
        self._behavior_musd = None
        # Jitted, K-batched implementations of the per-step denoiser loops. The
        # late steps all share cond/denoiser and differ only in (x_t, x_prev, t),
        # so the K forwards fold into a single fused call over a K*B batch (the
        # denoiser is per-example -- GroupNorm, conv, cross-attn, FiLM -- so this is
        # numerically the same as the Python loop, just compiled and parallel).
        # Built lazily so the graphdef/nonparam captured are the final ones.
        self._jit_log_probs = None
        self._jit_means_std = None
        # value_and_grad(self._ppo_loss) compiled once and reused every iteration.
        # The old per-iter `jax.value_and_grad(self._ppo_loss)(...)` re-wrapped a
        # fresh bound method each call, so JAX cached a new entry per iteration and
        # leaked ~0.6 GiB/iter (with the critic; probes 32647/32673, clear_caches
        # flattened it 33056). Built lazily in ppo_update once pg_scale/_reduce_n
        # are resolved, so their values are baked into the trace correctly.
        self._jit_ppo_grad = None
        # Jitted samplers keyed by rl_mode, each built once. The old collect/act
        # merged the trainable module *outside* jit and passed it into the
        # un-jitted sample_with_trace, so every collect re-traced the late scan
        # with that iteration's params baked in as constants -> a fresh JAX cache
        # entry per iter (~0.6 GiB/iter host leak, probe 33193, present even in
        # warmup where only collect runs). Passing params as a *traced* arg and
        # merging inside (as batched_log_probs already does) compiles once.
        # Keyed (rl_mode, num_sample_steps) -- see `_sampler_for` for why the
        # step count has to be part of the key and not just a closure constant.
        self._jit_samplers: dict[tuple[bool, int | None], Any] = {}
        # Jitted teacher-forced unroll kernels, keyed by the Python constants that
        # change the traced program (loop length, gradient window, DDIM eta). Same
        # discipline as `_jit_samplers`: params are a *traced* argument merged
        # inside, so one entry is compiled for the run instead of one per iteration.
        self._jit_unrolls: dict[tuple[int, int, float], Any] = {}

    # ------------------------------------------------------------------ #
    def _sampler_for(self, rl_mode: bool):
        """Leak-free jitted sampler (params traced, merged inside), one per mode.

        Keyed on ``(rl_mode, num_sample_steps)``, NOT on ``rl_mode`` alone.
        ``num_sample_steps`` is read here and captured as a closure constant, so a
        cache keyed only on the mode would hand back a sampler compiled against
        whatever the attribute happened to be on the FIRST call and silently
        ignore every later assignment to it. That is not hypothetical: it made
        ``tests/bench_ddim_ab.py`` measure nothing -- it warms a collect (so the
        ``rl_mode=True`` entry is compiled at the starting value) and then sweeps
        by assigning ``actor.num_sample_steps = S``, which after that first call
        could not change the sampler. Every row of that A/B was the same full
        chain, so it reported ~1.0x speedup and flat quality at every step count,
        which reads exactly like "subsampling does not help". Including the value
        in the key is what makes the attribute live, and it is also what lets one
        run collect with a strided prefix while eval keeps the full chain.
        """
        mode = bool(rl_mode)
        num_sample_steps = self.num_sample_steps
        key = (mode, num_sample_steps)
        cached = self._jit_samplers.get(key)
        if cached is not None:
            return cached
        diffusion = self.diffusion
        tgdef, tnonparam = self._tgdef, self._tnonparam
        k_trainable, prefix_len = self.k_trainable, self.prefix_len
        sigma_s, sigma_lp = self.sigma_sample_floor, self.sigma_logprob_floor
        in_ch, horizon = diffusion.denoise_fn.in_ch, diffusion.denoise_fn.horizon

        def sampler(params, cond1, cond2, cond2_mask, rng):
            denoise = nnx.merge(tgdef, params, tnonparam)
            shape = (cond1.shape[0], in_ch, horizon)
            return diffusion.sample_with_trace(
                shape, cond1, cond2, cond2_mask, rng=rng,
                k_trainable=k_trainable, trainable_denoise_fn=denoise,
                rl_mode=mode, sigma_sample_floor=sigma_s,
                sigma_logprob_floor=sigma_lp, prefix_len=prefix_len,
                num_sample_steps=num_sample_steps)

        jitted = jax.jit(sampler)
        self._jit_samplers[key] = jitted
        return jitted

    # ------------------------------------------------------------------ #
    def _trainable_fn(self, params):
        """Reconstitute the trainable denoiser from a params pytree."""
        return nnx.merge(self._tgdef, params, self._tnonparam)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _tile_leading(cond1, cond2, cond2_mask, k):
        """Tile each condition tensor ``k`` times along a new leading block.

        ``x.reshape(k*B, ...)`` lays the K steps out as contiguous B-blocks
        ``[step0_b0..b(B-1), step1_b0..]``, so the matching condition is the whole
        ``[B, ...]`` block repeated k times -- ``jnp.tile``, not ``jnp.repeat``.
        """
        c1 = jnp.tile(cond1, (k,) + (1,) * (cond1.ndim - 1))
        c2 = jnp.tile(cond2, (k,) + (1,) * (cond2.ndim - 1))
        cm = jnp.tile(cond2_mask, (k,) + (1,) * (cond2_mask.ndim - 1))
        return c1, c2, cm

    def _ensure_jitted(self):
        """Build the K-batched jitted loop kernels once (idempotent)."""
        if self._jit_log_probs is not None:
            return
        diffusion = self.diffusion
        tgdef, tnonparam = self._tgdef, self._tnonparam
        prefix_len = self.prefix_len
        sigma_lp = self.sigma_logprob_floor
        tile = self._tile_leading

        # `x_sc` is the self-conditioning input the SAMPLER used at each recorded
        # transition (trace["x_sc"]). It is `None` -- an empty pytree, so the jitted
        # signature is unchanged -- for the default architecture. Re-deriving it
        # here instead of reading the trace would evaluate a different policy than
        # the one that acted.
        def batched_log_probs(params, cond1, cond2, cond2_mask, x_t, x_prev, timestep, x_sc=None):
            fn = nnx.merge(tgdef, params, tnonparam)
            k, b = x_t.shape[0], x_t.shape[1]
            xt = x_t.reshape((k * b,) + x_t.shape[2:])
            xp = x_prev.reshape((k * b,) + x_prev.shape[2:])
            tt = timestep.reshape((k * b,))
            xsc = None if x_sc is None else x_sc.reshape((k * b,) + x_sc.shape[2:])
            c1, c2, cm = tile(cond1, cond2, cond2_mask, k)
            lp = diffusion.transition_log_prob(
                xt, xp, tt, c1, c2, cm,
                sigma_floor=sigma_lp, prefix_len=prefix_len, denoise_fn=fn, x_sc=xsc)
            return lp.reshape((k, b))

        def batched_means_std(params, cond1, cond2, cond2_mask, x_t, timestep, x_sc=None):
            fn = nnx.merge(tgdef, params, tnonparam)
            k, b = x_t.shape[0], x_t.shape[1]
            xt = x_t.reshape((k * b,) + x_t.shape[2:])
            tt = timestep.reshape((k * b,))
            xsc = None if x_sc is None else x_sc.reshape((k * b,) + x_sc.shape[2:])
            c1, c2, cm = tile(cond1, cond2, cond2_mask, k)
            mean, std, _ = diffusion._posterior_mean_std(xt, tt, c1, c2, cm, denoise_fn=fn, x_sc=xsc)
            # `std` is the per-t schedule std -- shape (K*B, 1, 1), broadcast over
            # C and T -- so it reshapes with its own (1, 1) tail, not the mean's.
            return mean.reshape((k, b) + mean.shape[1:]), std.reshape((k, b) + std.shape[1:])

        self._jit_log_probs = jax.jit(batched_log_probs)
        self._jit_means_std = jax.jit(batched_means_std)

    # ------------------------------------------------------------------ #
    def collect(self, cond1, cond2, cond2_mask, *, rng=None):
        """Sample a trajectory + record the last k_trainable transitions.

        Uses the frozen denoiser for early steps and the current trainable copy
        for late steps, in RL sampling mode. Returns ``(trajectory, trace)``.
        """
        if rng is None:
            self._rng, rng = jax.random.split(self._rng)
        # Freeze the behavior policy for this data batch: the KL gate compares the
        # PPO candidate against these params, so it must be the params that draw
        # the trajectory. Collection within an iteration never updates params, so
        # every replan step in the iteration shares one snapshot.
        self._behavior_tparams = self._tparams
        # New data batch: the cached behavior posterior (keyed to the old batch) is
        # stale. Recomputed on this iteration's first `ppo_update`.
        self._behavior_musd = None
        traj, trace = self._sampler_for(True)(self._tparams, cond1, cond2, cond2_mask, rng)
        return traj, trace

    # ------------------------------------------------------------------ #
    def act(self, cond1, cond2, cond2_mask, *, rng, deterministic: bool = False):
        """Sample a trajectory for *evaluation* only (no trace, no training).

        ``deterministic=True`` runs the standard (non-RL) sampler over the
        trainable late steps -- the trajectory the fine-tuned policy would emit at
        deployment, with no exploration-floor inflation. ``deterministic=False``
        draws from the same stochastic policy PPO optimizes. Both are reproducible
        for a fixed ``rng``.
        """
        traj, _ = self._sampler_for(not deterministic)(
            self._tparams, cond1, cond2, cond2_mask, rng)
        return traj

    # ------------------------------------------------------------------ #
    def log_probs(self, params, cond1, cond2, cond2_mask, trace):
        """Per-step transition log-probs ``[K, B]`` under a given params pytree.

        The K late steps fold into one fused, jitted denoiser call over a K*B
        batch (see :meth:`_ensure_jitted`).
        """
        self._ensure_jitted()
        return self._jit_log_probs(
            params, cond1, cond2, cond2_mask,
            trace["x_t"], trace["x_prev"], trace["timestep"], trace.get("x_sc"),
        )

    # ------------------------------------------------------------------ #
    def _expert_loss(self, params, expert, alive_b, time_weight=None):
        """Ordinary diffusion loss of the trainable denoiser on expert targets.

        ``expert`` is ``(target_btd, cond1, cond2, cond2_mask, rng)`` -- the
        normalized expert (log) trajectory and its frozen condition. Anchors the
        trainable copy to the demonstration (DPPO + expert anchor). Averaged over
        the *alive* samples only, so padded post-terminal replans do not tug the
        anchor toward a stale expert target.

        ``time_weight`` is an optional ``[T]`` weight over the model horizon, used
        to restrict the anchor to the window the env will actually execute (see
        ``train_safety --anchor_window``). It reweights the LOSS only: the
        denoiser still predicts, and is still noised over, the whole horizon."""
        target_btd, ec1, ec2, emask, erng = expert
        fn = self._trainable_fn(params)
        x0 = jnp.transpose(target_btd, (0, 2, 1))  # [B,T,D] -> [B,D,T]
        lpe = self.diffusion.loss_per_example(x0, ec1, ec2, emask, rng=erng, denoise_fn=fn,
                                              time_weight=time_weight)  # [B]
        return jnp.sum(lpe * alive_b) / jnp.maximum(jnp.sum(alive_b), 1.0)

    def base_anchor_loss(self, params, target_btd, cond1, cond2, cond2_mask,
                         alive_b, rng, time_weight=None, t_max=None):
        """Do-no-harm tether to the FROZEN BASE policy (not the human log).

        Gaussian KL under the shared posterior std between the trainable and the
        frozen base denoiser's one-step posterior means, at states ``x_t`` formed
        by noising ``target_btd`` at a single shared ``(t, eps)`` draw. Unlike
        ``_expert_loss`` -- whose target is the human log, which the pretrained
        policy already departs from *while still succeeding*, so pulling toward it
        perturbs a working policy -- this term's minimizer is the base policy
        itself: it is exactly zero in value AND gradient when the trainable copy
        equals the base, so it penalizes only *movement away from the pretrained
        behavior* and contributes no force where the policy already matches base.
        That makes it the do-no-harm leash on the successes the safety term is not
        trying to change. Averaged over the alive samples only.

        ``rng`` MUST be split once per gradient step by the caller and passed in
        (never a literal constant): `q_sample` turns it into both ``t`` and the
        forward noise, so a frozen key descends one Monte-Carlo slice of the KL
        rather than the KL -- the same frozen-rng bug that made the 09-02 anchor
        control the worst arm of six.

        ``time_weight`` is an optional ``[T]`` weight over the model horizon, so
        the tether can be charged on the executed window only (``train_safety
        --anchor_window``). Both denoiser passes still run the whole horizon; only
        the reduction is windowed.

        ``t_max`` (exclusive) restricts the draw to ``t ~ U[0, t_max)``. ``None``
        is the whole schedule, as every earlier run. The preference path passes
        ``k_trainable``: at ``t >= k_trainable`` the deployed sampler runs the
        frozen base, so a tether charged there protects nothing that is executed.
        """
        x0 = jnp.transpose(target_btd, (0, 2, 1))          # [B,T,D] -> [B,D,T]
        kt, ke = jax.random.split(rng)
        t = jax.random.randint(kt, (x0.shape[0],), 0,
                               self.diffusion.timesteps if t_max is None else int(t_max),
                               dtype=jnp.int32)
        x_t = self.diffusion.q_sample(x0, t, jax.random.normal(ke, x0.shape))
        fn = self._trainable_fn(params)
        mean_tr, var, _ = self.diffusion.p_mean_variance(
            x_t, t, cond1, cond2, cond2_mask, denoise_fn=fn)
        mean_base, _, _ = self.diffusion.p_mean_variance(
            x_t, t, cond1, cond2, cond2_mask, denoise_fn=self.frozen_denoise)
        mean_base = jax.lax.stop_gradient(mean_base)
        std = jnp.maximum(jnp.sqrt(var), self.sigma_logprob_floor)
        sq = 0.5 * ((mean_tr - mean_base) / std) ** 2                   # [B, D, T]
        if time_weight is None:
            per_b = jnp.mean(sq, axis=(1, 2))                           # [B]
        else:
            # Weighted mean over time (divided by the weight sum, not by T) so a
            # window mask does not double as a weight cut on the term -- same
            # convention as `GaussianDiffusion._compute_loss_per_example`.
            w = jnp.broadcast_to(jnp.asarray(time_weight, sq.dtype), sq.shape[-1:])
            per_b = (jnp.sum(sq * w, axis=(1, 2))
                     / (jnp.maximum(jnp.sum(w), 1e-8) * sq.shape[1]))
        return jnp.sum(per_b * alive_b) / jnp.maximum(jnp.sum(alive_b), 1.0)

    def late_denoise_loss(self, params, x0_bct, cond1, cond2, cond2_mask, w_b, rng,
                          step_w_bt=None):
        """x0-prediction MSE of the trainable copy at ``t ~ U[0, k_trainable)`` only.

        The ordinary diffusion loss draws t over the whole schedule, but the
        trainable copy only ever runs the last ``k_trainable`` reverse steps at
        sampling time -- a target taught at t=60 is taught to a step the frozen
        base drives. This is the loss for a TARGET that should come out of those
        low-noise steps: a verified repair (``counterfactual --mode repair``),
        whose coarse manoeuvre is already set by the frozen prefix. ``w_b`` weights
        rows (0 = padding); ``step_w_bt`` weights model STEPS, since a repair target
        is only recoverable where the world time base covered it (see
        ``branching.model_step_valid``). ``rng`` must be split per gradient step,
        never fixed.
        """
        kt, ke = jax.random.split(rng)
        t = jax.random.randint(kt, (x0_bct.shape[0],), 0, int(self.k_trainable), dtype=jnp.int32)
        x_t = self.diffusion.q_sample(x0_bct, t, jax.random.normal(ke, x0_bct.shape))
        _, _, x0_pred = self.diffusion.p_mean_variance(
            x_t, t, cond1, cond2, cond2_mask, denoise_fn=self._trainable_fn(params))
        sq = (x0_pred - x0_bct) ** 2                                    # [B, C, T]
        if step_w_bt is None:
            per_b = jnp.mean(sq, axis=(1, 2))
        else:
            sw = jnp.asarray(step_w_bt, sq.dtype)[:, None, :]            # [B, 1, T]
            per_b = (jnp.sum(sq * sw, axis=(1, 2))
                     / (jnp.maximum(jnp.sum(sw, axis=(1, 2)), 1e-6) * sq.shape[1]))
        w = jnp.asarray(w_b, per_b.dtype)
        return jnp.sum(per_b * w) / jnp.maximum(jnp.sum(w), 1.0)

    def _eps_from_pred(self, x_t, t, pred):
        """The noise a denoiser output implies, for every ``predict_type``.

        The base checkpoint predicts ``v``; ``v = sqrt(ab) eps - sqrt(1-ab) x0``
        and ``x_t = sqrt(ab) x0 + sqrt(1-ab) eps`` give
        ``eps = sqrt(ab) v + sqrt(1-ab) x_t``, exactly, with no clip."""
        d = self.diffusion
        sqrt_ab = d._extract(d.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = d._extract(d.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        if d.predict_type == "eps":
            return pred
        if d.predict_type == "v":
            return sqrt_ab * pred + sqrt_1mab * x_t
        if d.predict_type == "mu":
            return (x_t - sqrt_ab * pred) / sqrt_1mab
        raise ValueError(f"Unsupported predict_type: {d.predict_type}")

    def pref_denoise_losses(self, params, good_bct, bad_bct, cond1, cond2, cond2_mask,
                            step_w_bt, rng, *, t_min: int = 0, t_max: int | None = None):
        """Per-pair eps-prediction MSE of the trainable copy AND the frozen base.

        Diffusion-DPO's four denoising losses for a preference pair
        ``(good, bad)`` under one condition. Each pair draws ONE ``t ~ U[t_min,
        t_max)`` and ONE ``eps``, shared by both trajectories and both models, so
        the comparison is between trajectories and models, not between noise
        draws. ``t_max=None`` is ``k_trainable``: the steps the trainable copy
        drives at deploy time (a preference taught at t >= k_trainable is taught
        to a step the frozen base runs). ``step_w_bt`` weights model steps -- the
        covered-steps mask, optionally scaled up where the pair differs -- and
        the reduction is the weighted mean over steps and channels.

        Returns ``(model_good, model_bad, ref_good, ref_bad)``, each ``[B]``. The
        reference side is stop-gradient'd; it is the frozen base module, whose
        params are not in ``params`` anyway."""
        d = self.diffusion
        if d.self_cond or d.distortion_enabled():
            raise ValueError("pref_denoise_losses: self-conditioning / distortion "
                             "checkpoints are not supported (their loss needs key_aug)")
        B = good_bct.shape[0]
        kt, ke = jax.random.split(rng)
        hi = int(self.k_trainable) if t_max is None else int(t_max)
        t = jax.random.randint(kt, (B,), int(t_min), hi, dtype=jnp.int32)
        eps = jax.random.normal(ke, good_bct.shape)
        # Good and bad stacked into one 2B batch: one forward per model.
        x0 = jnp.concatenate([good_bct, bad_bct], axis=0)
        t2 = jnp.concatenate([t, t], axis=0)
        eps2 = jnp.concatenate([eps, eps], axis=0)
        c1, c2, cm = self._tile_leading(cond1, cond2, cond2_mask, 2)
        x_t = d.q_sample(x0, t2, eps2)
        sw = jnp.asarray(step_w_bt, x0.dtype)
        sw = jnp.concatenate([sw, sw], axis=0)[:, None, :]                # [2B, 1, T]
        den = jnp.maximum(jnp.sum(sw, axis=(1, 2)), 1e-6) * x0.shape[1]

        def per_b(fn):
            pred = d._apply_denoiser(fn, x_t, t2, c1, c2, cm, None)
            eps_hat = self._eps_from_pred(x_t, t2, pred)
            return jnp.sum((eps_hat - eps2) ** 2 * sw, axis=(1, 2)) / den   # [2B]

        lm = per_b(self._trainable_fn(params))
        lr = jax.lax.stop_gradient(per_b(self.frozen_denoise))
        return lm[:B], lm[B:], lr[:B], lr[B:]

    def pref_loss(self, params, good_bct, bad_bct, cond1, cond2, cond2_mask, step_w_bt,
                  rng, *, beta: float, t_min: int = 0, t_max: int | None = None):
        """Diffusion-DPO: ``mean softplus(-z)``,
        ``z = beta * [(l_model_bad - l_ref_bad) - (l_model_good - l_ref_good)]``.

        ``z > 0`` iff, relative to the frozen base, the model got better at
        denoising the good trajectory than the bad one. At initialization the
        model IS the base, so ``z = 0`` and the loss is ``log 2``. Returns
        ``(loss, aux)`` for ``value_and_grad(has_aux=True)``."""
        mg, mb, rg, rb = self.pref_denoise_losses(
            params, good_bct, bad_bct, cond1, cond2, cond2_mask, step_w_bt, rng,
            t_min=t_min, t_max=t_max)
        z = float(beta) * ((mb - rb) - (mg - rg))
        loss = jnp.mean(jax.nn.softplus(-z))
        aux = {"good_model_loss": jnp.mean(mg), "good_ref_loss": jnp.mean(rg),
               "bad_model_loss": jnp.mean(mb), "bad_ref_loss": jnp.mean(rb),
               "logit_z": jnp.mean(z), "z_abs": jnp.mean(jnp.abs(z)),
               "accuracy": jnp.mean((z > 0).astype(jnp.float32)),
               # The base's own preference before any training: > 0 means the
               # base already denoises the edit worse than its own draw.
               "ref_gap": jnp.mean(rg - rb),
               # Largest model-vs-ref gap on either side: ~float noise at init.
               "init_gap": jnp.max(jnp.abs(jnp.concatenate([mg - rg, mb - rb])))}
        return loss, aux

    @staticmethod
    def _wmean_kb(x_kb, w_b):
        """Mean of ``[K, B]`` over the alive samples (weight ``[B]``, broadcast over K)."""
        w = w_b[None, :]
        return jnp.sum(x_kb * w) / jnp.maximum(jnp.sum(w) * x_kb.shape[0], 1.0)

    def _ppo_loss(self, params, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b, alive_b,
                  expert=None, expert_weight=0.0):
        new_lp = self.log_probs(params, cond1, cond2, cond2_mask, trace)  # [K, B]
        log_ratio = new_lp - old_log_probs_kb                             # [K, B]
        ratio = jnp.exp(log_ratio)                                        # [K, B]
        adv = adv_b[None, :]                                              # [1, B]
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv
        surrogate = jnp.minimum(unclipped, clipped)
        # All reductions are over the *alive* samples: post-terminal padded replans
        # carry adv==0 but still have a ratio, so an unweighted mean would dilute
        # the PG term and mis-report KL/clip_frac. `_wmean_kb` divides by the alive
        # count, not the padded n*B.
        # `_reduce_logdensity` averages the per-element log density over
        # channels*prefix_len, which keeps `ratio` near 1 (intended) but shrinks
        # the PG gradient by that same factor. `pg_scale` restores it so the PG
        # term is not swamped by the expert anchor; the clip still acts on the
        # un-scaled ratio, so the trust region is unchanged.
        pg_raw = -self._wmean_kb(surrogate, alive_b)
        pg_loss = self.pg_scale * pg_raw
        # Schulman's nonnegative KL estimator `ratio - 1 - log(ratio)` -- always
        # >= 0 and lower-variance than the raw `old - new` difference. Reported for
        # diagnostics; the *gate* uses the exact Gaussian KL in `ppo_update`.
        approx_kl = self._wmean_kb(ratio - 1.0 - log_ratio, alive_b)
        # `ratio` above is on the *averaged* log density (what the clip sees). The
        # per-transition ratio compounds over all `reduce_n` coordinates, so report
        # the trust region as the policy actually moves it: log_ratio * reduce_n.
        lr_t = log_ratio * self._reduce_n
        ratio_t = jnp.exp(lr_t)
        approx_kl_transition = self._wmean_kb(ratio_t - 1.0 - lr_t, alive_b)
        # `expert` is a Python-level None-or-tuple at trace time, so a plain branch
        # is correct (and keeps the expert forward out of the graph when unused).
        if expert is not None:
            expert_loss = self._expert_loss(params, expert, alive_b)
        else:
            expert_loss = jnp.asarray(0.0, dtype=jnp.float32)
        expert_term = expert_weight * expert_loss
        loss = pg_loss + expert_term
        clip_frac = self._wmean_kb((jnp.abs(ratio - 1.0) > self.ppo_clip).astype(jnp.float32), alive_b)
        return loss, {"loss": loss, "pg_loss": pg_loss, "pg_raw": pg_raw,
                      "expert_loss": expert_loss, "expert_term": expert_term,
                      "approx_kl": approx_kl, "approx_kl_transition": approx_kl_transition,
                      "mean_ratio": self._wmean_kb(ratio, alive_b),
                      "mean_ratio_transition": self._wmean_kb(ratio_t, alive_b),
                      "clip_frac": clip_frac}

    # ------------------------------------------------------------------ #
    def _posterior_means_std(self, params, cond1, cond2, cond2_mask, trace):
        """Per-transition posterior mean and std, both ``[K, B, C, T]``.

        The std is the (param-independent) schedule posterior std; the mean is the
        denoiser-dependent posterior mean. Used by the exact Gaussian KL gate. The
        K late steps fold into one fused, jitted denoiser call over a K*B batch
        (see :meth:`_ensure_jitted`).
        """
        self._ensure_jitted()
        return self._jit_means_std(
            params, cond1, cond2, cond2_mask, trace["x_t"], trace["timestep"],
            trace.get("x_sc"),
        )

    def _transition_kl(self, mu_old, mu_new, std, alive_b):
        """Exact KL(old || candidate) of the reverse-diffusion policy, in nats.

        Both policies are diagonal Gaussians with the *same* (param-independent)
        std, so the KL is ``0.5 * sum((mu_old - mu_new)/sigma)^2`` over the executed
        prefix coordinates -- no ``exp`` of a large log-ratio, so it cannot overflow
        or go negative. Summed over the ``K`` late steps (the trajectory KL
        factorizes over the reverse chain) and averaged over the alive samples.
        """
        std_lp = jnp.maximum(std, self.sigma_logprob_floor)          # [K, B, C, T]
        per_elt = 0.5 * ((mu_new - mu_old) / std_lp) ** 2            # [K, B, C, T]
        t_dim = per_elt.shape[3]
        if self.prefix_len is None:
            kl_kb = jnp.sum(per_elt, axis=(2, 3))                    # [K, B]
        else:
            mt = (jnp.arange(t_dim) < int(self.prefix_len)).astype(per_elt.dtype)
            kl_kb = jnp.sum(per_elt * mt[None, None, None, :], axis=(2, 3))
        kl_b = jnp.sum(kl_kb, axis=0)                                # [B], trajectory KL
        return jnp.sum(kl_b * alive_b) / jnp.maximum(jnp.sum(alive_b), 1.0)

    def ppo_update(self, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b, alive_b,
                   *, expert=None, expert_weight=0.0, target_kl=None):
        """One clipped-PPO gradient step on the trainable denoiser only.

        With ``expert`` given, adds ``expert_weight`` * (expert diffusion loss) --
        the DPPO + expert-anchor variant.

        The trust-region gate is on the *candidate*: grads are formed at the current
        params, a candidate params + optimizer state are constructed, and the exact
        Gaussian KL of the candidate policy against the behavior snapshot
        (``_behavior_tparams``, frozen in ``collect``) is measured. If it exceeds
        ``target_kl`` the candidate is discarded and ``metrics["applied"] == 0.0``,
        so the step that would cross the trust region is never committed -- unlike a
        pre-update KL, which is ~0 at epoch 0 and only rejects the *next* step.
        ``metrics["kl_gauss"]`` is the realized post-update KL (0 when discarded).
        """
        if self.pg_scale is None or self._reduce_n is None:
            # trace["x_t"] is [K, B, C, T]; the log density was averaged over
            # C and the executed prefix (or all of T when prefix_len is None).
            _, _, c, t_dim = trace["x_t"].shape
            pl = t_dim if self.prefix_len is None else int(self.prefix_len)
            self._reduce_n = float(int(c) * pl)
            if self.pg_scale is None:
                self.pg_scale = self._reduce_n
        if self._jit_ppo_grad is None:
            # static_argnums=9 -> expert_weight (a Python float) is baked in; expert
            # (arg 8) is a None-or-tuple pytree, so toggling the anchor recompiles
            # once but the constant-shape steady state never does.
            self._jit_ppo_grad = jax.jit(
                jax.value_and_grad(self._ppo_loss, has_aux=True), static_argnums=(9,))
        (loss, metrics), grads = self._jit_ppo_grad(
            self._tparams, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b, alive_b,
            expert, expert_weight,
        )
        metrics = dict(metrics)
        # Build the candidate (params + optimizer state) but do not commit yet.
        cand_updates, cand_opt_state = self.tx.update(grads, self._opt_state, self._tparams)
        cand_params = optax.apply_updates(self._tparams, cand_updates)
        # Exact KL of the candidate against the behavior policy (the data-generating
        # snapshot), not against the current params. The behavior mean/std are
        # constant across this iteration's epochs (frozen params + fixed data), so
        # compute them once and reuse; only the candidate mean moves per epoch.
        if self._behavior_musd is None:
            self._behavior_musd = self._posterior_means_std(
                self._behavior_tparams, cond1, cond2, cond2_mask, trace)
        mu_old, std = self._behavior_musd
        mu_cand, _ = self._posterior_means_std(cand_params, cond1, cond2, cond2_mask, trace)
        kl_gauss = self._transition_kl(mu_old, mu_cand, std, alive_b)
        applied = target_kl is None or float(np.asarray(kl_gauss)) <= float(target_kl)
        if applied:
            self._opt_state = cand_opt_state
            self._tparams = cand_params
        metrics["grad_global_norm"] = optax.global_norm(grads)
        metrics["kl_gauss"] = kl_gauss if applied else jnp.asarray(0.0, dtype=jnp.float32)
        metrics["kl_gauss_candidate"] = kl_gauss
        metrics["applied"] = jnp.asarray(1.0 if applied else 0.0, dtype=jnp.float32)
        return metrics

    # ------------------------------------------------------------------ #
    def expert_update(self, expert, alive_b, *, measure_kl_vs_behavior: bool = False):
        """One gradient step on the trainable denoiser against the expert loss only.

        The behavior-cloning half of the expert:DPPO *step ratio*. This is a
        different lever from ``ppo_update(expert_weight=...)``, which mixes the two
        objectives inside one gradient: there the balance is a coefficient on
        incommensurable loss scales (the calibration that went wrong at 0.1, where
        the anchor was ~1000x the PG term), here it is a count of steps, which is
        directly interpretable and does not need rescaling when the PG magnitude
        moves.

        Deliberately **not** gated by the PPO trust region. The trust region exists
        to keep the importance ratio valid for the data collected under the
        behavior policy; the expert loss uses no importance ratio, so gating it
        would only couple it to the sigma/target_kl bind. The trainer therefore runs
        the expert steps *after* the PPO epochs of an iteration, so no PPO epoch
        ever computes a ratio against a policy the expert steps have since moved.

        ``measure_kl_vs_behavior`` additionally reports the exact Gaussian KL of the
        post-step policy against the collection-time snapshot, i.e. how much of this
        iteration's policy motion the expert term is responsible for -- the number
        the ratio sweep is actually read by. It costs one extra denoiser pass, so it
        is off unless asked for (the trainer asks on the last expert step only).
        """
        if self._jit_expert_grad is None:
            # argnums belongs to value_and_grad (differentiate w.r.t. _tparams,
            # the first arg), NOT to jit -- jax.jit has no such kwarg and raises
            # TypeError on the first expert step. That crash is why every arm of
            # the 20260828 ratio sweep with expert_steps>0 died ~90s in while the
            # x0 control ran fine: this line is only reached when expert_steps>0,
            # so no run before that sweep had ever executed it.
            self._jit_expert_grad = jax.jit(
                jax.value_and_grad(self._expert_loss, argnums=0))
        loss, grads = self._jit_expert_grad(self._tparams, expert, alive_b)
        updates, self._opt_state_expert = self.tx_expert.update(
            grads, self._opt_state_expert, self._tparams)
        self._tparams = optax.apply_updates(self._tparams, updates)
        metrics = {
            "expert_loss": float(np.asarray(loss)),
            "expert_grad_norm": float(np.asarray(optax.global_norm(grads))),
        }
        if (measure_kl_vs_behavior and self._behavior_musd is not None
                and self._expert_kl_cond is not None):
            cond1, cond2, cond2_mask = self._expert_kl_cond
            mu_old, std = self._behavior_musd
            mu_new, _ = self._posterior_means_std(
                self._tparams, cond1, cond2, cond2_mask, self._expert_kl_trace)
            metrics["kl_gauss_expert"] = float(
                np.asarray(self._transition_kl(mu_old, mu_new, std, alive_b)))
        return metrics

    def safety_update(self, loss_fn, *, lr_scale: float = 1.0):
        """One gradient step on the trainable denoiser against an arbitrary
        differentiable loss over the *sampled* trajectory.

        This is the analytic-gradient counterpart to :meth:`ppo_update`. PPO needs
        a score-function estimator because the simulator reward is a black box;
        the geometric safety loss is a differentiable function of the trajectory,
        so its gradient can be taken directly through the sampler and is not an
        estimate at all. ``loss_fn(params) -> (scalar, aux)`` is built by the
        caller because it needs the environment's postprocess and roadgraph, which
        the actor deliberately knows nothing about.

        Like :meth:`expert_update`, this is **not** gated by the PPO trust region:
        that region exists to keep an importance ratio valid for data drawn under
        the behaviour policy, and there is no ratio here. It reuses the expert
        optimizer state so that the anchor and the safety term -- which are
        stepped together as one objective -- share one Adam moment estimate rather
        than fighting over two.
        """
        grads, loss, aux = self.safety_grads(loss_fn)
        out = {"loss": float(np.asarray(loss))}
        out.update({k: float(np.asarray(v)) for k, v in aux.items()})
        out.update(self.apply_safety_grads(grads, lr_scale=lr_scale))
        return out

    def safety_grads(self, loss_fn):
        """``loss_fn``'s gradient w.r.t. the trainable denoiser, *unapplied*.

        Split out of :meth:`safety_update` so a caller can average the gradient
        over several replan contexts and take ONE optimizer step instead of one
        step per context. That distinction matters more than it looks: an Adam
        update is ``lr * m/(sqrt(v)+eps)``, which is O(lr) per parameter *whatever
        the gradient norm is*, so the number of steps -- not the size of the
        signal -- is what sets how far the policy travels from the pretrained
        weights. Stepping once per context therefore both shortens the averaging
        window and multiplies the distance walked, and with a weak signal the
        walk is mostly noise. Returns ``(grads, loss, aux)``.
        """
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(self._tparams)
        return grads, loss, aux

    def apply_safety_grads(self, grads, *, lr_scale: float = 1.0):
        """Take one optimizer step on an already-averaged safety gradient.

        Note that ``tx_expert``'s ``clip_by_global_norm`` sits *before* Adam and
        so barely constrains the update: Adam divides by its own running RMS, and
        a uniform pre-scaling of the gradient cancels once the moments have warmed
        up. The clip bounds the gradient, not the distance travelled. Whatever
        limits the latter has to be the step count or the learning rate.
        """
        updates, self._opt_state_expert = self.tx_expert.update(
            grads, self._opt_state_expert, self._tparams)
        if lr_scale != 1.0:
            updates = jax.tree_util.tree_map(lambda g: g * lr_scale, updates)
        self._tparams = optax.apply_updates(self._tparams, updates)
        return {"grad_norm": float(np.asarray(optax.global_norm(grads)))}

    def sample_world_for(self, params, cond1, cond2, cond2_mask, rng, *, rl_mode=True):
        """Differentiable sample under an explicit params pytree, ``[B, C, T]``.

        Exposes the traced sampler so a caller can build a loss that backpropagates
        into the denoiser. ``collect``/``act`` bind ``self._tparams``; this does
        not, which is what lets ``jax.grad`` reach the parameters.
        """
        return self._sampler_for(bool(rl_mode))(params, cond1, cond2, cond2_mask, rng)[0]

    def _prefix_samplers(self):
        """Jitted ``(with_prefix, late_only, with_trace)`` for the current step count.

        ``with_prefix(params, c1, c2, m, rng) -> (x, (x_{k-1}, key_late))`` is the
        rl_mode sampler that also hands back the frozen prefix's output;
        ``late_only(params, c1, c2, m, x_{k-1}, key_late) -> x`` replays just the
        ``k_trainable`` late steps from it. The prefix runs the frozen base
        denoiser, so late_only reproduces with_prefix's trajectory for the same
        params -- and is what a gradient pass differentiates.

        ``with_trace`` is ``with_prefix`` plus the late chain's own
        ``(x_t, timestep)`` records, which ``sample_with_trace`` already computes
        and the safety path otherwise throws away. It is a separate compiled
        program only because returning the extra arrays changes the output
        signature; the sampler math is identical, so a collection rollout can
        harvest genuine diffusion states for
        :mod:`rl.diffusion_ft.transition_buffer` at no extra denoiser cost."""
        key = ("prefix", self.num_sample_steps)
        cached = self._jit_samplers.get(key)
        if cached is not None:
            return cached
        diffusion = self.diffusion
        tgdef, tnonparam = self._tgdef, self._tnonparam
        kw = dict(k_trainable=self.k_trainable, rl_mode=True,
                  sigma_sample_floor=self.sigma_sample_floor,
                  sigma_logprob_floor=self.sigma_logprob_floor,
                  prefix_len=self.prefix_len,
                  num_sample_steps=self.num_sample_steps)
        in_ch, horizon = diffusion.denoise_fn.in_ch, diffusion.denoise_fn.horizon

        def with_prefix(params, cond1, cond2, cond2_mask, rng):
            denoise = nnx.merge(tgdef, params, tnonparam)
            x, _, pre = diffusion.sample_with_trace(
                (cond1.shape[0], in_ch, horizon), cond1, cond2, cond2_mask, rng=rng,
                trainable_denoise_fn=denoise, return_prefix=True, **kw)
            return x, pre

        def late_only(params, cond1, cond2, cond2_mask, x_prefix, key_late):
            denoise = nnx.merge(tgdef, params, tnonparam)
            x, _ = diffusion.sample_with_trace(
                (cond1.shape[0], in_ch, horizon), cond1, cond2, cond2_mask,
                rng=key_late, trainable_denoise_fn=denoise,
                x_prefix=x_prefix, key_late=key_late, **kw)
            return x

        def with_trace(params, cond1, cond2, cond2_mask, rng):
            denoise = nnx.merge(tgdef, params, tnonparam)
            x, trace, pre = diffusion.sample_with_trace(
                (cond1.shape[0], in_ch, horizon), cond1, cond2, cond2_mask, rng=rng,
                trainable_denoise_fn=denoise, return_prefix=True, **kw)
            # Only the two arrays a replayable transition needs. `mean`/`std`/
            # `log_prob` are DPPO's and are dropped inside the jit, so they are
            # never transferred off the device.
            return x, pre, trace["x_t"], trace["timestep"]

        pair = (jax.jit(with_prefix), jax.jit(late_only), jax.jit(with_trace))
        self._jit_samplers[key] = pair
        return pair

    def sample_world_with_prefix(self, params, cond1, cond2, cond2_mask, rng):
        """:meth:`sample_world_for` (rl_mode) plus ``(x_{k-1}, key_late)``."""
        return self._prefix_samplers()[0](params, cond1, cond2, cond2_mask, rng)

    def sample_world_with_prefix_trace(self, params, cond1, cond2, cond2_mask, rng):
        """:meth:`sample_world_with_prefix` plus the late chain's transitions.

        Returns ``(x, (x_prefix, key_late), x_t_kbct, t_kb)`` where ``x_t_kbct[j]``
        is the sampler's own state entering late step ``j`` and ``t_kb[j]`` its
        diffusion timestep, running ``k_trainable-1 .. 0``. These are the states
        the deployed sampler actually passes through, which is what distinguishes
        a stored transition from ``q_sample(x0_expert, t)``: same timestep,
        different distribution.
        """
        return self._prefix_samplers()[2](params, cond1, cond2, cond2_mask, rng)

    def sample_late_for(self, params, cond1, cond2, cond2_mask, x_prefix, key_late):
        """Differentiable late chain from a stored prefix, ``[B, C, T]``."""
        return self._prefix_samplers()[1](params, cond1, cond2, cond2_mask,
                                          x_prefix, key_late)

    # ------------------------------------------------------------------ #
    def _unroll_for(self, k_steps: int, grad_steps: int, eta: float):
        """Jitted teacher-forced unroll, one compiled program per (K, G, eta)."""
        key = (int(k_steps), int(grad_steps), float(eta))
        cached = self._jit_unrolls.get(key)
        if cached is not None:
            return cached
        diffusion = self.diffusion
        tgdef, tnonparam = self._tgdef, self._tnonparam

        def merge(p):
            return nnx.merge(tgdef, p, tnonparam)

        def run(params, cond1, cond2, cond2_mask, x0_bct, t_grid_kb, rng):
            return teacher_forced_unroll(
                diffusion, merge, params, x0_bct=x0_bct, t_grid_kb=t_grid_kb,
                cond1=cond1, cond2=cond2, cond2_mask=cond2_mask,
                grad_steps=key[1], eta=key[2], rng=rng)

        jitted = jax.jit(run)
        self._jit_unrolls[key] = jitted
        return jitted

    def unrolled_sample_for(self, params, cond1, cond2, cond2_mask, x0_bct,
                            t_grid_kb, rng, *, grad_steps: int = 1, eta: float = 0.0):
        """Teacher-forced unrolled candidate under explicit params, ``[B, C, T]``.

        The cheap counterpart to :meth:`sample_world_for`: instead of denoising
        from pure noise through the whole chain, corrupt the expert trajectory to
        ``t_grid_kb[0]`` and unroll ``K = t_grid_kb.shape[0]`` steps back down to
        ``t_grid_kb[-1]``. Differentiable in ``params`` through the last
        ``grad_steps`` evaluations. See :mod:`rl.diffusion_ft.unrolled_prediction`.

        The caller is responsible for keeping ``t_grid_kb`` inside the trainable
        window (``t < k_trainable``): every step here runs the *trainable* copy,
        and the sampler only ever invokes it at ``t <= k_trainable - 1``.
        """
        k = int(t_grid_kb.shape[0])
        return self._unroll_for(k, grad_steps, eta)(
            params, cond1, cond2, cond2_mask, x0_bct, t_grid_kb, rng)

    def _clean_from_xt_for(self, k_steps: int, grad_steps: int, eta: float):
        """Jitted clean-prediction-from-``x_t``, one program per (K, G, eta)."""
        key = ("from_xt", int(k_steps), int(grad_steps), float(eta))
        cached = self._jit_unrolls.get(key)
        if cached is not None:
            return cached
        diffusion = self.diffusion
        tgdef, tnonparam = self._tgdef, self._tnonparam

        def merge(p):
            return nnx.merge(tgdef, p, tnonparam)

        def run(params, cond1, cond2, cond2_mask, x_t, t_grid_kb, rng):
            return unrolled_clean_prediction(
                diffusion, merge, params, x_t=x_t, t_grid_kb=t_grid_kb,
                cond1=cond1, cond2=cond2, cond2_mask=cond2_mask,
                grad_steps=key[2], eta=key[3], rng=rng)

        jitted = jax.jit(run)
        self._jit_unrolls[key] = jitted
        return jitted

    def clean_prediction_from_xt(self, params, cond1, cond2, cond2_mask, x_t,
                                 t_grid_kb, rng, *, grad_steps: int = 1,
                                 eta: float = 0.0):
        """Clean prediction ``x0_hat`` reached from an EXPLICIT ``x_t``, ``[B, C, T]``.

        :meth:`unrolled_sample_for` without the teacher forcing: the caller
        supplies the noised state instead of having it built as
        ``q_sample(x0_expert, t)``. That is the entry point a stored sampler
        transition uses -- the state is the one the policy's own chain passed
        through, so the diffusion-state mismatch
        :mod:`rl.diffusion_ft.unrolled_prediction` documents does not apply.

        ``K = t_grid_kb.shape[0]``: ``K=1`` is the plain one-step clean prediction
        at the stored timestep (no unroll, one network evaluation); ``K>1`` runs
        ``K-1`` eta-DDIM steps down to ``t_grid_kb[-1]`` and returns the clean
        prediction of the final evaluation. ``grad_steps`` bounds how many of the
        final evaluations carry a gradient, so ``grad_steps=1`` differentiates
        exactly one transition however long the unroll is.
        """
        k = int(t_grid_kb.shape[0])
        return self._clean_from_xt_for(k, grad_steps, eta)(
            params, cond1, cond2, cond2_mask, x_t, t_grid_kb, rng)

    def set_expert_kl_probe(self, cond1, cond2, cond2_mask, trace) -> None:
        """Give ``expert_update`` the batch to measure its KL-vs-behavior on.

        Same (cond, trace) the PPO epochs used, so the expert-side and DPPO-side KLs
        are on one scale and their sum is the iteration's total policy move.
        """
        self._expert_kl_cond = (cond1, cond2, cond2_mask)
        self._expert_kl_trace = trace

    # ------------------------------------------------------------------ #
    def frozen_params_unchanged(self, reference_state) -> bool:
        """True iff the frozen base denoiser still equals ``reference_state``.

        Used to assert the update touched only the trainable copy.
        """
        _, cur = nnx.split(self.frozen_denoise)
        leaves_a = jax.tree_util.tree_leaves(cur)
        leaves_b = jax.tree_util.tree_leaves(reference_state)
        return all(bool(jnp.array_equal(a, b)) for a, b in zip(leaves_a, leaves_b))

    def frozen_state(self):
        _, st = nnx.split(self.frozen_denoise)
        return jax.tree_util.tree_map(lambda x: jnp.array(x), st)

    # ------------------------------------------------------------------ #
    def reset_trainable_to_frozen(self) -> None:
        """Restore the trainable denoiser to the pretrained (frozen) weights.

        The trainable copy starts life as a clone of ``frozen_denoise``, so the
        pretrained policy is always recoverable from the frozen module -- no
        separate snapshot needed. Lets one actor instance measure the base policy
        and a fine-tuned checkpoint in the same process without arm order leaking
        one into the other. The optimizer state is re-initialized to match.
        """
        _, tparams, _ = nnx.split(_clone_module(self.frozen_denoise), nnx.Param, ...)
        self._tparams = tparams
        self._opt_state = self.tx.init(self._tparams)
        self._opt_state_expert = self.tx_expert.init(self._tparams)
        self._behavior_tparams = self._tparams
        self._behavior_musd = None

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        return {
            "tparams": self._tparams,
            "opt_state": self._opt_state,
            "opt_state_expert": self._opt_state_expert,
            "rng": self._rng,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._tparams = sd["tparams"]
        self._opt_state = sd["opt_state"]
        # Absent in checkpoints written before the expert step existed; keep the
        # freshly initialized state rather than failing to resume them.
        if "opt_state_expert" in sd:
            self._opt_state_expert = sd["opt_state_expert"]
        self._rng = sd["rng"]
