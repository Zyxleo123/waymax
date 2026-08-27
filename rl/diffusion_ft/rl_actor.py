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
        self._jit_samplers: dict[bool, Any] = {}

    # ------------------------------------------------------------------ #
    def _sampler_for(self, rl_mode: bool):
        """Leak-free jitted sampler (params traced, merged inside), one per mode."""
        key = bool(rl_mode)
        cached = self._jit_samplers.get(key)
        if cached is not None:
            return cached
        diffusion = self.diffusion
        tgdef, tnonparam = self._tgdef, self._tnonparam
        k_trainable, prefix_len = self.k_trainable, self.prefix_len
        sigma_s, sigma_lp = self.sigma_sample_floor, self.sigma_logprob_floor
        num_sample_steps = self.num_sample_steps
        in_ch, horizon = diffusion.denoise_fn.in_ch, diffusion.denoise_fn.horizon

        def sampler(params, cond1, cond2, cond2_mask, rng):
            denoise = nnx.merge(tgdef, params, tnonparam)
            shape = (cond1.shape[0], in_ch, horizon)
            return diffusion.sample_with_trace(
                shape, cond1, cond2, cond2_mask, rng=rng,
                k_trainable=k_trainable, trainable_denoise_fn=denoise,
                rl_mode=key, sigma_sample_floor=sigma_s,
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

        def batched_log_probs(params, cond1, cond2, cond2_mask, x_t, x_prev, timestep):
            fn = nnx.merge(tgdef, params, tnonparam)
            k, b = x_t.shape[0], x_t.shape[1]
            xt = x_t.reshape((k * b,) + x_t.shape[2:])
            xp = x_prev.reshape((k * b,) + x_prev.shape[2:])
            tt = timestep.reshape((k * b,))
            c1, c2, cm = tile(cond1, cond2, cond2_mask, k)
            lp = diffusion.transition_log_prob(
                xt, xp, tt, c1, c2, cm,
                sigma_floor=sigma_lp, prefix_len=prefix_len, denoise_fn=fn)
            return lp.reshape((k, b))

        def batched_means_std(params, cond1, cond2, cond2_mask, x_t, timestep):
            fn = nnx.merge(tgdef, params, tnonparam)
            k, b = x_t.shape[0], x_t.shape[1]
            xt = x_t.reshape((k * b,) + x_t.shape[2:])
            tt = timestep.reshape((k * b,))
            c1, c2, cm = tile(cond1, cond2, cond2_mask, k)
            mean, std, _ = diffusion._posterior_mean_std(xt, tt, c1, c2, cm, denoise_fn=fn)
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
            trace["x_t"], trace["x_prev"], trace["timestep"],
        )

    # ------------------------------------------------------------------ #
    def _expert_loss(self, params, expert, alive_b):
        """Ordinary diffusion loss of the trainable denoiser on expert targets.

        ``expert`` is ``(target_btd, cond1, cond2, cond2_mask, rng)`` -- the
        normalized expert (log) trajectory and its frozen condition. Anchors the
        trainable copy to the demonstration (DPPO + expert anchor). Averaged over
        the *alive* samples only, so padded post-terminal replans do not tug the
        anchor toward a stale expert target."""
        target_btd, ec1, ec2, emask, erng = expert
        fn = self._trainable_fn(params)
        x0 = jnp.transpose(target_btd, (0, 2, 1))  # [B,T,D] -> [B,D,T]
        lpe = self.diffusion.loss_per_example(x0, ec1, ec2, emask, rng=erng, denoise_fn=fn)  # [B]
        return jnp.sum(lpe * alive_b) / jnp.maximum(jnp.sum(alive_b), 1.0)

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
            self._jit_expert_grad = jax.jit(
                jax.value_and_grad(self._expert_loss), argnums=0)
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
