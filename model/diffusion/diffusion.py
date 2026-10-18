from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


def make_beta_schedule(timesteps: int, schedule: str = "cosine") -> jnp.ndarray:
    if schedule == "linear":
        return jnp.linspace(1e-4, 2e-2, timesteps, dtype=jnp.float32)
    if schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = jnp.linspace(0, timesteps, steps, dtype=jnp.float32)
        alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return jnp.clip(betas, 0.0, 0.999).astype(jnp.float32)
    raise ValueError(f"Unknown schedule: {schedule}")


class GaussianDiffusion(nnx.Module):
    """DDPM forward process with DDIM sampling.

    Convention:
        t = 0 : data
        t = T - 1 : most noised latent

        x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * noise

    The network output is interpreted with predict_type ("eps", "mu", or "v").
    Discrete timestep indices are passed to the network unchanged.
    Sampling starts from noise at t = T - 1 and steps down to t = 0.
    """

    def __init__(
        self,
        denoise_fn: nnx.Module,
        timesteps: int = 100,
        beta_schedule: str = "cosine",
        predict_type: str = "v",
        clip_denoised: bool = False,
    ):
        self.denoise_fn = denoise_fn
        self.timesteps = int(timesteps)
        self.predict_type = predict_type
        self.clip_denoised = clip_denoised

        betas = make_beta_schedule(self.timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate(
            [jnp.array([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]],
            axis=0,
        )

        self.betas = nnx.Variable(betas)
        self.alphas = nnx.Variable(alphas)
        self.alphas_cumprod = nnx.Variable(alphas_cumprod)
        self.alphas_cumprod_prev = nnx.Variable(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = nnx.Variable(jnp.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = nnx.Variable(jnp.sqrt(1.0 - alphas_cumprod))

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = jnp.maximum(posterior_var, 1e-20)
        self.posterior_variance = nnx.Variable(posterior_variance)
        self.posterior_log_variance_clipped = nnx.Variable(jnp.log(posterior_variance))

        self.posterior_mean_coef1 = nnx.Variable(
            betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = nnx.Variable(
            (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Sequence[int]) -> jnp.ndarray:
        """Gather schedule values at discrete t and broadcast to [B, 1, 1]."""
        out = a[t]
        return out.reshape((x_shape[0], 1, 1))

    @staticmethod
    def _temp_schedule(sampling_temp: float, temp_mode: str, num_steps: int) -> jnp.ndarray:
        """Precompute per-timestep effective temperature.

        Returns array of shape [num_steps] indexed by diffusion loop step index
        (0 = highest noise timestep, num_steps - 1 = lowest).
        """
        t_idx = jnp.arange(num_steps, dtype=jnp.float32)
        denom = jnp.maximum(num_steps - 1, 1)
        if temp_mode == "early":
            return 1.0 + (sampling_temp - 1.0) * (1.0 - t_idx / denom)
        if temp_mode == "late":
            return 1.0 + (sampling_temp - 1.0) * (t_idx / denom)
        return jnp.full((num_steps,), sampling_temp, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        """Sample x_t from the closed-form Gaussian forward process."""
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / sqrt_ab

    def p_mean_variance(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pred = self.predict(x_t, t, cond)

        if self.predict_type == "eps":
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
        elif self.predict_type == "mu":
            x0_pred = pred
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
            x0_pred = sqrt_ab * x_t - sqrt_1mab * pred
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        if self.clip_denoised:
            x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        mean = (
            self._extract(self.posterior_mean_coef1.value, t, x_t.shape) * x0_pred
            + self._extract(self.posterior_mean_coef2.value, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance.value, t, x_t.shape)
        return mean, var, x0_pred

    # ------------------------------------------------------------------
    # Network prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the denoiser at a discrete diffusion timestep.

        t is the integer index in [0, T). It is not rescaled.
        """
        return self.denoise_fn(x_t, t, cond)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        x0: jnp.ndarray,
        cond: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        x_t = self.q_sample(x0, t, noise=noise)
        pred = self.predict(x_t, t, cond)

        if self.predict_type == "eps":
            return jnp.mean((pred - noise) ** 2)
        if self.predict_type == "mu":
            return jnp.mean((pred - x0) ** 2)
        if self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
            v = sqrt_ab * noise - sqrt_1mab * x0
            return jnp.mean((pred - v) ** 2)
        raise ValueError(f"Unsupported predict_type: {self.predict_type}")

    def _loss_impl(
        self,
        x0: jnp.ndarray,
        cond: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:
        batch = x0.shape[0]
        key_t, key_noise = jax.random.split(rng)
        t = jax.random.randint(key_t, (batch,), 0, self.timesteps, dtype=jnp.int32)
        noise = jax.random.normal(key_noise, x0.shape, dtype=jnp.float32)
        return self._compute_loss(x0, cond, noise, t)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def p_sample(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
        noise: jnp.ndarray,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
    ) -> jnp.ndarray:
        """One DDIM step from t toward t - 1."""
        _, _, x0_pred = self.p_mean_variance(x_t, t, cond)

        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_1mab

        eps_pred = eps_pred * sampling_temp
        x0_pred = (x_t - sqrt_1mab * eps_pred) / sqrt_ab

        post_var = self._extract(self.posterior_variance.value, t, x_t.shape)
        sigma = eta * jnp.sqrt(post_var)

        abar_prev = self._extract(self.alphas_cumprod_prev.value, t, x_t.shape)
        dir_coeff = jnp.sqrt(jnp.maximum(1.0 - abar_prev - sigma ** 2, 0.0))

        nonzero_mask = (t != 0).astype(x_t.dtype).reshape((-1, 1, 1))
        return jnp.sqrt(abar_prev) * x0_pred + dir_coeff * eps_pred + nonzero_mask * sigma * noise

    def _sample_impl(
        self,
        shape: Tuple[int, int, int],
        cond: jnp.ndarray,
        rng: jax.Array,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        batch_size = shape[0]
        key_x, key_steps = jax.random.split(rng)
        x_init = jax.random.normal(key_x, shape, dtype=jnp.float32)
        num_steps = self.timesteps
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, num_steps)

        def body(i: int, carry: tuple[jnp.ndarray, jax.Array]) -> tuple[jnp.ndarray, jax.Array]:
            x_curr, key = carry
            timestep = num_steps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
            eff_temp = temp_sched[i]
            x_next = self.p_sample(
                x_curr,
                t,
                cond,
                noise,
                eta=eta,
                sampling_temp=eff_temp,
            )
            return x_next, key

        x_final, _ = jax.lax.fori_loop(0, num_steps, body, (x_init, key_steps))
        return x_final

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _resample_impl(
        self,
        proposals: jnp.ndarray,
        cond: jnp.ndarray,
        n_timesteps: int,
        rng: jax.Array,
        noise_scale: float = 1.0,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        """Noise proposals to t = n_timesteps - 1, then denoise back to t = 0."""
        batch_size = proposals.shape[0]
        key_q, key_steps = jax.random.split(rng)
        q_noise = jax.random.normal(key_q, proposals.shape, dtype=jnp.float32) * noise_scale
        t0 = jnp.full((batch_size,), n_timesteps - 1, dtype=jnp.int32)
        x = self.q_sample(proposals, t0, noise=q_noise)
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, n_timesteps)

        def body(i: int, carry: tuple[jnp.ndarray, jax.Array]) -> tuple[jnp.ndarray, jax.Array]:
            x_curr, key = carry
            timestep = n_timesteps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, proposals.shape, dtype=jnp.float32)
            eff_temp = temp_sched[i]
            x_next = self.p_sample(
                x_curr,
                t,
                cond,
                noise,
                eta=eta,
                sampling_temp=eff_temp,
            )
            return x_next, key

        x, _ = jax.lax.fori_loop(0, n_timesteps, body, (x, key_steps))
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond: jnp.ndarray,
        *,
        rng: jax.Array,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        return self._sample_impl(
            shape,
            cond,
            rng,
            eta=eta,
            sampling_temp=sampling_temp,
            temp_mode=temp_mode,
        )

    def loss(
        self,
        x0: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        rng: jax.Array,
    ) -> jnp.ndarray:
        return self._loss_impl(x0, cond, rng)

    def resample(
        self,
        proposals: jnp.ndarray,
        cond: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: jax.Array,
        noise_scale: float = 1.0,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        if n_timesteps <= 0 or n_timesteps > self.timesteps:
            raise ValueError(f"n_timesteps must be in [1, {self.timesteps}], got {n_timesteps}")
        return self._resample_impl(
            proposals,
            cond,
            n_timesteps,
            rng,
            noise_scale=noise_scale,
            eta=eta,
            sampling_temp=sampling_temp,
            temp_mode=temp_mode,
        )
