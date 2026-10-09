from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

# Normalized FM time t in [0, 1] is scaled into this range before the
# sinusoidal timestep embedding. Sampling step count does not change this.
FLOW_TIME_EMBEDDING_SCALE = 1000.0


class FlowMatching(nnx.Module):
    """Linear Flow Matching / Rectified Flow.

    Convention
    ----------
    t = 0:
        Data distribution.

    t = 1:
        Standard Gaussian distribution.

    Forward probability path:

        x_t = (1 - t) * x_0 + t * eps

        eps ~ N(0, I)

    Therefore,

        dx_t / dt = eps - x_0

    and the model learns

        v_theta(x_t, t, cond) ~= eps - x_0

    Sampling starts from Gaussian noise at t=1 and integrates the learned
    vector field backward toward t=0.
    """

    def __init__(
        self,
        velocity_fn: nnx.Module,
        num_denoising_steps: int = 50,
        clip_output: bool = False,
    ):
        self.velocity_fn = velocity_fn
        self.num_denoising_steps = int(num_denoising_steps)
        self.clip_output = bool(clip_output)

    # ------------------------------------------------------------------
    # Time utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_t(
        t: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Expand [B] time to [B, 1, 1] for trajectory broadcasting."""
        return t.reshape((x.shape[0], 1, 1))

    def _model_time(
        self,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map normalized FM time t in [0, 1] to the sinusoidal embedding input.

        Training and sampling both use t * FLOW_TIME_EMBEDDING_SCALE. The
        number of ODE steps never changes this scale.
        """
        return t * FLOW_TIME_EMBEDDING_SCALE

    # ------------------------------------------------------------------
    # Probability path
    # ------------------------------------------------------------------

    def interpolate(
        self,
        x0: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Linear interpolation between data and Gaussian noise.

        Args
        ----
        x0:
            [B, D, T]

        noise:
            [B, D, T]

        t:
            [B], continuous values in [0, 1]
        """
        t_expanded = self._expand_t(t, x0)

        return (
            (1.0 - t_expanded) * x0
            + t_expanded * noise
        )

    @staticmethod
    def target_velocity(
        x0: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Exact conditional velocity for the linear path."""
        return noise - x0

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------

    def predict_velocity(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the learned vector field at scene-conditioned time t."""
        model_t = self._model_time(t)
        return self.velocity_fn(x_t, model_t, cond)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0: jnp.ndarray,
        cond: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Flow Matching MSE objective."""
        x_t = self.interpolate(x0=x0, noise=noise, t=t)
        pred_v = self.predict_velocity(x_t=x_t, t=t, cond=cond)
        target_v = self.target_velocity(x0=x0, noise=noise)
        return jnp.mean((pred_v - target_v) ** 2)

    def loss(
        self,
        x0: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        rng: jax.Array,
    ) -> jnp.ndarray:
        """Sample continuous t and a Gaussian endpoint, then compute FM loss."""
        batch_size = x0.shape[0]
        key_t, key_noise = jax.random.split(rng)
        t = jax.random.uniform(
            key_t,
            shape=(batch_size,),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        noise = jax.random.normal(key_noise, shape=x0.shape, dtype=x0.dtype)
        return self.compute_loss(x0=x0, cond=cond, noise=noise, t=t)

    # ------------------------------------------------------------------
    # ODE solvers
    # ------------------------------------------------------------------

    def _euler_step(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        cond: jnp.ndarray,
    ) -> jnp.ndarray:
        """One backward Euler step: x(t - dt) = x(t) - dt * v(x, t)."""
        v = self.predict_velocity(x_t=x, t=t, cond=cond)
        return x - dt * v

    def _heun_step(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        cond: jnp.ndarray,
    ) -> jnp.ndarray:
        """Second-order Heun ODE step."""
        v1 = self.predict_velocity(x_t=x, t=t, cond=cond)
        x_euler = x - dt * v1
        t_next = jnp.maximum(t - dt, 0.0)
        v2 = self.predict_velocity(x_t=x_euler, t=t_next, cond=cond)
        return x - dt * 0.5 * (v1 + v2)

    def _integrate(
        self,
        x_init: jnp.ndarray,
        start_t: float,
        num_steps: int,
        cond: jnp.ndarray,
        solver: str,
    ) -> jnp.ndarray:
        """Integrate the vector field from start_t down to 0."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if solver not in ("euler", "heun"):
            raise ValueError(f"Unknown solver: {solver}")

        batch_size = x_init.shape[0]
        dt = jnp.asarray(start_t / float(num_steps), dtype=x_init.dtype)

        def body(i: int, x_curr: jnp.ndarray) -> jnp.ndarray:
            i_float = i.astype(x_init.dtype)
            t_scalar = jnp.asarray(start_t, dtype=x_init.dtype) - i_float * dt
            t = jnp.full((batch_size,), t_scalar, dtype=x_init.dtype)
            if solver == "euler":
                return self._euler_step(x=x_curr, t=t, dt=dt, cond=cond)
            return self._heun_step(x=x_curr, t=t, dt=dt, cond=cond)

        x_final = jax.lax.fori_loop(0, num_steps, body, x_init)
        if self.clip_output:
            x_final = jnp.clip(x_final, -1.0, 1.0)
        return x_final

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond: jnp.ndarray,
        *,
        rng: jax.Array,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:
        """Generate trajectories from the standard Gaussian prior.

        shape is [B, D, T]. solver is "euler" or "heun".
        """
        if num_steps is None:
            num_steps = self.num_denoising_steps
        x_init = jax.random.normal(rng, shape=shape, dtype=jnp.float32)
        return self._integrate(
            x_init=x_init,
            start_t=1.0,
            num_steps=num_steps,
            cond=cond,
            solver=solver,
        )

    # ------------------------------------------------------------------
    # Proposal refinement
    # ------------------------------------------------------------------

    def resample(
        self,
        proposals: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        rng: jax.Array,
        t_start: float = 0.2,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:
        """Corrupt proposals to t_start and integrate back to data."""
        if not (0.0 < t_start <= 1.0):
            raise ValueError(f"t_start must be in (0, 1], got {t_start}")
        if num_steps is None:
            num_steps = max(1, int(round(self.num_denoising_steps * t_start)))

        noise = jax.random.normal(rng, shape=proposals.shape, dtype=proposals.dtype)
        batch_size = proposals.shape[0]
        t = jnp.full((batch_size,), t_start, dtype=proposals.dtype)
        x_init = self.interpolate(x0=proposals, noise=noise, t=t)
        return self._integrate(
            x_init=x_init,
            start_t=t_start,
            num_steps=num_steps,
            cond=cond,
            solver=solver,
        )
