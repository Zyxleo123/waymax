from __future__ import annotations

import math
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from model.diffusion.modules.attention_2 import CrossAttention, SelfAttention, AttentionBlock


def sinusoidal_timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10_000) -> jnp.ndarray:
    t = jnp.asarray(t)
    if not jnp.issubdtype(t.dtype, jnp.floating):
        t = t.astype(jnp.float32)
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
    return emb


class Conv1dBlock(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, rngs: nnx.Rngs, kernel: int = 3, groups: int = 8):
        self.conv = nnx.Conv(
            in_features=in_ch,
            out_features=out_ch,
            kernel_size=(kernel,),
            padding="SAME",
            rngs=rngs,
        )
        self.gn = nnx.GroupNorm(num_features=out_ch, num_groups=min(groups, out_ch), epsilon=1e-5, rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        h_btc = self.conv(x_btc)
        h_btc = self.gn(h_btc)
        return nnx.silu(h_btc)


class Identity1d(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class ResBlock1d(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, cond_ch: int, hidden_dim: int, rngs: nnx.Rngs, groups: int = 8):
        self.time = nnx.Conv(in_features=time_ch, out_features=in_ch, kernel_size=(1,), padding="SAME", rngs=rngs)
        self.block1 = Conv1dBlock(in_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.skip = Identity1d() if in_ch == out_ch else nnx.Conv(in_features=in_ch, out_features=out_ch, kernel_size=(1,), padding="SAME", rngs=rngs)
        # Cross-attention for condition integration
        self.cross_attn = CrossAttention(query_dim=out_ch, context_dim=cond_ch, num_heads=8, hidden_dim=hidden_dim, rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray, time_feat_b1c: jnp.ndarray) -> jnp.ndarray:
        """Apply residual block with cross-attention conditioning.
        
        Args:
            x_btc: [B, T, C_in] input features
            time_feat_b1c: [B, 1, C_time] time embedding
            context_bnc: [B, n_tokens, C_cond] condition tokens
        
        Returns:
            [B, T, C_out] output features
        """
        time_b1c = self.time(time_feat_b1c)
        h = x_btc + time_b1c
        h = self.block1(h)
        h = self.block2(h)

        if isinstance(self.skip, Identity1d):
            return h + x_btc
        skip = self.skip(x_btc)
        return h + skip


class Downsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(4,), strides=(2,), padding=((1, 1),), rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x_btc)


class Upsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        x_btc = jax.image.resize(x_btc, shape=(x_btc.shape[0], x_btc.shape[1] * 2, x_btc.shape[2]), method="nearest")
        return self.conv(x_btc)


class Projector(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, in_feat: jnp.ndarray) -> jnp.ndarray:
        h = self.fc1(in_feat)
        h = nnx.silu(h)
        return self.fc2(h)
    

class FiLMBlock(nnx.Module):
    def __init__(self, in_ch: int, cond_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(cond_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, 2 * in_ch, rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray, cond_bc: jnp.ndarray, cond_mask: jnp.ndarray) -> jnp.ndarray:
        """Apply FiLM conditioning.
        
        Args:
            x_btc: [B, T, C] input features
            cond_bc: [B, C_cond] condition vectors
            cond_mask: [B] condition masks
        
        Returns:
            [B, T, C] modulated features
        """
        # Pool condition tokens to get a single vector per batch item
        gamma_beta = self.fc1(cond_bc)  # [B, hidden_dim]
        gamma_beta = nnx.silu(gamma_beta)
        
        gamma_beta = self.fc2(gamma_beta)  # [B, 2*in_ch]
        # After fc2: split to gamma/beta
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)  # each [B, C]
       
        return x_btc * (1 + gamma[:, None, :] * cond_mask[:, None, None]) + beta[:, None, :] * cond_mask[:, None, None]


class UNet1DConditioned(nnx.Module):
    def __init__(
        self,
        in_ch: int,
        rngs: nnx.Rngs,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        cond2_dim: int = 256,
        time_emb_dim: int = 128,
        hidden_dim: int = 512,
        groups: int = 8,
        horizon: int = 25,
    ):
        self.in_ch = in_ch
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.n_down = len(ch_mult)
        self.horizon = horizon

        # Condition projection: project tokens to base_ch dimension for cross-attention
        self.cond_proj = nnx.Linear(cond_dim, base_ch, rngs=rngs)
        self.time_proj = Projector(time_emb_dim, base_ch, hidden_dim=hidden_dim, rngs=rngs)
        self.stem = nnx.Conv(in_features=in_ch, out_features=base_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

        ch = base_ch
        down_blocks = []
        downsamples = []
        skip_channels = []
        down_film_blocks = []
        down_attn_blocks = []

        for i, m in enumerate(ch_mult):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                down_blocks.append(ResBlock1d(ch, out_ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups))
                down_attn_blocks.append(AttentionBlock(out_ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs))
                down_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
                ch = out_ch
            skip_channels.append(ch)
            downsamples.append(Downsample1d(ch, rngs=rngs) if i != len(ch_mult) - 1 else Identity1d())\
            
        mid_film_blocks = []
        self.mid_block1 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups)
        self.mid_attn_block1 = AttentionBlock(ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs)
        mid_film_blocks.append(FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
        self.mid_block2 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups)
        self.mid_attn_block2 = AttentionBlock(ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs)
        mid_film_blocks.append(FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))

        up_blocks = []
        upsamples = []
        up_film_blocks = []
        up_attn_blocks = []
        for i, m in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                up_blocks.append(ResBlock1d(ch + skip_channels[i], out_ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups))
                up_attn_blocks.append(AttentionBlock(out_ch, context_dim=base_ch, attn_hidden_dim=hidden_dim, ffn_hidden_dim=hidden_dim, n_layers=2, rngs=rngs))
                up_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
                ch = out_ch
            upsamples.append(Upsample1d(ch, rngs=rngs) if i != 0 else Identity1d())

        # Register containers as NNX modules in a single assignment.
        self.down_blocks = nnx.List(down_blocks)
        self.down_attn_blocks = nnx.List(down_attn_blocks)
        self.downsamples = nnx.List(downsamples)
        self.skip_channels = tuple(skip_channels)
        self.up_blocks = nnx.List(up_blocks)
        self.up_attn_blocks = nnx.List(up_attn_blocks)
        self.upsamples = nnx.List(upsamples)
        self.down_film_blocks = nnx.List(down_film_blocks)
        self.mid_film_blocks = nnx.List(mid_film_blocks)
        self.up_film_blocks = nnx.List(up_film_blocks)
        self.out_norm = nnx.GroupNorm(num_features=ch, num_groups=min(groups, ch), epsilon=1e-5, rngs=rngs)
        self.out_conv = nnx.Conv(in_features=ch, out_features=in_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, c1: jnp.ndarray, c2: jnp.ndarray, c2_mask: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with cross-attention conditioning.
        
        Args:
            x: [B, T, 5] input trajectory
            t: [B] timestep indices
            c1: [B, n_cond_tokens, cond_dim] condition tokens for cross-attention
            c2: [B, cond_dim] second condition tokens
            c2_mask: [B] mask for second condition tokens

        Returns:
            [B, T, 5] denoised trajectory
        """
        divisor = 2 ** (self.n_down - 1)
        if self.horizon % divisor != 0:
            target_t = ((self.horizon // divisor) + 1) * divisor
            x = jnp.pad(x, ((0, 0), (0, 0), (0, target_t - self.horizon)), mode="constant")

        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        # Project condition tokens: [B, n_tokens, cond_dim] -> [B, n_tokens, base_ch]
        c1 = self.cond_proj(c1)  # [B, n_tokens, base_ch]
        time_feat = self.time_proj(t_emb)[:, None, :]

        h = self.stem(jnp.swapaxes(x, 1, 2))

        skips = []
        rb = 0
        for i in range(self.n_down):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[rb](h, time_feat)
                h = self.down_attn_blocks[rb](h, c1)
                h = self.down_film_blocks[rb](h, c2, c2_mask)
                rb += 1
            skips.append(h)
            h = self.downsamples[i](h)

        h = self.mid_block1(h, time_feat)
        h = self.mid_attn_block1(h, c1)
        h = self.mid_film_blocks[0](h, c2, c2_mask)
        h = self.mid_block2(h, time_feat)
        h = self.mid_attn_block2(h, c1)
        h = self.mid_film_blocks[1](h, c2, c2_mask)
        
        rb_up = 0
        for i in range(self.n_down):
            skip = skips.pop()
            if h.shape[1] != skip.shape[1]:
                min_len = min(h.shape[1], skip.shape[1])
                h = h[:, :min_len, :]
                skip = skip[:, :min_len, :]

            for _ in range(self.num_res_blocks):
                h = jnp.concatenate([h, skip], axis=-1)
                h = self.up_blocks[rb_up](h, time_feat)
                h = self.up_attn_blocks[rb_up](h, c1)
                h = self.up_film_blocks[rb_up](h, c2, c2_mask)
                rb_up += 1

            h = self.upsamples[i](h)

        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)
        return jnp.swapaxes(h[:, :self.horizon, :], 1, 2)


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

class FlowMatching(nnx.Module):
    """Linear Flow Matching / Rectified Flow.

    Convention:
        t = 0 : data
        t = 1 : Gaussian noise

        x_t = (1 - t) * x0 + t * noise

    The network predicts the forward velocity

        v = dx_t / dt = noise - x0

    Sampling starts from noise at t=1 and integrates backward to t=0.
    """

    def __init__(
        self,
        denoise_fn: nnx.Module,
        flow_time_embedding_scale: int = 1000,
        flow_num_steps: int = 50,
        clip_denoised: bool = False,
    ):
        self.denoise_fn = denoise_fn
        self.flow_time_embedding_scale = int(flow_time_embedding_scale)
        self.flow_num_steps = int(flow_num_steps)
        self.clip_denoised = clip_denoised

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_t(
        t: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert [B] timestep to broadcastable [B, 1, 1]."""
        return t.reshape((x.shape[0], 1, 1))

    def _model_time(
        self,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert normalized FM time [0, 1] to the range previously used
        by the diffusion timestep embedding.

        Your UNet currently feeds t directly into sinusoidal_timestep_embedding.
        Using [0, 1] directly makes the time embeddings vary only weakly.

        Scaling it back to approximately [0, timesteps-1] lets you keep the
        existing UNet implementation unchanged.
        """
        return t * float(self.flow_time_embedding_scale - 1)

    # ------------------------------------------------------------------
    # Forward flow
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: jnp.ndarray,
        t: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample a point on the linear probability path.

        Args:
            x0:
                [B, ...] clean samples.

            t:
                [B] continuous time in [0, 1].

            noise:
                Gaussian noise with same shape as x0.

        Returns:
            x_t = (1-t) x0 + t noise
        """
        t_expand = self._expand_t(t, x0)

        return (
            (1.0 - t_expand) * x0
            + t_expand * noise
        )

    @staticmethod
    def target_velocity(
        x0: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Ground-truth velocity for the linear path."""
        return noise - x0

    # ------------------------------------------------------------------
    # Network prediction
    # ------------------------------------------------------------------

    def predict_velocity(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict dx/dt at x_t."""
        model_t = self._model_time(t)

        return self.denoise_fn(
            x_t,
            model_t,
            cond1,
            cond2,
            cond2_mask,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:

        # Linear interpolation between data and noise
        x_t = self.q_sample(
            x0,
            t,
            noise,
        )

        # Network predicts velocity
        pred_v = self.predict_velocity(
            x_t,
            t,
            cond1,
            cond2,
            cond2_mask,
        )

        # Exact target velocity of the linear path
        target_v = self.target_velocity(
            x0,
            noise,
        )

        return jnp.mean(
            (pred_v - target_v) ** 2
        )

    def _loss_impl(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:

        batch = x0.shape[0]

        key_t, key_noise = jax.random.split(rng)

        # Continuous time rather than discrete diffusion timestep
        t = jax.random.uniform(
            key_t,
            shape=(batch,),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )

        noise = jax.random.normal(
            key_noise,
            x0.shape,
            dtype=x0.dtype,
        )

        return self._compute_loss(
            x0,
            cond1,
            cond2,
            cond2_mask,
            noise,
            t,
        )

    # ------------------------------------------------------------------
    # ODE sampling
    # ------------------------------------------------------------------

    def _ode_step(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: float,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """One backward Euler step.

        Model predicts forward velocity dx/dt.
        Since sampling goes t=1 -> 0,

            x_{t-dt} = x_t - dt * v(x_t, t)
        """

        velocity = self.predict_velocity(
            x,
            t,
            cond1,
            cond2,
            cond2_mask,
        )

        return x - dt * velocity

    def _sample_impl(
        self,
        shape: Tuple[int, int, int],
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        rng: jax.Array,
        sampling_temp: float = 1.0,
    ) -> jnp.ndarray:

        batch_size = shape[0]

        # Initial distribution p_1(x) = N(0, I)
        x_init = (
            jax.random.normal(
                rng,
                shape,
                dtype=jnp.float32,
            )
            * sampling_temp
        )

        num_steps = self.flow_num_steps
        dt = 1.0 / num_steps

        def body(
            i: int,
            x_curr: jnp.ndarray,
        ) -> jnp.ndarray:

            # i = 0 -> t = 1
            # i = N-1 -> t = 1/N
            t_scalar = 1.0 - (
                i.astype(jnp.float32)
                / float(num_steps)
            )

            t = jnp.full(
                (batch_size,),
                t_scalar,
                dtype=jnp.float32,
            )

            x_next = self._ode_step(
                x_curr,
                t,
                dt,
                cond1,
                cond2,
                cond2_mask,
            )

            return x_next

        x_final = jax.lax.fori_loop(
            0,
            num_steps,
            body,
            x_init,
        )

        if self.clip_denoised:
            x_final = jnp.clip(
                x_final,
                -1.0,
                1.0,
            )

        return x_final

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _resample_impl(
        self,
        proposals: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        n_timesteps: int,
        rng: jax.Array,
        noise_scale: float = 1.0,
    ) -> jnp.ndarray:
        """Partially noise proposals and integrate back to t=0.

        This is the Flow Matching analogue of your previous diffusion
        q_sample + partial denoising procedure.

        Example:
            timesteps = 100
            n_timesteps = 20

        means:
            proposal -> t=0.2 -> integrate 0.2 -> 0
        """

        batch_size = proposals.shape[0]

        # Amount of corruption
        t_start = (
            float(n_timesteps)
            / float(self.flow_num_steps)
        )

        noise = (
            jax.random.normal(
                rng,
                proposals.shape,
                dtype=proposals.dtype,
            )
            * noise_scale
        )

        t0 = jnp.full(
            (batch_size,),
            t_start,
            dtype=jnp.float32,
        )

        # Move proposal toward Gaussian noise.
        x_init = self.q_sample(
            proposals,
            t0,
            noise,
        )

        # Because
        #
        #   t_start = n_timesteps / total_timesteps
        #
        # this gives exactly the same dt used during full sampling.
        dt = 1.0 / self.flow_num_steps

        def body(
            i: int,
            x_curr: jnp.ndarray,
        ) -> jnp.ndarray:

            t_scalar = (
                float(n_timesteps)
                - i.astype(jnp.float32)
            ) / float(self.flow_num_steps)

            t = jnp.full(
                (batch_size,),
                t_scalar,
                dtype=jnp.float32,
            )

            return self._ode_step(
                x_curr,
                t,
                dt,
                cond1,
                cond2,
                cond2_mask,
            )

        x_final = jax.lax.fori_loop(
            0,
            n_timesteps,
            body,
            x_init,
        )

        if self.clip_denoised:
            x_final = jnp.clip(
                x_final,
                -1.0,
                1.0,
            )

        return x_final

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        sampling_temp: float = 1.0,
    ) -> jnp.ndarray:

        return self._sample_impl(
            shape,
            cond1,
            cond2,
            cond2_mask,
            rng,
            sampling_temp=sampling_temp,
        )

    def loss(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
    ) -> jnp.ndarray:

        return self._loss_impl(
            x0,
            cond1,
            cond2,
            cond2_mask,
            rng,
        )

    def resample(
        self,
        proposals: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: jax.Array,
        noise_scale: float = 1.0,
    ) -> jnp.ndarray:

        if (
            n_timesteps <= 0
            or n_timesteps > self.flow_num_steps
        ):
            raise ValueError(
                f"n_timesteps must be in "
                f"[1, {self.flow_num_steps}], "
                f"got {n_timesteps}"
            )

        return self._resample_impl(
            proposals,
            cond1,
            cond2,
            cond2_mask,
            n_timesteps,
            rng,
            noise_scale=noise_scale,
        )