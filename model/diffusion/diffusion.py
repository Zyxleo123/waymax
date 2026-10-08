from __future__ import annotations

import math
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from model.diffusion.modules.attention import CrossAttention


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

    def __call__(self, x_btc: jnp.ndarray, time_feat_b1c: jnp.ndarray, context_bnc: jnp.ndarray) -> jnp.ndarray:
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
        # Apply cross-attention with condition tokens
        # Provide an explicit boolean mask for condition tokens (assume all present)
        context_mask_bn = jnp.ones((context_bnc.shape[0], context_bnc.shape[1]), dtype=bool)
        h = h + self.cross_attn(h, context_bnc, context_mask_bn)

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
        self_cond: bool = False,
    ):
        self.in_ch = in_ch
        # FlowMol3 self-conditioning: the denoiser additionally sees its OWN clean
        # prediction from the previous integration step. `in_ch` stays the
        # SEMANTIC channel count (5) -- callers size sample shapes off it and the
        # output conv still emits 5 -- so only the stem's fan-in changes.
        self.self_cond = bool(self_cond)
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.n_down = len(ch_mult)
        self.horizon = horizon

        # Condition projection: project tokens to base_ch dimension for cross-attention
        self.cond_proj = nnx.Linear(cond_dim, base_ch, rngs=rngs)
        self.time_proj = Projector(time_emb_dim, base_ch, hidden_dim=hidden_dim, rngs=rngs)
        stem_in = in_ch * 2 if self.self_cond else in_ch
        self.stem = nnx.Conv(in_features=stem_in, out_features=base_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

        ch = base_ch
        down_blocks = []
        downsamples = []
        skip_channels = []
        down_film_blocks = []

        for i, m in enumerate(ch_mult):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                down_blocks.append(ResBlock1d(ch, out_ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups))
                down_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
                ch = out_ch
            skip_channels.append(ch)
            downsamples.append(Downsample1d(ch, rngs=rngs) if i != len(ch_mult) - 1 else Identity1d())\
            
        mid_film_blocks = []
        self.mid_block1 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups)
        mid_film_blocks.append(FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
        self.mid_block2 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups)
        mid_film_blocks.append(FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))

        up_blocks = []
        upsamples = []
        up_film_blocks = []
        for i, m in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                up_blocks.append(ResBlock1d(ch + skip_channels[i], out_ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups))
                up_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
                ch = out_ch
            upsamples.append(Upsample1d(ch, rngs=rngs) if i != 0 else Identity1d())

        # Register containers as NNX modules in a single assignment.
        self.down_blocks = nnx.List(down_blocks)
        self.downsamples = nnx.List(downsamples)
        self.skip_channels = tuple(skip_channels)
        self.up_blocks = nnx.List(up_blocks)
        self.upsamples = nnx.List(upsamples)
        self.down_film_blocks = nnx.List(down_film_blocks)
        self.mid_film_blocks = nnx.List(mid_film_blocks)
        self.up_film_blocks = nnx.List(up_film_blocks)
        self.out_norm = nnx.GroupNorm(num_features=ch, num_groups=min(groups, ch), epsilon=1e-5, rngs=rngs)
        self.out_conv = nnx.Conv(in_features=ch, out_features=in_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, c1: jnp.ndarray, c2: jnp.ndarray, c2_mask: jnp.ndarray,
                 x_sc: jnp.ndarray | None = None, tail_hook=None) -> jnp.ndarray:
        """Forward pass with cross-attention conditioning.

        Args:
            x: [B, 5, T] input trajectory (channels on axis 1, horizon on axis 2)
            x_sc: [B, 5, T] self-conditioning input -- the clean prediction from the
                PREVIOUS integration step, or None/zeros for the first step (and for
                the ``1 - sc_train_prob`` share of training steps). Ignored unless
                the module was built with ``self_cond=True``.
            t: [B] timestep indices
            c1: [B, n_cond_tokens, cond_dim] condition tokens for cross-attention
            c2: [B, cond_dim] second condition tokens
            c2_mask: [B] mask for second condition tokens
            tail_hook: optional ``hook(j, h, time_feat) -> h`` called after every up
                block (ResBlock + FiLM), ``j`` counted from the END (-1 = the last
                block), ``h`` ``[B, T_pad, C]`` and ``time_feat`` ``[B, base_ch]``.
                The gated adapter (:mod:`rl.diffusion_ft.adapter`) inserts its
                residual here. None (default) leaves the traced graph unchanged.

        Returns:
            [B, T, 5] denoised trajectory
        """
        if self.self_cond:
            # Channel-concat on axis 1. Zeros is the documented "no previous
            # prediction" token, not a special branch, so the first integration
            # step and the unconditioned training share use the same graph.
            x = jnp.concatenate([x, jnp.zeros_like(x) if x_sc is None else x_sc], axis=1)

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
                h = self.down_blocks[rb](h, time_feat, c1)
                h = self.down_film_blocks[rb](h, c2, c2_mask)
                rb += 1
            skips.append(h)
            h = self.downsamples[i](h)

        h = self.mid_block1(h, time_feat, c1)
        h = self.mid_film_blocks[0](h, c2, c2_mask)
        h = self.mid_block2(h, time_feat, c1)
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
                h = self.up_blocks[rb_up](h, time_feat, c1)
                h = self.up_film_blocks[rb_up](h, c2, c2_mask)
                if tail_hook is not None:
                    h = tail_hook(rb_up - len(self.up_blocks), h, time_feat[:, 0, :])
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


class GaussianDiffusion(nnx.Module):
    def __init__(
        self,
        denoise_fn: nnx.Module,
        timesteps: int = 100,
        beta_schedule: str = "cosine",
        predict_type: str = "v",
        clip_denoised: bool = False,
        self_cond: bool = False,
        sc_train_prob: float = 0.5,
        distort_p: float = 0.0,
        distort_sigma: float = 0.0,
        distort_mode: str = "relative",
        distort_t_max: int = -1,
        distort_t_min: int = 0,
        distort_channels: int = 2,
    ):
        self.denoise_fn = denoise_fn
        self.timesteps = int(timesteps)
        self.predict_type = predict_type
        self.clip_denoised = clip_denoised

        # ---- FlowMol3 inference-OOD mitigations (arXiv 2508.12629) ----------
        # Both are OFF by default, and with both off every method below is
        # bit-for-bit the pre-existing one (the self-cond carry is a `None` leaf,
        # which is an empty pytree, so the traced programs are unchanged).
        #
        # self_cond: the denoiser is fed its own clean prediction from the previous
        #   reverse step. Training draws a Bernoulli(`sc_train_prob`) mask per
        #   example: on the masked rows a FIRST, stop-gradient pass supplies the
        #   input; on the rest it is zeros (which is also what the first reverse
        #   step sees). Requires `denoise_fn` built with `self_cond=True`.
        #
        # distortion: sparse Gaussian kicks added to `x_t` in the LATE (low-t,
        #   near-data) part of the chain at training time only, so the network is
        #   asked to map an off-path state back to the SAME clean target instead of
        #   only ever seeing exact points of the forward path. `t_distort=0.5` in
        #   FlowMol3's flow time (t=0 noise -> t=1 data) is the low-noise half,
        #   i.e. `t <= timesteps//2 - 1` on this DDPM index.
        self.self_cond = bool(self_cond)
        if self.self_cond and not bool(getattr(denoise_fn, "self_cond", False)):
            raise ValueError(
                "GaussianDiffusion(self_cond=True) needs a denoise_fn built with "
                "self_cond=True (its stem takes 2*in_ch channels). Build the UNet "
                "with self_cond=True, or expand an existing checkpoint with "
                "rl.diffusion_ft.self_cond_init.expand_params_for_self_cond.")
        self.sc_train_prob = float(sc_train_prob)
        self.distort_p = float(distort_p)
        self.distort_sigma = float(distort_sigma)
        # "relative": sigma is a MULTIPLE of the forward noise sqrt(1-abar_t) at that
        # timestep, so the kick is always commensurate with the noise already there
        # and the corrected v/eps target stays O(sigma). "absolute": sigma is in the
        # trajectory's own normalized units (1.0 == ego_range == 100 m), matching
        # FlowMol3's fixed sigma_distort. Relative is the default because on a
        # v-parameterized DDPM an absolute kick makes the corrected target blow up
        # like 1/sqrt(1-abar_t), which at t=0 is a factor of ~40.
        if distort_mode not in ("relative", "absolute"):
            raise ValueError(f"distort_mode must be 'relative' or 'absolute', got {distort_mode!r}")
        self.distort_mode = str(distort_mode)
        self.distort_t_max = (self.timesteps // 2 - 1) if int(distort_t_max) < 0 else int(distort_t_max)
        self.distort_t_min = int(distort_t_min)
        # How many leading channels get kicked. The trajectory is
        # [x/ego_range, y/ego_range, vx, vy, yaw/pi]; FlowMol3 distorts POSITIONS
        # only, so 2 is the faithful port.
        self.distort_channels = int(distort_channels)

        betas = make_beta_schedule(self.timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]], axis=0)

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

        self.posterior_mean_coef1 = nnx.Variable(betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = nnx.Variable((1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod))

    @staticmethod
    def _extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Sequence[int]) -> jnp.ndarray:
        out = a[t]
        return out.reshape((x_shape[0], 1, 1))

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / sqrt_ab

    def _apply_denoiser(self, fn, x_t, t, cond1, cond2, cond2_mask, x_sc):
        """Call the denoiser, passing the self-conditioning input only when the
        module was built for it.

        Off the self-cond path this is the exact 5-argument call every caller and
        every test double already implements, so nothing about the default
        architecture's tracing changes.
        """
        if self.self_cond:
            return fn(x_t, t, cond1, cond2, cond2_mask, x_sc)
        return fn(x_t, t, cond1, cond2, cond2_mask)

    def _pred_to_x0(self, x_t: jnp.ndarray, t: jnp.ndarray, pred: jnp.ndarray) -> jnp.ndarray:
        """Network output -> clean-sample prediction, for every ``predict_type``."""
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
        return x0_pred

    def p_mean_variance(self, x_t: jnp.ndarray, t: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, denoise_fn=None, x_sc: jnp.ndarray | None = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # ``denoise_fn`` lets a caller substitute a different denoiser (e.g. the
        # DPPO-trainable copy for the late steps) without changing default
        # inference, which keeps using ``self.denoise_fn``.
        fn = self.denoise_fn if denoise_fn is None else denoise_fn
        pred = self._apply_denoiser(fn, x_t, t, cond1, cond2, cond2_mask, x_sc)
        x0_pred = self._pred_to_x0(x_t, t, pred)

        mean = self._extract(self.posterior_mean_coef1.value, t, x_t.shape) * x0_pred + self._extract(self.posterior_mean_coef2.value, t, x_t.shape) * x_t
        var = self._extract(self.posterior_variance.value, t, x_t.shape)
        return mean, var, x0_pred

    @staticmethod
    def _temp_schedule(sampling_temp: float, temp_mode: str, num_steps: int) -> jnp.ndarray:
        """Precompute per-timestep effective temperature.

        Returns array of shape [num_steps] indexed by diffusion loop step index
        (0 = highest noise timestep, num_steps-1 = lowest).
        """
        t_idx = jnp.arange(num_steps, dtype=jnp.float32)
        denom = jnp.maximum(num_steps - 1, 1)  # avoid /0 when num_steps==1
        if temp_mode == "early":
            # Full temp at high-noise (step 0 = high t), taper to 1.0 at low-noise
            return 1.0 + (sampling_temp - 1.0) * (1.0 - t_idx / denom)
        elif temp_mode == "late":
            # 1.0 at high-noise, full temp at low-noise (step num_steps-1)
            return 1.0 + (sampling_temp - 1.0) * (t_idx / denom)
        else:  # "uniform"
            return jnp.full((num_steps,), sampling_temp, dtype=jnp.float32)

    def _ddim_update(self, x_t: jnp.ndarray, t: jnp.ndarray, x0_pred: jnp.ndarray,
                     noise: jnp.ndarray, eta: float = 1.0, sampling_temp: float = 1.0) -> jnp.ndarray:
        """The eta-DDIM reverse step given an already-predicted ``x0``.

        Factored out of :meth:`p_sample` so a caller that has already run the
        denoiser once (e.g. :meth:`sample_with_trace`, which also needs the
        posterior stats) can reuse the single prediction instead of invoking the
        network twice -- the network is not purely functional across calls (nnx
        rng streams), so a second call would perturb the trajectory.
        """
        # Predicted noise direction
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_1mab

        # Scale noise direction by sampling temperature
        eps_pred = eps_pred * sampling_temp
        x0_pred = (x_t - sqrt_1mab * eps_pred) / sqrt_ab

        # DDIM variance: σ_t = η * sqrt(posterior_variance_t)
        post_var = self._extract(self.posterior_variance.value, t, x_t.shape)
        sigma = eta * jnp.sqrt(post_var)

        # Direction coefficient: sqrt(max(1 - ᾱ_{t-1} - σ², 0))
        abar_prev = self._extract(self.alphas_cumprod_prev.value, t, x_t.shape)
        dir_coeff = jnp.sqrt(jnp.maximum(1.0 - abar_prev - sigma ** 2, 0.0))

        # DDIM update: sqrt(ᾱ_{t-1}) * x̂₀ + dir_coeff * ε_pred + σ * noise
        nonzero_mask = (t != 0).astype(x_t.dtype).reshape((-1, 1, 1))
        return jnp.sqrt(abar_prev) * x0_pred + dir_coeff * eps_pred + nonzero_mask * sigma * noise

    def p_sample_step(self, x_t: jnp.ndarray, t: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray,
                      noise: jnp.ndarray, eta: float = 1.0, sampling_temp: float = 1.0, denoise_fn=None,
                      x_sc: jnp.ndarray | None = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """:meth:`p_sample` that also returns the step's clean prediction.

        The sampler needs that prediction as the NEXT step's self-conditioning
        input, and the network must not be called twice to get it (nnx rng streams
        aside, it would double the cost). ``p_sample`` is this with the second
        element dropped, so the two cannot drift apart.
        """
        _, _, x0_pred = self.p_mean_variance(x_t, t, cond1, cond2, cond2_mask, denoise_fn=denoise_fn, x_sc=x_sc)
        return self._ddim_update(x_t, t, x0_pred, noise, eta=eta, sampling_temp=sampling_temp), x0_pred

    def p_sample(self, x_t: jnp.ndarray, t: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray,
                 noise: jnp.ndarray, eta: float = 1.0, sampling_temp: float = 1.0, denoise_fn=None,
                 x_sc: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.p_sample_step(x_t, t, cond1, cond2, cond2_mask, noise, eta=eta,
                                  sampling_temp=sampling_temp, denoise_fn=denoise_fn, x_sc=x_sc)[0]

    def _next_sc(self, x0_pred: jnp.ndarray | None) -> jnp.ndarray | None:
        """The self-conditioning input for the next reverse step.

        ``None`` off the self-cond path -- an empty pytree, so a scan/fori carry
        that holds it is identical to one that does not. Stop-gradient matches
        training, where the first (self-cond-producing) pass is explicitly cut off
        from the graph; letting the gradient run back through the whole chain of
        previous predictions would also defeat the truncated-backprop budgets the
        unroll and DRaFT-K paths are built around.
        """
        if not self.self_cond:
            return None
        return jax.lax.stop_gradient(x0_pred)

    def _ddim_step_strided(self, x_t: jnp.ndarray, from_t: jnp.ndarray, to_t: jnp.ndarray,
                           cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray,
                           noise: jnp.ndarray, eta: float = 1.0, sampling_temp: float = 1.0,
                           denoise_fn=None, x_sc: jnp.ndarray | None = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """A single eta-DDIM reverse step between *non-adjacent* timesteps ``from_t
        -> to_t`` (both ``>= 0``, ``to_t < from_t``).

        This is the general strided DDIM update. For a unit step ``to_t = from_t-1``
        and ``eta=1`` it is algebraically identical to :meth:`_ddim_update`: the DDIM
        sigma ``eta*sqrt((1-abar_to)/(1-abar_from) * (1 - abar_from/abar_to))`` reduces
        to ``sqrt(posterior_variance[from_t])`` because ``abar_from/abar_to = alpha``
        and ``1-alpha = beta``. Used only for the *frozen* early portion when the
        sampler is asked to subsample the reverse chain; the trainable late steps
        keep the unit-step math untouched, so the PPO likelihood is unaffected.
        """
        _, _, x0_pred = self.p_mean_variance(x_t, from_t, cond1, cond2, cond2_mask, denoise_fn=denoise_fn, x_sc=x_sc)
        x0_raw = x0_pred
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, from_t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, from_t, x_t.shape)
        eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_1mab
        # Match _ddim_update's temperature handling: scale the noise direction, then
        # re-derive x0 so the two coefficients stay consistent.
        eps_pred = eps_pred * sampling_temp
        x0_pred = (x_t - sqrt_1mab * eps_pred) / sqrt_ab

        abar_from = self._extract(self.alphas_cumprod.value, from_t, x_t.shape)
        abar_to = self._extract(self.alphas_cumprod.value, to_t, x_t.shape)
        sigma2 = (eta ** 2) * (1.0 - abar_to) / (1.0 - abar_from) * (1.0 - abar_from / abar_to)
        sigma = jnp.sqrt(jnp.maximum(sigma2, 0.0))
        dir_coeff = jnp.sqrt(jnp.maximum(1.0 - abar_to - sigma ** 2, 0.0))
        # Callers only use this for the frozen prefix, where every ``to_t >= 1`` (the
        # late/unit region owns ``t=0``), so noise is always injected here.
        # ``x0_raw`` (pre-temperature) is the network's own clean prediction and is
        # what the next step self-conditions on; the returned state is unchanged.
        return jnp.sqrt(abar_to) * x0_pred + dir_coeff * eps_pred + sigma * noise, x0_raw

    def _sample_impl(self, shape: Tuple[int, int, int], cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, 
                     rng: jax.Array, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> jnp.ndarray:
        batch_size = shape[0]
        key_x, key_steps = jax.random.split(rng)
        x_init = jax.random.normal(key_x, shape, dtype=jnp.float32)
        num_steps = self.timesteps
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, num_steps)

        def body(i: int, carry) -> tuple:
            x_curr, sc, key = carry
            timestep = num_steps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
            eff_temp = temp_sched[i]
            x_next, x0_pred = self.p_sample_step(
                x_curr, t, cond1, cond2, cond2_mask, noise, eta=eta, sampling_temp=eff_temp, x_sc=sc)
            return x_next, self._next_sc(x0_pred), key

        # `sc` is None off the self-cond path: an empty pytree, so this carry is
        # the two-element one this loop has always had.
        sc0 = jnp.zeros(shape, dtype=jnp.float32) if self.self_cond else None
        x_final, _sc, _ = jax.lax.fori_loop(0, num_steps, body, (x_init, sc0, key_steps))
        return x_final

    def distortion_enabled(self) -> bool:
        return self.distort_p > 0.0 and self.distort_sigma > 0.0

    def _draw_distortion(self, x_t: jnp.ndarray, t: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """FlowMol3's late-stage geometry distortion, as an additive ``[B, C, T]``.

        ``M_i ~ Bernoulli(p_distort)`` over WAYPOINTS (the waypoint is this model's
        "atom": one masked waypoint is kicked on all its distorted channels at
        once, which is the local, correctable defect the paper perturbs), times
        ``eps ~ N(0, sigma^2)`` on the first ``distort_channels`` channels, gated to
        ``distort_t_min <= t <= distort_t_max``.

        In ``relative`` mode sigma is scaled by ``sqrt(1 - abar_t)``, the forward
        noise already present at that timestep.
        """
        kb, kn = jax.random.split(key)
        b, c, tt = x_t.shape
        mask_bt = jax.random.bernoulli(kb, self.distort_p, (b, 1, tt)).astype(x_t.dtype)
        eps = jax.random.normal(kn, x_t.shape, dtype=x_t.dtype)
        ch = (jnp.arange(c) < self.distort_channels).astype(x_t.dtype)[None, :, None]
        sigma = jnp.asarray(self.distort_sigma, x_t.dtype)
        if self.distort_mode == "relative":
            sigma = sigma * self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        active = ((t >= self.distort_t_min) & (t <= self.distort_t_max)).astype(x_t.dtype).reshape((b, 1, 1))
        return active * mask_bt * ch * sigma * eps

    def _distorted_target(self, target: jnp.ndarray, t: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        """Correct the regression target for a distortion ``d`` added to ``x_t``.

        The whole point of the technique is that the network must map the
        DISTORTED state back to the SAME clean trajectory, so the clean target is
        held fixed and the parameterized target is re-derived from it. Leaving the
        target alone would instead teach the network to REPRODUCE the kick:
        with an unchanged ``v`` target the optimum satisfies
        ``x0_pred = x0 + sqrt(abar)*d``, i.e. the drift is propagated rather than
        corrected -- exactly backwards.

            mu :  target = x0                                 (no correction)
            eps:  target = noise + d / sqrt(1 - abar)
            v  :  target = v_std + sqrt(abar) * d / sqrt(1 - abar)
        """
        if self.predict_type == "mu":
            return target
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, target.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, target.shape)
        if self.predict_type == "eps":
            return target + d / sqrt_1mab
        return target + sqrt_ab * d / sqrt_1mab

    def _self_cond_train_input(self, fn, x_t, t, cond1, cond2, cond2_mask, key) -> jnp.ndarray:
        """The self-conditioning input for a TRAINING step.

        FlowMol3 splits training 50/50: on one half the network is scored on its
        zero-self-cond prediction, on the other half a FIRST pass supplies the
        input and is kept out of the gradient graph. The coin is flipped PER
        EXAMPLE here rather than per step, which costs a second denoiser
        evaluation on every step (the first pass runs for the whole batch and is
        then masked) but gives a lower-variance estimate and keeps the program
        shape static under jit/pmap.
        """
        zeros = jnp.zeros_like(x_t)
        pred0 = fn(x_t, t, cond1, cond2, cond2_mask, zeros)
        x0_0 = self._pred_to_x0(x_t, t, pred0)
        use = jax.random.bernoulli(key, self.sc_train_prob, (x_t.shape[0], 1, 1))
        return jax.lax.stop_gradient(jnp.where(use, x0_0, zeros))

    def _compute_loss_per_example(self, x0: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray, denoise_fn=None, time_weight: jnp.ndarray | None = None, key_aug: jax.Array | None = None) -> jnp.ndarray:
        """Per-example diffusion loss, averaged over channels and time but NOT
        over the batch. Returns ``[B]``. ``_compute_loss`` is its batch-mean.

        ``time_weight`` is an optional ``[T]`` weight over the horizon axis. The
        reduction stays a *weighted mean* (divided by the weight sum, not by T),
        so a 0/1 window mask is on the same scale as the unweighted loss rather
        than being shrunk by the fraction of the horizon it keeps -- otherwise
        narrowing the window would silently also lower the term's weight in a
        multi-term objective. ``None`` reduces over the whole horizon and is
        bit-for-bit the pre-existing behaviour.
        """
        fn = self.denoise_fn if denoise_fn is None else denoise_fn
        x_t = self.q_sample(x0, t, noise=noise)

        if self.predict_type == "eps":
            target = noise
        elif self.predict_type == "mu":
            target = x0
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
            target = sqrt_ab * noise - sqrt_1mab * x0
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        needs_key = self.distortion_enabled() or self.self_cond
        if needs_key and key_aug is None:
            raise ValueError(
                "self_cond / distortion need `key_aug`; call through loss()/"
                "loss_per_example(), which derives it from the loss rng.")
        if self.distortion_enabled():
            d = self._draw_distortion(x_t, t, jax.random.fold_in(key_aug, 1))
            x_t = x_t + d
            target = self._distorted_target(target, t, d)

        x_sc = None
        if self.self_cond:
            x_sc = self._self_cond_train_input(
                fn, x_t, t, cond1, cond2, cond2_mask, jax.random.fold_in(key_aug, 2))
        pred = self._apply_denoiser(fn, x_t, t, cond1, cond2, cond2_mask, x_sc)

        se = (pred - target) ** 2
        if time_weight is None:
            return jnp.mean(se, axis=tuple(range(1, se.ndim)))  # mean over C, T -> [B]
        # se is [B, C, T]; weight the trailing (time) axis only.
        w = jnp.asarray(time_weight, se.dtype)
        w = jnp.broadcast_to(w, se.shape[-1:])
        num = jnp.sum(se * w, axis=tuple(range(1, se.ndim)))
        den = jnp.maximum(jnp.sum(w), 1e-8) * se.shape[1]
        return num / den

    def _compute_loss(self, x0: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray, key_aug: jax.Array | None = None) -> jnp.ndarray:
        return jnp.mean(self._compute_loss_per_example(x0, cond1, cond2, cond2_mask, noise, t, key_aug=key_aug))

    def _loss_impl(self, x0: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, rng: jax.Array, per_example: bool = False, denoise_fn=None, time_weight: jnp.ndarray | None = None) -> jnp.ndarray:
        batch = x0.shape[0]
        key_t, key_noise = jax.random.split(rng)
        # `fold_in` branches a THIRD stream off `rng` without consuming it, so the
        # `t` and `eps` draws above are bit-for-bit what they were before
        # distortion/self-conditioning existed -- which is what makes the
        # flags-off path a usable equivalence reference.
        key_aug = jax.random.fold_in(rng, 0x5C0D)
        t = jax.random.randint(key_t, (batch,), 0, self.timesteps, dtype=jnp.int32)
        noise = jax.random.normal(key_noise, x0.shape, dtype=jnp.float32)
        per = self._compute_loss_per_example(x0, cond1, cond2, cond2_mask, noise, t, denoise_fn=denoise_fn, time_weight=time_weight, key_aug=key_aug)
        return per if per_example else jnp.mean(per)

    def _resample_impl(self, proposals: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, n_timesteps: int, rng: jax.Array, noise_scale: float = 1.0, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> jnp.ndarray:
        batch_size = proposals.shape[0]
        key_q, key_steps = jax.random.split(rng)
        q_noise = jax.random.normal(key_q, proposals.shape, dtype=jnp.float32) * noise_scale
        t0 = jnp.full((batch_size,), n_timesteps - 1, dtype=jnp.int32)
        x = self.q_sample(proposals, t0, noise=q_noise)
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, n_timesteps)

        def body(i: int, carry) -> tuple:
            x_curr, sc, key = carry
            timestep = n_timesteps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, proposals.shape, dtype=jnp.float32)
            eff_temp = temp_sched[i]
            x_next, x0_pred = self.p_sample_step(
                x_curr, t, cond1, cond2, cond2_mask, noise, eta=eta, sampling_temp=eff_temp, x_sc=sc)
            return x_next, self._next_sc(x0_pred), key

        sc0 = jnp.zeros(proposals.shape, dtype=jnp.float32) if self.self_cond else None
        x, _sc, _ = jax.lax.fori_loop(0, n_timesteps, body, (x, sc0, key_steps))
        return x

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        return self._sample_impl(shape, cond1, cond2, cond2_mask, rng, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def loss(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
    ) -> jnp.ndarray:
        return self._loss_impl(x0, cond1, cond2, cond2_mask, rng)

    def loss_per_example(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        denoise_fn=None,
        time_weight: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Diffusion loss per batch element, ``[B]`` (mean over channels + time).

        ``loss(...) == loss_per_example(...).mean()`` for the same rng, so this is
        a drop-in for reward-weighted / filtered fine-tuning where each example
        carries a different weight. ``denoise_fn`` substitutes the denoiser (e.g.
        the DPPO-trainable copy for an expert-anchor minibatch). ``time_weight``
        is an optional ``[T]`` weight over the horizon (see
        :meth:`_compute_loss_per_example`); ``None`` scores the whole horizon.
        """
        return self._loss_impl(x0, cond1, cond2, cond2_mask, rng, per_example=True, denoise_fn=denoise_fn, time_weight=time_weight)

    # ----------------------------------------------------------------------- #
    # RL / DPPO support: reverse-transition statistics, log-probabilities, and
    # a sampler that records the last K' denoising transitions.
    # ----------------------------------------------------------------------- #
    _LOG_2PI = float(math.log(2.0 * math.pi))

    def _posterior_mean_std(
        self, x_t: jnp.ndarray, t: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray, denoise_fn=None, x_sc: jnp.ndarray | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """DDPM reverse-transition mean and std at step ``t``.

        Returns ``(mean, std, x0_pred)`` where ``x_{t-1} ~ N(mean, std**2 I)``.
        The mean is the posterior mean ``coef1*x0 + coef2*x_t`` (independent of any
        exploration-noise floor), and ``std = sqrt(posterior_variance_t)`` (0 at
        ``t=0``; callers floor it for likelihoods). Equivalent to the eta=1 DDIM
        update the default sampler uses, so likelihoods are consistent with it.
        """
        mean, var, x0_pred = self.p_mean_variance(x_t, t, cond1, cond2, cond2_mask, denoise_fn=denoise_fn, x_sc=x_sc)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        return mean, std, x0_pred

    @staticmethod
    def _gaussian_logdensity(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        """Elementwise diagonal-Gaussian log density (same shape as ``x``)."""
        return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - 0.5 * GaussianDiffusion._LOG_2PI

    @staticmethod
    def _reduce_logdensity(ld: jnp.ndarray, prefix_len: int | None) -> jnp.ndarray:
        """Average a per-element log density over channels and (executed) time.

        ``ld`` is ``[B, C, T]``. We average over channels and time rather than
        summing all C*T dims, so the PPO ratio stays a sane per-transition scalar
        (DPPO note: "do not sum all 125 dimensions into one enormous ratio").
        With ``prefix_len`` set, only the first ``prefix_len`` time points (the
        executed prefix that receives policy-gradient credit) are averaged.
        """
        c = ld.shape[1]
        t_dim = ld.shape[2]
        if prefix_len is None:
            return jnp.mean(ld, axis=(1, 2))
        pl = int(prefix_len)
        mask_t = (jnp.arange(t_dim) < pl).astype(ld.dtype)  # [T]
        summed = jnp.sum(ld * mask_t[None, None, :], axis=(1, 2))
        return summed / (c * pl)

    def transition_log_prob(
        self,
        x_t: jnp.ndarray,
        x_prev: jnp.ndarray,
        t: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        sigma_floor: float = 0.1,
        prefix_len: int | None = None,
        denoise_fn=None,
        x_sc: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Log-prob of the reverse transition ``x_t -> x_prev`` at step ``t``.

        Under self-conditioning the transition is ALSO a function of the previous
        step's clean prediction, so ``x_sc`` must be the value the sampler used --
        it is recorded in the trace as ``x_sc`` for exactly this reason. Passing
        the wrong one (or ``None``) silently evaluates a different policy.

        Converts the current prediction to ``x0`` and then to the posterior mean
        (see :meth:`_posterior_mean_std`), floors the std at ``sigma_floor`` for a
        stable likelihood, and averages the Gaussian log density over channels and
        the executed time prefix. Returns ``[B]``.
        """
        mean, std, _ = self._posterior_mean_std(x_t, t, cond1, cond2, cond2_mask, denoise_fn=denoise_fn, x_sc=x_sc)
        std_lp = jnp.maximum(std, sigma_floor)
        ld = self._gaussian_logdensity(x_prev, mean, std_lp)
        return self._reduce_logdensity(ld, prefix_len)

    def sample_with_trace(
        self,
        shape: Tuple[int, int, int],
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        k_trainable: int,
        trainable_denoise_fn=None,
        rl_mode: bool = False,
        sigma_sample_floor: float = 0.0,
        sigma_logprob_floor: float = 0.1,
        prefix_len: int | None = None,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
        num_sample_steps: int | None = None,
        scan_late: bool = True,
        x_prefix: jnp.ndarray | None = None,
        key_late: jax.Array | None = None,
        return_prefix: bool = False,
    ):
        """Sample and record the last ``k_trainable`` denoising transitions.

        The early ``timesteps - k_trainable`` steps run in a compact ``fori_loop``
        with the frozen ``self.denoise_fn`` (the default sampler math). The final
        ``k_trainable`` steps (diffusion timesteps ``k_trainable-1 .. 0``) are
        unrolled so each transition is recorded, using ``trainable_denoise_fn`` if
        given (the DPPO-trainable copy) else ``self.denoise_fn``.

        ``num_sample_steps`` optionally DDIM-subsamples the *frozen* early portion:
        with ``num_sample_steps < timesteps`` the ``timesteps - k`` unit steps that
        bring pure noise down to ``x_{k-1}`` are replaced by ``num_sample_steps - k``
        strided DDIM steps over the same span (fewer denoiser evals -> cheaper
        collection). The ``k`` trainable late steps are always the unit-step chain
        ``k-1 .. 0``, so the recorded transitions and the PPO likelihood are
        unchanged. ``None`` (default) keeps the full unit-step chain and is
        bit-identical to the previous sampler.

        With ``rl_mode=False`` and default settings the produced trajectory is
        bit-identical to :meth:`sample` under the same rng (Gate 2). With
        ``rl_mode=True`` the late steps are stochastic Gaussian transitions with an
        exploration-std floor -- including at ``t=0``, where the default sampler
        suppresses noise -- matching DPPO's training sampler.

        ``scan_late`` (default True) runs the ``rl_mode`` late chain as one compiled
        ``lax.scan`` instead of ``k`` eager steps; set False to force the unrolled
        path (tests use this to check the two are numerically identical).

        Returns ``(trajectory, trace)`` where ``trace`` is a dict of arrays stacked
        over the ``k_trainable`` steps (leading axis K, ordered high-t -> t=0):
        ``x_t, x_prev, timestep, mean, std, log_prob``.

        The early steps use the frozen ``self.denoise_fn`` only, so their output is
        a function of ``rng`` alone. ``return_prefix=True`` appends ``(x_{k-1},
        key_late)`` to the return; passing those back as ``x_prefix``/``key_late``
        skips the early steps and reproduces the same late chain -- a gradient
        pass then pays ``k`` denoiser evals instead of ``timesteps``.

        Under ``self_cond`` the late chain also needs the prefix's last clean
        prediction, so ``x_prefix`` becomes the PAIR ``(x_{k-1}, x0_hat_{k})``
        instead of a bare array. It stays a single opaque pytree argument, so every
        caller that just stores and hands it back is unaffected; only code that
        indexes into it would have to change, and nothing does.
        """
        batch_size = shape[0]
        num_steps = self.timesteps
        k = int(k_trainable)
        if k < 0 or k > num_steps:
            raise ValueError(f"k_trainable must be in [0, {num_steps}], got {k}.")
        n_early = num_steps - k

        S = num_steps if num_sample_steps is None else int(num_sample_steps)
        if not (k <= S <= num_steps):
            raise ValueError(f"num_sample_steps must be in [{k}, {num_steps}], got {S}.")
        stride_frozen = S != num_steps

        temp_sched = self._temp_schedule(sampling_temp, temp_mode, num_steps)
        if x_prefix is not None:
            if self.self_cond:
                x, sc = x_prefix
            else:
                x, sc = x_prefix, None
            key = key_late
        else:
            key_x, key_steps = jax.random.split(rng)
            x = jax.random.normal(key_x, shape, dtype=jnp.float32)
            sc = jnp.zeros(shape, dtype=jnp.float32) if self.self_cond else None

        # --- early (frozen) steps: compact loop, no trace --------------------- #
        if x_prefix is not None:
            pass
        elif not stride_frozen:
            # Full unit-step chain (t = num_steps-1 .. k), bit-identical to before.
            def body(i, carry):
                x_curr, sc_curr, key = carry
                timestep = num_steps - 1 - i
                t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
                key, step_key = jax.random.split(key)
                noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
                x_next, x0_pred = self.p_sample_step(
                    x_curr, t, cond1, cond2, cond2_mask, noise,
                    eta=eta, sampling_temp=temp_sched[i], x_sc=sc_curr,
                )
                return x_next, self._next_sc(x0_pred), key

            x, sc, key = jax.lax.fori_loop(0, n_early, body, (x, sc, key_steps))
        else:
            # DDIM-subsampled frozen prefix: n_frozen strided steps from the noisiest
            # timestep down to x_{k-1} (the input the unit-step late region expects).
            if temp_mode != "uniform":
                raise ValueError("num_sample_steps striding requires temp_mode='uniform'.")
            n_frozen = S - k
            # `from_ts` descends from num_steps-1 to k; each step targets the next
            # `from` (and the last targets k-1), so the prefix ends exactly at x_{k-1}.
            hi, lo = num_steps - 1, k
            denom = max(n_frozen - 1, 1)
            from_ts = [int(round(hi - (hi - lo) * j / denom)) for j in range(n_frozen)]
            if len(set(from_ts)) != n_frozen:
                raise ValueError(
                    f"num_sample_steps={S} (k={k}) yields duplicate strided timesteps; "
                    f"pick a smaller value so the {n_frozen} frozen steps are distinct.")
            to_ts = from_ts[1:] + [k - 1]
            from_arr = jnp.asarray(from_ts, dtype=jnp.int32)
            to_arr = jnp.asarray(to_ts, dtype=jnp.int32)

            def fbody(i, carry):
                x_curr, sc_curr, key = carry
                from_t = jnp.full((batch_size,), from_arr[i], dtype=jnp.int32)
                to_t = jnp.full((batch_size,), to_arr[i], dtype=jnp.int32)
                key, step_key = jax.random.split(key)
                noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
                x_next, x0_pred = self._ddim_step_strided(
                    x_curr, from_t, to_t, cond1, cond2, cond2_mask, noise,
                    eta=eta, sampling_temp=sampling_temp, x_sc=sc_curr,
                )
                return x_next, self._next_sc(x0_pred), key

            x, sc, key = jax.lax.fori_loop(0, n_frozen, fbody, (x, sc, key_steps))

        prefix_out = ((x, sc) if self.self_cond else x, key)

        # --- late (trainable) steps: record transitions ---------------------- #
        # Collection (rl_mode) runs the sequential late chain as a compiled
        # lax.scan -- one fused program instead of k eager denoiser dispatches,
        # which dominates collect time once the frozen prefix is subsampled. The
        # scan captures `trainable_denoise_fn` the same way the frozen fori_loop
        # captures `self.denoise_fn`. The deterministic (rl_mode=False) path keeps
        # the unrolled math below so it stays bit-identical to sample() (Gate 2).
        if rl_mode and k > 0 and scan_late:
            ts = jnp.arange(k - 1, -1, -1, dtype=jnp.int32)  # timesteps k-1 .. 0

            def late_body(carry, timestep):
                x_in, sc_in, key = carry
                t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
                key, step_key = jax.random.split(key)
                noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
                mean, _var, _x0 = self.p_mean_variance(
                    x_in, t, cond1, cond2, cond2_mask, denoise_fn=trainable_denoise_fn, x_sc=sc_in)
                std = jnp.sqrt(jnp.maximum(_var, 0.0))
                # Exploration noise on every trainable step, including t=0.
                std_s = jnp.maximum(std, sigma_sample_floor)
                x_out = mean + std_s * noise
                std_lp = jnp.maximum(std, sigma_logprob_floor)
                lp = self._reduce_logdensity(
                    self._gaussian_logdensity(x_out, mean, std_lp), prefix_len)
                # `sc_in`, not the new prediction, is recorded: it is the input
                # this transition was computed from, and re-evaluating the
                # log-prob needs exactly that.
                return (x_out, self._next_sc(_x0), key), (x_in, x_out, t, mean, std_s, lp, sc_in)

            (x, _sc, _key), (xt_s, xprev_s, t_s, mean_s, std_s, lp_s, sc_s) = jax.lax.scan(
                late_body, (x, sc, key), ts)
            trace = {"x_t": xt_s, "x_prev": xprev_s, "timestep": t_s,
                     "mean": mean_s, "std": std_s, "log_prob": lp_s}
            if self.self_cond:
                trace["x_sc"] = sc_s
            return (x, trace, prefix_out) if return_prefix else (x, trace)

        # --- late (trainable) steps: unrolled, record transitions ------------ #
        xt_l, xprev_l, t_l, mean_l, std_l, lp_l, sc_l = [], [], [], [], [], [], []
        for j in range(k):
            i = n_early + j
            timestep = num_steps - 1 - i  # runs k-1 .. 0
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
            x_in = x

            # ONE denoiser call per step; both the update and the recorded
            # posterior stats derive from this single prediction (calling the
            # network twice would perturb the trajectory via its nnx rng streams).
            sc_in = sc
            mean, _var, x0_pred = self.p_mean_variance(
                x_in, t, cond1, cond2, cond2_mask, denoise_fn=trainable_denoise_fn, x_sc=sc_in
            )
            sc = self._next_sc(x0_pred)
            std = jnp.sqrt(jnp.maximum(_var, 0.0))

            if rl_mode:
                std_s = jnp.maximum(std, sigma_sample_floor)
                # Exploration noise on every trainable step, including t=0 (the
                # default sampler suppresses it there); DPPO relies on this.
                x_out = mean + std_s * noise
                rec_std = std_s
            else:
                # Standard sampler math (bit-identical to sample()).
                x_out = self._ddim_update(x_in, t, x0_pred, noise, eta=eta, sampling_temp=temp_sched[i])
                rec_std = std

            std_lp = jnp.maximum(std, sigma_logprob_floor)
            lp = self._reduce_logdensity(
                self._gaussian_logdensity(x_out, mean, std_lp), prefix_len
            )

            xt_l.append(x_in); xprev_l.append(x_out); t_l.append(t)
            mean_l.append(mean); std_l.append(rec_std); lp_l.append(lp)
            sc_l.append(sc_in)
            x = x_out

        def _stack(lst, empty_shape):
            return jnp.stack(lst, axis=0) if lst else jnp.zeros(empty_shape, dtype=jnp.float32)

        trace = {
            "x_t": _stack(xt_l, (0,) + tuple(shape)),
            "x_prev": _stack(xprev_l, (0,) + tuple(shape)),
            "timestep": jnp.stack(t_l, axis=0) if t_l else jnp.zeros((0, batch_size), jnp.int32),
            "mean": _stack(mean_l, (0,) + tuple(shape)),
            "std": _stack(std_l, (0,) + tuple(shape)),
            "log_prob": jnp.stack(lp_l, axis=0) if lp_l else jnp.zeros((0, batch_size), jnp.float32),
        }
        if self.self_cond:
            trace["x_sc"] = _stack(sc_l, (0,) + tuple(shape))
        return (x, trace, prefix_out) if return_prefix else (x, trace)

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
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        if n_timesteps <= 0 or n_timesteps > self.timesteps:
            raise ValueError(f"n_timesteps must be in [1, {self.timesteps}], got {n_timesteps}")
        return self._resample_impl(proposals, cond1, cond2, cond2_mask, n_timesteps, rng,
                                   noise_scale=noise_scale, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)