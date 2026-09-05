from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .modules.attention import CrossAttention


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    t = torch.as_tensor(t)
    if not torch.is_floating_point(t):
        t = t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _safe_num_groups(num_channels: int, max_groups: int) -> int:
    return max(1, math.gcd(num_channels, max_groups))


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, groups: int = 8) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding)
        self.gn = nn.GroupNorm(num_groups=_safe_num_groups(out_ch, groups), num_channels=out_ch, eps=1e-5)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        h = self.conv(x_btc.transpose(1, 2)).transpose(1, 2)
        h = self.gn(h.transpose(1, 2)).transpose(1, 2)
        return F.silu(h)


class Identity1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, cond_ch: int, hidden_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.time = nn.Conv1d(time_ch, in_ch, kernel_size=1)
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel=3, groups=groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel=3, groups=groups)
        self.skip = Identity1d() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.cross_attn = CrossAttention(query_dim=out_ch, context_dim=cond_ch, num_heads=8, hidden_dim=hidden_dim)

    def forward(self, x_btc: torch.Tensor, time_feat_b1c: torch.Tensor, context_bnc: torch.Tensor) -> torch.Tensor:
        time_b1c = self.time(time_feat_b1c.transpose(1, 2)).transpose(1, 2)
        h = x_btc + time_b1c
        h = self.block1(h)
        h = self.block2(h)
        context_mask_bn = torch.ones((context_bnc.shape[0], context_bnc.shape[1]), dtype=torch.bool, device=context_bnc.device)
        h = h + self.cross_attn(h, context_bnc, context_mask_bn)

        if isinstance(self.skip, Identity1d):
            return h + x_btc
        skip = self.skip(x_btc.transpose(1, 2)).transpose(1, 2)
        return h + skip


class Downsample1d(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        return self.conv(x_btc.transpose(1, 2)).transpose(1, 2)


class Upsample1d(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_btc.transpose(1, 2), scale_factor=2.0, mode="nearest")
        return self.conv(x).transpose(1, 2)


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, in_feat: torch.Tensor) -> torch.Tensor:
        h = self.fc1(in_feat)
        h = F.silu(h)
        return self.fc2(h)


class FiLMBlock(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * in_ch)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x_btc: torch.Tensor, cond_bc: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        if cond_bc is None:
            return x_btc
        gamma_beta = self.fc2(F.silu(self.fc1(cond_bc)))
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        if cond_mask is None:
            mask = torch.ones((x_btc.shape[0],), device=x_btc.device, dtype=x_btc.dtype)
        else:
            mask = cond_mask.to(dtype=x_btc.dtype)
        return x_btc * (1 + gamma[:, None, :] * mask[:, None, None]) + beta[:, None, :] * mask[:, None, None]


class UNet1DConditioned(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        cond2_dim: int = 256,
        time_emb_dim: int = 128,
        hidden_dim: int = 512,
        groups: int = 8,
        horizon: int = 25,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.n_down = len(ch_mult)
        self.horizon = horizon

        self.cond_proj = nn.Linear(cond_dim, base_ch)
        self.time_proj = Projector(time_emb_dim, base_ch, hidden_dim=hidden_dim)
        self.stem = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)

        ch = base_ch
        down_blocks = []
        downsamples = []
        skip_channels = []
        down_film_blocks = []

        for i, m in enumerate(ch_mult):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                down_blocks.append(ResBlock1d(ch, out_ch, base_ch, base_ch, hidden_dim=hidden_dim, groups=groups))
                down_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim))
                ch = out_ch
            skip_channels.append(ch)
            downsamples.append(Downsample1d(ch) if i != len(ch_mult) - 1 else Identity1d())

        self.mid_block1 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, groups=groups)
        self.mid_film_blocks = nn.ModuleList([FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim), FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim)])
        self.mid_block2 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, groups=groups)

        up_blocks = []
        upsamples = []
        up_film_blocks = []
        for i, m in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                up_blocks.append(ResBlock1d(ch + skip_channels[i], out_ch, base_ch, base_ch, hidden_dim=hidden_dim, groups=groups))
                up_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim))
                ch = out_ch
            upsamples.append(Upsample1d(ch) if i != 0 else Identity1d())

        self.down_blocks = nn.ModuleList(down_blocks)
        self.downsamples = nn.ModuleList(downsamples)
        self.skip_channels = tuple(skip_channels)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.upsamples = nn.ModuleList(upsamples)
        self.down_film_blocks = nn.ModuleList(down_film_blocks)
        self.up_film_blocks = nn.ModuleList(up_film_blocks)
        self.out_norm = nn.GroupNorm(num_groups=_safe_num_groups(ch, groups), num_channels=ch, eps=1e-5)
        self.out_conv = nn.Conv1d(ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor = None, c2_mask: torch.Tensor = None) -> torch.Tensor:
        divisor = 2 ** (self.n_down - 1)
        if self.horizon % divisor != 0:
            target_t = ((self.horizon // divisor) + 1) * divisor
            if x.shape[-1] < target_t:
                x = F.pad(x, (0, target_t - x.shape[-1]))
            else:
                x = x[..., :target_t]

        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        c1 = self.cond_proj(c1)
        time_feat = self.time_proj(t_emb)[:, None, :]

        h = self.stem(x).transpose(1, 2)

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
                h = torch.cat([h, skip], dim=-1)
                h = self.up_blocks[rb_up](h, time_feat, c1)
                h = self.up_film_blocks[rb_up](h, c2, c2_mask)
                rb_up += 1

            h = self.upsamples[i](h)

        h = self.out_norm(h.transpose(1, 2)).transpose(1, 2)
        h = F.silu(h)
        h = self.out_conv(h.transpose(1, 2)).transpose(1, 2)
        return h[:, : self.horizon, :].transpose(1, 2)


def make_beta_schedule(timesteps: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0, 0.999).float()
    raise ValueError(f"Unknown schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        timesteps: int = 100,
        beta_schedule: str = "cosine",
        predict_type: str = "v",
        clip_denoised: bool = False,
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = int(timesteps)
        self.predict_type = predict_type
        self.clip_denoised = clip_denoised

        betas = make_beta_schedule(self.timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_var, min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(x_shape[0], 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / sqrt_ab

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond1: torch.Tensor,
        cond2: torch.Tensor = None,
        cond2_mask: torch.Tensor = None,
        guidance_scale: float = 1.0,
    ):
        pred = self._guided_denoise(x_t, t, cond1, cond2, cond2_mask, guidance_scale=guidance_scale)

        if self.predict_type == "eps":
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
        elif self.predict_type == "mu":
            x0_pred = pred
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            x0_pred = sqrt_ab * x_t - sqrt_1mab * pred
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        if self.clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0_pred

    def _guided_denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond1: torch.Tensor,
        cond2: torch.Tensor,
        cond2_mask: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        cond_pred = self.denoise_fn(x_t, t, cond1, cond2, cond2_mask)
        if cond2 is None or guidance_scale == 1.0:
            return cond_pred

        uncond_pred = self.denoise_fn(x_t, t, cond1, None, None)
        return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    @staticmethod
    def _temp_schedule(sampling_temp: float, temp_mode: str, num_steps: int) -> torch.Tensor:
        t_idx = torch.arange(num_steps, dtype=torch.float32)
        denom = max(num_steps - 1, 1)
        if temp_mode == "early":
            return 1.0 + (sampling_temp - 1.0) * (1.0 - t_idx / denom)
        if temp_mode == "late":
            return 1.0 + (sampling_temp - 1.0) * (t_idx / denom)
        return torch.full((num_steps,), sampling_temp, dtype=torch.float32)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond1: torch.Tensor,
        cond2: torch.Tensor,
        cond2_mask: torch.Tensor,
        noise: torch.Tensor,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        pred = self._guided_denoise(x_t, t, cond1, cond2, cond2_mask, guidance_scale=guidance_scale)

        if self.predict_type == "eps":
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
        elif self.predict_type == "mu":
            x0_pred = pred
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            x0_pred = sqrt_ab * x_t - sqrt_1mab * pred
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_1mab

        eps_pred = eps_pred * sampling_temp
        x0_pred = (x_t - sqrt_1mab * eps_pred) / sqrt_ab

        post_var = self._extract(self.posterior_variance, t, x_t.shape)
        sigma = eta * torch.sqrt(post_var)
        abar_prev = self._extract(self.alphas_cumprod_prev, t, x_t.shape)
        dir_coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma ** 2, min=0.0))

        nonzero_mask = (t != 0).to(dtype=x_t.dtype).reshape(-1, 1, 1)
        return torch.sqrt(abar_prev) * x0_pred + dir_coeff * eps_pred + nonzero_mask * sigma * noise

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond1: torch.Tensor,
        cond2: torch.Tensor = None,
        cond2_mask: torch.Tensor = None,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        batch_size = shape[0]
        device = cond1.device
        x_init = torch.randn(shape, device=device, dtype=torch.float32)
        num_steps = self.timesteps
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, num_steps).to(device=device)

        x_curr = x_init
        for i in range(num_steps):
            timestep = num_steps - 1 - i
            t = torch.full((batch_size,), timestep, dtype=torch.long, device=device)
            noise = torch.randn(shape, device=device, dtype=torch.float32)
            x_curr = self.p_sample(
                x_curr,
                t,
                cond1,
                cond2,
                cond2_mask,
                noise,
                eta=eta,
                sampling_temp=float(temp_sched[i].item()),
                guidance_scale=guidance_scale,
            )
        return x_curr

    def _compute_loss(self, x0: torch.Tensor, cond1: torch.Tensor, cond2: torch.Tensor, cond2_mask: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_t = self.q_sample(x0, t, noise=noise)
        pred = self.denoise_fn(x_t, t, cond1, cond2, cond2_mask)

        if self.predict_type == "eps":
            return torch.mean((pred - noise) ** 2)
        if self.predict_type == "mu":
            return torch.mean((pred - x0) ** 2)
        if self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
            v = sqrt_ab * noise - sqrt_1mab * x0
            return torch.mean((pred - v) ** 2)
        raise ValueError(f"Unsupported predict_type: {self.predict_type}")

    def loss(
        self, 
        x0: torch.Tensor, 
        cond1: torch.Tensor, 
        cond2: torch.Tensor = None, 
        cond2_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.timesteps, (batch,), device=device, dtype=torch.long)
        noise = torch.randn(x0.shape, device=device, dtype=x0.dtype)
        return self._compute_loss(x0, cond1, cond2, cond2_mask, noise, t)

    def resample(
        self,
        proposals: torch.Tensor,
        cond1: torch.Tensor,
        cond2: torch.Tensor = None,
        cond2_mask: torch.Tensor = None,
        n_timesteps: int = 3,
        noise_scale: float = 1.0,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        batch_size = proposals.shape[0]
        device = proposals.device
        q_noise = torch.randn(proposals.shape, device=device, dtype=proposals.dtype) * noise_scale
        t0 = torch.full((batch_size,), n_timesteps - 1, dtype=torch.long, device=device)
        x = self.q_sample(proposals, t0, noise=q_noise)
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, n_timesteps).to(device=device)

        x_curr = x
        for i in range(n_timesteps):
            timestep = n_timesteps - 1 - i
            t = torch.full((batch_size,), timestep, dtype=torch.long, device=device)
            noise = torch.randn(proposals.shape, device=device, dtype=proposals.dtype)
            x_curr = self.p_sample(
                x_curr,
                t,
                cond1,
                cond2,
                cond2_mask,
                noise,
                eta=eta,
                sampling_temp=float(temp_sched[i].item()),
                guidance_scale=guidance_scale,
            )
        return x_curr