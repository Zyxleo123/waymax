"""Gated tail adapter: the deploy-time repair, distilled into the sampler's last K steps.

WHAT. Two small residual adapters inside the FROZEN denoiser, after the UNet's
last two up blocks (the trajectory-processing blocks at full horizon resolution,
already cross-attended to the scene tokens and FiLM-ed by the instruction), plus
one scalar gate per sampled plan::

    adapter(h) = Linear(32 -> d)( SiLU( Linear(d -> 32)( LayerNorm(h) ) ) )
    h' = h + strength * gate * adapter(h)          # strength in [0, 1], external

The up-projection starts at exactly zero, so at initialization the policy IS the
base policy for any strength and any gate. ``strength = 0`` bypasses every
residual. The adapters are active only in the final ``K`` steps of the deployed
late chain (``DiffusionRLActor`` runs ``k_trainable`` unit steps after the frozen
prefix; the tail is its last ``K``). Sampler, noise schedule, v-parameterization,
conditioning and decoding are the base's.

THE GATE is computed ONCE per plan, at the first adapted step, from the frozen
hidden features immediately before the first residual is added: mean and max over
the horizon of those tokens, concatenated with the timestep feature the UNet
already uses (``time_feat``). The tokens are scene-conditioned already (every
block cross-attends to the scene tokens), so no separate scene summary is added.
``Linear(in, 64) -> SiLU -> Linear(64, 1) -> sigmoid``; the scalar is shared by
both adapters and the whole K-step tail, and recomputed at every replan. It sees
only what the deployed policy sees.

WHY THE TAIL CAN SEE ITS OWN DRAW. Bank BC (docs/bank_bc_distillation.md) failed
because a denoising loss on targets never shows the student which of ITS draws
are bad. Here the student starts from the state the base sampler actually reached
before the last K steps (the frozen prefix is shared, the adapters are off until
then) and replays the same late noise, so the supervised object is the trajectory
the sampler produces for THIS draw, paired with the repair of THIS draw.

RANDOMNESS. ``sample_with_trace`` draws one ``[B, C, T]`` noise tensor per late
step from ``key_late`` (split, then normal). A row's noise therefore depends on its
batch, so pairs store the per-row tensors (:func:`late_noise`), not the key.

See ``train_adapter.py`` for the staged training and ``docs/adapter_distill.md``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

__all__ = ["AdapterConfig", "GatedTailAdapter", "GATE_MODES", "late_noise", "base_steps",
           "adapted_steps", "tail_from_prefix", "gate_features", "traj_distance",
           "edit_coords", "make_repair_objective", "save_adapter", "load_adapter",
           "param_count", "split_params"]

# learned = the trained sigmoid; open = held at 1 (the adapter fully on); closed =
# held at 0 (exactly the base whatever the strength).
GATE_MODES = ("learned", "open", "closed")


@dataclass(frozen=True)
class AdapterConfig:
    d: int = 128             # hidden width at the insertion point (UNet base_ch * ch_mult[0])
    time_dim: int = 128      # width of the UNet's projected timestep feature (base_ch)
    n_blocks: int = 2        # adapters after the last n up blocks
    width: int = 32
    gate_hidden: int = 64
    K: int = 5               # adapted sampling steps: timesteps K-1 .. 0
    k_trainable: int = 10    # the deployed late chain the tail lives in
    horizon: int = 25
    seed: int = 0

    @classmethod
    def for_unet(cls, unet, *, K: int, k_trainable: int, **kw) -> "AdapterConfig":
        """Read the insertion width off the UNet instead of trusting a default."""
        n_blocks = int(kw.pop("n_blocks", 2))
        if n_blocks > int(unet.num_res_blocks):
            raise ValueError(f"n_blocks={n_blocks} spans more than the top up level "
                             f"({unet.num_res_blocks} blocks): widths would differ")
        if not 1 <= int(K) <= int(k_trainable):
            raise ValueError(f"K={K} must lie in [1, k_trainable={k_trainable}]")
        return cls(d=int(unet.out_conv.in_features), time_dim=int(unet.cond_proj.out_features),
                   n_blocks=n_blocks, K=int(K), k_trainable=int(k_trainable),
                   horizon=int(unet.horizon), **kw)


class TailAdapter(nnx.Module):
    """``LayerNorm(d) -> Linear(d, w) -> SiLU -> Linear(w, d)``, last layer zero."""

    def __init__(self, d: int, width: int, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(d, rngs=rngs)
        self.down = nnx.Linear(d, width, rngs=rngs)
        self.up = nnx.Linear(width, d, kernel_init=nnx.initializers.zeros_init(),
                             bias_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, h):
        return self.up(nnx.silu(self.down(self.norm(h))))


class PlanGate(nnx.Module):
    """``Linear(in, 64) -> SiLU -> Linear(64, 1)``: the gate's LOGIT, ``[B]``."""

    def __init__(self, in_dim: int, hidden: int, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, feat):
        return self.l2(nnx.silu(self.l1(feat)))[..., 0]


class GatedTailAdapter(nnx.Module):
    def __init__(self, cfg: AdapterConfig, *, rngs: nnx.Rngs | None = None):
        rngs = nnx.Rngs(int(cfg.seed)) if rngs is None else rngs
        self.adapters = nnx.List([TailAdapter(cfg.d, cfg.width, rngs=rngs)
                                  for _ in range(cfg.n_blocks)])
        self.gate = PlanGate(2 * cfg.d + cfg.time_dim, cfg.gate_hidden, rngs=rngs)


def split_params(module: GatedTailAdapter):
    """``(graphdef, params, rest)``; every parameter is trainable here."""
    return nnx.split(module, nnx.Param, ...)


def param_count(params) -> int:
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(params)))


def gate_features(h, time_feat, horizon: int):
    """``[B, 2d + time_dim]``: mean and max over the (unpadded) horizon tokens, and
    the timestep feature. ``h`` is ``[B, T_pad, d]``; the UNet pads 25 -> 28."""
    hh = h[:, :int(horizon), :]
    return jnp.concatenate([jnp.mean(hh, axis=1), jnp.max(hh, axis=1), time_feat], axis=-1)


def _gate_from_logit(logit, mode: str):
    if mode == "learned":
        return jax.nn.sigmoid(logit)
    if mode == "open":
        return jnp.ones_like(logit)
    if mode == "closed":
        return jnp.zeros_like(logit)
    raise ValueError(f"gate mode {mode!r} not in {GATE_MODES}")


class _TailHook:
    """``UNet1DConditioned(tail_hook=...)`` for one forward of one adapted step.

    With ``gate=None`` (the first adapted step) the hook computes the gate from the
    hidden state it receives at the first adapter's position, BEFORE adding that
    adapter's residual, and keeps it on ``self.gate`` for the caller to carry.
    Python-side bookkeeping inside one trace only; nothing escapes the transform.
    """

    def __init__(self, module, cfg: AdapterConfig, strength, gate, gate_mode: str):
        self.module, self.n, self.horizon = module, int(cfg.n_blocks), int(cfg.horizon)
        self.strength, self.gate, self.gate_mode = strength, gate, gate_mode
        self.logit = None

    def __call__(self, j, h, time_feat):
        i = int(j) + self.n
        if i < 0:
            return h
        if self.gate is None:
            self.logit = self.module.gate(gate_features(h, time_feat, self.horizon))
            self.gate = _gate_from_logit(self.logit, self.gate_mode)
        scale = jnp.asarray(self.strength, h.dtype) * self.gate
        return h + scale[:, None, None] * self.module.adapters[i](h)


# --------------------------------------------------------------------------- #
# The deployed late chain, split at the tail
# --------------------------------------------------------------------------- #
def late_noise(key_late, shape, k: int):
    """``[k, *shape]``: the noise ``sample_with_trace``'s rl_mode late scan draws from
    ``key_late`` -- ``key, step_key = split(key)``, then ``normal(step_key, shape)`` --
    in step order (timestep k-1 first)."""
    out, key = [], key_late
    for _ in range(int(k)):
        key, sk = jax.random.split(key)
        out.append(jax.random.normal(sk, tuple(shape), dtype=jnp.float32))
    return jnp.stack(out, axis=0)


def _check_diffusion(diffusion):
    if getattr(diffusion, "self_cond", False):
        raise NotImplementedError("the tail adapter does not carry the self-conditioning "
                                  "input; the base checkpoint has self_cond off")


def base_steps(diffusion, cond1, cond2, mask, x, noise_kbct, *, t_hi: int, sigma_floor: float):
    """Frozen unit steps ``t_hi .. t_hi - k + 1`` from ``x`` (the state entering
    ``t_hi``), ``k = noise_kbct.shape[0]``: the rl_mode late scan's math verbatim
    (posterior mean + max(std, floor) * noise, noise on every step)."""
    _check_diffusion(diffusion)
    k = int(noise_kbct.shape[0])
    if k == 0:
        return x
    B = x.shape[0]
    ts = jnp.arange(int(t_hi), int(t_hi) - k, -1, dtype=jnp.int32)

    def body(x_in, inp):
        timestep, noise = inp
        t = jnp.full((B,), timestep, dtype=jnp.int32)
        mean, var, _ = diffusion.p_mean_variance(x_in, t, cond1, cond2, mask)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        return mean + jnp.maximum(std, sigma_floor) * noise, None

    x, _ = jax.lax.scan(body, x, (ts, noise_kbct))
    return x


def adapted_steps(diffusion, module, cfg: AdapterConfig, cond1, cond2, mask, x, noise_kbct, *,
                  strength, gate_mode: str = "learned", sigma_floor: float, gate=None):
    """Timesteps ``K-1 .. 0`` with the adapter in the denoiser, from ``x`` (the state
    entering ``K-1``). Returns ``(x_0, gate [B], logit [B] or None)``.

    ``gate=None`` computes it in-chain at the first step (deployment); passing one
    (training, from precomputed frozen features) skips that. ``strength`` may be a
    traced scalar: at 0 every residual adds exactly 0.
    """
    _check_diffusion(diffusion)
    unet = diffusion.denoise_fn
    B = x.shape[0]
    logit = None
    for j in range(int(cfg.K)):
        hook = _TailHook(module, cfg, strength, gate, gate_mode)

        def fn(xx, tt, c1, c2, m, _h=hook):
            return unet(xx, tt, c1, c2, m, tail_hook=_h)

        t = jnp.full((B,), int(cfg.K) - 1 - j, dtype=jnp.int32)
        mean, var, _ = diffusion.p_mean_variance(x, t, cond1, cond2, mask, denoise_fn=fn)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        x = mean + jnp.maximum(std, sigma_floor) * noise_kbct[j]
        if gate is None:
            gate, logit = hook.gate, hook.logit
    return x, gate, logit


def tail_from_prefix(diffusion, module, cfg: AdapterConfig, cond1, cond2, mask, x_prefix,
                     noise_late, *, strength, gate_mode: str = "learned", sigma_floor: float):
    """The whole deployed late chain from the frozen prefix output ``x_{k-1}``:
    ``k_trainable - K`` frozen steps, then the adapted tail. ``noise_late`` is
    ``[k_trainable, B, C, T]`` (:func:`late_noise` of the sampler's ``key_late``).
    Returns ``(x_0, gate, logit)``."""
    k, K = int(cfg.k_trainable), int(cfg.K)
    x = base_steps(diffusion, cond1, cond2, mask, x_prefix, noise_late[:k - K],
                   t_hi=k - 1, sigma_floor=sigma_floor)
    return adapted_steps(diffusion, module, cfg, cond1, cond2, mask, x, noise_late[k - K:],
                         strength=strength, gate_mode=gate_mode, sigma_floor=sigma_floor)


def first_step_gate_features(diffusion, cfg: AdapterConfig, cond1, cond2, mask, x_tail):
    """Frozen gate inputs for rows whose tail starts at ``x_tail``: the base UNet at
    timestep ``K-1``, read at the first adapter's position. Identical to what the
    in-chain hook computes (the adapters have added nothing before that point)."""
    unet = diffusion.denoise_fn
    box = {}

    def grab(j, h, time_feat):
        if int(j) + int(cfg.n_blocks) == 0:
            box["f"] = gate_features(h, time_feat, cfg.horizon)
        return h

    t = jnp.full((x_tail.shape[0],), int(cfg.K) - 1, dtype=jnp.int32)
    unet(x_tail, t, cond1, cond2, mask, tail_hook=grab)
    return box["f"]


# --------------------------------------------------------------------------- #
# Losses on the final sampled output (world coordinates)
# --------------------------------------------------------------------------- #
def traj_distance(world_bw5, target_bw5, step_w_bw, *, s_pos=1.0, s_yaw=0.1, s_speed=1.0):
    """``[B]`` per-row distance: ``train_safety.preserve_loss``'s per-step formula
    (squared position error / s_pos^2 + squared WRAPPED heading error / s_yaw^2 +
    squared speed error / s_speed^2) as a ``step_w``-weighted mean over steps.
    ``step_w`` carries the scored window (1 on steps 1..n, 0 on the anchor and past
    the window), so rows with different windows share one batch. The target is
    held constant."""
    g = jax.lax.stop_gradient(target_bw5)
    dpos2 = jnp.sum((world_bw5[..., :2] - g[..., :2]) ** 2, axis=-1)
    d = world_bw5[..., 2] - g[..., 2]
    dyaw = jnp.arctan2(jnp.sin(d), jnp.cos(d))
    sp = jnp.sqrt(jnp.sum(world_bw5[..., 3:5] ** 2, axis=-1) + 1e-6)
    sg = jnp.sqrt(jnp.sum(g[..., 3:5] ** 2, axis=-1) + 1e-6)
    per_step = dpos2 / s_pos ** 2 + dyaw ** 2 / s_yaw ** 2 + (sp - sg) ** 2 / s_speed ** 2
    w = jnp.asarray(step_w_bw, per_step.dtype)
    return jnp.sum(per_step * w, axis=1) / jnp.maximum(jnp.sum(w, axis=1), 1.0)


def edit_coords(world_s, world_b, n: int, *, lon: bool = True):
    """``[B, 2n]`` (lateral d, then along-path s) or ``[B, n]``: the student's plan as
    an EDIT of the base plan, in ``lateral_opt``'s coordinates -- the displacement
    at world steps 1..n projected on the base's own heading normal and tangent. The
    linearization of ``lateral_opt.project_onto_path``; exact on a straight path."""
    dxy = world_s[:, 1:n + 1, :2] - world_b[:, 1:n + 1, :2]
    yaw = world_b[:, 1:n + 1, 2]
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    lat = -s * dxy[..., 0] + c * dxy[..., 1]
    if not lon:
        return lat
    return jnp.concatenate([lat, c * dxy[..., 0] + s * dxy[..., 1]], axis=1)


def make_repair_objective(objcfg, opt_cfg, *, n: int, dt: float, neighbor_pred: str = "cv",
                          world_dt_method: str = "median"):
    """``objective(world_s, world_b, state, t0_b) -> (per_row [B], terms)``: the repair
    optimizer's own loss (``lateral_opt.make_lateral_optimizer``) evaluated on a
    GIVEN plan -- the training barrier (with ``opt_cfg``'s shape, edge geometry and
    margins) over world steps 1..n, plus the comfort/closeness quadratic of the
    plan's edit relative to the base plan, ``0.5 x^T H x`` with H = ``edit_hessian``.
    ``t0_b`` is per row (a minibatch mixes replans); ``n`` is static."""
    from data.preprocess import compute_world_dt_seconds
    from rl.diffusion_ft.lateral_opt import edit_hessian
    from rl.diffusion_ft.safety_objective import candidate_safety

    # The same loss config lateral_opt builds for its optimizer.
    margins = {k: v for k, v in (("edge_margin_m", opt_cfg.edge_margin_m),
                                 ("agent_margin_m", opt_cfg.agent_margin_m)) if v is not None}
    loss_cfg = dataclasses.replace(
        objcfg, safety=dataclasses.replace(objcfg.safety, barrier_shape=opt_cfg.barrier_shape,
                                           edge_geometry=opt_cfg.edge_geometry, **margins))
    H = jnp.asarray(edit_hessian(int(n), float(dt), opt_cfg), jnp.float32)

    def objective(world_s, world_b, state, t0_b):
        dt_b = (None if neighbor_pred == "log"
                else compute_world_dt_seconds(state, 0.1, world_dt_method))
        terms = candidate_safety(loss_cfg, world_s, state=state, t0=t0_b, n_steps=int(n),
                                 alive_b=None, neighbor_pred=neighbor_pred, neighbor_dt_b=dt_b)
        x = edit_coords(world_s, jax.lax.stop_gradient(world_b), int(n), lon=bool(opt_cfg.lon))
        barrier = terms["loss_b"] / int(n)
        comfort = 0.5 * jnp.einsum("bi,ij,bj->b", x, H, x)
        return barrier + comfort, {"barrier": barrier, "comfort": comfort}

    return objective


# --------------------------------------------------------------------------- #
# Checkpoints: flattened leaves + the config, train_safety's convention
# --------------------------------------------------------------------------- #
def save_adapter(path, cfg: AdapterConfig, params, extra: dict | None = None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    flat = jax.tree_util.tree_flatten(params)[0]
    # Zero-padded keys: savez names sort lexically on load.
    np.savez(path / "adapter.npz", **{f"p{i:05d}": np.asarray(x) for i, x in enumerate(flat)})
    (path / "adapter.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "n_params": param_count(params),
         **(extra or {})}, indent=2))


def load_adapter(path):
    """``(cfg, module)`` with the saved parameters loaded."""
    path = Path(path)
    meta = json.loads((path / "adapter.json").read_text())
    cfg = AdapterConfig(**meta["config"])
    module = GatedTailAdapter(cfg)
    gdef, params, rest = split_params(module)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    z = np.load(path / "adapter.npz")
    vals = [jnp.asarray(z[k]) for k in sorted(z.files)]
    if len(vals) != len(leaves) or any(v.shape != l.shape for v, l in zip(vals, leaves)):
        raise SystemExit(f"{path}: checkpoint does not match AdapterConfig {cfg}")
    return cfg, nnx.merge(gdef, jax.tree_util.tree_unflatten(treedef, vals), rest)
