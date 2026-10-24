"""Gated tail adapter (rl/diffusion_ft/adapter.py): sampler identity, gate, gradients.

Tiny synthetic UNet, CPU, seconds. What each check pins, and the silent failure it
would otherwise be:

* **hook is inert.** ``tail_hook=None`` and an identity hook give the same output:
  the adapter path does not perturb the base by merely existing.
* **noise replay.** ``late_noise(key_late)`` + ``base_steps`` from the prefix lands
  on ``sample_with_trace``'s own rl_mode sample. Wrong noise = every pair trains on
  a draw the sampler never made.
* **zero init / strength 0.** Fresh adapter (any gate) and a PERTURBED adapter at
  strength 0 reproduce the base sample.
* **gate once.** The in-chain gate equals the gate from precomputed frozen
  features (what training uses), and passing it in gives the same trajectory.
* **gradients.** Gate open, a fresh adapter: the up-projections get finite nonzero
  gradients, the down-projections and the gate get exactly zero.
* **groups.** ``train_adapter.param_group`` separates adapters from gate.
* **distance.** ``traj_distance`` with the full window equals
  ``train_safety.preserve_loss``.
* **edit coords** recover an ``edit_world`` edit on a straight path.
* **checkpoint** round trip; ``parse_arms`` adapter tokens.

    python -m rl.diffusion_ft.tests.test_adapter
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model.diffusion.diffusion import GaussianDiffusion, UNet1DConditioned
from rl.diffusion_ft import adapter as A

FAILED = []
C, H, CD, STEPS, KTR, K = 5, 25, 16, 20, 6, 3
FLOOR = 1e-4


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _diffusion():
    unet = UNet1DConditioned(in_ch=C, cond_dim=CD, cond2_dim=CD, time_emb_dim=CD, horizon=H,
                             hidden_dim=32, base_ch=16, rngs=nnx.Rngs(0))
    return GaussianDiffusion(unet, timesteps=STEPS, predict_type="v")


def _cond(B=4, seed=1):
    ks = jax.random.split(jax.random.PRNGKey(seed), 2)
    return (jax.random.normal(ks[0], (B, 4, CD)), jax.random.normal(ks[1], (B, CD)),
            jnp.ones((B,), jnp.float32))


def _perturb(module, scale=0.3, seed=5):
    gdef, params, rest = A.split_params(module)
    leaves, td = jax.tree_util.tree_flatten(params)
    ks = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    params = jax.tree_util.tree_unflatten(
        td, [l + scale * jax.random.normal(k, l.shape) for l, k in zip(leaves, ks)])
    return nnx.merge(gdef, params, rest)


def main():
    diff = _diffusion()
    unet = diff.denoise_fn
    c1, c2, m = _cond()
    B = c1.shape[0]
    shape = (B, C, H)

    # --- hook is inert -------------------------------------------------------
    x = jax.random.normal(jax.random.PRNGKey(3), shape)
    t = jnp.full((B,), 2, jnp.int32)
    out0 = unet(x, t, c1, c2, m)
    out1 = unet(x, t, c1, c2, m, tail_hook=lambda j, h, tf: h)
    check(bool(jnp.array_equal(out0, out1)), "tail_hook identity == no hook (bitwise)")
    seen = []
    unet(x, t, c1, c2, m, tail_hook=lambda j, h, tf: (seen.append((j, h.shape, tf.shape)), h)[1])
    check([s[0] for s in seen][-2:] == [-2, -1] and seen[-1][1][-1] == 16 and seen[-1][2] == (B, 16),
          f"hook sees the last two up blocks at width base_ch: {seen[-2:]}")

    cfg = A.AdapterConfig.for_unet(unet, K=K, k_trainable=KTR)
    check(cfg.d == 16 and cfg.time_dim == 16 and cfg.horizon == H,
          f"config read off the UNet: d={cfg.d} time_dim={cfg.time_dim}")
    mod = A.GatedTailAdapter(cfg)
    gdef, params, rest = A.split_params(mod)
    ups = [np.asarray(a.up.kernel.value) for a in mod.adapters]
    check(all(np.all(u == 0) for u in ups) and all(np.all(np.asarray(a.up.bias.value) == 0)
                                                  for a in mod.adapters),
          "final projections start at exactly zero")
    want = 2 * (2 * cfg.d + cfg.d * 32 + 32 + 32 * cfg.d + cfg.d) + (2 * cfg.d + cfg.time_dim) * 64 + 64 + 65
    check(A.param_count(params) == want, f"param count {A.param_count(params)} == {want}")

    # --- noise replay against the real sampler --------------------------------
    rng = jax.random.PRNGKey(11)
    x_ref, trace, (x_prefix, key_late) = diff.sample_with_trace(
        shape, c1, c2, m, rng=rng, k_trainable=KTR, rl_mode=True, sigma_sample_floor=FLOOR,
        sigma_logprob_floor=FLOOR, return_prefix=True)
    noise = A.late_noise(key_late, shape, KTR)
    x_rep = A.base_steps(diff, c1, c2, m, x_prefix, noise, t_hi=KTR - 1, sigma_floor=FLOOR)
    e = float(jnp.max(jnp.abs(x_rep - x_ref)))
    check(e < 1e-5, f"late_noise + base_steps replays sample_with_trace's late chain (max {e:.1e})")
    x_tail = A.base_steps(diff, c1, c2, m, x_prefix, noise[:KTR - K], t_hi=KTR - 1, sigma_floor=FLOOR)
    e = float(jnp.max(jnp.abs(x_tail - trace["x_t"][KTR - K])))
    check(e < 1e-5, f"the tail's entry state is the sampler's own x_t at t=K-1 (max {e:.1e})")

    # --- zero init / strength 0 ----------------------------------------------
    for mode in A.GATE_MODES:
        xa, g, lg = A.tail_from_prefix(diff, mod, cfg, c1, c2, m, x_prefix, noise, strength=1.0,
                                       gate_mode=mode, sigma_floor=FLOOR)
        e = float(jnp.max(jnp.abs(xa - x_ref)))
        check(e < 1e-5 and g.shape == (B,), f"fresh adapter, gate {mode}: base sample (max {e:.1e})")
    modp = _perturb(mod)
    xs0, _, _ = A.tail_from_prefix(diff, modp, cfg, c1, c2, m, x_prefix, noise, strength=0.0,
                                   gate_mode="open", sigma_floor=FLOOR)
    e = float(jnp.max(jnp.abs(xs0 - x_ref)))
    check(e < 1e-5, f"perturbed adapter at strength 0: base sample (max {e:.1e})")
    xs1, _, _ = A.tail_from_prefix(diff, modp, cfg, c1, c2, m, x_prefix, noise, strength=1.0,
                                   gate_mode="open", sigma_floor=FLOOR)
    xsc, _, _ = A.tail_from_prefix(diff, modp, cfg, c1, c2, m, x_prefix, noise, strength=1.0,
                                   gate_mode="closed", sigma_floor=FLOOR)
    check(float(jnp.max(jnp.abs(xs1 - x_ref))) > 1e-4 and float(jnp.max(jnp.abs(xsc - x_ref))) < 1e-5,
          "perturbed adapter moves the sample when open, not when closed")
    xh, _, _ = A.tail_from_prefix(diff, modp, cfg, c1, c2, m, x_prefix, noise, strength=0.5,
                                  gate_mode="open", sigma_floor=FLOOR)
    d_half, d_full = float(jnp.mean(jnp.abs(xh - x_ref))), float(jnp.mean(jnp.abs(xs1 - x_ref)))
    check(0.0 < d_half < d_full, f"strength 0.5 moves less than 1 ({d_half:.2e} < {d_full:.2e})")

    # --- the gate is computed once, from frozen features ---------------------
    xg, g_in, lg_in = A.adapted_steps(diff, modp, cfg, c1, c2, m, x_tail, noise[KTR - K:],
                                      strength=1.0, gate_mode="learned", sigma_floor=FLOOR)
    feat = A.first_step_gate_features(diff, cfg, c1, c2, m, x_tail)
    g_pre = jax.nn.sigmoid(modp.gate(feat))
    check(float(jnp.max(jnp.abs(g_pre - g_in))) < 1e-6, "in-chain gate == gate of precomputed features")
    xg2, g2, _ = A.adapted_steps(diff, modp, cfg, c1, c2, m, x_tail, noise[KTR - K:], strength=1.0,
                                 gate=g_pre, sigma_floor=FLOOR)
    check(float(jnp.max(jnp.abs(xg2 - xg))) < 1e-6 and g2 is g_pre,
          "passing the gate in reproduces the in-chain trajectory")
    calls = []
    orig = A.PlanGate.__call__

    def counting(self, f):
        calls.append(1)
        return orig(self, f)
    A.PlanGate.__call__ = counting
    try:
        A.adapted_steps(diff, modp, cfg, c1, c2, m, x_tail, noise[KTR - K:], strength=1.0,
                        gate_mode="learned", sigma_floor=FLOOR)
    finally:
        A.PlanGate.__call__ = orig
    check(len(calls) == 1, f"gate evaluated once per {K}-step tail (got {len(calls)})")

    # --- gradients with the gate open ----------------------------------------
    target = x_ref + 0.05

    def loss(p):
        mm = nnx.merge(gdef, p, rest)
        xo, _, _ = A.adapted_steps(diff, mm, cfg, c1, c2, m, x_tail, noise[KTR - K:], strength=1.0,
                                   gate_mode="open", sigma_floor=FLOOR)
        return jnp.mean((xo - target) ** 2)

    g = jax.grad(loss)(params)
    flat = {"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): np.asarray(v)
            for path, v in jax.tree_util.tree_flatten_with_path(g)[0]}
    up = [np.linalg.norm(v) for k, v in flat.items() if "up" in k]
    down = [np.linalg.norm(v) for k, v in flat.items() if "down" in k]
    gate_g = [np.linalg.norm(v) for k, v in flat.items() if k.startswith("gate")]
    check(len(up) == 4 and all(np.isfinite(u) and u > 0 for u in up),
          f"up-projection grads finite and nonzero: {[f'{u:.2e}' for u in up]}")
    check(all(d == 0 for d in down) and all(x == 0 for x in gate_g),
          "down-projection and gate grads exactly zero at init with the gate open")
    from rl.diffusion_ft.train_adapter import param_group
    groups = {param_group(path) for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]}
    check(groups == {"adapters", "gate"}, f"param groups {groups}")
    check(A.param_count(params["gate"]) == (2 * cfg.d + cfg.time_dim) * 64 + 64 + 65,
          "params['gate'] indexes the gate subtree")

    # --- distance == preserve_loss over the full window ----------------------
    from rl.diffusion_ft.train_safety import preserve_loss
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    wa = jax.random.normal(k1, (3, 12, 5)); wb = wa + 0.1 * jax.random.normal(k2, (3, 12, 5))
    wb = wb.at[..., 2].add(2 * np.pi)          # heading wrap must not count
    sw = jnp.zeros((3, 12)).at[:, 1:12].set(1.0)
    d = A.traj_distance(wa, wb, sw)
    ref, _ = preserve_loss(wa, wb, jnp.ones(3), n=11)
    check(abs(float(jnp.mean(d)) - float(ref)) < 1e-5, f"traj_distance == preserve_loss ({float(jnp.mean(d)):.5f} vs {float(ref):.5f})")

    # --- edit coordinates ------------------------------------------------------
    from rl.diffusion_ft.lateral_opt import edit_world
    W, n, dt = 31, 20, 0.1
    tt = jnp.arange(W) * dt
    yaw0 = 0.4
    world = jnp.stack([10 * tt * np.cos(yaw0), 10 * tt * np.sin(yaw0), jnp.full(W, yaw0),
                       jnp.full(W, 10 * np.cos(yaw0)), jnp.full(W, 10 * np.sin(yaw0))], -1)[None]
    xe = jnp.concatenate([0.3 * jnp.sin(jnp.linspace(0, 1.5, n)), 0.2 * jnp.linspace(0, 1, n)])[None]
    we = edit_world(world, xe, n=n, dt=dt, lon=1)
    rec = A.edit_coords(we, world, n, lon=True)
    e = float(jnp.max(jnp.abs(rec - xe)))
    check(e < 1e-3, f"edit_coords recovers an edit_world edit on a straight path (max {e:.1e} m)")

    # --- checkpoint round trip --------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        _, pp, _ = A.split_params(modp)
        A.save_adapter(tmp, cfg, pp, {"note": "test"})
        cfg2, mod2 = A.load_adapter(tmp)
        _, pp2, _ = A.split_params(mod2)
        same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                   zip(jax.tree_util.tree_leaves(pp), jax.tree_util.tree_leaves(pp2)))
        check(cfg2 == cfg and same, "save_adapter / load_adapter round trip")

    # --- arm parsing ---------------------------------------------------------------
    from rl.diffusion_ft.proposal_select import parse_arms
    arms = parse_arms("base,adp=adapter:learned,adp_open=adapter:open:1,adp_s5=adapter:learned:0.5")
    a = {x.name: x for x in arms}
    check(a["adp"].primary == "adapter" and a["adp"].gate == "learned" and a["adp"].strength == 1.0
          and a["adp_s5"].strength == 0.5 and a["adp_open"].gate == "open" and a["adp"].alt is None
          and not a["adp"].needs_ft, "parse_arms adapter tokens")
    for bad in ("x=adapter:maybe", "x=adapter:learned:1.5", "x=adapter"):
        try:
            parse_arms(bad)
            check(False, f"parse_arms rejects {bad}")
        except SystemExit:
            check(True, f"parse_arms rejects {bad}")

    # --- the trainer's CLI composes with train_safety's (no duplicate flags) ---------
    from rl.diffusion_ft.train_adapter import build_parser
    try:
        args = build_parser().parse_args(["--out_dir", "x", "--pairs", "y"])
        check(args.adapter_repair_weight == 1.0 and args.repair_weight == 0.0
              and args.adapter_K == 5, "train_adapter parser builds; BC --repair_weight stays 0")
    except Exception as e:  # argparse raises ArgumentError on a duplicate flag
        check(False, f"train_adapter parser: {e!r}")

    print(f"\n{len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
