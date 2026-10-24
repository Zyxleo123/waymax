"""Gates for the FlowMol3 inference-OOD port: self-conditioning + late-stage distortion.

FlowMol3 (arXiv 2508.12629) attacks a pathology this repo already names in
``rl/diffusion_ft/unrolled_prediction.py``: the sampler's ``x_t`` is whatever the
previous reverse steps produced, which is NOT the ``q_sample(x0, t)`` the network
was trained on. Two train-time changes address it -- the network sees its own
previous clean prediction, and the training state is sparsely kicked off the
forward path in the low-noise half so the network learns to come back.

Everything here runs on a tiny synthetic UNet, no checkpoint, CPU only.

The gates, in the order they matter:

1. flags OFF is the OLD code. `sample` reproduces a hand-written reference loop
   and `loss` a hand-written reference objective, with the reference drawing
   ``(t, eps)`` from ``jax.random.split(rng)`` exactly as the pre-change
   `_loss_impl` did. This is what lets a self-cond run be compared against the
   base at all.
2. a ZERO-EXPANDED self-cond model is the base model. Same rng, bit-identical
   trajectory -- so `--init_from` + `--self_cond` starts a fine-tune at the base
   policy's behaviour instead of at a perturbed one.
3. the new channels are actually wired (a non-zero expansion DOES change output).
4. distortion mechanics: gated by t, sparse at the right rate, positions only,
   and scaled by sqrt(1-abar_t) in relative mode.
5. the distorted TARGET is the corrected one: a network that emits it recovers
   the clean x0 from the distorted state. Getting this backwards would teach the
   model to reproduce the kick instead of undoing it.
6. the self-cond first pass is cut out of the gradient.
7. under self-cond, the recorded trace replays: stored log-probs are reproduced
   from ``trace["x_sc"]`` (PPO ratio 1), and a prefix replay reproduces the chain.

Submit it; do not run it on a login node (the XLA CPU compile gets the process
killed -- observed exit 137)::

    bash rl/diffusion_ft/slurm/submit_test_self_cond.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model.diffusion.diffusion import GaussianDiffusion, UNet1DConditioned

C, T, COND, HID, BASE_CH, STEPS, B = 5, 8, 16, 32, 16, 10, 3


def _unet(seed: int, self_cond: bool) -> UNet1DConditioned:
    return UNet1DConditioned(
        in_ch=C, cond_dim=COND, cond2_dim=COND, time_emb_dim=COND,
        horizon=T, hidden_dim=HID, base_ch=BASE_CH, self_cond=self_cond,
        rngs=nnx.Rngs(seed),
    )


def _cond():
    c1 = jax.random.normal(jax.random.PRNGKey(1), (B, 4, COND))
    c2 = jax.random.normal(jax.random.PRNGKey(2), (B, COND))
    m = jnp.ones((B,), dtype=jnp.float32)
    return c1, c2, m


def _close(a, b, atol=1e-5, rtol=1e-5):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol))


def _exact(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _copy_params_zero_expanded(src: UNet1DConditioned, dst: UNet1DConditioned) -> None:
    """Copy every leaf of ``src`` into ``dst``, zero-filling the stem's new block.

    The in-memory twin of ``rl.diffusion_ft.self_cond_init.expand_params_for_self_cond``
    (which does the same thing to a restored checkpoint tree).
    """
    sg, sp, _ = nnx.split(src, nnx.Param, ...)
    dg, dp, dn = nnx.split(dst, nnx.Param, ...)
    sp_d, dp_d = nnx.to_pure_dict(sp), nnx.to_pure_dict(dp)

    def walk(a, b, path=()):
        for k, v in a.items():
            if isinstance(v, dict):
                walk(v, b[k], path + (k,))
            else:
                arr = np.asarray(v)
                if path and path[-1] == "stem" and k == "kernel":
                    tgt = np.zeros(np.asarray(b[k]).shape, dtype=arr.dtype)
                    tgt[:, : arr.shape[1], :] = arr
                    b[k] = jnp.asarray(tgt)
                else:
                    b[k] = jnp.asarray(arr)

    walk(sp_d, dp_d)
    nnx.replace_by_pure_dict(dp, dp_d)
    nnx.update(dst, dp, dn)


# --------------------------------------------------------------------------- #
def gate_flags_off_is_old_code() -> bool:
    """With both features off, sample() and loss() are the pre-change functions.

    The sampler is checked THREE ways, because no single one is both decisive
    and achievable:

      (a) against a ``fori_loop`` reference written the pre-change way -- the old
          two-element carry calling the public ``p_sample``. Same fusion context,
          so this is required EXACT: it is what catches a threading or carry bug
          introduced by the self-cond plumbing.
      (b) against a fully hand-written EAGER loop, to a tolerance. This one is
          immune to (a)'s circularity (if ``p_sample`` itself were wrong, (a)
          would be wrong the same way in both arms) but it cannot be exact:
          XLA fuses a loop body differently from op-by-op dispatch, so eager and
          `fori_loop` disagree at fp32 rounding. A SEMANTIC error -- a dropped
          coefficient, the wrong x0, a mis-threaded timestep -- is O(1) here, not
          O(1e-6), so a tolerance still catches it. The measured gap is printed
          rather than hidden.
      (c) against ``sample_with_trace(k_trainable=0, rl_mode=False)``, the
          repo's own Gate 2 invariant, which is exact by construction.
    """
    unet = _unet(0, self_cond=False)
    diff = GaussianDiffusion(unet, timesteps=STEPS, predict_type="v")
    c1, c2, m = _cond()
    shape = (B, C, T)
    rng = jax.random.PRNGKey(7)
    got = diff.sample(shape, c1, c2, m, rng=rng)

    # (a) the pre-change loop, verbatim: two-element carry, public `p_sample`.
    temp_sched = diff._temp_schedule(1.0, "uniform", STEPS)
    key_x, key_steps = jax.random.split(rng)
    x_init = jax.random.normal(key_x, shape, dtype=jnp.float32)

    def old_body(i, carry):
        x_curr, key = carry
        t = jnp.full((B,), STEPS - 1 - i, dtype=jnp.int32)
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, shape, dtype=jnp.float32)
        return diff.p_sample(x_curr, t, c1, c2, m, noise,
                             eta=1.0, sampling_temp=temp_sched[i]), key

    x_old, _ = jax.lax.fori_loop(0, STEPS, old_body, (x_init, key_steps))
    old_ok = _exact(got, x_old)

    # (b) eager, tolerance only.
    x, key = x_init, key_steps
    for i in range(STEPS):
        t = jnp.full((B,), STEPS - 1 - i, dtype=jnp.int32)
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, shape, dtype=jnp.float32)
        _, _, x0p = diff.p_mean_variance(x, t, c1, c2, m)
        x = diff._ddim_update(x, t, x0p, noise, eta=1.0, sampling_temp=1.0)
    eager_gap = float(np.max(np.abs(np.asarray(got) - np.asarray(x))))
    scale = float(np.max(np.abs(np.asarray(got))))
    # 1% of the trajectory scale. Deliberately loose: this check's job is to
    # catch an O(1) semantic error, and fp32 rounding here is amplified by the
    # 1/sqrt(1-abar_t) in the eps round-trip (~6x per step at the bottom of a
    # 10-step schedule), so a tight bound would fail on arithmetic rather than
    # on meaning. The measured gap is printed, so a drift into 1e-3 territory is
    # visible even while the check passes.
    eager_ok = eager_gap <= 1e-2 * max(scale, 1.0)

    # (c) the repo's existing exactness invariant.
    traced, _ = diff.sample_with_trace(shape, c1, c2, m, rng=rng,
                                       k_trainable=0, rl_mode=False)
    trace_ok = _exact(got, traced)

    sample_ok = old_ok and eager_ok and trace_ok
    print(f"  sample == pre-change fori_loop (exact): {old_ok}")
    print(f"  sample ~= eager reference: {eager_ok} "
          f"(max|diff|={eager_gap:.3e} on scale {scale:.3f}; "
          f"fp32 loop-vs-eager fusion, not semantics)")
    print(f"  sample == sample_with_trace(k=0) (exact): {trace_ok}")

    # --- loss: hand-written reference, drawing (t, eps) the ORIGINAL way ------
    x0 = jax.random.normal(jax.random.PRNGKey(21), shape)
    lrng = jax.random.PRNGKey(22)
    key_t, key_noise = jax.random.split(lrng)
    t = jax.random.randint(key_t, (B,), 0, STEPS, dtype=jnp.int32)
    noise = jax.random.normal(key_noise, shape, dtype=jnp.float32)
    x_t = diff.q_sample(x0, t, noise=noise)
    pred = unet(x_t, t, c1, c2, m)
    sqrt_ab = diff._extract(diff.sqrt_alphas_cumprod.value, t, shape)
    sqrt_1mab = diff._extract(diff.sqrt_one_minus_alphas_cumprod.value, t, shape)
    target = sqrt_ab * noise - sqrt_1mab * x0
    ref = float(jnp.mean((pred - target) ** 2))
    loss_ok = _close(float(diff.loss(x0, c1, c2, m, rng=lrng)), ref, atol=1e-6, rtol=1e-6)

    print(f"  loss   == reference objective (rng contract intact): {loss_ok}")
    return sample_ok and loss_ok


def gate_zero_expansion_is_base() -> bool:
    """A zero-expanded self-cond model samples bit-identically to the base."""
    base = _unet(0, self_cond=False)
    sc = _unet(0, self_cond=True)
    _copy_params_zero_expanded(base, sc)

    d_base = GaussianDiffusion(base, timesteps=STEPS, predict_type="v")
    d_sc = GaussianDiffusion(sc, timesteps=STEPS, predict_type="v", self_cond=True)
    c1, c2, m = _cond()
    shape = (B, C, T)
    rng = jax.random.PRNGKey(7)

    a = np.asarray(d_base.sample(shape, c1, c2, m, rng=rng))
    b = np.asarray(d_sc.sample(shape, c1, c2, m, rng=rng))
    exact = _exact(a, b)
    close = _close(a, b, atol=1e-6, rtol=1e-6)
    print(f"  stem fan-in {np.asarray(nnx.to_pure_dict(nnx.split(base, nnx.Param, ...)[1])['stem']['kernel']).shape[1]}"
          f" -> {np.asarray(nnx.to_pure_dict(nnx.split(sc, nnx.Param, ...)[1])['stem']['kernel']).shape[1]}")
    print(f"  trajectories identical: exact={exact} close={close} "
          f"max|diff|={float(np.max(np.abs(a - b))):.3e}")
    # Float ops are re-associated by the wider conv, so require close, report exact.
    return close


def gate_new_channels_are_wired() -> bool:
    """A non-zero self-cond block DOES change the output -- the concat is live."""
    base = _unet(0, self_cond=False)
    sc = _unet(0, self_cond=True)
    _copy_params_zero_expanded(base, sc)
    gd, params, nonp = nnx.split(sc, nnx.Param, ...)
    pure = nnx.to_pure_dict(params)
    k = np.asarray(pure["stem"]["kernel"]).copy()
    k[:, C:, :] = 0.5  # was zeros
    pure["stem"]["kernel"] = jnp.asarray(k)
    nnx.replace_by_pure_dict(params, pure)
    nnx.update(sc, params, nonp)

    d_base = GaussianDiffusion(base, timesteps=STEPS, predict_type="v")
    d_sc = GaussianDiffusion(sc, timesteps=STEPS, predict_type="v", self_cond=True)
    c1, c2, m = _cond()
    rng = jax.random.PRNGKey(7)
    a = d_base.sample((B, C, T), c1, c2, m, rng=rng)
    b = d_sc.sample((B, C, T), c1, c2, m, rng=rng)
    changed = not _close(a, b, atol=1e-4)
    print(f"  non-zero self-cond block changes the trajectory: {changed} "
          f"(max|diff|={float(np.max(np.abs(np.asarray(a) - np.asarray(b)))):.3e})")
    return changed


def gate_distortion_mechanics() -> bool:
    """t-gating, sparsity rate, channel restriction, relative scaling."""
    unet = _unet(0, self_cond=False)
    t_max = STEPS // 2 - 1
    diff = GaussianDiffusion(unet, timesteps=STEPS, predict_type="v",
                             distort_p=0.2, distort_sigma=0.8,
                             distort_mode="relative", distort_channels=2)
    nb = 4096
    x_t = jnp.zeros((nb, C, T))

    hi = jnp.full((nb,), t_max + 1, dtype=jnp.int32)
    d_hi = diff._draw_distortion(x_t, hi, jax.random.PRNGKey(0))
    gated = _exact(d_hi, jnp.zeros_like(d_hi))

    lo = jnp.full((nb,), t_max, dtype=jnp.int32)
    d_lo = np.asarray(diff._draw_distortion(x_t, lo, jax.random.PRNGKey(0)))
    kicked = (np.abs(d_lo[:, 0, :]) > 0).mean()
    rate_ok = abs(kicked - 0.2) < 0.02
    chan_ok = bool(np.all(d_lo[:, 2:, :] == 0.0)) and bool(np.any(d_lo[:, :2, :] != 0.0))

    # relative scaling: std over the KICKED entries == sigma * sqrt(1-abar_t)
    nz = d_lo[:, 0, :][np.abs(d_lo[:, 0, :]) > 0]
    expect = 0.8 * float(np.asarray(diff.sqrt_one_minus_alphas_cumprod.value)[t_max])
    scale_ok = abs(float(nz.std()) - expect) / expect < 0.05

    # per-waypoint mask: the two position channels of a waypoint are kicked together
    both = ((np.abs(d_lo[:, 0, :]) > 0) == (np.abs(d_lo[:, 1, :]) > 0)).all()

    print(f"  gated off above t_max: {gated}")
    print(f"  kicked waypoint rate {kicked:.4f} (want 0.20): {rate_ok}")
    print(f"  positions only: {chan_ok}; per-waypoint mask shared across channels: {bool(both)}")
    print(f"  relative sigma {float(nz.std()):.4f} vs expected {expect:.4f}: {scale_ok}")
    return bool(gated and rate_ok and chan_ok and both and scale_ok)


def gate_distorted_target_recovers_x0() -> bool:
    """The corrected target is the one that undoes the kick, for every predict_type.

    A network emitting ``_distorted_target(target, t, d)`` from the DISTORTED state
    must decode to the ORIGINAL clean x0. With an uncorrected target the same
    decode lands on ``x0 + sqrt(abar)*d`` -- the kick reproduced instead of
    corrected -- so the second assertion below is the one that catches a sign or
    scale slip.
    """
    unet = _unet(0, self_cond=False)
    shape = (B, C, T)
    x0 = jax.random.normal(jax.random.PRNGKey(31), shape)
    noise = jax.random.normal(jax.random.PRNGKey(32), shape)
    t = jnp.array([1, 3, 4] , dtype=jnp.int32)[:B]
    ok = True
    for ptype in ("v", "eps", "mu"):
        diff = GaussianDiffusion(unet, timesteps=STEPS, predict_type=ptype,
                                 distort_p=1.0, distort_sigma=0.5, distort_mode="absolute",
                                 distort_t_max=STEPS - 1)
        x_t = diff.q_sample(x0, t, noise=noise)
        d = diff._draw_distortion(x_t, t, jax.random.PRNGKey(33))
        sqrt_ab = diff._extract(diff.sqrt_alphas_cumprod.value, t, shape)
        sqrt_1mab = diff._extract(diff.sqrt_one_minus_alphas_cumprod.value, t, shape)
        base_target = {"v": sqrt_ab * noise - sqrt_1mab * x0, "eps": noise, "mu": x0}[ptype]
        corrected = diff._distorted_target(base_target, t, d)
        got = diff._pred_to_x0(x_t + d, t, corrected)
        good = _close(got, x0, atol=1e-4, rtol=1e-4)
        # ... and the UNcorrected target does not (except for `mu`, where the
        # target is the clean sample itself and no correction is needed).
        naive = diff._pred_to_x0(x_t + d, t, base_target)
        distinct = (ptype == "mu") or not _close(naive, x0, atol=1e-3)
        print(f"  predict_type={ptype}: corrected recovers x0={good}; uncorrected differs={distinct}")
        ok = ok and good and distinct
    return ok


def gate_self_cond_first_pass_is_detached() -> bool:
    """No gradient flows through the self-conditioning input."""
    unet = _unet(0, self_cond=True)
    diff = GaussianDiffusion(unet, timesteps=STEPS, predict_type="v", self_cond=True)
    c1, c2, m = _cond()
    gd, params, nonp = nnx.split(unet, nnx.Param, ...)

    def sc_sum(p):
        fn = nnx.merge(gd, p, nonp)
        x_t = jnp.ones((B, C, T))
        t = jnp.full((B,), 3, dtype=jnp.int32)
        sc = diff._self_cond_train_input(fn, x_t, t, c1, c2, m, jax.random.PRNGKey(0))
        return jnp.sum(sc)

    g = jax.grad(sc_sum)(params)
    norms = [float(jnp.sum(jnp.abs(x))) for x in jax.tree_util.tree_leaves(g)]
    detached = max(norms) == 0.0
    # The mask really does mix both branches at sc_train_prob.
    fn = nnx.merge(gd, params, nonp)
    big = jnp.ones((512, C, T))
    sc = np.asarray(diff._self_cond_train_input(
        fn, big, jnp.full((512,), 3, jnp.int32),
        jnp.tile(c1[:1], (512, 1, 1)), jnp.tile(c2[:1], (512, 1)), jnp.ones((512,)),
        jax.random.PRNGKey(5)))
    frac = float((np.abs(sc).sum(axis=(1, 2)) > 0).mean())
    mix_ok = abs(frac - 0.5) < 0.06
    print(f"  grad through first pass = {max(norms):.3e} -> detached={detached}")
    print(f"  self-conditioned share {frac:.3f} (want {0.5}): {mix_ok}")
    return bool(detached and mix_ok)


def gate_trace_and_prefix_replay() -> bool:
    """Under self-cond the trace replays: log-probs reproduce, prefix reproduces."""
    unet = _unet(0, self_cond=True)
    diff = GaussianDiffusion(unet, timesteps=STEPS, predict_type="v", self_cond=True)
    c1, c2, m = _cond()
    shape = (B, C, T)
    rng = jax.random.PRNGKey(41)

    x, tr, pre = diff.sample_with_trace(
        shape, c1, c2, m, rng=rng, k_trainable=3, rl_mode=True,
        sigma_sample_floor=0.05, sigma_logprob_floor=0.1, return_prefix=True)
    has_sc = "x_sc" in tr
    ratios_ok = True
    for j in range(int(tr["timestep"].shape[0])):
        re = diff.transition_log_prob(
            tr["x_t"][j], tr["x_prev"][j], tr["timestep"][j], c1, c2, m,
            sigma_floor=0.1, x_sc=tr["x_sc"][j])
        ratios_ok = ratios_ok and _close(re, tr["log_prob"][j], atol=1e-5)
    # Passing the WRONG self-cond input must NOT reproduce it (otherwise the
    # argument is being ignored and the replay check above is vacuous).
    wrong = diff.transition_log_prob(
        tr["x_t"][0], tr["x_prev"][0], tr["timestep"][0], c1, c2, m,
        sigma_floor=0.1, x_sc=jnp.zeros_like(tr["x_sc"][0]))
    sensitive = not _close(wrong, tr["log_prob"][0], atol=1e-4)

    x_prefix, key_late = pre
    is_pair = isinstance(x_prefix, tuple) and len(x_prefix) == 2
    x2, _ = diff.sample_with_trace(
        shape, c1, c2, m, rng=rng, k_trainable=3, rl_mode=True,
        sigma_sample_floor=0.05, sigma_logprob_floor=0.1,
        x_prefix=x_prefix, key_late=key_late)
    replay_ok = _exact(x, x2)

    print(f"  trace carries x_sc: {has_sc}; stored log-probs reproduce: {ratios_ok}")
    print(f"  log-prob depends on x_sc (wrong input differs): {sensitive}")
    print(f"  prefix is (x, x_sc) pair: {is_pair}; prefix replay bit-identical: {replay_ok}")
    return bool(has_sc and ratios_ok and sensitive and is_pair and replay_ok)


def main() -> int:
    print(f"[self_cond] devices={jax.devices()}")
    checks = [
        ("flags off == pre-change code", gate_flags_off_is_old_code),
        ("zero-expanded self-cond == base", gate_zero_expansion_is_base),
        ("self-cond channels are wired", gate_new_channels_are_wired),
        ("distortion mechanics", gate_distortion_mechanics),
        ("distorted target recovers x0", gate_distorted_target_recovers_x0),
        ("self-cond first pass detached", gate_self_cond_first_pass_is_detached),
        ("trace + prefix replay under self-cond", gate_trace_and_prefix_replay),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n=== {name} ===")
        try:
            results[name] = bool(fn())
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    print("\n=========== self-cond / distortion summary ===========")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allp = all(results.values())
    print(f"\nOverall: {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
