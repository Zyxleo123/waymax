"""Diffusion-DPO term (`train_safety --pref_weight`): loss math and pair loading.

Tiny synthetic UNet, CPU, seconds. What each check pins, and the silent failure
it would otherwise be:

* **init is log 2.** The trainable copy starts AS the reference, so z = 0. A
  loss that starts elsewhere means the two sides are not the same model (or not
  the same (t, eps)) and every z after it is an offset, not a preference.
* **shared (t, eps).** A pair with good == bad must give z = 0 for ANY params.
  Independent noise per side would make z a noise-draw difference.
* **eps from v.** The checkpoint predicts v; the loss is eps-MSE. The conversion
  must be exact, or the "eps loss" is a re-weighted v loss.
* **t window.** t ~ U[t_min, t_max), shared by both sides; checked against a
  hand recomputation at a one-value window.
* **step weights.** The covered-steps mask (and the edit weight) must act as a
  weighted mean over model steps.
* **direction.** A few descent steps on a fixed batch must push z > 0: the
  model gets better at the good trajectory than at the bad one, relative to ref.
* **base_anchor_loss(t_max=None)** is bit-identical to the old call.
* **_init_pref** pools: pairs = switched (& solved), anchor = success_train
  kept rows; refuses data without `rejected`, and pairs that differ off the edit.

    python -m rl.diffusion_ft.tests.test_pref_loss
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from model.diffusion.diffusion import GaussianDiffusion, UNet1DConditioned
from rl.diffusion_ft import train_safety as T
from rl.diffusion_ft.rl_actor import DiffusionRLActor

FAILED = []
C, H, CD, STEPS, K = 5, 8, 16, 20, 6


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _actor(predict_type="v"):
    unet = UNet1DConditioned(in_ch=C, cond_dim=CD, cond2_dim=CD, time_emb_dim=CD,
                             horizon=H, hidden_dim=32, base_ch=16, rngs=nnx.Rngs(0))
    diff = GaussianDiffusion(unet, timesteps=STEPS, predict_type=predict_type)
    return DiffusionRLActor(diff, k_trainable=K, sigma_sample_floor=0.1,
                            sigma_logprob_floor=0.1, actor_lr=1e-3)


def _batch(B=4, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 4)
    good = 0.1 * jax.random.normal(ks[0], (B, C, H))
    bad = good + 0.05 * jax.random.normal(ks[1], (B, C, H))
    c1 = jax.random.normal(ks[2], (B, 4, CD))
    c2 = jax.random.normal(ks[3], (B, CD))
    return good, bad, c1, c2, jnp.ones((B,), jnp.float32), jnp.ones((B, H), jnp.float32)


def _perturbed(actor, scale=1e-2):
    leaves, td = jax.tree_util.tree_flatten(actor._tparams)
    keys = jax.random.split(jax.random.PRNGKey(99), len(leaves))
    return jax.tree_util.tree_unflatten(
        td, [l + scale * jax.random.normal(k, l.shape) for l, k in zip(leaves, keys)])


def init_is_log2():
    a = _actor()
    good, bad, c1, c2, m, w = _batch()
    loss, aux = a.pref_loss(a._tparams, good, bad, c1, c2, m, w, jax.random.PRNGKey(1), beta=1.0)
    check(abs(float(loss) - np.log(2.0)) < 1e-6,
          f"init: loss {float(loss):.8f} == log 2 ({np.log(2.0):.8f})")
    check(float(aux["z_abs"]) < 1e-6, f"init: |z| {float(aux['z_abs']):.2e} ~ 0")
    check(float(aux["init_gap"]) < 1e-5, f"init: max |model - ref| {float(aux['init_gap']):.2e}")
    check(0.0 <= float(aux["accuracy"]) <= 1.0, "init: accuracy in [0, 1]")


def shared_noise():
    a = _actor()
    good, _, c1, c2, m, w = _batch()
    p = _perturbed(a)
    for s in range(3):
        mg, mb, rg, rb = a.pref_denoise_losses(p, good, good, c1, c2, m, w,
                                               jax.random.PRNGKey(10 + s))
        same = float(jnp.max(jnp.abs(mg - mb))) + float(jnp.max(jnp.abs(rg - rb)))
        check(same == 0.0, f"good == bad -> identical losses on both sides (key {s}, gap {same:.1e})")
    _, aux = a.pref_loss(p, good, good, c1, c2, m, w, jax.random.PRNGKey(3), beta=50.0)
    check(float(aux["z_abs"]) == 0.0, "good == bad -> z == 0 exactly, for perturbed params")


def eps_conversion():
    for pt in ("v", "eps", "mu"):
        a = _actor(pt)
        d = a.diffusion
        x0 = 0.3 * jax.random.normal(jax.random.PRNGKey(5), (3, C, H))
        eps = jax.random.normal(jax.random.PRNGKey(6), (3, C, H))
        t = jnp.asarray([0, 7, 19], jnp.int32)
        x_t = d.q_sample(x0, t, eps)
        sab = d._extract(d.sqrt_alphas_cumprod.value, t, x0.shape)
        s1m = d._extract(d.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
        pred = {"v": sab * eps - s1m * x0, "eps": eps, "mu": x0}[pt]
        err = float(jnp.max(jnp.abs(a._eps_from_pred(x_t, t, pred) - eps)))
        check(err < 1e-4, f"eps from exact {pt}-prediction recovers eps (max err {err:.1e})")


def t_window_and_weights():
    a = _actor()
    d = a.diffusion
    good, bad, c1, c2, m, _ = _batch(B=3)
    p = _perturbed(a)
    w = jnp.zeros((3, H)).at[:, 2].set(1.0).at[:, 5].set(3.0)
    key = jax.random.PRNGKey(21)
    mg, mb, rg, rb = a.pref_denoise_losses(p, good, bad, c1, c2, m, w, key, t_min=4, t_max=5)
    # Hand recomputation: same key split, t fixed at 4, shared eps.
    kt, ke = jax.random.split(key)
    eps = jax.random.normal(ke, good.shape)
    t = jnp.full((3,), 4, jnp.int32)

    def manual(fn, x0):
        x_t = d.q_sample(x0, t, eps)
        e = a._eps_from_pred(x_t, t, fn(x_t, t, c1, c2, m))
        se = jnp.mean((e - eps) ** 2, axis=1)                # [B, T] (mean over C)
        return (se[:, 2] * 1.0 + se[:, 5] * 3.0) / 4.0
    tf = a._trainable_fn(p)
    ref = a.frozen_denoise
    errs = [float(jnp.max(jnp.abs(x - y))) for x, y in
            ((mg, manual(tf, good)), (mb, manual(tf, bad)),
             (rg, manual(ref, good)), (rb, manual(ref, bad)))]
    check(max(errs) < 1e-5, f"t window [4,5) + shared eps + step weights == hand "
                            f"recomputation (max err {max(errs):.1e})")
    # t_max=None means k_trainable: a window of one value at K-1 vs the default
    # cannot be told apart from outside, so pin the default bound instead.
    try:
        a.pref_denoise_losses(p, good, bad, c1, c2, m, w, key, t_min=0, t_max=None)
        check(True, "t_max=None runs (draws over [0, k_trainable))")
    except Exception as e:                     # pragma: no cover
        check(False, f"t_max=None raised {e!r}")


def descent_raises_z():
    a = _actor()
    good, bad, c1, c2, m, w = _batch(B=8, seed=3)
    tx = optax.adam(1e-3)
    p = a._tparams
    st = tx.init(p)
    vg = jax.jit(jax.value_and_grad(
        lambda q, k: a.pref_loss(q, good, bad, c1, c2, m, w, k, beta=1.0), has_aux=True))
    for i in range(60):
        (_, aux), g = vg(p, jax.random.PRNGKey(1000 + i))
        upd, st = tx.update(g, st, p)
        p = optax.apply_updates(p, upd)
    zs, acc, dg, db = [], [], [], []
    for i in range(16):
        _, aux = a.pref_loss(p, good, bad, c1, c2, m, w, jax.random.PRNGKey(5000 + i), beta=1.0)
        zs.append(float(aux["logit_z"])); acc.append(float(aux["accuracy"]))
        dg.append(float(aux["good_model_loss"] - aux["good_ref_loss"]))
        db.append(float(aux["bad_model_loss"] - aux["bad_ref_loss"]))
    check(np.mean(zs) > 0, f"60 descent steps: mean z {np.mean(zs):+.3e} > 0 "
                           f"(accuracy {np.mean(acc):.2f})")
    check(np.mean(dg) < np.mean(db),
          f"relative to ref, good improved more than bad ({np.mean(dg):+.2e} < {np.mean(db):+.2e})")


def anchor_t_max_default():
    a = _actor()
    good, _, c1, c2, m, _ = _batch()
    p = _perturbed(a)
    x = jnp.transpose(good, (0, 2, 1))
    k = jax.random.PRNGKey(4)
    old = a.base_anchor_loss(p, x, c1, c2, m, m, k)
    new = a.base_anchor_loss(p, x, c1, c2, m, m, k, t_max=None)
    check(float(old) == float(new), "base_anchor_loss(t_max=None) bit-identical to the old call")
    z0 = a.base_anchor_loss(a._tparams, x, c1, c2, m, m, k, t_max=K)
    check(abs(float(z0)) < 1e-8, f"anchor at init (t_max=k) ~ 0 ({float(z0):.1e})")


def _rows(n, *, sw, split, solved, seed=0, rejected=True, bad_kept=False):
    r = np.random.default_rng(seed)
    tgt = (0.1 * r.normal(size=(n, H, C))).astype(np.float32)
    rej = tgt.copy()
    valid = np.ones((n, H), bool)
    valid[:, -2:] = False                     # an uncovered tail
    for i in np.nonzero(sw)[0]:
        rej[i, :-2, :2] += 0.004              # 0.4 m at ego_range 100, covered steps only
    if bad_kept:
        rej[np.nonzero(~sw)[0][0], 0, 0] += 1e-3
    d = {"cond1": r.normal(size=(n, 4, CD)).astype(np.float32),
         "cond2": r.normal(size=(n, CD)).astype(np.float32),
         "mask": np.ones((n,), np.float32), "target": tgt, "valid": valid,
         "scene_id": np.arange(n, dtype=np.int64), "switched": sw,
         "split": np.asarray(split, dtype="U32"), "ep_solved": solved,
         "rt_err_m": np.full(n, 0.002, np.float32)}
    if rejected:
        d["rejected"] = rej
    return d


def _stub(actor, data_glob, **kw):
    t = T.SafetyFineTuner.__new__(T.SafetyFineTuner)
    t.args = SimpleNamespace(pref_weight=1.0, pref_data=data_glob, grad_accum=0, epochs=1,
                             pref_rows="solved", pref_edit_weight=0.0, pref_edit_delta_m=0.1,
                             pref_anchor_weight=100.0, splits_json="", pref_t_min=0,
                             pref_t_max=-1, pref_beta=1.0, pref_batch=4, pref_anchor_batch=4,
                             pref_steps=1, data_seed=0)
    for k, v in kw.items():
        setattr(t.args, k, v)
    t.unroll_steps = 0
    t.actor = actor
    t.ckpt = SimpleNamespace(preprocess_cfg=SimpleNamespace(ego_range=100.0))
    t.rng = jax.random.PRNGKey(0)
    t._adam_steps = 0
    return t


def init_pref_pools_and_grads():
    a = _actor()
    n = 12
    sw = np.zeros(n, bool); sw[[0, 1, 2, 7]] = True
    split = ["failure_train"] * 6 + ["success_train"] * 6
    solved = np.ones(n, bool); solved[1] = False
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / "a.npz", **_rows(n, sw=sw, split=split, solved=solved))
        t = _stub(a, str(Path(td) / "*.npz"))
        t._init_pref()
        check(sorted(t._pref_ix.tolist()) == [0, 2, 7],
              f"pairs = switched & solved, any split ({t._pref_ix.tolist()})")
        check(sorted(t._pref_anchor_ix.tolist()) == [6, 8, 9, 10, 11],
              f"anchor = success_train kept rows ({t._pref_anchor_ix.tolist()})")
        check(t.pref_t == (0, K), f"default t window = [0, k_trainable) ({t.pref_t})")
        sw_w = t._pref["step_w"][0]
        check(sw_w[:-2].min() == 1.0 and sw_w[-2:].max() == 0.0,
              "edit_weight 0 -> step weights are the covered-steps mask")
        g, mtr = t.pref_grads(0)
        check(abs(mtr["pref/loss"] - np.log(2.0)) < 1e-5, f"pref_grads at init: loss "
                                                         f"{mtr['pref/loss']:.6f} ~ log 2")
        check(mtr["pref/grad_norm"] > 0, f"pref gradient is non-zero at init "
                                         f"({mtr['pref/grad_norm']:.2e})")
        check(mtr["pref/anchor_grad_norm"] < 1e-4 * max(mtr["pref/grad_norm"], 1e-12) + 1e-9,
              f"anchor gradient ~0 at init ({mtr['pref/anchor_grad_norm']:.1e})")
        g1, _ = t.pref_grads(0)
        g2, _ = t.pref_grads(1)
        check(len(jax.tree_util.tree_leaves(g1)) == len(jax.tree_util.tree_leaves(g2)),
              "pref_grads returns a params-shaped pytree every call")

        t2 = _stub(a, str(Path(td) / "*.npz"), pref_rows="all", pref_edit_weight=4.0,
                   pref_edit_delta_m=0.3)
        t2._init_pref()
        check(sorted(t2._pref_ix.tolist()) == [0, 1, 2, 7], "rows=all keeps unsolved pairs")
        w0 = t2._pref["step_w"][0]
        check(np.allclose(w0[:-2], 5.0) and np.allclose(w0[-2:], 0.0),
              "edit_weight 4 @ 0.3 m: edited covered steps weigh 5, uncovered 0")
        w6 = t2._pref["step_w"][6]
        check(np.allclose(w6[:-2], 1.0), "a kept row is not up-weighted")

    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / "a.npz", **_rows(n, sw=sw, split=split, solved=solved,
                                             rejected=False))
        try:
            _stub(a, str(Path(td) / "*.npz"))._init_pref()
            check(False, "data without `rejected` is refused")
        except SystemExit as e:
            check("rejected" in str(e), "data without `rejected` is refused")
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / "a.npz", **_rows(n, sw=sw, split=split, solved=solved,
                                             bad_kept=True))
        try:
            _stub(a, str(Path(td) / "*.npz"))._init_pref()
            check(False, "a kept row with target != rejected is refused")
        except SystemExit as e:
            check("did not edit" in str(e), "a kept row with target != rejected is refused")


def init_check_is_beta_free():
    """Job 71201: at beta 1e4 a 4.8e-6 float gap became |z| 0.01 and tripped a
    loss == log 2 check. The check must test the params, not a beta-scaled loss."""
    a = _actor()
    n = 12
    sw = np.zeros(n, bool); sw[[0, 2]] = True
    split = ["failure_train"] * 6 + ["success_train"] * 6
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / "a.npz", **_rows(n, sw=sw, split=split, solved=np.ones(n, bool)))
        t = _stub(a, str(Path(td) / "*.npz"), pref_beta=1e4)
        t._init_pref()
        try:
            t.pref_grads(0)
            check(True, "beta 1e4 at init passes the check")
        except SystemExit as e:
            check(False, f"beta 1e4 at init passes the check ({e})")
        t2 = _stub(a, str(Path(td) / "*.npz"), pref_beta=1.0)
        t2._init_pref()
        good = a._tparams
        a._tparams = _perturbed(a, 1e-3)
        try:
            t2.pref_grads(0)
            check(False, "params that differ from the base at step 0 are refused")
        except SystemExit as e:
            check("init check FAILED" in str(e), "params that differ from the base at step 0 are refused")
        finally:
            a._tparams = good


def main():
    for f in (init_is_log2, shared_noise, eps_conversion, t_window_and_weights,
              descent_raises_z, anchor_t_max_default, init_pref_pools_and_grads,
              init_check_is_beta_free):
        print(f"--- {f.__name__}")
        f()
    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
