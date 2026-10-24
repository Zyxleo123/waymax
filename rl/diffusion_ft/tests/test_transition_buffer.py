"""Transition replay: buffer mechanics, trace harvest, clean-prediction scoring.

Tiny synthetic model (fast, CPU), same one Gate 2 uses. Covers the parts of
``--tb_weight`` that can be checked without a checkpoint or a simulator:

* the ring buffer itself -- FIFO overwrite, use-capped eviction, ragged rejection,
  and "too few to draw" returning None rather than a short batch;
* ``sample_world_with_prefix_trace`` returns the SAME trajectory as
  ``sample_world_with_prefix`` on the same key (harvesting must not perturb the
  rollout that fills the buffer) and a trace whose timesteps run ``k-1..0``;
* the stored transition really is replayable: a ``K=1`` clean prediction from
  ``(x_t, t)`` equals the denoiser's own ``x0_pred`` at that state, so what the
  loss is charged on at replay time is the quantity the sampler computed there;
* truncated backprop is real -- ``grad_steps=1`` and ``grad_steps=K`` give
  different gradients through the same unroll, which is what lets a K-step unroll
  differentiate exactly one transition;
* per-row ``t0`` in ``candidate_safety``'s window slice agrees with the scalar
  path it generalises (the replayed minibatch mixes replan boundaries).

Submit it; do not run it on a login node (the XLA CPU compile gets the process
killed -- exit 137).
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
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.safety_objective import _window_slice
from rl.diffusion_ft.transition_buffer import TransitionBuffer
from rl.diffusion_ft.unrolled_prediction import unroll_timestep_grid

K_TRAINABLE = 4


def _tiny_actor():
    C, T, cond_dim, hidden = 5, 8, 16, 32
    rngs = nnx.Rngs(0)
    unet = UNet1DConditioned(
        in_ch=C, cond_dim=cond_dim, cond2_dim=cond_dim, time_emb_dim=cond_dim,
        horizon=T, hidden_dim=hidden, base_ch=16, rngs=rngs,
    )
    diff = GaussianDiffusion(unet, timesteps=10, predict_type="v")
    B = 3
    cond1 = jax.random.normal(jax.random.PRNGKey(1), (B, 4, cond_dim))
    cond2 = jax.random.normal(jax.random.PRNGKey(2), (B, cond_dim))
    mask = jnp.ones((B,), dtype=jnp.float32)
    actor = DiffusionRLActor(diff, k_trainable=K_TRAINABLE, seed=0)
    actor.reset_trainable_to_frozen()
    return actor, cond1, cond2, mask


def _close(a, b, atol=1e-5):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol))


def _rows(n, *, scene0=0, t0=0, C=5, T=8, F=(4, 16)):
    return {
        "scene_id": np.arange(scene0, scene0 + n, dtype=np.int64),
        "t0": np.full(n, t0, np.int32),
        "t": np.arange(n, dtype=np.int32) % K_TRAINABLE,
        "x_t": np.zeros((n, C, T), np.float32),
        "cond1": np.zeros((n,) + F, np.float32),
        "cond2": np.zeros((n, F[1]), np.float32),
        "cond2_mask": np.ones((n,), np.float32),
        "aux/origin_xy": np.zeros((n, 2), np.float32),
    }


# --------------------------------------------------------------------------- #
def gate_buffer_ring() -> bool:
    """FIFO overwrite, capacity, and the fields of a datum survive a round trip."""
    buf = TransitionBuffer(4, seed=0)
    buf.add(_rows(3, scene0=0))
    ok = len(buf) == 3
    buf.add(_rows(3, scene0=10))                 # wraps: evicts the 2 oldest
    ok &= len(buf) == 4 and buf.n_added == 6 and buf.n_evicted == 2
    drawn = buf.sample(4)
    ok &= drawn is not None and drawn["scene_id"].shape == (4,)
    # Slots 0,1 were overwritten by scenes 11,12; scene 2 and scenes 10-12 remain.
    ok &= set(drawn["scene_id"].tolist()) == {2, 10, 11, 12}
    ok &= drawn["x_t"].shape == (4, 5, 8) and "aux/origin_xy" in drawn
    print(f"  ring: size={len(buf)} added={buf.n_added} evicted={buf.n_evicted} "
          f"ids={sorted(drawn['scene_id'].tolist())} -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_buffer_uses() -> bool:
    """`max_uses` retires a slot, and a short buffer refuses to draw."""
    buf = TransitionBuffer(8, max_uses=2, seed=0)
    buf.add(_rows(4))
    ok = buf.sample(4) is not None and buf.sample(4) is not None
    ok &= len(buf) == 0                            # both draws spent every slot
    ok &= buf.sample(4) is None                    # nothing left -> None, not short
    buf2 = TransitionBuffer(8, seed=0)
    buf2.add(_rows(2))
    ok &= buf2.sample(4) is None                   # fewer than asked -> None
    ok &= buf2.sample(2) is not None
    ragged = _rows(3)
    ragged["t"] = np.zeros(2, np.int32)
    try:
        buf2.add(ragged)
        ok = False                                 # must have raised
    except ValueError:
        pass
    print(f"  uses/short: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def gate_trace_matches_rollout(actor, c1, c2, m) -> bool:
    """Harvesting the trace changes neither the trajectory nor the prefix."""
    key = jax.random.PRNGKey(7)
    x_a, pre_a = actor.sample_world_with_prefix(actor._tparams, c1, c2, m, key)
    x_b, pre_b, x_t_kbct, t_kb = actor.sample_world_with_prefix_trace(
        actor._tparams, c1, c2, m, key)
    ok = _close(x_a, x_b) and _close(pre_a[0], pre_b[0])
    B = c1.shape[0]
    ok &= tuple(x_t_kbct.shape) == (K_TRAINABLE, B, 5, 8)
    ok &= tuple(t_kb.shape) == (K_TRAINABLE, B)
    # Trace row j is timestep k-1-j, identical across the batch.
    want = np.arange(K_TRAINABLE - 1, -1, -1)
    ok &= bool(np.array_equal(np.asarray(t_kb)[:, 0], want))
    ok &= bool(np.all(np.asarray(t_kb) == np.asarray(t_kb)[:, :1]))
    # x_t[0] is the state ENTERING the late chain, i.e. the stored prefix.
    ok &= _close(x_t_kbct[0], pre_b[0])
    print(f"  trace: traj_identical={_close(x_a, x_b)} t={np.asarray(t_kb)[:, 0].tolist()} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_replay_is_the_same_state(actor, c1, c2, m) -> bool:
    """K=1 from a stored (x_t, t) == the denoiser's own x0_pred at that state.

    This is the claim the whole mechanism rests on: a slot holds enough to
    recompute, later and under different weights, exactly the clean prediction
    the network would make at the state the sampler passed through.
    """
    key = jax.random.PRNGKey(7)
    _, _, x_t_kbct, t_kb = actor.sample_world_with_prefix_trace(
        actor._tparams, c1, c2, m, key)
    j = 1                                           # a mid-chain transition
    x_t, t_b = x_t_kbct[j], t_kb[j]
    got = actor.clean_prediction_from_xt(
        actor._tparams, c1, c2, m, x_t, t_b[None, :], jax.random.PRNGKey(0))
    _, _, want = actor.diffusion.p_mean_variance(
        x_t, t_b, c1, c2, m, denoise_fn=actor._trainable_fn(actor._tparams))
    ok = _close(got, want)
    print(f"  replay K=1 == x0_pred at the stored state: max|d|="
          f"{float(jnp.max(jnp.abs(got - want))):.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_truncated_backprop(actor, c1, c2, m) -> bool:
    """A K-step unroll with grad_steps=1 differentiates ONE transition.

    Checked two ways: the gradient is non-zero (the chain reaches the weights at
    all) and it differs from the full-unroll gradient (the truncation is real,
    not silently ignored).
    """
    key = jax.random.PRNGKey(7)
    _, _, x_t_kbct, t_kb = actor.sample_world_with_prefix_trace(
        actor._tparams, c1, c2, m, key)
    x_t, t_b = x_t_kbct[0], t_kb[0]                 # t = k-1, the noisiest stored
    K = 3
    grid = unroll_timestep_grid(t_b, K, 0)

    def loss(params, g):
        x0 = actor.clean_prediction_from_xt(
            params, c1, c2, m, x_t, grid, jax.random.PRNGKey(0), grad_steps=g)
        return jnp.sum(x0 ** 2)

    g1 = jax.grad(lambda p: loss(p, 1))(actor._tparams)
    gK = jax.grad(lambda p: loss(p, K))(actor._tparams)
    n1 = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g1))))
    nK = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(gK))))
    diff = float(jnp.sqrt(sum(jnp.sum((a - b) ** 2) for a, b in zip(
        jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(gK)))))
    ok = n1 > 0 and nK > 0 and diff > 1e-8
    print(f"  truncation: |g(grad_steps=1)|={n1:.4e} |g(K)|={nK:.4e} "
          f"|diff|={diff:.4e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_grid_clamp() -> bool:
    """A stored state already at/below t_end unrolls as a no-op, never upward."""
    t_b = jnp.asarray([0, 1, 3], jnp.int32)
    t_end = 1
    grid = np.asarray(unroll_timestep_grid(jnp.maximum(t_b, t_end), 3, t_end))
    ok = bool(np.all(np.diff(grid, axis=0) <= 0))   # never ascends => never re-noises
    ok &= bool(np.all(grid[:, 0] == t_end))         # the clamped row is constant
    ok &= bool(np.all(grid[-1] == t_end))
    print(f"  grid clamp: {grid.tolist()} -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_per_row_window() -> bool:
    """Per-row t0 reproduces the scalar slice when every row shares a t0."""
    arr = jnp.asarray(np.arange(2 * 3 * 12, dtype=np.float32).reshape(2, 3, 12))
    n = 4
    same = _window_slice(arr, 5, n, axis=2)
    per_row = _window_slice(arr, jnp.asarray([5, 5], jnp.int32), n, axis=2)
    ok = _close(same, per_row)
    # And a genuinely mixed batch slices each row at its own offset.
    mixed = _window_slice(arr, jnp.asarray([2, 6], jnp.int32), n, axis=2)
    ok &= _close(mixed[0], arr[0, :, 3:3 + n]) and _close(mixed[1], arr[1, :, 7:7 + n])
    z = jnp.asarray(np.arange(2 * 12, dtype=np.float32).reshape(2, 12))
    zr = _window_slice(z, jnp.asarray([2, 6], jnp.int32), n, axis=1)
    ok &= _close(zr[0], z[0, 3:3 + n]) and _close(zr[1], z[1, 7:7 + n])
    print(f"  per-row window slice: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=== transition replay ===", flush=True)
    results = {}
    results["buffer_ring"] = gate_buffer_ring()
    results["buffer_uses"] = gate_buffer_uses()
    results["grid_clamp"] = gate_grid_clamp()
    results["per_row_window"] = gate_per_row_window()
    actor, c1, c2, m = _tiny_actor()
    results["trace_matches_rollout"] = gate_trace_matches_rollout(actor, c1, c2, m)
    results["replay_is_same_state"] = gate_replay_is_the_same_state(actor, c1, c2, m)
    results["truncated_backprop"] = gate_truncated_backprop(actor, c1, c2, m)
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} gates passed", flush=True)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
