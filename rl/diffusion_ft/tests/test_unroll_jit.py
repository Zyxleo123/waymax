"""The jitted teacher-forced unroll must draw the SAME stream as the eager one.

Background (measured 2026-09-14, proposal sweep `20260914_162021_1495255`): the
unroll went through ``_loss_for``'s un-jitted closure, which re-traced the whole
unrolled sampler + geometry graph once per context per iteration. It cost
**34.5 s/iter** against the jitted free-running path's **5.96** -- the proposal
whose entire claim is ``K`` network evaluations instead of 100 ran 6x slower than
the thing it was meant to undercut. Routing it through the cached
``_jit_free_grad`` fixes that, and ``--unroll_eager`` keeps the old closure as the
equivalence reference.

That reference is only worth having if the two paths are seed-identical, and the
risk is entirely in the KEY ORDER: ``_loss_for`` splits anchor, then bc, then
base, then the corruption level ``t``, and lets the proposal consume what is
left. ``_grads_for`` has to reproduce that exactly. A mismatch would not crash or
even look wrong -- both arms would train, on different noise draws, and the
"equivalence reference" would quietly be a different objective.

No model, no corpus, no GPU: both paths are stubbed down to their key derivation
and their argument routing, as ``test_bc_term`` does for the BC term.

    python -m rl.diffusion_ft.tests.test_unroll_jit
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft import train_safety as T

FAILED = []
B, H, K_UNROLL = 2, 4, 3


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _key_ints(k):
    # Legacy raw uint32 keys (what `jax.random.PRNGKey` returns here) array
    # directly; typed keys need `key_data`. Handle both so the check does not
    # silently compare two objects that are never equal.
    try:
        return tuple(int(x) for x in np.asarray(k).reshape(-1))
    except (TypeError, ValueError):
        return tuple(int(x) for x in np.asarray(jax.random.key_data(k)).reshape(-1))


def _ctx():
    c = T.ReplanContext(t0=3, cond1=jnp.zeros((B, 2)), cond2=jnp.zeros((B, 2)),
                        mask=jnp.ones((B, 1)), aux=None,
                        alive=jnp.ones((B,), jnp.float32), n_steps=10)
    c.expert_target = jnp.arange(B * H * 2, dtype=jnp.float32).reshape(B, H, 2)
    return c


def _tuner(*, anchor_weight=0.0, bc_weight=0.0, base_anchor_weight=0.0):
    """A SafetyFineTuner with only what the two gradient paths touch."""
    t = T.SafetyFineTuner.__new__(T.SafetyFineTuner)
    t.args = SimpleNamespace(
        anchor_weight=anchor_weight, bc_weight=bc_weight,
        base_anchor_weight=base_anchor_weight, safety_weight=1.0,
        unroll_eta=0.0, unroll_eager=False, base_anchor_success_only=False)
    t.objcfg = SimpleNamespace(safety_weight=1.0)
    t.unroll_steps = K_UNROLL
    t.unroll_grad_steps = 1
    t.t_end = 0
    t.t_start_max = 9
    t._anchor_tw = None
    t._anchor_mask = None
    t._steps_train = None
    t.actor = SimpleNamespace(
        _tparams={"w": jnp.zeros((1,))},
        num_sample_steps=None,
        diffusion=SimpleNamespace(timesteps=100,
                                  denoise_fn=SimpleNamespace(horizon=H)),
        _expert_loss=lambda *a, **k: jnp.asarray(0.0),
        base_anchor_loss=lambda *a, **k: jnp.asarray(0.0),
        safety_grads=None)
    t.env = SimpleNamespace(norm_to_world_with=lambda n, aux: n)
    return t


def _run_eager(tuner, ctx, rng, bc_ctx=None):
    """`_loss_for` + `safety_grads`: capture what the proposal is handed."""
    seen = {}

    def propose_unrolled(params, c, x0_bct, t_grid_kb, r):
        seen["rng"] = r
        seen["t_grid"] = np.asarray(t_grid_kb)
        seen["x0_bct"] = np.asarray(x0_bct)
        return jnp.zeros((B, 6, 5))

    tuner._propose_unrolled = propose_unrolled
    real_cs = T.candidate_safety
    T.candidate_safety = lambda cfg, w, **kw: {
        "total": jnp.asarray(0.0), "edge_loss": jnp.zeros((B, 1)),
        "agent_loss": jnp.zeros((B, 1))}
    try:
        loss_fn = tuner._loss_for(object(), ctx, rng, bc_ctx)
        loss_fn(tuner.actor._tparams)          # forces the propose call
    finally:
        T.candidate_safety = real_cs
    return seen


def _run_jitted(tuner, ctx, rng, bc_ctx=None):
    """`_grads_for`: capture the `dyn`/`keys` handed to the traced core."""
    seen = {}

    def fake_grad_fn(params, state, dyn, keys, **static):
        seen["keys"], seen["dyn"], seen["static"] = keys, dyn, static
        return (jnp.asarray(0.0), {}), params

    tuner._free_grad_fn = lambda: fake_grad_fn
    tuner._grads_for(object(), ctx, rng, bc_ctx)
    return seen


# --------------------------------------------------------------------------- #
def gate_streams_match(**weights) -> None:
    label = ",".join(f"{k.split('_')[0]}={v}" for k, v in weights.items()) or "safety only"
    rng = jax.random.PRNGKey(4242)
    # The BC term only splits a key when it HAS a context, so supply one whenever
    # its weight is on -- otherwise the `bc_weight=1` case would test nothing.
    bc = _ctx() if float(weights.get("bc_weight", 0.0)) > 0 else None
    eager = _run_eager(_tuner(**weights), _ctx(), rng, bc)
    jit = _run_jitted(_tuner(**weights), _ctx(), rng, bc)

    check(jit["static"]["has_unroll"] is True,
          f"[{label}] the unroll routes through the jitted core (has_unroll)")
    check(_key_ints(jit["keys"]["propose"]) == _key_ints(eager["rng"]),
          f"[{label}] the PROPOSAL key matches the eager closure's")
    check(np.array_equal(jit["dyn"]["t_grid"], eager["t_grid"]),
          f"[{label}] the corruption grid matches ({eager['t_grid'][0].tolist()} -> "
          f"{eager['t_grid'][-1].tolist()})")
    check(np.array_equal(np.asarray(jit["dyn"]["x0_bct"]), eager["x0_bct"]),
          f"[{label}] the teacher-forced target matches")


def gate_prefix_is_dropped() -> None:
    """A stored frozen prefix is meaningless under the unroll and must not ride along."""
    ctx = _ctx()
    ctx.x_prefix, ctx.key_late = jnp.zeros((B, 5, H)), jax.random.PRNGKey(0)
    jit = _run_jitted(_tuner(), ctx, jax.random.PRNGKey(1))
    check(jit["static"]["has_prefix"] is False,
          "a stored x_prefix is ignored under the unroll (nothing samples the chain)")
    check("x_prefix" not in jit["dyn"],
          "... and is not carried in `dyn` either")


def gate_eager_flag_still_works() -> None:
    """`--unroll_eager` must actually reach the old closure, or it is not a reference."""
    t = _tuner()
    t.args.unroll_eager = True
    called = {}
    t.actor.safety_grads = lambda fn: called.setdefault("hit", True) or (None, 0.0, {})
    t._propose_unrolled = lambda *a, **k: jnp.zeros((B, 6, 5))
    real_cs = T.candidate_safety
    T.candidate_safety = lambda cfg, w, **kw: {
        "total": jnp.asarray(0.0), "edge_loss": jnp.zeros((B, 1)),
        "agent_loss": jnp.zeros((B, 1))}
    try:
        t._grads_for(object(), _ctx(), jax.random.PRNGKey(0))
    finally:
        T.candidate_safety = real_cs
    check(called.get("hit") is True, "--unroll_eager reaches the un-jitted closure")


def main() -> int:
    print("=== jitted vs eager teacher-forced unroll ===")
    # Every combination of the terms that consume the key stream BEFORE the
    # proposal does: each one shifts the proposal key, so the orders must agree
    # under all of them, not just the single-term default.
    gate_streams_match()
    gate_streams_match(anchor_weight=1.0)
    gate_streams_match(bc_weight=1.0)
    gate_streams_match(base_anchor_weight=1.0)
    gate_streams_match(anchor_weight=1.0, bc_weight=1.0, base_anchor_weight=1.0)
    gate_prefix_is_dropped()
    gate_eager_flag_still_works()
    print("\nverdict:", "FAILED" if FAILED else "PASSED")
    for m in FAILED:
        print("  FAIL", m)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
