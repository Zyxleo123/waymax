"""Pure-logic tests for the BC term (`--bc_weight`), the pretraining loss.

No model, no corpus, no GPU: `_loss_for` is stubbed down to its arithmetic and
its argument routing. Every failure mode below produces a *plausible* training
curve, which is why they are worth pinning:

* **conditioning on the wrong states.** The BC term exists precisely because the
  anchor conditions on policy-visited states. If `_loss_for` read `ctx` instead
  of `bc_ctx`, `--bc_weight` would silently BE `--anchor_weight` and the whole
  experiment would compare an arm against itself.
* **the wrong alive mask.** Weighting the human's states by which scenes the
  POLICY was still driving is a subtle version of the same error.
* **a shared rng key.** `diffusion._loss_impl` splits one key into BOTH the
  timestep and the noise. Anchor and BC sharing a key would correlate two terms
  that must be independent draws; a *frozen* key is what made the 09-02 BC
  control the worst arm of six.

    python -m rl.diffusion_ft.tests.test_bc_term
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


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _ctx(tag, alive):
    c = T.ReplanContext(t0=0, cond1=jnp.asarray([[float(tag)]]),
                        cond2=jnp.asarray([[float(tag)]]),
                        mask=jnp.asarray([[1.0]]), aux=None,
                        alive=jnp.asarray(alive, jnp.float32), n_steps=10)
    # [B, T, D]; the horizon slice in _loss_for must keep the tag readable.
    c.expert_target = jnp.full((1, 4, 2), float(tag))
    return c


def _tuner(*, bc_weight, anchor_weight, safety_weight=1.0, calls=None):
    t = T.SafetyFineTuner.__new__(T.SafetyFineTuner)
    t.args = SimpleNamespace(anchor_weight=anchor_weight, bc_weight=bc_weight,
                             safety_weight=safety_weight)
    t.objcfg = SimpleNamespace(safety_weight=safety_weight)
    t.unroll_steps = 0
    # Whole-horizon anchor window (None), as `_init_windows` resolves --anchor_window -1.
    t._anchor_tw = None

    def expert_loss(params, expert, alive_b, time_weight=None):
        target, c1, c2, mask, rng = expert
        calls.append({"tag": float(np.asarray(target).reshape(-1)[0]),
                      "alive": float(np.asarray(alive_b).sum()),
                      "key": tuple(int(x) for x in np.asarray(rng).reshape(-1))})
        # A distinct constant per source so the total is attributable.
        return jnp.asarray(float(np.asarray(target).reshape(-1)[0]))

    t.actor = SimpleNamespace(
        _expert_loss=expert_loss,
        diffusion=SimpleNamespace(timesteps=100,
                                  denoise_fn=SimpleNamespace(horizon=4)))
    t._propose = lambda params, ctx, rng: jnp.zeros((1, 4, 5))
    return t


def run(tuner, ctx, bc_ctx):
    """_loss_for with candidate_safety stubbed to a constant 2.0."""
    orig = T.candidate_safety
    T.candidate_safety = lambda cfg, world, **kw: {
        "total": jnp.asarray(2.0), "edge_loss": jnp.zeros(1),
        "agent_loss": jnp.zeros(1)}
    try:
        fn = tuner._loss_for(state=None, ctx=ctx, rng=jax.random.PRNGKey(0),
                             bc_ctx=bc_ctx)
        total, aux = fn({})
        return float(total), {k: float(v) for k, v in aux.items()}
    finally:
        T.candidate_safety = orig


def main():
    # ---- 1. the term is added, with the right weight ------------------------
    calls = []
    t = _tuner(bc_weight=3.0, anchor_weight=0.0, calls=calls)
    total, aux = run(t, _ctx(7, [1.0]), _ctx(5, [1.0]))
    # safety 2.0 * sw 1.0  +  bc 5.0 * bw 3.0
    check(abs(total - (2.0 + 15.0)) < 1e-6,
          f"1 total = sw*safety + bw*bc ({total})")
    check(abs(aux["bc"] - 5.0) < 1e-6, "1 aux['bc'] reports the raw BC loss")
    check(abs(aux["anchor"]) < 1e-9, "1 anchor stays 0 when anchor_weight=0")

    # ---- 2. it conditions on bc_ctx, not on the policy context --------------
    check(len(calls) == 1 and calls[0]["tag"] == 5.0,
          f"2 the BC term reads bc_ctx (tag 5), not ctx (tag 7): {calls}")

    # ---- 3. it uses bc_ctx's own alive mask ---------------------------------
    calls = []
    t = _tuner(bc_weight=1.0, anchor_weight=0.0, calls=calls)
    run(t, _ctx(7, [1.0, 1.0, 1.0]), _ctx(5, [1.0, 0.0, 0.0]))
    check(calls[0]["alive"] == 1.0,
          f"3 BC is weighted by bc_ctx.alive (1), not ctx.alive (3): {calls}")

    # ---- 4. anchor and BC coexist, on different contexts and keys -----------
    calls = []
    t = _tuner(bc_weight=1.0, anchor_weight=2.0, calls=calls)
    total, aux = run(t, _ctx(7, [1.0]), _ctx(5, [1.0]))
    tags = sorted(c["tag"] for c in calls)
    check(tags == [5.0, 7.0], f"4 anchor reads ctx and BC reads bc_ctx ({tags})")
    check(abs(total - (2.0 + 2.0 * 7.0 + 1.0 * 5.0)) < 1e-6,
          f"4 total = safety + aw*anchor + bw*bc ({total})")
    keys = [c["key"] for c in calls]
    check(keys[0] != keys[1],
          "4 anchor and BC draw independent (t, eps) keys")

    # ---- 5. off means off ---------------------------------------------------
    calls = []
    t = _tuner(bc_weight=0.0, anchor_weight=0.0, calls=calls)
    total, aux = run(t, _ctx(7, [1.0]), _ctx(5, [1.0]))
    check(not calls and abs(total - 2.0) < 1e-6,
          "5 bc_weight=0 computes no BC term at all")
    calls = []
    t = _tuner(bc_weight=3.0, anchor_weight=0.0, calls=calls)
    total, _ = run(t, _ctx(7, [1.0]), None)
    check(not calls and abs(total - 2.0) < 1e-6,
          "5 a missing bc_ctx is a no-op, not a crash")

    print("\nverdict:", "PASSED" if not FAILED else f"FAILED ({len(FAILED)})")
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
