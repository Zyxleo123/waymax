"""Pure-logic tests for the resume path, which multi-day runs depend on.

No model, no corpus, no GPU: `save`/`load`/`_resume_iter` are exercised on a
stub actor whose "weights" and optimizer trees are a handful of small arrays.

WHY THESE. A run long enough to cross the partition's time limit is a CHAIN of
jobs, and every one of these failures is silent -- the job restarts, the log
continues, the curve looks continuous:

* **optimizer state dropped.** Adam's moments and step counter reset, and bias
  correction then makes the first steps after every restart full-size whatever
  the gradient is. Measured on this model: the null objective moves drift
  0.412 -> 0.717 m in ONE iteration, then only -> 0.810 over the next 80. One
  uncontrolled kick per restart, in a project where displacement rank-orders
  damage (Spearman -1.0).
* **the batch stream restarting.** A resumed run re-draws the same batches it
  already trained on, so a many-restart run sees the first few hundred batches
  over and over and the rest never.
* **tparams.npz gaining keys.** `load`, `eval_reach` and
  `probe_param_distance` all read it with `sorted(z.files)`; an extra key there
  permutes the pytree in each of them.

    python -m rl.diffusion_ft.tests.test_resume_state
"""

from __future__ import annotations

import json
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

from rl.diffusion_ft.train_safety import SafetyFineTuner

FAILED = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _tree(scale):
    """A stand-in for a params / Adam-moment pytree."""
    return {"w": jnp.arange(6, dtype=jnp.float32).reshape(2, 3) * scale,
            "b": jnp.asarray([scale, 2.0 * scale], jnp.float32),
            "count": jnp.asarray(int(scale), jnp.int32)}


def _tuner(out_dir, *, tparams=1.0, opt=2.0, opte=3.0, adam_steps=17,
           resume=1, keep_every=0):
    t = SafetyFineTuner.__new__(SafetyFineTuner)
    t.args = SimpleNamespace(
        out_dir=str(out_dir), resume=resume, ckpt="CKPT", k_trainable=10,
        anchor_weight=0.0, edge_margin=0.3, agent_margin=0.75,
        anchor_states="policy", grad_accum=0, lr=1e-5, eval_seed=12345,
        eval_repeats=2, unroll_eta=0.0, keep_every=keep_every)
    t.unroll_steps = 0
    t.unroll_grad_steps = 1
    t.t_start_max = 9
    t.t_end = 0
    t._adam_steps = adam_steps
    t.rng = jax.random.PRNGKey(4242)
    actor = SimpleNamespace(_tparams=_tree(tparams), _opt_state=_tree(opt),
                            _opt_state_expert=_tree(opte),
                            _rng=jax.random.PRNGKey(99))
    actor.state_dict = lambda: {"tparams": actor._tparams,
                                "opt_state": actor._opt_state,
                                "opt_state_expert": actor._opt_state_expert,
                                "rng": actor._rng}
    t.actor = actor
    return t


def leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_flatten(tree)[0]]


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(leaves(a), leaves(b)))


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---- 1. round trip: weights AND optimizer state -------------------
        src = _tuner(td)
        src.save(td / "latest")
        dst = _tuner(td, tparams=99.0, opt=99.0, opte=99.0, adam_steps=0)
        dst.rng = jax.random.PRNGKey(0)
        dst.load(td / "latest")
        check(same(dst.actor._tparams, _tree(1.0)), "1 weights round-trip")
        check(same(dst.actor._opt_state, _tree(2.0)),
              "1 Adam moments round-trip (opt_state)")
        check(same(dst.actor._opt_state_expert, _tree(3.0)),
              "1 Adam moments round-trip (opt_state_expert)")
        check(dst._adam_steps == 17, "1 the Adam step COUNT survives a resume")
        check(np.array_equal(np.asarray(dst.rng),
                             np.asarray(jax.random.PRNGKey(4242))),
              "1 the training rng survives a resume")
        check(np.asarray(dst.actor._opt_state["count"]).dtype == np.int32,
              "1 integer optimizer leaves keep their dtype")

        # ---- 2. tparams.npz keeps its exact key set ------------------------
        z = np.load(td / "latest" / "tparams.npz")
        check(all(k.startswith("p") for k in z.files),
              f"2 tparams.npz holds only p-keys ({sorted(z.files)})")
        check((td / "latest" / "opt_state.npz").exists(),
              "2 optimizer state goes to its own file")

        # ---- 3. an old checkpoint still loads, loudly -----------------------
        (td / "old").mkdir()
        src.save(td / "old")
        (td / "old" / "opt_state.npz").unlink()
        (td / "old" / "train_state.json").unlink()
        d2 = _tuner(td, tparams=99.0, opt=7.0, opte=7.0)
        d2.load(td / "old")
        check(same(d2.actor._tparams, _tree(1.0)),
              "3 a pre-optimizer-state checkpoint still restores weights")
        check(same(d2.actor._opt_state, _tree(7.0)),
              "3 ... and keeps the fresh optimizer state rather than crashing")

        # ---- 4. snapshots are weights only ---------------------------------
        src.save(td / "iter_000500", with_opt=False)
        check((td / "iter_000500" / "tparams.npz").exists()
              and not (td / "iter_000500" / "opt_state.npz").exists(),
              "4 a kept snapshot omits the optimizer state")

        # ---- 5. the resume point, and the batch stream that follows it ------
        log = td / "train_log.jsonl"
        log.write_text("".join(json.dumps({"iter": i, "n_steps": 1}) + "\n"
                               for i in range(40)) + '{"iter": 40, "n_ste')
        check(_tuner(td)._resume_iter() == 40,
              "5 the resume point ignores a torn last line")
        check(_tuner(td, resume=0)._resume_iter() == 0,
              "5 --resume 0 starts from scratch")
        seeds = {int(0 + 100003 * _tuner(td)._resume_iter()),
                 int(0 + 100003 * 0)}
        check(len(seeds) == 2,
              "5 a resumed run seeds the sampler differently, so the batch "
              "stream continues instead of repeating")

    with tempfile.TemporaryDirectory() as td:
        check(_tuner(Path(td))._resume_iter() == 0,
              "5 no log and no checkpoint means iteration 0")

    print("\nverdict:", "PASSED" if not FAILED else f"FAILED ({len(FAILED)})")
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
