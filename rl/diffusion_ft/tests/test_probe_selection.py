"""Pure-logic tests for the training pool and the probe's scene choice.

No model, no corpus, no GPU: `_resolve_train_pool` and `_build_probe` are the two
places where a run can *claim* to have trained on a scene it never touched, so
they are worth testing without paying for a rollout. Both are exercised on a
stub sampler and a hand-written splits JSON.

    python -m rl.diffusion_ft.tests.test_probe_selection
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

import numpy as np

from rl.diffusion_ft.train_safety import SafetyFineTuner


SPEC = {
    # No "shard_glob" key: the corpus check is skipped, which is what lets this
    # run without the 1000-shard WOMD tree present.
    "splits": {
        "failure_train": {"scene_ids": list(range(0, 100))},
        "success_train": {"scene_ids": list(range(100, 1100))},
        "failure_heldout": {"scene_ids": list(range(1100, 1120))},
        "success_heldout": {"scene_ids": list(range(1120, 1140))},
    }
}


def _args(**over):
    a = dict(splits_json="", shard_glob="glob", data_seed=0,
             train_splits="", train_split_cap="-1",
             probe_json="", probe_seed=0, probe_per_split=3,
             probe_splits="failure_train,success_train,"
                          "failure_heldout,success_heldout",
             probe_every=0, eval_every=10)
    a.update(over)
    return SimpleNamespace(**a)


def _tuner(args, pool=None, held=()):
    """A SafetyFineTuner with only the fields these two methods touch."""
    t = SafetyFineTuner.__new__(SafetyFineTuner)
    t.args = args
    t.train_pool_ids = pool
    t.sampler = SimpleNamespace(
        _eval_set=set(int(i) for i in held),
        load_explicit=lambda ids: ("state", list(ids)))
    return t


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        js = Path(td) / "splits.json"
        js.write_text(json.dumps(SPEC))
        ok = True

        # 1. no --train_splits == the historical behaviour: no pool at all.
        pool = _tuner(_args(splits_json=str(js)))._resolve_train_pool()
        assert pool is None, pool
        print("PASS  1 no --train_splits -> uniform corpus (pool is None)")

        # 2. caps are honoured per split and the union is deduplicated/sorted.
        a = _args(splits_json=str(js), train_splits="failure_train,success_train",
                  train_split_cap="40,60")
        pool = _tuner(a)._resolve_train_pool()
        n_fail = int(((pool >= 0) & (pool < 100)).sum())
        n_succ = int(((pool >= 100) & (pool < 1100)).sum())
        assert (n_fail, n_succ) == (40, 60), (n_fail, n_succ)
        assert (np.diff(pool) > 0).all(), "pool must be sorted and unique"
        print(f"PASS  2 per-split caps: failure {n_fail}  success {n_succ}")

        # 3. the cap subsample is deterministic under --data_seed, and moves with it.
        p2 = _tuner(a)._resolve_train_pool()
        assert (p2 == pool).all(), "same data_seed must give the same pool"
        a3 = _args(splits_json=str(js), train_splits="failure_train",
                   train_split_cap="40", data_seed=1)
        assert not np.array_equal(_tuner(a3)._resolve_train_pool(), pool[:40])
        print("PASS  3 cap subsample is deterministic in --data_seed")

        # 4. a single cap value applies to every split; a wrong-length list is
        #    an error, not a silent recycle.
        a4 = _args(splits_json=str(js),
                   train_splits="failure_train,success_train", train_split_cap="10")
        assert _tuner(a4)._resolve_train_pool().size == 20
        a5 = _args(splits_json=str(js),
                   train_splits="failure_train,success_train,failure_heldout",
                   train_split_cap="10,20")
        try:
            _tuner(a5)._resolve_train_pool()
            ok = False
            print("FAIL  4 mismatched --train_split_cap length was accepted")
        except SystemExit:
            print("PASS  4 one cap for all; a mismatched list is rejected")

        # 5. THE ONE THAT MATTERS: a probe scene is labelled `trained` only when
        #    it is in the pool AND not in the sampler's own held-out ids.
        a6 = _args(splits_json=str(js), train_splits="failure_train,success_train",
                   train_split_cap="40,60", probe_per_split=3)
        pool = _tuner(a6)._resolve_train_pool()
        held = [int(i) for i in pool[:35]]        # most of the failure half
        pr = _tuner(a6, pool=pool, held=held)._build_probe()
        by = dict(zip(pr["ids"], zip(pr["labels"], pr["trained"])))
        for i, (lab, tr) in by.items():
            in_pool = int(i) in set(int(x) for x in pool)
            expect = in_pool and int(i) not in set(held)
            if lab.endswith("_train") and tr != expect:
                ok = False
                print(f"FAIL  5 scene {i} ({lab}) trained={tr} expected {expect}")
            if lab.endswith("_heldout") and tr:
                ok = False
                print(f"FAIL  5 held-out scene {i} was labelled trained")
        n_tr = sum(pr["trained"])
        assert all(int(i) in set(int(x) for x in pool)
                   for i, (l, t) in by.items() if t)
        assert all(int(i) not in set(held) for i, (l, t) in by.items() if t)
        print(f"PASS  5 probe labels: {n_tr}/{len(pr['ids'])} trained-on, none of "
              f"them in the sampler's held-out ids")

        # 6. with no pool at all, NOTHING is labelled trained -- which is the
        #    honest reading of every run before 2026-09-04.
        pr = _tuner(_args(splits_json=str(js)))._build_probe()
        assert not any(pr["trained"]), pr["trained"]
        print("PASS  6 uniform-corpus run labels no probe scene as trained-on")

        # 7. the probe is deterministic in --probe_seed and independent of it only
        #    through the choice of scenes.
        p1 = _tuner(_args(splits_json=str(js), probe_seed=0))._build_probe()["ids"]
        p2 = _tuner(_args(splits_json=str(js), probe_seed=0))._build_probe()["ids"]
        p3 = _tuner(_args(splits_json=str(js), probe_seed=7))._build_probe()["ids"]
        assert p1 == p2 and p1 != p3
        print("PASS  7 probe scene choice is deterministic in --probe_seed")

        # 8. a missing split name is a hard error, not a skipped panel.
        try:
            _tuner(_args(splits_json=str(js), train_splits="nope"))._resolve_train_pool()
            ok = False
            print("FAIL  8 an unknown split name was accepted")
        except SystemExit:
            print("PASS  8 an unknown split name is rejected")

        print("\nverdict:", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
