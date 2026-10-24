"""Pure-logic tests for the 1000-scene eval and the balanced training batch.

No model, no corpus, no GPU. Three things are worth pinning without paying for a
rollout, because all three fail SILENTLY -- they produce a plausible number:

* **Chunked eval must equal one batch.** `--split_cap 1000` is rolled in chunks
  of `--eval_chunk`; if the aggregate were a plain mean over chunks it would
  silently mis-weight a ragged tail, and `arc_ratio_vs_log` (a ratio of two
  means) would be wrong even with equal chunks.
* **Balanced batches must hold their quota.** A rounding bug that drops the
  minority group turns "trained on failures" back into the uniform draw the
  whole 09-04 sweep existed to escape, and nothing in the curve says so.
* **The rebuild must drop ambiguous scenes.** A scene the base solves half the
  time belongs in neither pool.

    python -m rl.diffusion_ft.tests.test_eval_chunking
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from rl.diffusion_ft.scene_sampler import SceneSampler
from rl.diffusion_ft.train_safety import SafetyFineTuner

FAILED = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


# --------------------------------------------------------------------------- #
def test_chunked_eval_equals_one_batch():
    """Per-scene truth -> the chunked aggregate is the whole-batch number."""
    solved = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], float)
    arc = np.arange(10, 101, 10, dtype=float)
    arclog = np.array([12, 18, 33, 44, 45, 66, 70, 88, 99, 90], float)

    class Stub(SafetyFineTuner):
        def __init__(self):
            pass

        def evaluate(self, state, rng=None):
            i = state
            return {"solved_rate": solved[i].mean(), "arc_m": arc[i].mean(),
                    "arc_ratio_vs_log": arc[i].mean() / arclog[i].mean(),
                    "_n": float(len(i)), "_arc_log_m": arclog[i].mean()}

    t = Stub()
    whole = t.evaluate(np.arange(10))
    # Deliberately ragged: 4 + 4 + 2, which is what --split_cap 1000 with
    # --eval_chunk 64 produces on any split that is not a multiple of 64.
    got = t._eval_chunks([(np.arange(0, 4), 4), (np.arange(4, 8), 4),
                          (np.arange(8, 10), 2)])
    for k in ("solved_rate", "arc_m", "arc_ratio_vs_log"):
        check(abs(whole[k] - got[k]) < 1e-9,
              f"1 chunked {k} == single batch ({got[k]:.6f})")
    check(got["_n"] == 10, "1 chunked eval counts every scene")

    # A plain (unweighted) mean over chunks would get the ragged tail wrong --
    # confirm the test would actually catch that.
    naive = np.mean([np.mean(solved[0:4]), np.mean(solved[4:8]), np.mean(solved[8:10])])
    check(abs(naive - whole["solved_rate"]) > 1e-3,
          "1 the ragged-tail case does discriminate weighted from naive")


# --------------------------------------------------------------------------- #
def _sampler_stub(groups, weights, eval_ids=(), batch=16, seed=0):
    s = SceneSampler.__new__(SceneSampler)
    s.rng = np.random.default_rng(seed)
    s._eval_set = set(int(i) for i in eval_ids)
    s.groups = [np.asarray(g, dtype=np.int64) for g in groups]
    s.group_weights = np.asarray(weights, float) / np.sum(weights)
    s.batch_size = batch
    return s


def test_balanced_batches():
    fail, succ = np.arange(0, 400), np.arange(1000, 24000)
    s = _sampler_stub([fail, succ], [1, 1])
    check(s._group_quota(16) == [8, 8], "2 equal weights split a 16-batch 8/8")
    check(sum(s._group_quota(15)) == 15, "2 quotas sum to the batch size (odd)")
    s2 = _sampler_stub([fail, succ], [0.02, 0.98])
    check(sum(s2._group_quota(16)) == 16, "2 quotas sum to the batch size (skewed)")

    for _ in range(5):
        ids = s._draw_ids(16)
        n_fail = int((ids < 400).sum())
        check(n_fail == 8 and len(set(ids.tolist())) == 16,
              f"2 every batch is 8 failures + 8 successes, all distinct "
              f"(got {n_fail})")

    # Held-out ids never appear, and a group too small for its quota raises
    # instead of looping forever.
    held = list(range(0, 396))
    s3 = _sampler_stub([fail, succ], [1, 1], eval_ids=held)
    try:
        s3._draw_ids(16)
        check(False, "2 an undersized group raises")
    except ValueError:
        check(True, "2 an undersized group raises instead of hanging")

    s4 = _sampler_stub([fail, succ], [1, 1], eval_ids=range(0, 100))
    ids = s4._draw_ids(16)
    check(not (set(ids.tolist()) & set(range(0, 100))),
          "2 held-out eval ids are never drawn for training")


# --------------------------------------------------------------------------- #
def test_rebuild_drops_ambiguous():
    """0/4 stays a failure, 4/4 stays a success, 2/4 is dropped from both."""
    spec = {"splits": {
        "failure_heldout": {"scene_ids": [1, 2]},
        "failure_train": {"scene_ids": [3, 4]},
        "success_heldout": {"scene_ids": [10, 11]},
        "success_train": {"scene_ids": [12, 13, 14]},
    }}
    rows = [
        {"scene_id": 1, "solved_frac": 0.0, "clean_frac": 1.0},    # keep
        {"scene_id": 2, "solved_frac": 0.5, "clean_frac": 1.0},    # ambiguous
        {"scene_id": 3, "solved_frac": 0.0, "clean_frac": 0.0},    # expert-dirty
        {"scene_id": 4, "solved_frac": 0.0, "clean_frac": 1.0},    # keep
        {"scene_id": 10, "solved_frac": 1.0, "clean_frac": 1.0},   # keep
        {"scene_id": 11, "solved_frac": 0.75, "clean_frac": 1.0},  # ambiguous
        {"scene_id": 12, "solved_frac": 1.0, "clean_frac": 1.0},   # keep
        # 13, 14 unverified -> pool only, never held out
    ]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "rows.jsonl").write_text(
            "\n".join(json.dumps(dict(r, split="x", repeats=4, verify_seed=777))
                      for r in rows) + "\n")
        (td / "in.json").write_text(json.dumps(spec))
        r = subprocess.run(
            [sys.executable, "-m", "rl.diffusion_ft.tools.rebuild_verified_splits",
             "--rows", str(td / "rows.jsonl"), "--in_json", str(td / "in.json"),
             "--out_json", str(td / "out.json"), "--heldout_failure", "1",
             "--heldout_success", "1"],
            cwd=REPO_ROOT, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu",
                 "PYTHONPATH": str(REPO_ROOT)})
        if r.returncode != 0:
            check(False, f"3 rebuild exited {r.returncode}: {r.stderr[-400:]}")
            return
        out = json.loads((td / "out.json").read_text())
        sp = {k: set(v["scene_ids"]) for k, v in out["splits"].items()}
        fails = sp["failure_heldout"] | sp["failure_train"]
        check(fails == {1, 4}, f"3 only 0/4 expert-clean scenes are failures ({fails})")
        check(2 not in fails and 2 not in sp["success_train"],
              "3 a sometimes-solved scene is dropped from BOTH pools")
        check(sp["success_heldout"] <= {10, 12},
              f"3 held-out successes are verified 4/4 ({sp['success_heldout']})")
        check(11 not in sp["success_train"],
              "3 a sometimes-failed success is not in the training pool")
        check({13, 14} <= sp["success_train"],
              "3 unverified successes stay in the training pool")
        check(len(sp["failure_heldout"]) == 1 and len(sp["failure_train"]) == 1,
              "3 --heldout_failure partitions the verified failures")


def main():
    test_chunked_eval_equals_one_batch()
    test_balanced_batches()
    test_rebuild_drops_ambiguous()
    print("\nverdict:", "PASSED" if not FAILED else f"FAILED ({len(FAILED)})")
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
