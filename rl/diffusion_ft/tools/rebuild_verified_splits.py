"""Rebuild the eval splits from scenes the base policy VERIFIABLY fails.

The splits in `hard_womd/scan_seed0/eval_splits.json` were labelled by
`scan_womd_headroom`, which rolls the base checkpoint DETERMINISTICALLY at
`sigma_floor=0.03` under its own reward config. The safety training curve scores
the same checkpoint through `train_safety.rollout(train=False)`: a stochastic
sampler at `sigma_floor=1e-4`, 8 replans, and the expert-clean gate. Those are
different measurements, and the disagreement is large -- the base scores
`solved=0.328` on `failure_heldout`, a split defined as "the base fails it".

A third of the failure split being pre-solved is not a rounding error: it caps
the visible headroom, and any movement on those scenes is re-measurement noise
rather than a repair. This script consumes `train_safety --verify_only` rows
(the real eval path, at a seed disjoint from `--eval_seed`) and keeps only:

  * FAILURES: the base solved the scene in **0 of N** verification rollouts, and
    the scene is expert-clean. Anything it solves sometimes is dropped from both
    pools rather than reassigned -- a scene the policy solves half the time is
    not evidence for or against a repair.
  * SUCCESSES (held out): solved in **N of N**, expert-clean. This is the
    do-no-harm set, and it has to be unambiguous for a drop in it to mean
    anything.

The TRAINING success pool is deliberately NOT filtered to verified scenes: the
gradient should see all the data (`--train_split_ratio` balances the mix), and
requiring 4/4 rollouts on 23k scenes would cost more compute than the fine-tune.
Only the held-out success set, which is a measurement, is verified.

CPU only; reads saved rows, so it is re-runnable at any threshold without
touching the GPU. Usage (submitted, never inline):

    python -m rl.diffusion_ft.tools.rebuild_verified_splits \
        --rows OUT/verify_rows.jsonl --in_json .../eval_splits.json \
        --out_json .../eval_splits_verified.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, help="train_safety --verify_only output")
    p.add_argument("--in_json", required=True, help="the splits file that was verified")
    p.add_argument("--out_json", required=True)
    p.add_argument("--heldout_failure", type=int, default=-1,
                   help="failures held out for eval; -1 = half of the verified "
                        "failures, the rest going to the training pool")
    p.add_argument("--heldout_success", type=int, default=1000,
                   help="verified successes held out as the do-no-harm eval set")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the failure held-out/train partition")
    p.add_argument("--fail_max_solved", type=float, default=0.0,
                   help="keep a failure only if solved_frac <= this (0 = the "
                        "base never solved it in any verification rollout)")
    p.add_argument("--success_min_solved", type=float, default=1.0,
                   help="keep a held-out success only if solved_frac >= this")
    return p.parse_args()


def read_rows(path: Path) -> dict[int, dict]:
    """Per-scene verification rows, last write wins (a resumed job may repeat)."""
    rows: dict[int, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn last line from a killed task
            rows[int(r["scene_id"])] = r
    return rows


def main() -> None:
    a = _parse_args()
    rows = read_rows(Path(a.rows))
    spec = json.loads(Path(a.in_json).read_text())
    src = {n: [int(i) for i in v["scene_ids"]] for n, v in spec["splits"].items()}
    print(f"[rebuild] {len(rows)} verified scenes; source splits: "
          + ", ".join(f"{n}={len(v)}" for n, v in src.items()))

    fail_src = sorted(set(src.get("failure_heldout", []) + src.get("failure_train", [])))
    succ_src = sorted(set(src.get("success_heldout", []) + src.get("success_train", [])))

    # ---- failures: verified, expert-clean, never solved ---------------------
    kept_fail, drop = [], Counter()
    for sid in fail_src:
        r = rows.get(sid)
        if r is None:
            drop["unverified"] += 1
        elif r.get("clean_frac", 1.0) < 1.0:
            drop["expert_dirty"] += 1
        elif r["solved_frac"] > a.fail_max_solved:
            drop["base_solves_it"] += 1
        else:
            kept_fail.append(sid)
    print(f"[rebuild] failures: {len(fail_src)} candidates -> {len(kept_fail)} "
          f"verified base-failures  (dropped: {dict(drop)})")
    if not kept_fail:
        raise SystemExit("[rebuild] no verified failures -- refusing to write a "
                         "splits file whose failure splits are empty")

    # ---- successes: verified, expert-clean, always solved -------------------
    kept_succ, sdrop = [], Counter()
    for sid in succ_src:
        r = rows.get(sid)
        if r is None:
            sdrop["unverified"] += 1
        elif r.get("clean_frac", 1.0) < 1.0:
            sdrop["expert_dirty"] += 1
        elif r["solved_frac"] < a.success_min_solved:
            sdrop["base_fails_it"] += 1
        else:
            kept_succ.append(sid)
    print(f"[rebuild] successes: {len(succ_src)} candidates -> {len(kept_succ)} "
          f"verified base-solves  (dropped: {dict(sdrop)})")

    rng = np.random.default_rng(int(a.seed))
    fail = np.asarray(kept_fail, dtype=np.int64)
    rng.shuffle(fail)
    n_fh = (fail.size // 2) if a.heldout_failure < 0 else min(int(a.heldout_failure),
                                                              fail.size - 1)
    failure_heldout = sorted(int(i) for i in fail[:n_fh])
    failure_train = sorted(int(i) for i in fail[n_fh:])

    succ = np.asarray(kept_succ, dtype=np.int64)
    rng.shuffle(succ)
    n_sh = min(int(a.heldout_success), succ.size)
    success_heldout = sorted(int(i) for i in succ[:n_sh])

    # The success TRAINING pool keeps every source success that is not held out
    # and is not a verified base-failure: the gradient sees all the data, and
    # `--train_split_ratio` (not a cap) decides how much of each batch it gets.
    held = set(success_heldout)
    ambiguous = {sid for sid in succ_src
                 if sid in rows and rows[sid]["solved_frac"] < a.success_min_solved}
    success_train = sorted(sid for sid in succ_src
                           if sid not in held and sid not in ambiguous)

    out = dict(spec)
    out["splits"] = {
        "failure_heldout": {
            "criteria": f"verified: base solved_frac <= {a.fail_max_solved} over "
                        f"N rollouts of the eval path, expert-clean, held out",
            "scene_ids": failure_heldout},
        "success_heldout": {
            "criteria": f"verified: base solved_frac >= {a.success_min_solved}, "
                        f"expert-clean, held out",
            "scene_ids": success_heldout},
        "failure_train": {
            "criteria": "verified base-failure, in the training pool",
            "scene_ids": failure_train},
        "success_train": {
            "criteria": "source successes minus held-out and minus verified "
                        "base-failures; the training pool, NOT individually "
                        "verified",
            "scene_ids": success_train},
    }
    out["counts"] = {k: len(v["scene_ids"]) for k, v in out["splits"].items()}

    # ---- population weights, so a MARGINAL rate is computable ---------------
    # The splits are a STRATIFIED sample, and wildly disproportionate: the
    # verified-failure stratum is ~1.5% of the corpus and the verified-success
    # stratum ~98.5%. Per-split curves therefore hide the only number a
    # deployment cares about -- did the overall solve rate go up? Measured on the
    # first arm: failure 0.040 -> 0.364 (+0.324, worth +0.005 marginal) against
    # success 0.998 -> 0.962 (-0.036, worth -0.035), i.e. a NET REGRESSION that
    # both per-split curves looked like a success.
    #
    # `failure_heldout` is a random sample of the verified-failure stratum and
    # `success_heldout` of the verified-success stratum, so each split's rate
    # estimates its stratum's rate and the marginal is just the weighted sum.
    #
    # The success stratum's size is EXTRAPOLATED: only `n_sampled` of its
    # candidates were verified, so its size is scaled by the verified fraction.
    # The ambiguous scenes (solved sometimes) are in NEITHER stratum and are not
    # covered -- `coverage` says how much of the corpus the marginal spans.
    n_sampled = len(succ_src) - sdrop["unverified"]
    p_verified = (len(kept_succ) / n_sampled) if n_sampled else 0.0
    n_succ_stratum = len(succ_src) * p_verified
    n_fail_stratum = float(len(kept_fail))
    corpus = float(len(fail_src) + len(succ_src))
    denom = n_fail_stratum + n_succ_stratum
    out["population"] = {
        "corpus_expert_clean": corpus,
        "coverage": (denom / corpus) if corpus else 0.0,
        "success_verified_fraction": p_verified,
        "success_sampled": n_sampled,
        "strata": {
            "failure_heldout": {"n": n_fail_stratum,
                                "weight": (n_fail_stratum / denom) if denom else 0.0},
            "success_heldout": {"n": n_succ_stratum,
                                "weight": (n_succ_stratum / denom) if denom else 0.0},
        },
        "note": ("weights for the marginal (corpus-level) rate: "
                 "sum_s weight_s * rate_s over the HELD-OUT splits. The *_train "
                 "splits estimate the same strata but are trained on."),
    }
    out["verification"] = {
        "rows": str(a.rows), "source_splits_json": str(a.in_json),
        "n_rows": len(rows), "seed": int(a.seed),
        "fail_max_solved": float(a.fail_max_solved),
        "success_min_solved": float(a.success_min_solved),
        "failure_candidates": len(fail_src), "failure_kept": len(kept_fail),
        "failure_dropped": dict(drop),
        "success_candidates": len(succ_src), "success_kept": len(kept_succ),
        "success_dropped": dict(sdrop),
        "verify_repeats": int(next(iter(rows.values())).get("repeats", 0)) if rows else 0,
        "verify_seed": int(next(iter(rows.values())).get("verify_seed", -1)) if rows else -1,
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    print("[rebuild] counts: " + json.dumps(out["counts"]))
    pop = out["population"]
    print(f"[rebuild] strata: failure w={pop['strata']['failure_heldout']['weight']:.6f} "
          f"(n={pop['strata']['failure_heldout']['n']:.0f})  "
          f"success w={pop['strata']['success_heldout']['weight']:.6f} "
          f"(n={pop['strata']['success_heldout']['n']:.0f})  "
          f"coverage={pop['coverage']:.4f}")
    wf = pop["strata"]["failure_heldout"]["weight"]
    print(f"[rebuild] a +0.10 gain on the failure split is worth {0.10 * wf:+.4f} "
          f"marginal; break-even needs the success split within "
          f"{0.10 * wf / max(1 - wf, 1e-9):.4f} of its base")
    print(f"[rebuild] wrote -> {a.out_json}")


if __name__ == "__main__":
    main()
