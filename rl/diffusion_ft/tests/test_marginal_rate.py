"""Pure-logic tests for the corpus-level ("marginal") rate.

No model, no corpus, no GPU. The marginal exists because the eval splits are a
stratified sample with wildly disproportionate strata (verified failures ~1.5% of
the corpus, verified successes ~98.5%), so per-split curves can BOTH look like
wins while the overall solve rate falls. Test 3 is that exact case, taken from
the first measured arm.

The two ways this can silently lie:

* **a partial marginal.** If one stratum's split is not evaluated, summing the
  rest reweights the corpus to whatever happened to be measured. It has to
  refuse, not renormalize.
* **a break-even that does not break even.** The reported
  `breakeven_success_rate` is only useful if a success split AT that value puts
  the marginal exactly back on the base.

    python -m rl.diffusion_ft.tests.test_marginal_rate
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.diffusion_ft.tools.marginal_rate import marginal as tool_marginal
from rl.diffusion_ft.train_safety import SafetyFineTuner

FAILED = []
# The measured weights (rebuild_verified_splits, 2026-09-08).
W = {"failure_heldout": 0.014966, "success_heldout": 0.985034}


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def _tuner(w=W, cov=0.9654):
    t = SafetyFineTuner.__new__(SafetyFineTuner)
    t._marg_w = dict(w)
    t._marg_cov = cov
    return t


def main():
    t = _tuner()

    # ---- 1. weights, and the weighted sum -------------------------------
    check(abs(sum(W.values()) - 1.0) < 1e-9, "1 stratum weights sum to 1")
    per = {"failure_heldout": {"solved_rate": 0.364},
           "success_heldout": {"solved_rate": 0.962}}
    got = t._marginal(per)["marginal/solved_rate"]
    want = 0.014966 * 0.364 + 0.985034 * 0.962
    check(abs(got - want) < 1e-12, f"1 marginal is the weighted sum ({got:.6f})")

    # ---- 2. every metric present in both splits is carried --------------
    per = {"failure_heldout": {"solved_rate": 0.3, "offroad_rate": 0.4,
                               "overlap_rate": 0.1},
           "success_heldout": {"solved_rate": 0.9, "offroad_rate": 0.05,
                               "overlap_rate": 0.0}}
    m = t._marginal(per)
    check({"marginal/solved_rate", "marginal/offroad_rate",
           "marginal/overlap_rate"} <= set(m),
          f"2 all shared metrics get a marginal ({sorted(m)})")
    check("marginal/reached_rate" not in m,
          "2 a metric missing from a split is skipped, not zero-filled")
    check(abs(m["marginal/coverage"] - 0.9654) < 1e-9,
          "2 coverage is reported alongside")

    # ---- 3. THE TRAP: both splits improve on their own terms, marginal falls
    base = {"failure_heldout": {"solved_rate": 0.040},
            "success_heldout": {"solved_rate": 0.998}}
    arm = {"failure_heldout": {"solved_rate": 0.364},
           "success_heldout": {"solved_rate": 0.962}}
    mb = t._marginal(base)["marginal/solved_rate"]
    ma = t._marginal(arm)["marginal/solved_rate"]
    check(arm["failure_heldout"]["solved_rate"] > base["failure_heldout"]["solved_rate"]
          and ma < mb,
          f"3 a +0.32 failure gain with a -0.036 success cost is a NET LOSS "
          f"({mb:.4f} -> {ma:.4f}, {ma - mb:+.4f})")
    check(abs((ma - mb) - (-0.0311)) < 5e-4,
          f"3 ... of the measured size ({ma - mb:+.4f})")

    # ---- 4. break-even actually breaks even ------------------------------
    wf = W["failure_heldout"]
    d_fail = arm["failure_heldout"]["solved_rate"] - base["failure_heldout"]["solved_rate"]
    breakeven = (base["success_heldout"]["solved_rate"]
                 - wf * d_fail / (1 - wf))
    at_be = t._marginal({"failure_heldout": arm["failure_heldout"],
                         "success_heldout": {"solved_rate": breakeven}})
    check(abs(at_be["marginal/solved_rate"] - mb) < 1e-9,
          f"4 a success rate at break-even ({breakeven:.4f}) restores the base "
          f"marginal exactly")
    check(breakeven > 0.99,
          f"4 break-even demands the success split stay above 0.99 ({breakeven:.4f})")

    # ---- 5. a partial marginal is refused, not renormalized --------------
    check(t._marginal({"failure_heldout": {"solved_rate": 0.5}}) == {},
          "5 a missing stratum returns nothing rather than a reweighted corpus")
    check(_tuner(w={})._marginal(per) == {},
          "5 no weights (a splits file with no population block) means no marginal")

    # ---- 6. the standalone tool agrees with the trainer -------------------
    row = {"failure_heldout/solved_rate": 0.364, "success_heldout/solved_rate": 0.962}
    check(abs(tool_marginal(row, W)["solved_rate"] - ma) < 1e-12,
          "6 tools/marginal_rate agrees with the trainer's own computation")
    check(tool_marginal({"failure_heldout/solved_rate": 0.3}, W) == {},
          "6 ... and refuses a partial row the same way")

    print("\nverdict:", "PASSED" if not FAILED else f"FAILED ({len(FAILED)})")
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
