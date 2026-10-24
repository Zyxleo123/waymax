"""The corpus-level ("marginal") solve rate for a finished or running arm.

WHY THIS EXISTS. The eval splits are a STRATIFIED sample, and the strata are
wildly disproportionate: verified failures are ~1.5% of the expert-clean corpus
and verified successes ~98.5%. Every curve this package draws is per-split, so
none of them answers the question a deployment asks -- did the overall solve rate
go up? Worse, the per-split view misleads about the SIGN. Measured on the first
arm (bc300, 300 iterations):

    failure_heldout  0.040 -> 0.364   (+0.324, worth +0.0049 marginal)
    success_heldout  0.998 -> 0.962   (-0.036, worth -0.0355 marginal)
    marginal         0.9837 -> 0.9530  ->  a NET REGRESSION of 0.031

Two curves that both look like wins, and the policy solves ~3 fewer scenes per
100 than the base. The break-even condition is brutal and worth internalizing:
a gain of `d` on the failure split is paid for only if the success split stays
within `d * w_fail / w_succ = d / 65.8` of its base -- a +0.30 failure gain
allows a success drop of **0.0046**.

Weights come from the `population` block that `rebuild_verified_splits` writes,
so they are a property of the splits file, not of this script. Scenes the base
solves only *sometimes* are in neither stratum; `coverage` reports how much of
the corpus the marginal spans.

CPU only, reads saved rows. `--in_place` appends the `marginal/*` keys into
train_log.jsonl so `wandb_backfill` pushes them to an existing run -- do NOT use
it on a job that is still appending to that log.

    python -m rl.diffusion_ft.tools.marginal_rate \
        --run OUT_DIR --splits_json .../eval_splits_verified.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("solved_rate", "offroad_rate", "overlap_rate", "reached_rate")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, nargs="+",
                   help="one or more run dirs (each holding train_log.jsonl)")
    p.add_argument("--splits_json", required=True,
                   help="the splits file the run was scored against; its "
                        "`population` block supplies the stratum weights")
    p.add_argument("--in_place", action="store_true",
                   help="rewrite train_log.jsonl with the marginal/* keys added, "
                        "so wandb_backfill can push them. Never do this while "
                        "the job is still running.")
    p.add_argument("--every", type=int, default=1,
                   help="print every Nth eval point")
    return p.parse_args()


def weights(splits_json: Path):
    spec = json.loads(splits_json.read_text())
    pop = spec.get("population")
    if not pop:
        raise SystemExit(
            f"{splits_json} has no `population` block -- it predates the "
            f"marginal rate. Re-run rebuild_verified_splits on the same rows "
            f"(deterministic: the split ids do not change).")
    w = {k: float(v["weight"]) for k, v in pop["strata"].items()}
    return w, float(pop.get("coverage", 0.0))


def marginal(row, w, prefix=""):
    """Weighted rates from one log row, or ``{}`` if a stratum is missing."""
    out = {}
    for m in METRICS:
        keys = [f"{prefix}{k}/{m}" for k in w]
        if all(k in row for k in keys):
            out[m] = sum(w[k] * row[f"{prefix}{k}/{m}"] for k in w)
    return out


def main() -> None:
    a = _parse_args()
    w, cov = weights(Path(a.splits_json))
    wf = w.get("failure_heldout", 0.0)
    print(f"[marginal] weights: " + "  ".join(f"{k}={v:.6f}" for k, v in w.items())
          + f"   coverage={cov:.4f} of the expert-clean corpus")
    print(f"[marginal] a +0.10 failure gain is worth {0.10 * wf:+.4f}; it is paid "
          f"for only if the success split stays within "
          f"{0.10 * wf / max(1 - wf, 1e-9):.4f} of its base\n")

    for run in a.run:
        run = Path(run)
        log = run / "train_log.jsonl"
        if not log.exists():
            print(f"[marginal] {run}: no train_log.jsonl, skipping")
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        base = next((marginal(r, w, prefix="base_") for r in rows
                     if marginal(r, w, prefix="base_")), {})
        b = base.get("solved_rate")
        print(f"=== {run.name}   base marginal "
              + (f"{b:.4f}" if b is not None else "n/a"))
        print(f"{'iter':>6} {'failure':>8} {'success':>8} {'marginal':>9} "
              f"{'vs base':>9} {'breakeven':>10}")
        out_rows, n = [], 0
        for r in rows:
            m = marginal(r, w)
            if not m:
                out_rows.append(r)
                continue
            r = dict(r)
            r.update({f"marginal/{k}": v for k, v in m.items()})
            r["marginal/coverage"] = cov
            if b is not None:
                r["marginal/delta_vs_base"] = m["solved_rate"] - b
                d_fail = (r["failure_heldout/solved_rate"]
                          - r.get("base_failure_heldout/solved_rate", 0.0))
                r["marginal/breakeven_success_rate"] = (
                    r.get("base_success_heldout/solved_rate", 0.0)
                    - wf * d_fail / max(1 - wf, 1e-9))
            out_rows.append(r)
            if n % max(1, a.every) == 0:
                print(f"{r['iter']:>6} {r['failure_heldout/solved_rate']:>8.3f} "
                      f"{r['success_heldout/solved_rate']:>8.3f} "
                      f"{m['solved_rate']:>9.4f} "
                      f"{r.get('marginal/delta_vs_base', float('nan')):>+9.4f} "
                      f"{r.get('marginal/breakeven_success_rate', float('nan')):>10.4f}")
            n += 1
        if n == 0:
            print("  (no eval points with both held-out splits)")
        elif a.in_place:
            log.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
            print(f"  rewrote {log} with marginal/* keys "
                  f"(push with tools/wandb_backfill.py)")
        print()


if __name__ == "__main__":
    main()
