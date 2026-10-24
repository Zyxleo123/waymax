"""Aggregate a DPPO reward-shaping sweep into one table and figure.

The question this answers: the `penalty_up` run (job 37016, offroad -1.0 /
collision -1.5 charged per world step) drove eval return from -24 to -2.7 while
*worsening* the off-road rate (0.28 -> 0.50) and reaching the goal in only 4/32
scenes. Under a per-step charge the cheapest way to raise return is to shorten
the excursion -- or to stop driving -- not to avoid the violation, and the
progress term is a bounded indicator worth at most +0.2 per world step, so at
-1.0 an off-road step forfeits five steps of the best possible progress. The
`noescape` run (job 36946, both penalties -0.2, i.e. exactly one progress step)
drove instead, reaching +9.3 return, but was never instrumented with the
reached/offroad/collided rates, so the two points cannot be compared today.

This reads the sweep those two runs bracket: penalty scale x goal bonus, every
arm logging the same rates. Input is a sweep root holding one subdirectory per
arm, each with an `arm.json` (written by the submitter) and a timestamped run
directory underneath holding `train_log.jsonl`. CPU only -- no model, no GPU --
so the table and the figure re-render cheaply from the artifacts.

Two things are checked rather than assumed:
  * an arm whose log is missing or has no eval rows is reported as missing,
    never silently dropped from the ranking;
  * the iteration-0 eval row is carried through as the *base policy* reference,
    so an arm that merely fails to move is distinguishable from one that hurts.

Reported per arm is the tail mean over the last `--tail` eval points rather than
the final point: single-seed eval on 32 scenes moves +-1 scene (0.031) between
adjacent points, and the s*/lr* sweep already lost time to reading such spikes
as trends.

Usage:
    python -m rl.diffusion_ft.tests.aggregate_reward_sweep SWEEP_DIR \
        [--tail 5] [--out_dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless compute node
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, the same slots viz_drive/aggregate_sigma_sweep use, so a metric
# keeps its colour across the whole diagnosis package.
C_OFFROAD = "#D55E00"
C_COLLIDED = "#CC79A7"
C_REACHED = "#009E73"
C_RETURN = "#0072B2"
C_SOLVE = "#E69F00"
C_REF = "#666666"

# Expert log replay on this 32-scene failure subset (job 36390): the ceiling any
# reward can be judged against. Reaching is not the hard part for the expert --
# staying clean is what the policy has to learn.
EXPERT_REACHED = 1.00
EXPERT_CLEAN = 0.94


def _run_dir(arm_dir: Path) -> Path | None:
    """The newest timestamped run directory under one arm.

    `train_dppo` appends its own `%Y%m%d_%H%M%S` under `--out_dir`, so an arm
    directory holds one (or, after a resubmit, several) run directories. The
    newest one that actually has a log is the arm's run.
    """
    cands = sorted(
        (d for d in arm_dir.iterdir() if d.is_dir() and (d / "train_log.jsonl").exists()),
        key=lambda d: d.name,
    )
    return cands[-1] if cands else None


def _eval_rows(log_path: Path) -> list[dict]:
    """Eval rows from one train_log.jsonl, in iteration order.

    A row is an eval row iff it carries the deterministic solve rate; the
    per-iteration rows in between hold only training diagnostics. A truncated
    final line (the job was killed mid-write) is skipped rather than fatal.
    """
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval_solve_deterministic" in row:
                rows.append(row)
    return sorted(rows, key=lambda r: r["iter"])


# The columns that make up one arm's verdict. `d` = deterministic eval (the
# policy that would be deployed), `s` = stochastic (what DPPO collects with).
_METRICS = [
    ("solve", "eval_solve_deterministic", "eval_solve_stochastic"),
    ("reach", "eval_reached_deterministic", "eval_reached_stochastic"),
    ("off", "eval_offroad_deterministic", "eval_offroad_stochastic"),
    ("col", "eval_collided_deterministic", "eval_collided_stochastic"),
    ("ret", "eval_return_deterministic", "eval_return_stochastic"),
]


def _tail_mean(rows: list[dict], key: str, tail: int) -> float:
    vals = [r[key] for r in rows[-tail:] if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(arm_dir: Path, tail: int) -> dict:
    """One arm -> its label, config, base-policy row, and tail-mean verdict."""
    meta_path = arm_dir / "arm.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out = {"arm": arm_dir.name, "meta": meta, "rows": [], "missing": None}

    run = _run_dir(arm_dir)
    if run is None:
        out["missing"] = "no run directory with train_log.jsonl"
        return out
    rows = _eval_rows(run / "train_log.jsonl")
    if not rows:
        out["missing"] = f"{run.name}/train_log.jsonl has no eval rows"
        return out

    out["run"] = run
    out["rows"] = rows
    out["n_eval"] = len(rows)
    out["last_iter"] = int(rows[-1]["iter"])
    # Iteration 0 is the base policy scored under *this arm's* reward, so the
    # return column is only comparable within an arm -- the rate columns are
    # comparable across arms (they do not depend on the weights at all).
    out["base"] = {name: rows[0].get(dk, float("nan")) for name, dk, _ in _METRICS}
    for name, dk, sk in _METRICS:
        out[f"{name}_d"] = _tail_mean(rows, dk, tail)
        out[f"{name}_s"] = _tail_mean(rows, sk, tail)
    # The across-seed spread, when the run was launched with eval_seeds > 1: the
    # band a difference between arms has to clear to be real.
    out["solve_d_std"] = _tail_mean(rows, "eval_solve_deterministic_std", tail)
    return out


def _fmt(x: float, width: int = 6, prec: int = 3) -> str:
    return f"{'--':>{width}}" if not np.isfinite(x) else f"{x:{width}.{prec}f}"


def print_table(summaries: list[dict], tail: int) -> None:
    ok = [s for s in summaries if not s["missing"]]
    bad = [s for s in summaries if s["missing"]]

    print(f"\n=== reward sweep: tail-{tail} mean of the eval rates ===")
    print(f"expert log replay on this set: reached={EXPERT_REACHED:.2f} "
          f"clean={EXPERT_CLEAN:.2f}  (the ceiling)")
    print(f"{'arm':>16} {'off':>6} {'col':>6} {'goal':>6} {'prog':>6} | "
          f"{'solve_d':>7} {'reach_d':>7} {'off_d':>6} {'col_d':>6} {'ret_d':>8} | "
          f"{'reach_s':>7} {'off_s':>6} {'col_s':>6} {'ret_s':>8} | {'its':>5}")
    # Rank by the real objective first (solve), then by goal-reaching: with solve
    # pinned near 0 across a whole sweep, the reached column is what separates
    # "drives to the goal dirty" from "learned to stop".
    for s in sorted(ok, key=lambda s: (-s["solve_d"], -s["reach_d"])):
        m = s["meta"]
        print(f"{s['arm']:>16} "
              f"{_fmt(m.get('offroad', float('nan')))} "
              f"{_fmt(m.get('collision', float('nan')))} "
              f"{_fmt(m.get('goal_bonus', float('nan')), 6, 1)} "
              f"{_fmt(m.get('progress', float('nan')))} | "
              f"{_fmt(s['solve_d'], 7)} {_fmt(s['reach_d'], 7)} "
              f"{_fmt(s['off_d'])} {_fmt(s['col_d'])} {_fmt(s['ret_d'], 8, 2)} | "
              f"{_fmt(s['reach_s'], 7)} {_fmt(s['off_s'])} {_fmt(s['col_s'])} "
              f"{_fmt(s['ret_s'], 8, 2)} | {s['last_iter']:5d}")

    if ok:
        print("\n--- base policy (iteration-0 eval, deterministic) ---")
        print("These are the same weights in every arm, so a spread here is eval "
              "seed noise and sets the band a difference has to clear.")
        print(f"{'arm':>16} {'solve':>7} {'reach':>7} {'off':>6} {'col':>6}")
        for s in sorted(ok, key=lambda s: s["arm"]):
            b = s["base"]
            print(f"{s['arm']:>16} {_fmt(b['solve'], 7)} {_fmt(b['reach'], 7)} "
                  f"{_fmt(b['off'])} {_fmt(b['col'])}")
        # A pre-instrumentation arm has no reached column at all; np.std over a
        # list holding one nan would report the whole band as nan.
        reach0 = [s["base"]["reach"] for s in ok if np.isfinite(s["base"]["reach"])]
        if len(reach0) > 1:
            print(f"base reached spread across arms (seed noise): "
                  f"std={float(np.std(reach0)):.3f} (1 scene = {1/32:.3f})")
        else:
            print(f"base reached: fewer than 2 arms reported it; no seed band "
                  f"(1 scene = {1/32:.3f})")

    for s in bad:
        print(f"MISSING {s['arm']}: {s['missing']}")
    if not ok:
        print("No arm produced eval rows -- nothing to rank.")


def plot(summaries: list[dict], out_path: Path, tail: int) -> None:
    ok = [s for s in summaries if not s["missing"]]
    if not ok:
        print("[rewardagg] no arms with eval rows; skipping the figure.")
        return
    panels = [
        ("eval_reached_deterministic", "reached (det)", C_REACHED, EXPERT_REACHED),
        ("eval_offroad_deterministic", "offroad-ever (det)", C_OFFROAD, 1 - EXPERT_CLEAN),
        ("eval_collided_deterministic", "collided-ever (det)", C_COLLIDED, 1 - EXPERT_CLEAN),
        ("eval_solve_deterministic", "solve (det)", C_SOLVE, EXPERT_CLEAN),
        ("eval_return_deterministic", "return (det)", C_RETURN, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    colors = {s["arm"]: cmap(i / max(1, len(ok) - 1)) for i, s in enumerate(sorted(ok, key=lambda s: s["arm"]))}

    for ax, (key, title, accent, ref) in zip(axes.flat, panels):
        for s in sorted(ok, key=lambda s: s["arm"]):
            it = [r["iter"] for r in s["rows"] if key in r]
            v = [r[key] for r in s["rows"] if key in r]
            ax.plot(it, v, lw=1.4, color=colors[s["arm"]], label=s["arm"])
        if ref is not None:
            ax.axhline(ref, color=C_REF, ls="--", lw=1.0)
        ax.set_title(title, color=accent, fontweight="bold")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    # The return panel is the one axis that is NOT comparable across arms: each
    # arm scores the same driving with different weights. Say so on the figure
    # rather than letting the curves invite the comparison.
    axes.flat[4].set_xlabel("iteration\n(return is per-arm units, not comparable)")

    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", frameon=False,
                        title=f"arm (tail-{tail} means in the table)")
    fig.suptitle("DPPO reward sweep: per-step penalty scale x goal bonus", fontweight="bold")
    fig.savefig(out_path, dpi=140)
    print(f"[rewardagg] figure -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", type=str)
    p.add_argument("--tail", type=int, default=5,
                   help="number of trailing eval points to average (default 5)")
    p.add_argument("--out_dir", type=str, default="",
                   help="where to write the figure/summary (default: sweep_dir)")
    a = p.parse_args()

    sweep = Path(a.sweep_dir)
    out_dir = Path(a.out_dir) if a.out_dir else sweep
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_dirs = sorted(d for d in sweep.iterdir() if d.is_dir())
    if not arm_dirs:
        print(f"[rewardagg] no arm directories under {sweep}")
        return
    summaries = [summarize(d, a.tail) for d in arm_dirs]
    print_table(summaries, a.tail)
    plot(summaries, out_dir / "reward_sweep.png", a.tail)

    payload = [
        {k: v for k, v in s.items() if k not in ("rows", "run")}
        for s in summaries
    ]
    (out_dir / "reward_sweep.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"[rewardagg] summary -> {out_dir / 'reward_sweep.json'}")


if __name__ == "__main__":
    main()
