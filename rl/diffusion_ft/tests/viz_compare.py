"""Redrawable comparison plot for a DPPO sweep group.

Reads the per-arm ``train_log.jsonl`` rows (written every iteration by
``train_dppo.py``) under a group directory and renders:

  * deterministic eval return vs iter, with the across-seed std as a band
    (the honest signal -- what the policy would score at deployment);
  * stochastic eval return vs iter (dashed, same colors);
  * a smoothed training return (mean_episode_return) for texture;
  * if the runs were profiled, the mean per-phase iteration time
    (collect sample / collect env / update) as a stacked bar per arm.

No model or GPU is needed -- it only reads the JSONL, so figures re-render
cheaply and belong on the ``cpu`` partition (see the companion .sbatch).

Usage:
    python -m rl.diffusion_ft.tests.viz_compare GROUP_DIR [--out PATH]

GROUP_DIR is the sweep's base dir; each arm is expected at
``GROUP_DIR/<arm>/<timestamp>/train_log.jsonl`` (the layout the submitter
writes). The newest timestamp per arm is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the compute node
import matplotlib.pyplot as plt
import numpy as np

# A small, fixed, colorblind-safe cycle so an arm keeps its color across panels.
_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _load_rows(jsonl: Path) -> list[dict]:
    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_arms(group_dir: Path) -> dict[str, Path]:
    """Map arm name -> newest train_log.jsonl under GROUP_DIR/<arm>/<ts>/."""
    arms: dict[str, Path] = {}
    for jsonl in sorted(group_dir.glob("*/*/train_log.jsonl")):
        arm = jsonl.parent.parent.name  # GROUP_DIR/<arm>/<ts>/train_log.jsonl
        # Keep the newest timestamp if an arm was rerun.
        if arm not in arms or jsonl.parent.name > arms[arm].parent.name:
            arms[arm] = jsonl
    return arms


def _series(rows: list[dict], key: str):
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r["iter"])
            ys.append(r[key])
    return np.asarray(xs), np.asarray(ys, dtype=float)


def _smooth(y: np.ndarray, w: int = 9) -> np.ndarray:
    if y.size < 3:
        return y
    w = min(w, y.size if y.size % 2 else y.size - 1)
    w = max(w, 1)
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("group_dir", type=str)
    p.add_argument("--out", type=str, default="", help="output PNG (default: GROUP_DIR/compare.png)")
    args = p.parse_args()

    group_dir = Path(args.group_dir)
    arms = _find_arms(group_dir)
    if not arms:
        raise SystemExit(f"No */*/train_log.jsonl under {group_dir}")
    out = Path(args.out) if args.out else group_dir / "compare.png"

    loaded = {arm: _load_rows(j) for arm, j in sorted(arms.items())}
    have_prof = any("t_sample" in r for rows in loaded.values() for r in rows)

    have_solve = any("eval_solve_deterministic" in r for rows in loaded.values() for r in rows)
    ncols = 1 + int(have_solve) + int(have_prof)
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 5.2), squeeze=False)
    col = 0
    axL = axes[0][col]; col += 1

    for i, (arm, rows) in enumerate(sorted(loaded.items())):
        c = _COLORS[i % len(_COLORS)]
        xd, yd = _series(rows, "eval_return_deterministic")
        if xd.size:
            _, ystd = _series(rows, "eval_return_deterministic_std")
            axL.plot(xd, yd, "-o", color=c, label=arm, lw=2, ms=4)
            if ystd.size == yd.size and ystd.size:
                axL.fill_between(xd, yd - ystd, yd + ystd, color=c, alpha=0.18, lw=0)
        xs, ys = _series(rows, "eval_return_stochastic")
        if xs.size:
            axL.plot(xs, ys, "--", color=c, lw=1, alpha=0.6)
        xt, yt = _series(rows, "mean_episode_return")
        if xt.size:
            axL.plot(xt, _smooth(yt), ":", color=c, lw=1, alpha=0.35)

    axL.axhline(0.0, color="0.6", lw=0.8, ls="-")
    axL.set_xlabel("iteration")
    axL.set_ylabel("episode return")
    axL.set_title("Deterministic eval (solid, ±seed std) · stochastic (dashed)\n"
                  "smoothed train return (dotted). Solve is strongly positive; 0 ≈ break-even.")
    axL.legend(fontsize=8, loc="lower right")
    axL.grid(True, alpha=0.25)

    if have_solve:
        axS = axes[0][col]; col += 1
        for i, (arm, rows) in enumerate(sorted(loaded.items())):
            c = _COLORS[i % len(_COLORS)]
            xs, ys = _series(rows, "eval_solve_deterministic")
            if xs.size:
                _, ystd = _series(rows, "eval_solve_deterministic_std")
                axS.plot(xs, ys, "-o", color=c, label=arm, lw=2, ms=4)
                if ystd.size == ys.size and ystd.size:
                    axS.fill_between(xs, ys - ystd, ys + ystd, color=c, alpha=0.18, lw=0)
        axS.set_ylim(-0.02, 1.02)
        axS.set_xlabel("iteration")
        axS.set_ylabel("solve rate")
        axS.set_title("Deterministic solve rate\n(reached goal & no collision & no offroad)")
        axS.legend(fontsize=8, loc="upper left")
        axS.grid(True, alpha=0.25)

    if have_prof:
        axR = axes[0][col]; col += 1
        names, s_samp, s_env, s_upd = [], [], [], []
        for arm, rows in sorted(loaded.items()):
            pr = [r for r in rows if "t_sample" in r and r["iter"] >= 5]  # skip compile/warmup
            if not pr:
                continue
            names.append(arm)
            s_samp.append(np.median([r["t_sample"] for r in pr]))
            s_env.append(np.median([r.get("t_env", 0.0) for r in pr]))
            s_upd.append(np.median([r.get("t_update", 0.0) for r in pr]))
        x = np.arange(len(names))
        s_samp, s_env, s_upd = map(np.asarray, (s_samp, s_env, s_upd))
        axR.bar(x, s_samp, label="collect: sample", color="#0072B2")
        axR.bar(x, s_env, bottom=s_samp, label="collect: env", color="#56B4E9")
        axR.bar(x, s_upd, bottom=s_samp + s_env, label="update", color="#D55E00")
        axR.set_xticks(x)
        axR.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axR.set_ylabel("median seconds / iter")
        axR.set_title("Per-phase iteration time (steady state)")
        axR.legend(fontsize=8)
        axR.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[viz] wrote {out}  ({len(arms)} arms: {', '.join(sorted(arms))})")


if __name__ == "__main__":
    main()
