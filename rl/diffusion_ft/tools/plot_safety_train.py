"""Redraw the safety fine-tune curves from ``train_log.jsonl``.

Reads only the JSONL the trainer appends, so figures are reproducible without
the checkpoint, the corpus or a GPU -- rerun this against any finished
``out_dir`` and it redraws in seconds.

The point of the layout is the degeneracy check. A single-term safety objective
can drive its own loss down by leaving the demonstrated line, so the safety
curve is meaningless on its own: read it against the eval panel, where a good
run shows offroad/overlap falling while drift and arc-ratio stay flat. Falling
loss with rising drift is the failure, not the success.

Usage: python -m rl.diffusion_ft.tools.plot_safety_train --out_dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def series(rows, key):
    """(iters, values) for the rows that carry ``key`` -- eval rows are sparse."""
    it = [r["iter"] for r in rows if key in r and r[key] is not None]
    v = [r[key] for r in rows if key in r and r[key] is not None]
    return np.asarray(it), np.asarray(v, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fig_dir", default="")
    args = p.parse_args()

    out = Path(args.out_dir)
    rows = load(out / "train_log.jsonl")
    if not rows:
        raise SystemExit(f"no rows in {out}/train_log.jsonl")
    fig_dir = Path(args.fig_dir) if args.fig_dir else out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    it, s = series(rows, "safety_last")
    ax[0, 0].plot(it, s, lw=1.2, color="tab:blue", label="safety (end of iter)")
    it0, s0 = series(rows, "safety_first")
    ax[0, 0].plot(it0, s0, lw=0.8, color="tab:blue", alpha=0.35, label="safety (start)")
    ax[0, 0].set_title("objective (single term)"); ax[0, 0].set_xlabel("iteration")
    ax[0, 0].set_ylabel("safety loss"); ax[0, 0].legend(fontsize=8)

    for k, c in (("edge_last", "tab:orange"), ("agent_last", "tab:green")):
        i, v = series(rows, k)
        if v.size:
            ax[0, 1].plot(i, v, lw=1.1, color=c, label=k.replace("_last", ""))
    ax[0, 1].set_title("safety by term"); ax[0, 1].set_xlabel("iteration")
    ax[0, 1].legend(fontsize=8)

    for k, c, lab in (("eval_offroad_rate", "tab:red", "offroad"),
                      ("eval_overlap_rate", "tab:purple", "collision")):
        i, v = series(rows, k)
        if v.size:
            ax[1, 0].plot(i, v, "o-", ms=3, lw=1.2, color=c, label=lab)
    ax[1, 0].set_title("held-out closed loop (what we actually want)")
    ax[1, 0].set_xlabel("iteration"); ax[1, 0].set_ylabel("rate"); ax[1, 0].legend(fontsize=8)

    # The degeneracy panel: clearance bought by wandering shows up here.
    i, d = series(rows, "eval_drift_m")
    if d.size:
        ax[1, 1].plot(i, d, "o-", ms=3, lw=1.2, color="tab:brown", label="drift from log (m)")
    i2, r = series(rows, "eval_arc_ratio_vs_log")
    if r.size:
        ax2 = ax[1, 1].twinx()
        ax2.plot(i2, r, "s--", ms=3, lw=1.0, color="tab:gray", label="arc / log arc")
        ax2.axhline(1.0, color="tab:gray", lw=0.6, ls=":")
        ax2.set_ylabel("arc ratio")
        ax2.legend(fontsize=8, loc="lower right")
    ax[1, 1].set_title("DEGENERACY CHECK -- flat is good")
    ax[1, 1].set_xlabel("iteration"); ax[1, 1].set_ylabel("m"); ax[1, 1].legend(fontsize=8, loc="upper left")

    fig.suptitle(f"safety fine-tune -- {out.name}  ({len(rows)} iters)")
    fig.savefig(fig_dir / "safety_train.png", dpi=140)
    print(f"[plot] wrote {fig_dir}/safety_train.png")

    # A one-line verdict beside the figure, so the chain leaves a readable trace.
    last = rows[-1]
    ev = {k: last.get(k) for k in ("eval_offroad_rate", "eval_overlap_rate",
                                   "eval_drift_m", "eval_arc_ratio_vs_log")}
    first_ev = next((r for r in rows if "eval_offroad_rate" in r), None)
    summary = {"iters": len(rows), "final": ev,
               "first_eval": {k: first_ev.get(k) for k in ev} if first_ev else None,
               "safety_first": rows[0].get("safety_first"),
               "safety_last": last.get("safety_last")}
    (fig_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
