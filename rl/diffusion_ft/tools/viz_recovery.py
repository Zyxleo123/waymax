"""Per-stratum headline bars FROM ``select_summary.json`` (CPU, no model, no re-roll).

Companion to ``plot_select.py``. That figure is the corpus-marginal story (the
splits reweighted to corpus proportions, so gains read as ~0.02 because failures
are ~1.5% of the corpus). THIS figure is the raw per-stratum story a report leads
with -- the two numbers the deploy-time line is judged on, side by side:

* FAILURE RECOVERY: solved rate on ``failure_heldout`` (base ~0.05 -> arm ~0.55).
* SUCCESS NON-REGRESSION: solved rate on ``success_heldout`` (the do-no-harm
  check, ~1.0), annotated with the fraction of successes the arm newly BROKE.
* INTERMITTENT (the middle stratum) for context.

Each panel is horizontal bars per arm with the base rate drawn as a reference
line. Reads any number of select_summary.json so several runs (e.g. the 3-seed
b_l24 run and the ramp/blend sweep) merge into one figure; an arm seen twice
keeps its first occurrence. Fine-tune arms use ``ckpt_mean`` (mean over seeds).

    python -m rl.diffusion_ft.tools.viz_recovery \
      --in_json A/select_summary.json B/select_summary.json \
      --arms base,ft,bf_w10,bf_w20,b_l24,c_r20 --out_png recovery.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, ORANGE, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
PANELS = (
    ("failure_heldout", "Failure recovery", (0.0, 1.0)),
    ("intermittent_heldout", "Intermittent", (0.0, 1.0)),
    ("success_heldout", "Success non-regression", (0.95, 1.005)),
)


def collect(in_jsons):
    """Return ordered {arm_label: {split: stats}}; first occurrence of an arm wins."""
    arms: dict[str, dict] = {}
    for jp in in_jsons:
        S = json.loads(Path(jp).read_text())
        # fine-tune arms: prefer the seed mean
        for arm, r in S.get("ckpt_mean", {}).items():
            if "splits" in r:
                arms.setdefault(arm, r["splits"])
        for k, r in S.get("policies", {}).items():
            if "splits" not in r:
                continue
            base = k.split("@")[0]
            if base in arms:                       # already have a ckpt_mean or earlier run
                continue
            if "@" in k and base in S.get("ckpt_mean", {}):
                continue                           # covered by its mean above
            arms.setdefault(base, r["splits"])
    return arms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", nargs="+", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--arms", default="", help="comma list to select & order; default = all found")
    p.add_argument("--title", default="Deploy-time repair: per-stratum solve rate vs base")
    a = p.parse_args()

    found = collect(a.in_json)
    order = [x for x in a.arms.split(",") if x] if a.arms else list(found)
    rows = [(x, found[x]) for x in order if x in found]
    if not rows:
        raise SystemExit(f"no matching arms in {a.in_json}; found {list(found)}")
    missing = [x for x in order if x not in found]
    if missing:
        print(f"[viz_recovery] not found, skipped: {missing}")

    labels = [lab for lab, _ in rows]
    y = list(range(len(rows)))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.5 * len(rows) + 1.8), sharey=True)
    fig.patch.set_facecolor(BG)

    for ax, (split, title, xlim) in zip(axes, PANELS):
        base_rate = next((st[split]["base_solved_rate"] for _, st in rows if split in st), None)
        for i, (lab, st) in enumerate(rows):
            s = st.get(split)
            if s is None:
                continue
            solved = s["solved_rate"]
            # colour: below base = regression (orange), else blue
            c = ORANGE if base_rate is not None and solved < base_rate - 1e-9 else BLUE
            ax.barh(y[i], solved, color=c, height=0.62)
            txt = f"{solved:.3f}"
            if split == "success_heldout" and s.get("broken", 0) > 0:
                txt += f"  (broke {s['broken']*100:.1f}%)"
            ax.annotate(txt, (min(solved, xlim[1]), y[i]), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8, color=MUTED, clip_on=False)
        if base_rate is not None:
            ax.axvline(base_rate, color=INK, lw=1.3, ls="--")
            ax.annotate(f"base {base_rate:.3f}", (base_rate, len(rows) - 0.4),
                        xytext=(3, 0), textcoords="offset points", fontsize=8,
                        color=INK, va="bottom")
        ax.set_xlim(*xlim)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("solved rate", color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_facecolor(BG)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y, labels, fontsize=9, color=INK)
    axes[0].legend(handles=[Patch(color=BLUE, label="at/above base"),
                            Patch(color=ORANGE, label="below base (regression)")],
                   frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(a.title, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out_png, dpi=150)
    print(f"[viz_recovery] {len(rows)} arms -> {a.out_png}")


if __name__ == "__main__":
    main()
