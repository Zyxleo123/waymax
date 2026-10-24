"""Render the post-hoc reach/solved eval of a safety checkpoint. CPU only.

Reads the artifacts ``train_safety.py --eval_only`` writes into a dump dir:

  reach_rollouts.npz   roadgraph + goal + human log + per-arm executed paths,
                       offroad marks and absorption step (fig_paths schema)
  reach_rows.jsonl     per (scene, arm) outcome flags (solved/reached/off/col)
  reach_summary.json   base-vs-ft aggregate rates

and produces two figures:

  reach_summary.png    base vs fine-tuned aggregate rates -- the headline the
                       training log never had (solved, reached, offroad, overlap)
  reach_paths.png      "how it drives": each eval scene on its roadgraph, human
                       log dashed, base and fine-tuned paths overlaid, offroad
                       steps ✕-marked, titled with each arm's S/R/O/C outcome

No GPU and no model: it only reads the dump, so the figures re-render cheaply
whenever the question changes. The GPU eval job produces the dump once.

Usage:
    python -m rl.diffusion_ft.tools.viz_reach DUMP_DIR [--out_dir DIR] [--scenes N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless compute node
import matplotlib.pyplot as plt
import numpy as np

from rl.diffusion_ft.tests.viz_drive import fig_paths, _SURFACE, _INK, _INK2

# Base vs fine-tuned keep a fixed identity across every panel (Okabe-Ito).
_ARM_COLOR = {"base": "#D55E00", "ft": "#0072B2"}
# The four rates that matter, and whether up is good (solved/reached) or bad
# (offroad/overlap) -- annotated so the bar chart is not read backwards.
_RATES = [("solved_rate", "solved", "good"),
          ("reached_rate", "reached goal", "good"),
          ("offroad_rate", "offroad", "bad"),
          ("overlap_rate", "collision", "bad")]


def fig_summary(summary: dict, out_path: Path, n_scenes: int) -> None:
    arms = [a for a in ("base", "ft") if a in summary]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=_SURFACE)
    x = np.arange(len(_RATES))
    w = 0.36
    for i, arm in enumerate(arms):
        vals = [summary[arm].get(k, float("nan")) for k, _, _ in _RATES]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=_ARM_COLOR[arm],
                      label=f"{arm}", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, color=_INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n({'↑ better' if d == 'good' else '↓ better'})"
                        for _, lab, d in _RATES], fontsize=9.5, color=_INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of eval scenes", fontsize=10, color=_INK)
    ax.set_title(f"Held-out closed-loop eval  ({n_scenes} scenes)  —  base vs fine-tuned",
                 fontsize=12, color=_INK)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", color="#eceff1", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5dade")
    ax.tick_params(colors=_INK2)
    # The honest caveat, on the figure: solved is over the whole eval batch;
    # solved_rate_clean uses only scenes whose own log is clean.
    sc = summary.get("ft", {})
    if "solved_rate_clean" in sc:
        ax.text(0.0, -0.22,
                f"ft solved over clean-expert scenes only: "
                f"{sc['solved_rate_clean']:.2f}  "
                f"(clean_frac {sc.get('clean_frac', float('nan')):.2f});  "
                f"drift {sc.get('drift_m', float('nan')):.2f} m, "
                f"arc {sc.get('arc_ratio_vs_log', float('nan')):.2f}×log",
                transform=ax.transAxes, fontsize=8.5, color=_INK2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump_dir")
    p.add_argument("--out_dir", default="")
    p.add_argument("--scenes", type=int, default=16,
                   help="scenes drawn in the path grid")
    args = p.parse_args()

    dump = Path(args.dump_dir)
    out = Path(args.out_dir) if args.out_dir else dump
    out.mkdir(parents=True, exist_ok=True)

    # Render every batch present. Each was dumped as reach_summary<suffix>.json
    # where suffix is "" (the run's own held-out) or "_<tag>" (train / a named
    # split like failure_heldout / success_heldout). Discover them by glob so any
    # set of splits renders without the viz needing to know their names.
    tags = []
    for jf in sorted(dump.glob("reach_summary*.json")):
        suffix = jf.stem[len("reach_summary"):]        # "" or "_<tag>"
        label = suffix[1:] if suffix else "heldout"
        tags.append((suffix, label))
    for suffix, label in tags:
        summary = json.loads((dump / f"reach_summary{suffix}.json").read_text())
        rows = [json.loads(l) for l in (dump / f"reach_rows{suffix}.jsonl").open()
                if l.strip()]
        npz = np.load(dump / f"reach_rollouts{suffix}.npz")
        n_scenes = int(npz["indices"].shape[0])
        fig_summary(summary, out / f"reach_summary_{label}.png", n_scenes)
        fig_paths(npz, rows, out / f"reach_paths_{label}.png",
                  arms=["base", "ft"], n_scenes=args.scenes)


if __name__ == "__main__":
    main()
