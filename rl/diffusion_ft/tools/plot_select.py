"""Redraw the proposal-selection figure FROM ``select_summary.json`` (CPU, no model).

Left: corpus-level (three-stratum) solve-rate delta vs the base, 95% scene
bootstrap CI, one row per arm (fine-tune arms as the checkpoint mean), with the
"any candidate succeeds" oracle gains as hollow markers for scale.
Right: how each delta was made -- population-weighted recovered failures (right
of zero) against newly broken successes (left of zero).

    python -m rl.diffusion_ft.tools.plot_select --in_json S.json --out_png S.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--marginal", default="marginal3", choices=("marginal2", "marginal3"))
    a = p.parse_args()
    S = json.loads(Path(a.in_json).read_text())
    mk = a.marginal

    rows = []   # (label, stats)
    for arm, r in S["ckpt_mean"].items():
        if mk in r:
            rows.append((f"{arm} (mean of {r['n_ckpts']} ckpts)", r[mk]))
    pooled = {k for k in S["ckpt_mean"]}
    for k, r in S["policies"].items():
        if "@" not in k and mk in r:          # checkpoint-independent arms
            rows.append((k, r[mk]))
        elif k.split("@")[0] not in pooled and mk in r:
            rows.append((k, r[mk]))
    oracles = [(k, o[mk]) for k, o in S["oracle"].items() if o and mk in o]
    if not rows:
        raise SystemExit(f"nothing with {mk} in {a.in_json}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.55 * (len(rows) + len(oracles)) + 1.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    labels = [r[0] for r in rows] + [f"oracle: {k}" for k, _ in oracles]
    y = list(range(len(labels)))[::-1]
    for i, (lab, st) in enumerate(rows):
        lo, hi = st["ci95"]
        ax1.plot([lo, hi], [y[i]] * 2, color=BLUE, lw=2, solid_capstyle="round")
        ax1.plot(st["delta"], y[i], "o", ms=8, color=BLUE, mec="#fcfcfb", mew=2)
        ax1.annotate(f"{st['delta']:+.4f}", (hi, y[i]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, color=MUTED)
    for j, (lab, o) in enumerate(oracles):
        yy = y[len(rows) + j]
        lo, hi = o["ci95"]
        ax1.plot([lo, hi], [yy] * 2, color=ORANGE, lw=2, solid_capstyle="round")
        ax1.plot(o["gain"], yy, "o", ms=8, mfc="#fcfcfb", mec=ORANGE, mew=2)
        ax1.annotate(f"{o['gain']:+.4f}", (hi, yy), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax1.axvline(0, color=INK, lw=1)
    ax1.set_yticks(y, labels, fontsize=9, color=INK)
    ax1.set_xlabel(f"{mk} solve-rate delta vs base (95% CI)", color=MUTED)
    ax1.grid(axis="x", color=GRID, lw=0.8)

    yr = list(range(len(rows)))[::-1]
    for i, (lab, st) in enumerate(rows):
        ax2.barh(yr[i], st["recovered"], color=BLUE, height=0.6)
        ax2.barh(yr[i], -st["broken"], color=ORANGE, height=0.6)
    ax2.axvline(0, color=INK, lw=1)
    ax2.set_yticks(yr, [r[0] for r in rows], fontsize=9, color=INK)
    ax2.set_xlabel("population-weighted rate: broken (left) | recovered (right)", color=MUTED)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=BLUE, label="recovered failures"),
                        Patch(color=ORANGE, label="newly broken successes")],
               frameon=False, fontsize=8, loc="lower right")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Proposal selection vs base  ({S['repeats']} paired repeats/scene)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out_png, dpi=150)
    print(f"[plot_select] wrote -> {a.out_png}")


if __name__ == "__main__":
    main()
