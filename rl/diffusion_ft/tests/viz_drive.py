"""Draw the DPPO driving diagnosis from `diagnose_drive` artifacts. CPU only.

Reads ``rollouts.jsonl`` + ``rollouts.npz`` and renders two figures plus a
plain-text table (so every plotted value is also readable without the image):

  drive_diagnosis.png
    a) Episode outcome, mutually exclusive: solved / reached-but-dirty /
       not-reached-clean / not-reached-dirty. The trainer only logs the first
       bucket; the split between the other three is the diagnosis.
    b) Return decomposition. If the rising eval return is progress-shaped while
       the offroad term stays pinned at its one-time -5, the policy is being paid
       for closing distance, not for driving well -- reward hacking, on one plot.
    c) How far each episode got: fraction of the initial goal distance closed,
       and the closest approach ever made.
    d) Offroad exposure vs distance closed. A hacking policy lives in the
       upper-right (closes distance *by* leaving the road); a competent one in
       the lower-right.

  drive_paths.png
    Executed paths on the roadgraph, per scene, base vs fine-tuned vs the human
    log, with offroad world-steps marked. This is the "look at how it drives"
    panel -- the numbers say what happened, this says where.

No GPU and no model: it only reads the artifacts, so figures re-render cheaply
whenever the question changes.

Usage:
    python -m rl.diffusion_ft.tests.viz_drive DIAG_DIR [--out_dir DIR] [--scenes N]
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless compute node
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots, fixed order, assigned to arms by identity (never by rank),
# so an arm keeps its color across every panel and across reruns. Okabe-Ito
# subset; adjacent-pair CVD separation >= 10.9 (OKLab dE x100), all above the
# chroma floor and inside the light-mode lightness band.
_ARM_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
# Reward terms are semantic, not identities: helped / hurt / goal.
_TERM_COLORS = OrderedDict([
    ("r_progress", "#0072B2"),
    ("r_goal_bonus", "#009E73"),
    ("r_offroad", "#D55E00"),
    ("r_collision", "#CC79A7"),
])
_TERM_LABEL = {"r_progress": "progress", "r_goal_bonus": "goal bonus",
               "r_offroad": "offroad (-5, once)", "r_collision": "collision (-10, once)"}
# Outcome buckets: mutually exclusive, so a stacked bar sums to 1 honestly.
_OUTCOME_COLORS = OrderedDict([
    ("solved", "#009E73"),
    ("reached_dirty", "#D55E00"),
    ("short_clean", "#7f9bb0"),
    ("short_dirty", "#3c4b57"),
])
_OUTCOME_LABEL = {
    "solved": "solved (reached, clean)",
    "reached_dirty": "reached goal, but offroad/collided",
    "short_clean": "never reached, stayed clean",
    "short_dirty": "never reached, offroad/collided",
}
_SURFACE = "#ffffff"
_GRID = dict(color="#d9d9d9", lw=0.6, alpha=1.0)  # solid hairline, recessive
_INK, _INK2 = "#1f2328", "#5b6570"


def _load(diag_dir: Path):
    rows = [json.loads(l) for l in open(diag_dir / "rollouts.jsonl") if l.strip()]
    npz = np.load(diag_dir / "rollouts.npz", allow_pickle=False)
    return rows, npz


def _bucket(r: dict) -> str:
    dirty = r["offroad"] or r["collided"]
    if r["reached"]:
        return "reached_dirty" if dirty else "solved"
    return "short_dirty" if dirty else "short_clean"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10, color=_INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=_INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK2)
    ax.grid(True, **_GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9ced4")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_INK2, labelsize=8.5, length=0)


# --------------------------------------------------------------------------- #
def fig_diagnosis(rows, out_path: Path, arms: list[str]):
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in arms}
    colors = {a: _ARM_COLORS[i % len(_ARM_COLORS)] for i, a in enumerate(arms)}
    x = np.arange(len(arms))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), facecolor=_SURFACE)
    fig.suptitle("How the DPPO-fine-tuned policy actually drives\n"
                 "solve rate is 0 while eval return climbs — which term is paying for that?",
                 fontsize=13, color=_INK, y=0.985)

    # (a) mutually exclusive outcome mix ------------------------------------ #
    ax = axes[0][0]
    bottom = np.zeros(len(arms))
    for key, c in _OUTCOME_COLORS.items():
        vals = np.array([np.mean([_bucket(r) == key for r in by_arm[a]]) if by_arm[a] else 0.0
                         for a in arms])
        # 2px surface gap between stacked segments, not a border around them.
        ax.bar(x, vals, bottom=bottom, color=c, width=0.55,
               edgecolor=_SURFACE, linewidth=2.0, label=_OUTCOME_LABEL[key])
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 0.09:   # only label a segment tall enough to hold the text
                ax.text(xi, b + v / 2, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=8.5, color=_SURFACE if key != "short_clean" else _INK)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=9, color=_INK)
    ax.set_ylim(0, 1)
    _style(ax, "(a) Episode outcome — mutually exclusive", ylabel="fraction of episodes")
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=2)

    # (b) return decomposition ---------------------------------------------- #
    # Only arms whose rows carry the decomposition: the expert-replay arm records
    # a total return but no per-term split, and an empty stack under a total dot
    # would read as "every term is zero".
    ax = axes[0][1]
    barms = [a for a in arms if any("r_progress" in r for r in by_arm[a])]
    xb = np.arange(len(barms))
    pos = np.zeros(len(barms)); neg = np.zeros(len(barms))
    for key, c in _TERM_COLORS.items():
        vals = np.array([np.mean([r.get(key, 0.0) for r in by_arm[a]]) for a in barms])
        base = np.where(vals >= 0, pos, neg)
        ax.bar(xb, vals, bottom=base, color=c, width=0.55,
               edgecolor=_SURFACE, linewidth=2.0, label=_TERM_LABEL[key])
        pos = pos + np.clip(vals, 0, None); neg = neg + np.clip(vals, None, 0)
    tot = np.array([np.mean([r["return"] for r in by_arm[a]]) for a in barms])
    ax.plot(xb, tot, "o", ms=9, color=_INK, mec=_SURFACE, mew=2.0,
            label="total return", ls="none", zorder=5)
    for xi, t in zip(xb, tot):
        # Offset up-and-right: at the top of the stack the dot sits exactly on a
        # segment boundary, and a purely horizontal label lands on the fill.
        ax.annotate(f"{t:+.1f}", (xi, t), textcoords="offset points", xytext=(13, 9),
                    fontsize=9, color=_INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc=_SURFACE, ec="none", alpha=0.85))
    ax.axhline(0, color="#c9ced4", lw=0.9)
    ax.set_xticks(xb); ax.set_xticklabels(barms, fontsize=9, color=_INK)
    _style(ax, "(b) What the return is made of", ylabel="mean reward contribution")
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=3)

    # (c) how far it got ----------------------------------------------------- #
    ax = axes[1][0]
    for i, a in enumerate(arms):
        v = np.array([r.get("frac_closed", np.nan) for r in by_arm[a]], dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        # Thin marks: a jittered strip + a mean rule, not a heavy box.
        jitter = (np.random.default_rng(0).random(v.size) - 0.5) * 0.26
        ax.plot(np.full(v.size, i) + jitter, v, "o", ms=4.5, color=colors[a],
                alpha=0.5, mec="none")
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, "-", color=_INK, lw=2.0, zorder=4)
        ax.annotate(f"{v.mean():.2f}", (i + 0.24, v.mean()), fontsize=9, color=_INK,
                    va="center")
    ax.axhline(1.0, color="#c9ced4", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=9, color=_INK)
    _style(ax, "(c) Fraction of the initial goal distance closed\n"
               "(1.0 = ended at the goal; the reward pays this term directly)",
           ylabel="1 - final_dist / init_dist")

    # (d) offroad exposure vs progress --------------------------------------- #
    ax = axes[1][1]
    for i, a in enumerate(arms):
        xs = np.array([r.get("frac_offroad_steps", np.nan) for r in by_arm[a]], dtype=float)
        ys = np.array([r.get("frac_closed", np.nan) for r in by_arm[a]], dtype=float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if not m.any():
            continue
        ax.plot(xs[m], ys[m], "o", ms=6, color=colors[a], alpha=0.55, mec=_SURFACE,
                mew=1.2, label=a, ls="none")
    ax.axhline(1.0, color="#c9ced4", lw=0.9)
    _style(ax, "(d) Offroad exposure vs distance closed\n"
               "upper-right = closes distance by leaving the road",
           xlabel="fraction of executed steps offroad", ylabel="1 - final_dist / init_dist")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=145, facecolor=_SURFACE)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# --------------------------------------------------------------------------- #
def fig_paths(npz, rows, out_path: Path, arms: list[str], n_scenes: int):
    indices = npz["indices"]
    goal = npz["goal_xy"]
    start = int(npz["start_step"])
    rg_x, rg_y = npz["rg_x"], npz["rg_y"]
    rg_t, rg_v = npz["rg_types"], npz["rg_valid"]
    path_arms = [a for a in arms if f"path_{a}" in npz.files]
    colors = {a: _ARM_COLORS[i % len(_ARM_COLORS)] for i, a in enumerate(arms)}

    n = min(int(n_scenes), len(indices))
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.0 * nrow),
                             facecolor=_SURFACE, squeeze=False)
    fig.suptitle("Executed paths on the roadgraph — road edges dark, lanes light, "
                 "human log dashed; ✕ marks an offroad step",
                 fontsize=12, color=_INK, y=0.995)

    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= n:
            ax.axis("off")
            continue
        v = rg_v[k].astype(bool)
        is_edge = np.isin(rg_t[k], [14, 15, 16]) & v
        is_lane = np.isin(rg_t[k], [1, 2, 3]) & v
        ax.plot(rg_x[k][is_lane], rg_y[k][is_lane], ".", ms=0.5, color="#dfe3e7",
                mec="none", zorder=0)
        ax.plot(rg_x[k][is_edge], rg_y[k][is_edge], ".", ms=1.2, color="#98a3ad",
                mec="none", zorder=1)

        if "log_path" in npz.files:
            lp = npz["log_path"][k]
            ax.plot(lp[start:, 0], lp[start:, 1], "--", lw=1.4, color="#8a929b",
                    zorder=2, label="human log")

        for a in path_arms:
            p = npz[f"path_{a}"][k]
            end = int(npz[f"done_{a}"][k])
            seg = p[start:end + 1]
            ax.plot(seg[:, 0], seg[:, 1], "-", lw=2.0, color=colors[a], zorder=3, label=a)
            off = npz[f"off_{a}"][k][start:end + 1]
            if off.any():
                ax.plot(seg[off, 0], seg[off, 1], "x", ms=5, mew=1.4,
                        color=colors[a], zorder=4)
            if seg.shape[0]:
                ax.plot(seg[-1, 0], seg[-1, 1], "o", ms=7, color=colors[a],
                        mec=_SURFACE, mew=2.0, zorder=5)

        ax.plot(goal[k, 0], goal[k, 1], "*", ms=15, color="#009E73", mec=_SURFACE,
                mew=1.2, zorder=6, label="goal")
        # Frame on the action, not the whole 30k-point map.
        pts = [npz[f"path_{a}"][k][start:int(npz[f'done_{a}'][k]) + 1, :2]
               for a in path_arms]
        pts.append(goal[k][None, :])
        allp = np.concatenate([p for p in pts if p.size], axis=0)
        cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
        half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.7 + 12.0
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e3e7ea")
        srow = [r for r in rows if r["scene"] == int(indices[k]) and r["seed"] == 778]
        sub = {r["arm"]: r for r in srow}
        note = "  ".join(
            f"{a}:{'S' if sub[a]['solved'] else ('R' if sub[a]['reached'] else '-')}"
            f"{'O' if sub[a]['offroad'] else ''}{'C' if sub[a]['collided'] else ''}"
            for a in path_arms if a in sub)
        ax.set_title(f"scene {int(indices[k])}   {note}", fontsize=8.5, color=_INK)
        if k == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# --------------------------------------------------------------------------- #
def write_table(rows, arms, out_path: Path):
    """Plain-text twin of the figures: every plotted number, readable without them."""
    lines = ["# DPPO drive diagnosis", "",
             "| arm | n | return | progress | offroad | collision | goal | "
             "reached | offroad-ep | collided | SOLVED | frac closed | "
             "min goal dist (m) | offroad steps | detour |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        rr = [r for r in rows if r["arm"] == a]
        if not rr:
            continue

        def g(k):
            v = [r[k] for r in rr if k in r]
            return float(np.mean(v)) if v else float("nan")

        lines.append(
            f"| {a} | {len(rr)} | {g('return'):+.2f} | {g('r_progress'):+.2f} | "
            f"{g('r_offroad'):+.2f} | {g('r_collision'):+.2f} | {g('r_goal_bonus'):+.2f} | "
            f"{g('reached'):.3f} | {g('offroad'):.3f} | {g('collided'):.3f} | "
            f"**{g('solved'):.3f}** | {g('frac_closed'):.2f} | {g('min_goal_dist'):.1f} | "
            f"{g('frac_offroad_steps'):.2f} | {g('detour_ratio'):.2f} |")
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[viz] wrote {out_path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("diag_dir", type=str)
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--scenes", type=int, default=16, help="scenes drawn in the path grid")
    args = p.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir) if args.out_dir else diag_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, npz = _load(diag_dir)

    # Deterministic rows are the deployment measurement; keep arm order stable
    # (first appearance) so a color belongs to an arm, not to its rank.
    det_arms: list[str] = []
    for r in rows:
        if r["arm"].endswith("/det") and r["arm"] not in det_arms:
            det_arms.append(r["arm"])
    short = [a.split("/")[0] for a in det_arms]

    det_rows = [dict(r, arm=r["arm"].split("/")[0]) for r in rows if r["arm"].endswith("/det")]
    fig_diagnosis(det_rows, out_dir / "drive_diagnosis.png", short)
    fig_paths(npz, det_rows, out_dir / "drive_paths.png",
              [a for a in short if a != "expert"], args.scenes)
    write_table(det_rows, short, out_dir / "drive_summary.md")


if __name__ == "__main__":
    main()
