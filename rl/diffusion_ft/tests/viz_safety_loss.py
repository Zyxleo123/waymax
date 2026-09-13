"""Draw the safety-loss probe from its artifacts. CPU only -- no model, no GPU.

Reads ``safety_probe.npz`` / ``.jsonl`` / ``_summary.json`` and renders the four
pictures that decide whether the loss design is sound, plus a markdown digest:

  1. **Calibration** -- clearance histograms for the policy and the human expert
     with the margin drawn on. This is where you see whether the margin taxes
     ordinary driving: if the expert's distribution sits largely to the left of
     the margin line, the margin is too aggressive regardless of what the gates
     said.
  2. **Transfer curve** -- loss against clearance, with the frames waymax flagged
     coloured in. The whole design claim is one picture: flat right of the
     margin, a knee at it, a ramp left of it, and the flagged frames living in
     the ramp.
  3. **Timelines** -- per-scene clearance and loss over the episode with the
     offroad/overlap bands shaded, so a spike can be read back to a moment.
  4. **Gradient field** -- the executed path over the roadgraph with the descent
     direction drawn at the at-risk frames. A loss can pass every scalar gate and
     still push sideways; this is the check that it points somewhere sensible.

Usage: ``python -m rl.diffusion_ft.tests.viz_safety_loss <probe_dir> [--scenes N]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROAD_EDGE_IDS = (14, 15, 16)


def _load(d: Path):
    npz = np.load(d / "safety_probe.npz", allow_pickle=False)
    with open(d / "safety_probe_summary.json") as f:
        summary = json.load(f)
    rows = [json.loads(l) for l in open(d / "safety_probe.jsonl")]
    return npz, summary, rows


# --------------------------------------------------------------------------- #
def fig_calibration(npz, summary, out: Path):
    edges = npz["hist_edges"]
    mid = 0.5 * (edges[:-1] + edges[1:])
    m_edge = summary["config"]["edge_margin_m"]
    m_agent = summary["config"]["agent_margin_m"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, (key_p, key_e, margin, title) in enumerate([
        ("hist_c_edge_policy", "hist_c_edge_expert", m_edge, "Road-edge clearance"),
        ("hist_c_agent_policy", None, m_agent, "Nearest-agent clearance (disc gap)"),
    ]):
        p = npz[key_p].astype(float)
        ax[a].bar(mid, p / max(p.sum(), 1), width=np.diff(edges), alpha=0.65,
                  label="policy (base)", color="#c0504d")
        if key_e is not None and key_e in npz:
            e = npz[key_e].astype(float)
            ax[a].bar(mid, e / max(e.sum(), 1), width=np.diff(edges), alpha=0.5,
                      label="expert (human log)", color="#4f81bd")
        ax[a].axvline(0, color="k", lw=1.2, ls="-", label="violation (metric flag)")
        ax[a].axvline(margin, color="g", lw=1.2, ls="--", label=f"margin {margin} m")
        ax[a].set_title(title)
        ax[a].set_xlabel("clearance (m), positive = safe")
        ax[a].set_ylabel("fraction of executed steps")
        ax[a].set_xlim(-3, 8)
        ax[a].legend(fontsize=8)
    fig.suptitle("Where the loss bites, vs where the policy and the human actually drive")
    fig.tight_layout()
    fig.savefig(out / "safety_calibration.png", dpi=140)
    plt.close(fig)


def fig_transfer(npz, summary, out: Path):
    """Loss vs clearance -- the design claim as a single scatter."""
    c = npz["c_edge_base"].ravel()
    l = npz["edge_loss_base"].ravel()
    live = npz["live_base"].ravel()
    off = npz["off_base"].ravel()
    c, l, off = c[live], l[live], off[live]
    m, tau = summary["config"]["edge_margin_m"], summary["config"]["edge_tau_m"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(c[~off], l[~off], s=5, alpha=0.25, color="#4f81bd",
               label=f"on road ({int((~off).sum())} steps)")
    ax.scatter(c[off], l[off], s=7, alpha=0.55, color="#c0504d",
               label=f"waymax offroad ({int(off.sum())} steps)")
    # The ideal single-corner barrier, for reference; the measured points sit on
    # or above it because the loss sums over all four corners.
    grid = np.linspace(max(c.min(), -6), min(c.max(), 8), 300)
    ax.plot(grid, tau * np.logaddexp(0.0, (m - grid) / tau), "k--", lw=1.2,
            label="single-corner barrier")
    ax.axvline(0, color="k", lw=1.0)
    ax.axvline(m, color="g", ls="--", lw=1.0)
    ax.set_xlabel("road-edge clearance c_edge (m)")
    ax.set_ylabel("edge loss at this step")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("Flat when clear, knee at the margin, ramp inside")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "safety_transfer.png", dpi=140)
    plt.close(fig)


def fig_timelines(npz, summary, out: Path, n: int):
    idx = npz["indices"]
    c = npz["c_edge_base"]; l = npz["loss_base"]; live = npz["live_base"]
    off = npz["off_base"]; col = npz["col_base"]
    # Show the scenes where the loss has the most to say.
    score = np.where(live, l, 0).sum(axis=1)
    pick = np.argsort(-score)[:n]
    rows = int(np.ceil(len(pick) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(17, 2.5 * rows), squeeze=False)
    m = summary["config"]["edge_margin_m"]
    for k, i in enumerate(pick):
        ax = axes[k // 4][k % 4]
        t = np.arange(c.shape[1])
        mask = live[i]
        ax.plot(t[mask], c[i][mask], color="#4f81bd", lw=1.3, label="c_edge (m)")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(m, color="g", ls="--", lw=0.8)
        ax2 = ax.twinx()
        ax2.plot(t[mask], l[i][mask], color="#c0504d", lw=1.0, alpha=0.8)
        ax2.set_ylabel("loss", color="#c0504d", fontsize=7)
        ax2.tick_params(labelsize=6)
        for flag, colr in ((off[i], "#c0504d"), (col[i], "#8064a2")):
            for s in np.where(flag)[0]:
                ax.axvspan(s - 0.5, s + 0.5, color=colr, alpha=0.18, lw=0)
        ax.set_title(f"scene {int(idx[i])}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(-3, 8)
    for k in range(len(pick), rows * 4):
        axes[k // 4][k % 4].axis("off")
    fig.suptitle("Clearance (blue, left) and loss (red, right); shaded = waymax "
                 "offroad (red) / overlap (purple)")
    fig.tight_layout()
    fig.savefig(out / "safety_timelines.png", dpi=130)
    plt.close(fig)


def fig_gradient(npz, out: Path, n: int = 8):
    """Executed path + descent direction at the at-risk frames."""
    if "grad_pose_base" not in npz:
        return
    idx = npz["indices"]
    pose = npz["ego_pose_base"]; grad = npz["grad_pose_base"]
    live = npz["live_base"]; off = npz["off_base"]
    near = npz["nearest_edge_base"] if "nearest_edge_base" in npz else None
    rgx, rgy = npz["rg_x"], npz["rg_y"]
    rgt, rgv = npz["rg_types"], npz["rg_valid"]

    gn = np.linalg.norm(grad[..., :2], axis=-1)
    pick = np.argsort(-np.where(live, gn, 0).max(axis=1))[:n]
    rows = int(np.ceil(len(pick) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(17, 4.0 * rows), squeeze=False)
    for k, i in enumerate(pick):
        ax = axes[k // 4][k % 4]
        m = rgv[i] & np.isin(rgt[i], ROAD_EDGE_IDS)
        ax.scatter(rgx[i][m], rgy[i][m], s=0.5, color="0.6", lw=0)
        sel = live[i]
        ax.plot(pose[i][sel, 0], pose[i][sel, 1], color="#4f81bd", lw=1.5, zorder=3)
        o = sel & off[i]
        ax.scatter(pose[i][o, 0], pose[i][o, 1], s=14, color="#c0504d", zorder=4)
        # Descent direction, scaled to be visible (arrows are directional only).
        g = grad[i][..., :2]
        big = sel & (gn[i] > 1e-3)
        if big.any():
            s = 3.0 / max(gn[i][big].max(), 1e-9)
            ax.quiver(pose[i][big, 0], pose[i][big, 1],
                      -g[big, 0] * s, -g[big, 1] * s,
                      color="#00a000", angles="xy", scale_units="xy", scale=1,
                      width=0.005, zorder=5)
        if near is not None:
            nb = near[i][big].reshape(-1, 2)
            ax.scatter(nb[:, 0], nb[:, 1], s=4, color="#ff8c00", zorder=4)
        if sel.any():
            cx, cy = pose[i][sel, 0].mean(), pose[i][sel, 1].mean()
            ax.set_xlim(cx - 45, cx + 45); ax.set_ylim(cy - 45, cy + 45)
        ax.set_aspect("equal"); ax.set_title(f"scene {int(idx[i])}", fontsize=8)
        ax.tick_params(labelsize=6)
    for k in range(len(pick), rows * 4):
        axes[k // 4][k % 4].axis("off")
    fig.suptitle("Descent direction (green) at at-risk frames; red = waymax offroad, "
                 "orange = nearest road-edge point, grey = road edges")
    fig.tight_layout()
    fig.savefig(out / "safety_gradient.png", dpi=130)
    plt.close(fig)


def write_digest(summary, rows, out: Path):
    lines = ["# Safety-loss probe", ""]
    cfg = summary["config"]
    lines += [f"Config: edge margin **{cfg['edge_margin_m']} m** (tau {cfg['edge_tau_m']}), "
              f"agent margin **{cfg['agent_margin_m']} m** (tau {cfg['agent_tau_m']}), "
              f"{cfg['n_discs']} discs/box.", "",
              f"{summary['n_scenes']} scenes x {summary['n_seeds']} seeds = "
              f"**{summary['pooled_steps']} executed steps**, "
              f"{summary['offroad_rate']*100:.1f}% offroad, "
              f"{summary['overlap_rate']*100:.1f}% overlap.", "",
              "## Gates", ""]
    for g in summary["gates"]:
        lines.append(f"- {'PASS' if g['passed'] else 'FAIL'} **{g['gate']}** — {g['detail']}")
    g = summary.get("gradient")
    if g:
        lines += ["", "## Gradient", "",
                  f"One {abs(g['total_after'] - g['total_before']):.2f}-lowering step takes "
                  f"the total loss {g['total_before']:.2f} → {g['total_after']:.2f}, moving "
                  f"c_edge **{g['d_c_edge']:+.4f} m** on {g['n_edge_at_risk']} at-risk frames "
                  f"and c_agent **{g['d_c_agent']:+.4f} m** on {g['n_agent_at_risk']}.", "",
                  f"**{g['n_wedged']} frames are wedged** — inside both margins at once, so "
                  f"the edge and agent terms pull against each other there. A negative "
                  f"per-term delta on those frames is the objective resolving a real "
                  f"conflict, not a defect; only the total is guaranteed to fall."]
    lines += ["", "## Calibration: what the loss charges", "",
              "| | policy (base) | expert (human log) |", "|---|---|---|"]
    p, e = summary["policy"], summary["expert"]
    lines += [
        f"| median c_edge | {p['median_c_edge']:+.3f} m | {e['median_c_edge']:+.3f} m |",
        f"| p05 c_edge | {p['p05_c_edge']:+.3f} m | {e['p05_c_edge']:+.3f} m |",
        f"| steps charged >0.01 | {p['frac_steps_charged']*100:.1f}% | "
        f"{e['frac_steps_charged']*100:.1f}% |",
        f"| mean loss/step | {p['mean_loss_per_step']:.4f} | {e['mean_loss_per_step']:.4f} |",
        "",
        f"Loss split: {p['mean_edge_loss_per_step']:.4f}/step road edge, "
        f"{p['mean_agent_loss_per_step']:.4f}/step agents.",
        "",
        "The expert row is the calibration reference, not a target: a margin that "
        "charges the human demonstration heavily is charging ordinary driving.",
        "",
        "## Worst scenes by total loss", "",
        "| scene | arm | loss | edge | agent | min c_edge | offroad steps |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted([r for r in rows if r["arm"] != "expert"],
                    key=lambda r: -r["loss"])[:15]:
        lines.append(f"| {r['scene']} | {r['arm']} | {r['loss']:.1f} | "
                     f"{r['edge_loss']:.1f} | {r['agent_loss']:.1f} | "
                     f"{r['min_c_edge']:+.2f} | {r['n_offroad_steps']} |")
    (out / "safety_probe_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe_dir", type=str)
    ap.add_argument("--scenes", type=int, default=12)
    args = ap.parse_args()
    d = Path(args.probe_dir)
    npz, summary, rows = _load(d)
    fig_calibration(npz, summary, d)
    fig_transfer(npz, summary, d)
    fig_timelines(npz, summary, d, args.scenes)
    fig_gradient(npz, d)
    write_digest(summary, rows, d)
    print(f"[viz] wrote safety_{{calibration,transfer,timelines,gradient}}.png "
          f"and safety_probe_summary.md in {d}")


if __name__ == "__main__":
    main()
