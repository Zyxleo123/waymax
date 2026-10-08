"""Pace experiment readout: does a faster ego, or an honest observation of its
own speed, buy back the long-trip goal block -- and what does it cost?

WHY BY TRIP LENGTH. The no-goal residual is trip length x pace: the goal is a
3 m ball at the END of the human's path (surplus +0.03 m), so a constant
fractional shortfall is free on a short trip and fatal on a long one (never-reach
0.03 under 40 m, 0.58 over 120 m; trip length alone AUC 0.793). A corpus average
is dominated by short, already-solved trips and washes a pace effect out, so the
headline is ``reached`` in the LONG-TRIP decile of each split, and the cost is
``solved`` on ``success_heldout``.

INPUT. One select run per CELL (``--cell name=DIR``), e.g. the two siblings of
``slurm/submit_pace.sh``: ``commanded`` and ``realised`` (``ego_vel_source``),
each running arms ``base`` / ``pace10`` / ``pace05``. A policy is
``<cell>:<arm>``. Cells are separate jobs, so arms are paired ACROSS them only
by common random numbers: same sampler seed, same chunking, same shards, same
scene order -- which this tool checks from every task's ``meta_shard*.json`` and
refuses to pair without. (Within a cell the pairing is exact.) Each row carries
``trip_m``; it must agree across every row of a scene, which is the check that
the join is joining the same scenes.

OUTPUT (``--out``, JSON; the figure is drawn FROM it by ``--plot``):

* ``select``: ``select_analysis.analyze`` over all policies against the control
  (default ``commanded:base``): per-split rates, recovered/broken, m2/m3 with
  stratified scene-bootstrap CIs -- identical definitions to every earlier run;
* ``by_trip``: per policy, split and trip bin (fixed edges 40/60/80/120 m, plus
  the split's top decile): reached / solved rates, closest approach to the goal
  (``goal_d_min``), realised pace vs the human (``pace_moving``);
* ``contrasts``: paired deltas (per scene, mean over repeats) with 95% scene
  bootstrap CIs, for every policy vs the control and every arm vs its own cell's
  ``base`` -- the latter is the pace effect with the observation held fixed;
* ``obs_check``: pooled shown-speed / driven-speed per policy. ``realised``
  cells must read 1.000 (the run was conditioned on what it claims);
  ``commanded`` base arms show the defect (~1.13).

    python -m rl.diffusion_ft.tools.pace_analysis --splits_json S \\
        --cell commanded=DIR_A --cell realised=DIR_B --out OUT/pace_summary.json
    python -m rl.diffusion_ft.tools.pace_analysis --plot OUT/pace_summary.json \\
        --out_png OUT/pace_summary.png
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl.diffusion_ft.tools.select_analysis import analyze

EDGES = (0.0, 40.0, 60.0, 80.0, 120.0, float("inf"))
BIN_NAMES = ("<40", "40-60", "60-80", "80-120", ">120")
# Meta keys that must match across every task of every cell for draws to pair.
PAIR_KEYS = ("seed", "repeats", "chunk", "cap", "splits", "splits_json", "shard")
OBS_TOL = 1e-3


def load_cell(select_dir: Path):
    """``(rows {arm: {split: {scene: {rep: row}}}}, metas)``, last write wins."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for f in sorted(glob.glob(str(select_dir / "tasks/*/rows.jsonl"))):
        for line in Path(f).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                 # torn last line from a killed task
            rows[r["arm"]][r["split"]][int(r["scene_id"])][int(r["repeat"])] = r
    metas = [json.loads(Path(m).read_text())
             for m in sorted(glob.glob(str(select_dir / "tasks/*/meta_shard*.json")))]
    return rows, metas


def pairing_signature(metas):
    """Per shard index, the meta values that decide which draws a scene gets."""
    sig = {}
    for m in metas:
        key = tuple(m.get("shard", [0, 1]))
        sig[key] = {k: m.get(k) for k in PAIR_KEYS}
    return sig


def mat(pol, split, scenes, repeats, key):
    """``[n_scenes, repeats]`` float, NaN where a row or value is missing."""
    m = np.full((len(scenes), repeats), np.nan)
    d = pol.get(split, {})
    for i, s in enumerate(scenes):
        for r, row in d.get(s, {}).items():
            v = row.get(key)
            if r < repeats and v is not None:
                m[i, r] = float(v)
    return m


def complete(pols, split, repeats):
    sets = [{s for s, reps in p.get(split, {}).items()
             if all(r in reps for r in range(repeats))} for p in pols]
    return sorted(set.intersection(*sets)) if sets else []


def boot_ci(d, n_boot, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return [float("nan"), float("nan")]
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    b = d[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


SHORT_DT_S = 0.09      # first-pair dt below this = the short-dt scenes


def trip_bins(trip, dt_first=None):
    """``{bin name: bool mask}`` over scenes: the fixed edges, the split's own
    top decile, all -- and, when rows carry ``dt_first``, the two time-base
    classes (``dt_short``: the scenes ``world_dt_method="first"`` runs at ~0.75x,
    see ``data.preprocess.compute_world_dt_seconds``)."""
    out = {"all": np.ones(len(trip), bool)}
    for name, lo, hi in zip(BIN_NAMES, EDGES[:-1], EDGES[1:]):
        out[name] = (trip >= lo) & (trip < hi)
    out["top_decile"] = trip >= np.percentile(trip, 90) if len(trip) else out["all"]
    if dt_first is not None and np.isfinite(dt_first).all():
        out["dt_short"] = dt_first < SHORT_DT_S
        out["dt_normal"] = ~out["dt_short"]
    return out


def run(cells: dict[str, Path], spec, *, repeats, control, n_boot):
    policies, metas_by_cell, report = {}, {}, {"cells": {}, "missing_cells": []}
    for cell, d in cells.items():
        rows, metas = load_cell(d)
        if not rows:
            print(f"[pace] cell '{cell}': NO ROWS under {d}/tasks -- skipped "
                  f"(check that sibling's .out: gated out, or never ran)")
            report["missing_cells"].append(cell)
            continue
        srcs = sorted({m.get("ego_vel_source", "commanded") for m in metas})
        sims = sorted({m.get("sim_agents", "none") for m in metas})
        wdts = sorted({m.get("world_dt_method", "first") for m in metas})
        report["cells"][cell] = {"dir": str(d), "ego_vel_source": srcs, "sim_agents": sims,
                                 "world_dt_method": wdts,
                                 "arms": sorted(rows), "n_tasks": len(metas)}
        if len(wdts) != 1:
            raise SystemExit(f"cell '{cell}' mixes world_dt_method {wdts}")
        if len(srcs) != 1 or len(sims) != 1:
            raise SystemExit(f"cell '{cell}' mixes ego_vel_source {srcs} / sim_agents {sims}")
        metas_by_cell[cell] = metas
        for arm, pol in rows.items():
            policies[f"{cell}:{arm}"] = pol
    if control not in policies:
        raise SystemExit(f"control policy '{control}' has no rows (have {sorted(policies)})")

    # ---- the cross-job pairing is only real if the draws line up -------------
    sigs = {c: pairing_signature(m) for c, m in metas_by_cell.items()}
    ref_cell = control.split(":", 1)[0]
    for c, s in sigs.items():
        if s != sigs[ref_cell]:
            raise SystemExit(f"cells '{c}' and '{ref_cell}' differ in seed/chunk/shard/cap/"
                             f"splits -- their draws are not paired:\n  {s}\n  {sigs[ref_cell]}")
    report["pairing"] = {"keys": list(PAIR_KEYS), "signature": {str(k): v for k, v in
                                                                 sigs[ref_cell].items()}}

    # ---- trip length must be a property of the SCENE -------------------------
    trip_of: dict[tuple, float] = {}
    worst = 0.0
    for pol in policies.values():
        for sp, scenes in pol.items():
            for s, reps in scenes.items():
                for row in reps.values():
                    v = row.get("trip_m")
                    if v is None:
                        continue
                    k = (sp, s)
                    if k in trip_of:
                        worst = max(worst, abs(trip_of[k] - v))
                    else:
                        trip_of[k] = float(v)
    if not trip_of:
        raise SystemExit("rows carry no trip_m -- they predate the pace columns "
                         "(proposal_select.episode_pace); re-run the select job")
    report["trip_consistency_max_m"] = worst
    if worst > 1e-3:
        raise SystemExit(f"trip_m disagrees across rows of one scene by {worst:.4f} m -- "
                         f"the cells are not scoring the same scenes")
    # dt_first is read off the LOG, so it is a scene property too, whichever
    # method a cell ran under.
    dt_of: dict[tuple, float] = {}
    for pol in policies.values():
        for sp, scenes in pol.items():
            for s, reps in scenes.items():
                for row in reps.values():
                    v = row.get("dt_first")
                    if v is None:
                        continue
                    if (sp, s) in dt_of and abs(dt_of[(sp, s)] - v) > 1e-6:
                        raise SystemExit(f"dt_first disagrees across rows of scene {s}")
                    dt_of[(sp, s)] = float(v)

    def dt_vec(sp, scenes):
        return (np.array([dt_of.get((sp, s), np.nan) for s in scenes]) if dt_of else None)

    # ---- the established readout, against the control ------------------------
    sa_rows = {(name, ""): pol for name, pol in policies.items()}
    report["select"] = analyze(sa_rows, spec, repeats=repeats, base_key=(control, ""),
                               n_boot=n_boot)

    splits = [sp for sp in ("failure_heldout", "intermittent_heldout", "success_heldout")
              if sp in policies[control]]
    names = sorted(policies)
    by_trip, obs_check = {}, {}
    for name in names:
        pol = policies[name]
        o = s = 0.0
        for sp in splits:
            for sc in pol.get(sp, {}).values():
                for row in sc.values():
                    o += row.get("obs_v_sum") or 0.0
                    s += row.get("real_v_sum") or 0.0
        cell = name.split(":", 1)[0]
        src = report["cells"][cell]["ego_vel_source"][0]
        ratio = o / s if s > 0 else float("nan")
        ok = (abs(ratio - 1.0) < OBS_TOL) if src == "realised" else None
        obs_check[name] = {"ego_vel_source": src, "shown_over_driven": ratio, "pass": ok}

    for sp in splits:
        scenes = complete([policies[n] for n in names], sp, repeats)
        if not scenes:
            continue
        trip = np.array([trip_of[(sp, s)] for s in scenes])
        bins = trip_bins(trip, dt_vec(sp, scenes))
        by_trip[sp] = {"n_scenes": len(scenes), "decile_edge_m": float(np.percentile(trip, 90)),
                       "bins": {b: int(m.sum()) for b, m in bins.items()}, "policies": {}}
        for name in names:
            pol = policies[name]
            reached, solved = mat(pol, sp, scenes, repeats, "reached"), mat(pol, sp, scenes, repeats, "solved")
            gmin, pace = mat(pol, sp, scenes, repeats, "goal_d_min"), mat(pol, sp, scenes, repeats, "pace_moving")
            wdt = mat(pol, sp, scenes, repeats, "world_dt")
            by_trip[sp]["policies"][name] = {
                b: {"reached": float(np.nanmean(reached[m])) if m.any() else None,
                    "solved": float(np.nanmean(solved[m])) if m.any() else None,
                    "goal_d_min_p50": float(np.nanmedian(gmin[m])) if m.any() else None,
                    "pace_moving_p50": float(np.nanmedian(pace[m]))
                    if m.any() and np.isfinite(pace[m]).any() else None,
                    # The dt this policy's run USED: the live check that a cell ran
                    # the method it claims (short-dt scenes: ~0.075 first, ~0.1 median).
                    "world_dt_p50": float(np.nanmedian(wdt[m]))
                    if m.any() and np.isfinite(wdt[m]).any() else None}
                for b, m in bins.items()}

    # ---- paired contrasts ------------------------------------------------------
    pairs = [(n, control) for n in names if n != control]
    for n in names:
        cell, arm = n.split(":", 1)
        own = f"{cell}:base"
        if arm != "base" and own in policies and own != control:
            pairs.append((n, own))
    contrasts = {}
    for a, b in pairs:
        key = f"{a} - {b}"
        contrasts[key] = {}
        for sp in splits:
            scenes = complete([policies[a], policies[b]], sp, repeats)
            if not scenes:
                continue
            trip = np.array([trip_of[(sp, s)] for s in scenes])
            out = {}
            for bname, m in trip_bins(trip, dt_vec(sp, scenes)).items():
                if not m.any():
                    continue
                cell = {"n_scenes": int(m.sum())}
                for metric in ("reached", "solved", "goal_d_min"):
                    da = np.nanmean(mat(policies[a], sp, scenes, repeats, metric), 1)
                    db = np.nanmean(mat(policies[b], sp, scenes, repeats, metric), 1)
                    d = (da - db)[m]
                    cell[metric] = {"delta": float(np.nanmean(d)), "ci95": boot_ci(d, n_boot)}
                sa = mat(policies[a], sp, scenes, repeats, "solved")[m]
                sb = mat(policies[b], sp, scenes, repeats, "solved")[m]
                cell["recovered"] = float(((sa > .5) & (sb < .5)).mean())
                cell["broken"] = float(((sa < .5) & (sb > .5)).mean())
                out[bname] = cell
            contrasts[key][sp] = out
    report.update({"repeats": repeats, "control": control, "splits": splits,
                   "bin_edges_m": list(EDGES[1:-1]), "by_trip": by_trip,
                   "contrasts": contrasts, "obs_check": obs_check})
    return report


def print_report(rep):
    print(f"[pace] control = {rep['control']}; trip_m consistent across rows to "
          f"{rep['trip_consistency_max_m']:.1e} m; pairing keys checked: {rep['pairing']['keys']}")
    if rep["missing_cells"]:
        print(f"[pace] MISSING cells (no rows): {rep['missing_cells']}")
    print("[pace] shown / driven ego speed (realised cells must read 1.000):")
    for n, o in rep["obs_check"].items():
        tag = {True: "PASS", False: "FAIL", None: "    "}[o["pass"]]
        print(f"  [{tag}] {n:<24} {o['ego_vel_source']:<9} {o['shown_over_driven']:.4f}")
    for sp in rep["splits"]:
        bt = rep["by_trip"].get(sp)
        if not bt:
            continue
        print(f"[pace] {sp}: {bt['n_scenes']} scenes, top decile >= {bt['decile_edge_m']:.0f} m "
              f"({bt['bins']['top_decile']} scenes); reached / solved / pace p50 by bin")
        if "dt_short" in bt["bins"]:
            wd = "  ".join(f"{n}={(per.get('dt_short') or {}).get('world_dt_p50') or float('nan'):.4f}"
                           for n, per in bt["policies"].items())
            print(f"  short-dt scenes: {bt['bins']['dt_short']}; world dt USED there (p50): {wd}")
        cols = list(BIN_NAMES) + ["top_decile", "all"] + [
            c for c in ("dt_short", "dt_normal") if c in bt["bins"]]
        print("  " + " " * 24 + "".join(f"{c:>17}" for c in cols))
        for n, per in bt["policies"].items():
            cells = []
            for c in cols:
                v = per.get(c) or {}
                if v.get("reached") is None:
                    cells.append(f"{'-':>17}")
                    continue
                pc = v.get("pace_moving_p50")
                cells.append(f"{v['reached']:.2f}/{v['solved']:.2f}/"
                             f"{(pc if pc is not None else float('nan')):.2f}".rjust(17))
            print(f"  {n:<24}" + "".join(cells))
    print("[pace] paired contrasts (per scene, mean over repeats; 95% scene bootstrap):")
    for key, per in rep["contrasts"].items():
        for sp, bins in per.items():
            for bname in ("top_decile", "all", "dt_short", "dt_normal"):
                c = bins.get(bname)
                if not c:
                    continue
                r, s, g = c["reached"], c["solved"], c["goal_d_min"]
                print(f"  {key:<42} {sp:<21} {bname:<10} n={c['n_scenes']:<4} "
                      f"reached {r['delta']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  "
                      f"solved {s['delta']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]  "
                      f"goal_d_min {g['delta']:+.2f} m  rec {c['recovered']:.3f} brk {c['broken']:.3f}")
    print("[pace] corpus (select_analysis, vs control):")
    for k, r in rep["select"]["policies"].items():
        m3 = r.get("marginal3")
        if m3:
            print(f"  {k:<24} m3 d {m3['delta']:+.4f} [{m3['ci95'][0]:+.4f},{m3['ci95'][1]:+.4f}]"
                  f"  rec {m3['recovered']:.4f} brk {m3['broken']:.4f}")


# --------------------------------------------------------------------------- #
# Figure, drawn from the JSON only
# --------------------------------------------------------------------------- #
ARM_COLOR = {"base": "#2a78d6", "pace10": "#eb6834", "pace05": "#1baf7a"}  # palette slots 1-3
EXTRA_COLORS = ["#e87ba4", "#4a3aa7", "#e34948"]                            # slots 5, 7, 8
# One line style + marker per CELL, in the order the cells were given, so any two
# experiments' cells (commanded/realised, first/median, ...) stay distinguishable.
CELL_STYLES = [("--", "o", "dashed"), ("-", "s", "solid"), (":", "^", "dotted"),
               ("-.", "D", "dash-dot")]
INK, MUTED, GRID, SURFACE = "#1a1a19", "#6b6a64", "#e4e3dc", "#fcfcfb"


def plot(json_path: Path, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads(json_path.read_text())
    splits = [sp for sp in rep["splits"] if sp in rep["by_trip"]]
    cells = list(rep["cells"])
    style = {c: CELL_STYLES[i % len(CELL_STYLES)] for i, c in enumerate(cells)}
    arms = sorted({n.split(":", 1)[1] for sp in splits for n in rep["by_trip"][sp]["policies"]})
    extra = [a for a in arms if a not in ARM_COLOR]
    colors = {**ARM_COLOR, **dict(zip(extra, EXTRA_COLORS))}
    # Trip-length bins, then the time-base classes when the rows carry them,
    # after a gap: both are scene groupings read on the same rate axis.
    has_dt = any("dt_short" in rep["by_trip"][sp]["bins"] for sp in splits)
    cols = list(BIN_NAMES) + (["dt_short", "dt_normal"] if has_dt else [])
    xs = list(range(len(BIN_NAMES))) + ([len(BIN_NAMES) + 0.6, len(BIN_NAMES) + 1.6]
                                        if has_dt else [])
    rows = (("reached", "reached rate"), ("solved", "solved rate"),
            ("pace_moving_p50", "pace vs human (p50)"))
    fig, axes = plt.subplots(len(rows), len(splits), figsize=(4.8 * len(splits), 9.6),
                             squeeze=False, facecolor=SURFACE)
    for j, sp in enumerate(splits):
        bt = rep["by_trip"][sp]
        for i, (metric, label) in enumerate(rows):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for name, per in bt["policies"].items():
                cell, arm = name.split(":", 1)
                ls, mk, _ = style.get(cell, CELL_STYLES[-1])
                y = [np.nan if (per.get(c) or {}).get(metric) is None else per[c][metric]
                     for c in cols]
                # Break the line between the trip bins and the dt classes.
                k = len(BIN_NAMES)
                for si, (seg_x, seg_y) in enumerate(((xs[:k], y[:k]), (xs[k:], y[k:]))):
                    if seg_x:
                        # Trip bins are ordinal (a line); the two dt classes are not.
                        ax.plot(seg_x, seg_y, ls=ls if si == 0 else "none", marker=mk,
                                ms=6 if si == 0 else 8, lw=2,
                                color=colors.get(arm, MUTED), markeredgecolor=SURFACE,
                                markeredgewidth=1.5, label=name if si == 0 else None)
            if has_dt:
                ax.axvline(len(BIN_NAMES) - 0.2, color=GRID, lw=1)
            if metric == "pace_moving_p50":
                ax.axhline(1.0, color=MUTED, lw=1, ls=":")
            ticks = [f"{c}\n(n={bt['bins'].get(c, 0)})" for c in cols]
            ax.set_xticks(xs, ticks, fontsize=7.5, color=MUTED)
            ax.tick_params(axis="y", colors=MUTED, labelsize=8)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.set_ylabel(label, color=INK, fontsize=9)
            if i == 0:
                ax.set_title(f"{sp}  ({bt['n_scenes']} scenes)", color=INK, fontsize=10,
                             loc="left")
            if i == len(rows) - 1:
                ax.set_xlabel("human trip length (m)" + ("   |   time-base class" if has_dt else ""),
                              color=MUTED, fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 6), frameon=False, fontsize=8,
               labelcolor=INK)
    key = ",  ".join(f"{c} = {style[c][2]}" for c in cells)
    fig.suptitle(f"Reach, solve and realised pace by scene group   ({key};  control {rep['control']})",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print(f"[pace] figure -> {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", action="append", default=[], help="name=select_dir (repeatable)")
    p.add_argument("--splits_json", default="")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--control", default="commanded:base")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", default="")
    p.add_argument("--plot", default="", help="draw the figure from this JSON and exit")
    p.add_argument("--out_png", default="")
    a = p.parse_args()
    if a.plot:
        plot(Path(a.plot), Path(a.out_png or Path(a.plot).with_suffix(".png")))
        return
    if not a.cell or not a.splits_json or not a.out:
        raise SystemExit("need --cell (>=1), --splits_json and --out")
    cells = {}
    for c in a.cell:
        name, d = c.split("=", 1)
        cells[name] = Path(d)
    spec = json.loads(Path(a.splits_json).read_text())
    rep = run(cells, spec, repeats=a.repeats, control=a.control, n_boot=a.n_boot)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print_report(rep)
    print(f"[pace] wrote -> {a.out}")


if __name__ == "__main__":
    main()
