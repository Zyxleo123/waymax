"""Why the filter did NOT save a scene: split the residual into named causes.

The `cons` filter can only ever act at a replan where the base's own proposal is
predicted to violate (``clear_p < 0``). So every episode it fails to save falls
into one of a small number of mutually exclusive cases, and each points at a
DIFFERENT fix. This reads the artefacts `proposal_select.py` already writes --
`rows.jsonl` (outcome per scene/arm/repeat) and `feat/*.npz` (per-replan
features, the chosen candidate, and EVERY candidate's clearance) -- so it needs
no GPU and no re-rolling.

BUCKETS (per unsolved episode, in priority order)

  no_goal         the episode never reached the goal and never violated. The
                  filter is a restriction: it can only reject a candidate, never
                  invent progress. Out of scope by construction -- and if this
                  bucket is large, the headroom is NOT a safety problem at all.
  no_alternative  the arm had no bank to choose from (`base`, or a single-
                  alternative arm): it predicted the violation but had nothing
                  to offer. Not evidence about the bank -- a reference line.
  unseen          a violation happened but NO alive replan predicted one. The
                  decision boundary never fired. Cause is horizon: the violation
                  is beyond the scoring window, or past the last replan.
  coverage        at some replan the base was predicted to violate and EVERY
                  candidate in the bank was predicted to violate too. No clean
                  option existed in the offered neighbourhood -> densify or
                  enrich the edit family.
  prog_blocked    a clean candidate existed but was rejected because it gave up
                  more than ``tol_m`` of goal progress -> the tolerance binds,
                  the cheapest knob in the system.
  fired_unsaved   the filter intervened wherever it could, every executed
                  candidate was predicted clean over its window, and the scene
                  violated anyway. Prediction was right locally and wrong
                  globally: compounding past the window.

WHAT THE SOLVE METRIC CANNOT SEE. `solved` is reached & !offroad & !overlap; it
says nothing about HOW a scene was recovered. `lateral_shift` eases the offset in
with a smoothstep over ``ramp_steps`` world steps, whose peak lateral
acceleration is ``6|A| / T**2`` (T = ramp_steps * world_dt) and peak lateral
speed ``1.5|A| / T`` -- so a +/-1.6 m candidate on a 1 s ramp is a ~9.6 m/s^2,
roughly 1 g, evasive swerve while +/-0.25 m is a 1.5 m/s^2 lane adjustment, and
the metric scores them identically. `bank_choice` executes the SMALLEST eligible
edit, so what matters is the distribution of edits actually executed, not the
menu on offer. This reports that distribution (per candidate, and binned by the
implied peak lateral acceleration), split by whether the episode was solved --
if the violent edits cluster in episodes that failed anyway, they are buying
nothing and the range can be trimmed.

THE NUMBER THAT PREDICTS THE DENSE-BANK SWEEP. At coverage-limited replans we
also report percentiles of ``max_k clear_k`` -- how far the BEST candidate fell
short. Near 0 means a slightly larger edit of the same family would have cleared
it and a denser bank (or a learned continuous edit) will pay. Strongly negative
means the 2-D lat x tw family is exhausted there, and no amount of densifying
helps -- that residual needs a richer edit space or a longer horizon.

Usage (CPU, no GPU):
  python -m rl.diffusion_ft.tools.residual_decomp --select_dir <dir> [--split failure_heldout]
  python -m rl.diffusion_ft.tools.residual_decomp --json <out.json> --plot_only
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Column order is meta['feat_cols']; asserted against it at load.
CLEAR_P, PROG_P = 0, 3
BUCKETS = ("no_goal", "unseen", "no_alternative", "coverage", "prog_blocked",
           "fired_unsaved")


# Comfort bands for peak lateral acceleration, m/s^2. Rough driving-comfort
# convention: <2 is an unremarkable lane adjustment, 2-4 brisk, 4-6 aggressive,
# 6-9 evasive, >9 at the dry-asphalt tyre limit.
ACC_BANDS = ((2.0, "<2 calm"), (4.0, "2-4 brisk"), (6.0, "4-6 aggr"),
             (9.0, "6-9 evasive"), (float("inf"), ">9 limit"))


def edit_physics(names: list[str], ramp_steps: int, world_dt: float) -> list[dict]:
    """Per bank candidate: signed amplitude and the peak lateral kinematics.

    Only `lat` has a lateral amplitude; `tw` rescales speed along the SAME path,
    so it gets no acceleration figure and is reported by its warp factor.
    """
    from rl.diffusion_ft.branching import Candidate
    T = max(int(ramp_steps), 1) * float(world_dt)
    out = []
    for nm in names:
        if nm.startswith("opt:"):
            # Loss-optimized (lateral_opt.py): the profile differs per replan, so
            # there is no analytic peak; its executed kinematics are in viz_repair.
            out.append({"name": nm, "family": "opt", "param": float("nan")})
            continue
        c = Candidate.parse(nm)
        d = {"name": nm, "family": c.family, "param": float(c.param)}
        if c.family == "lat":
            d["peak_lat_speed_mps"] = 1.5 * abs(c.param) / T
            d["peak_lat_acc_mps2"] = 6.0 * abs(c.param) / T ** 2
        out.append(d)
    return out


def _load_meta(select_dir: Path) -> dict:
    metas = sorted(select_dir.glob("tasks/*/meta_shard*.json"))
    if not metas:
        raise SystemExit(f"no tasks/*/meta_shard*.json under {select_dir}")
    m = json.loads(metas[0].read_text())
    cols = list(m.get("feat_cols", []))
    if cols[CLEAR_P] != "clear_p" or cols[PROG_P] != "prog_p":
        raise SystemExit(f"feat_cols changed: {cols}; fix CLEAR_P/PROG_P")
    return m


def _load_rows(select_dir: Path) -> dict:
    """(scene_id, arm, repeat) -> outcome dict. Bank/base arms carry ckpt ''."""
    out: dict[tuple, dict] = {}
    n = 0
    for p in sorted(select_dir.glob("tasks/*/rows.jsonl")):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue            # torn last line from a killed task
            out[(int(r["scene_id"]), r["arm"], int(r["repeat"]))] = r
            n += 1
    if not out:
        raise SystemExit(f"no rows under {select_dir}/tasks/*/rows.jsonl")
    print(f"[decomp] {n} row records -> {len(out)} unique (scene, arm, repeat)")
    return out


def _episodes(select_dir: Path, split: str, arms: set[str]):
    """Yield (arm, scene_id, repeat, feat[R,F], alive[R], choice[R]|None, clear_k[R,K]|None)."""
    files = sorted(glob.glob(str(select_dir / "tasks" / "*" / "feat" / f"{split}_s*.npz")))
    if not files:
        raise SystemExit(f"no feat npz for split '{split}' under {select_dir}")
    for f in files:
        z = np.load(f)
        sids = z["scene_ids"]
        keys = [k for k in z.keys() if k.endswith("/feat")]
        for k in keys:
            arm, rep, _ = k.split("/")
            if arms and arm not in arms:
                continue
            feat, alive = z[k], z[f"{arm}/{rep}/alive"]
            ch = z[f"{arm}/{rep}/choice"] if f"{arm}/{rep}/choice" in z else None
            ck = z[f"{arm}/{rep}/clear_k"] if f"{arm}/{rep}/clear_k" in z else None
            r = int(rep[1:])
            for b, sid in enumerate(sids):
                yield (arm, int(sid), r, feat[b], alive[b],
                       None if ch is None else ch[b],
                       None if ck is None else ck[b])


def classify(feat, alive, choice, clear_k, row) -> tuple[str, dict]:
    """Bucket one episode and return the per-replan counts behind the label."""
    violated = bool(row["offroad"]) or bool(row["overlap"])
    if not violated:
        return "no_goal", {}

    a = alive.astype(bool)
    pred = a & (feat[:, CLEAR_P] < 0.0)          # replans where the filter COULD act
    d = {"n_alive": int(a.sum()), "n_pred_viol": int(pred.sum())}
    if not pred.any():
        return "unseen", d

    if clear_k is None:
        # An arm with no bank: `base` has nothing to switch to, and for `ft` the
        # single alternative's clearance lives in the `_a` columns, not clear_k.
        # Either way this is not evidence about the bank, so it gets its own
        # label rather than being pooled into a bank diagnosis.
        return "no_alternative", d

    best = clear_k[pred].max(axis=1)             # [n_pred] best candidate clearance
    any_clean = best >= 0.0
    fired = (choice[pred] >= 0) if choice is not None else np.zeros_like(any_clean)
    d |= {"n_no_clean": int((~any_clean).sum()),
          "n_clean_avail": int(any_clean.sum()),
          "n_fired": int(fired.sum()),
          # how far short the BEST candidate fell where none was clean
          "shortfall": [float(x) for x in best[~any_clean]]}
    if (~any_clean).any():
        return "coverage", d
    if (any_clean & ~fired).any():               # clean existed, cons still said no
        return "prog_blocked", d                 # -> only prog_ok can have rejected it
    return "fired_unsaved", d


def decompose(select_dir: Path, split: str, arms: list[str],
              world_dt: float = 0.1) -> dict:
    meta = _load_meta(select_dir)
    ramp_steps = int(meta.get("ramp_steps", 10))
    rows = _load_rows(select_dir)
    want = set(arms)
    tally: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    short: dict[str, list] = defaultdict(list)
    nclean: dict[str, list] = defaultdict(list)
    uniq: dict[str, np.ndarray] = {}
    # executed candidate index -> count, split by episode outcome
    fired: dict[str, dict[str, np.ndarray]] = {}
    missing = 0

    for arm, sid, rep, feat, alive, choice, clear_k in _episodes(select_dir, split, want):
        row = rows.get((sid, arm, rep))
        if row is None:
            missing += 1
            continue
        tally[arm]["n"] += 1
        # WHICH edits actually ran, on solved and unsolved episodes alike: the
        # question is what the filter executes, not how the episode ended.
        if choice is not None and clear_k is not None:
            K = clear_k.shape[-1]
            f = fired.setdefault(arm, {"solved": np.zeros(K, np.int64),
                                       "unsolved": np.zeros(K, np.int64)})
            ex = choice[alive.astype(bool) & (choice >= 0)]
            if ex.size:
                f["solved" if row["solved"] else "unsolved"] += np.bincount(
                    ex.astype(int), minlength=K)
        if row["solved"]:
            tally[arm]["solved"] += 1
            continue
        bucket, d = classify(feat, alive, choice, clear_k, row)
        tally[arm][bucket] += 1
        short[arm].extend(d.get("shortfall", []))
        # Candidate-availability curve + per-candidate uniqueness, over ALL
        # actionable replans of this arm (solved episodes included: the question
        # is what the bank offers, not what the outcome was).
        if clear_k is not None:
            pred = alive.astype(bool) & (feat[:, CLEAR_P] < 0.0)
            if pred.any():
                clean = clear_k[pred] >= 0.0                      # [n, K]
                nclean[arm].extend(int(x) for x in clean.sum(1))
                u = uniq.setdefault(arm, np.zeros(clean.shape[1], np.int64))
                only = clean[clean.sum(1) == 1]
                if len(only):
                    u += only.sum(0).astype(np.int64)

    if missing:
        print(f"[decomp] WARNING: {missing} episodes in npz had no matching row (skipped)")

    def pct(v, q):
        return float(np.percentile(v, q)) if len(v) else float("nan")

    out = {"select_dir": str(select_dir), "split": split, "tol_m": meta.get("tol_m"),
           "ramp_steps": ramp_steps, "world_dt": world_dt,
           "banks": meta.get("banks", {}), "arms": {}}
    for arm in sorted(tally):
        t = tally[arm]
        n = max(int(t["n"]), 1)
        unsolved = n - int(t["solved"])
        rec = {"n": int(t["n"]), "solved": int(t["solved"]),
               "solve_rate": int(t["solved"]) / n, "unsolved": unsolved,
               "buckets": {b: int(t[b]) for b in BUCKETS},
               "buckets_frac_of_unsolved": {
                   b: (int(t[b]) / unsolved if unsolved else 0.0) for b in BUCKETS}}
        s = short[arm]
        if s:
            rec["shortfall_m"] = {"n": len(s), "p10": pct(s, 10), "p50": pct(s, 50),
                                  "p90": pct(s, 90), "mean": float(np.mean(s)),
                                  "frac_within_0.25m": float(np.mean(np.asarray(s) > -0.25)),
                                  "frac_within_0.50m": float(np.mean(np.asarray(s) > -0.50))}
        c = nclean[arm]
        if c:
            K = len(uniq.get(arm, []))
            hist = np.bincount(np.asarray(c, int), minlength=K + 1).tolist()
            rec["n_clean_candidates_hist"] = hist
            rec["actionable_replans"] = len(c)
            rec["frac_actionable_with_no_clean"] = float(np.mean(np.asarray(c) == 0))
            names = meta.get("banks", {}).get(
                next((a["bank"] for a in meta["arms"] if a["name"] == arm), ""), [])
            rec["sole_clean_by_candidate"] = {
                (names[i] if i < len(names) else f"k{i}"): int(v)
                for i, v in enumerate(uniq.get(arm, []))}
        f = fired.get(arm)
        if f is not None and (f["solved"].sum() + f["unsolved"].sum()):
            names = meta.get("banks", {}).get(
                next((a["bank"] for a in meta["arms"] if a["name"] == arm), ""), [])
            phys = edit_physics(names, ramp_steps, world_dt)
            tot = f["solved"] + f["unsolved"]
            rec["executed"] = {
                "ramp_steps": ramp_steps, "world_dt": world_dt,
                "n_interventions": int(tot.sum()),
                "by_candidate": [
                    dict(p, n=int(tot[i]), n_solved=int(f["solved"][i]),
                         n_unsolved=int(f["unsolved"][i]),
                         frac=float(tot[i] / max(tot.sum(), 1)))
                    for i, p in enumerate(phys) if i < len(tot)],
            }
            # Acceleration bands, over LATERAL interventions only.
            acc = np.array([p.get("peak_lat_acc_mps2", np.nan) for p in phys])
            m = ~np.isnan(acc)
            if m.any() and tot[m].sum():
                bands, lo = {}, 0.0
                for hi, lab in ACC_BANDS:
                    sel = m & (acc >= lo) & (acc < hi)
                    bands[lab] = {"n": int(tot[sel].sum()),
                                  "frac": float(tot[sel].sum() / tot[m].sum()),
                                  "n_solved": int(f["solved"][sel].sum())}
                    lo = hi
                rec["executed"]["lat_acc_bands"] = bands
                rec["executed"]["n_lat_interventions"] = int(tot[m].sum())
                # Amplitude-weighted percentiles of what actually ran.
                amps = np.repeat(np.abs(np.array([p["param"] for p in phys])[m]),
                                 tot[m].astype(int))
                if amps.size:
                    rec["executed"]["executed_abs_amp_m"] = {
                        "p50": float(np.percentile(amps, 50)),
                        "p90": float(np.percentile(amps, 90)),
                        "max": float(amps.max()), "mean": float(amps.mean())}
        out["arms"][arm] = rec
    return out


def render(d: dict, png: Path | None):
    print(f"\n=== residual decomposition: {d['split']}  (tol_m={d['tol_m']}) ===")
    hdr = f"{'arm':<14}{'n':>6}{'solve':>8}" + "".join(f"{b:>15}" for b in BUCKETS)
    print(hdr); print("-" * len(hdr))
    for arm, r in d["arms"].items():
        line = f"{arm:<14}{r['n']:>6}{r['solve_rate']:>8.3f}"
        for b in BUCKETS:
            f = r["buckets_frac_of_unsolved"][b]
            line += f"{r['buckets'][b]:>9d}{f * 100:>5.0f}%"
        print(line)
    for arm, r in d["arms"].items():
        if "shortfall_m" in r:
            s = r["shortfall_m"]
            print(f"\n[{arm}] coverage-limited replans: best candidate fell short by "
                  f"p10={s['p10']:.2f} p50={s['p50']:.2f} p90={s['p90']:.2f} m  "
                  f"(n={s['n']}); within 25cm of clean: {s['frac_within_0.25m']*100:.0f}%, "
                  f"within 50cm: {s['frac_within_0.50m']*100:.0f}%")
            print(f"[{arm}] actionable replans={r['actionable_replans']}  "
                  f"no clean candidate: {r['frac_actionable_with_no_clean']*100:.0f}%  "
                  f"#clean hist={r['n_clean_candidates_hist']}")
            print(f"[{arm}] sole clean option by candidate: {r['sole_clean_by_candidate']}")
    for arm, r in d["arms"].items():
        e = r.get("executed")
        if not e:
            continue
        print(f"\n[{arm}] EXECUTED edits: {e['n_interventions']} interventions "
              f"(ramp {e['ramp_steps']} steps x {e['world_dt']}s = "
              f"{e['ramp_steps'] * e['world_dt']:.1f}s)")
        for c in sorted(e["by_candidate"], key=lambda x: -x["n"]):
            if not c["n"]:
                continue
            acc = (f"  peak {c['peak_lat_acc_mps2']:5.1f} m/s^2"
                   if "peak_lat_acc_mps2" in c else "  (speed edit)      ")
            print(f"    {c['name']:<10} n={c['n']:>5} ({c['frac']*100:4.1f}%)"
                  f"  solved {c['n_solved']:>5} / unsolved {c['n_unsolved']:>5}{acc}")
        if "lat_acc_bands" in e:
            a = e["executed_abs_amp_m"]
            print(f"    lateral |amp| executed: p50={a['p50']:.2f} p90={a['p90']:.2f} "
                  f"max={a['max']:.2f} m  (n={e['n_lat_interventions']})")
            print("    " + "  ".join(
                f"{k} {v['frac']*100:.0f}%" for k, v in e["lat_acc_bands"].items()))
    if png is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = list(d["arms"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bot = np.zeros(len(arms))
    for b in BUCKETS:
        v = np.array([d["arms"][a]["buckets"][b] for a in arms], float)
        ax[0].bar(arms, v, bottom=bot, label=b)
        bot += v
    ax[0].set_ylabel("unsolved episodes"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"residual by cause -- {d['split']}")
    ax[0].tick_params(axis="x", rotation=30)
    for a in arms:
        s = d["arms"][a].get("shortfall_m")
        if s:
            ax[1].plot([s["p10"], s["p50"], s["p90"]], [a] * 3, "o-", label=a)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("best candidate clearance where none was clean (m)")
    ax[1].set_title("how far short the bank fell")
    fig.tight_layout(); fig.savefig(png, dpi=130)
    print(f"\n[decomp] figure -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select_dir", type=Path)
    ap.add_argument("--split", default="failure_heldout")
    ap.add_argument("--arms", default="", help="comma list; default = every arm present")
    ap.add_argument("--json", type=Path, default=None, help="output (or input with --plot_only)")
    ap.add_argument("--png", type=Path, default=None)
    ap.add_argument("--plot_only", action="store_true", help="redraw from --json, no npz reads")
    ap.add_argument("--world_dt", type=float, default=0.1,
                    help="world step, seconds; sets the ramp duration used for the "
                         "executed-edit acceleration figures (NOT stored in meta)")
    a = ap.parse_args()

    if a.plot_only:
        if not a.json or not a.json.exists():
            raise SystemExit("--plot_only needs an existing --json")
        render(json.loads(a.json.read_text()), a.png)
        return
    if not a.select_dir:
        raise SystemExit("--select_dir is required")
    arms = [x for x in a.arms.split(",") if x.strip()]
    d = decompose(a.select_dir, a.split, arms, world_dt=a.world_dt)
    out = a.json or (a.select_dir / f"residual_{a.split}.json")
    out.write_text(json.dumps(d, indent=1))
    print(f"[decomp] json -> {out}")
    render(d, a.png or out.with_suffix(".png"))


if __name__ == "__main__":
    main()
