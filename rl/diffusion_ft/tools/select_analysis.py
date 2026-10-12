"""Read proposal-selection rows; decide whether any arm beats the base.

Consumes ``proposal_select`` rows (one ``rows.jsonl`` per task; tasks are
checkpoint x shard) and the three-stratum splits file from
``add_intermittent_split``. Everything lands in one JSON so the figure is redrawn
from it on CPU.

A *policy* is ``(arm, ckpt)``. Arms that never touch the fine-tune (``base``,
``bb_*``) carry ``ckpt=""`` and are pooled across tasks; fine-tune arms are kept
per checkpoint and also averaged over them, paired by scene.

``ckpt_mean`` is a mean over whatever checkpoints were scored. It is a SEED mean
only when those checkpoints are seeds of one config -- point the runner at three
different configs (as the 09-14 checkpoint comparison did) and it averages apples
with oranges. Read the per-policy rows in that case; the key is named for what it
does, not for what it is usually used for.

Per policy and split:

* rates (solved / offroad / overlap / reached) and the per-replan switch rate;
* **paired labels vs the base**, per (scene, repeat) on the same draws:
  ``recovered`` = arm solves where base fails, ``broken`` = base solves where the
  arm fails. Population-weighted, ``recovered - broken`` IS the marginal delta, so
  the two numbers say how the delta was made;
* **marginal2** (failure + success, the trainer's number, comparable with every
  earlier run) and **marginal3** (adds the intermittent stratum), delta vs base,
  with a stratified scene-bootstrap 95% CI;
* **oracle "any candidate succeeds"**: ``any(base_r, alt_r)`` on the same seed and
  ``any(base_r, base_r')`` for two independent base episodes. A diagnostic upper
  bound at episode level, not a deployable policy. ``selection_gap`` = oracle -
  selector: large means the selector is the bottleneck, small (with a small
  oracle gain) means the proposals are.

Duplicate rows (a chunk re-run after a torn write) are resolved last-wins.

``--term`` scores the ``*_term`` columns instead (runner rows from 2026-09-30 on):
the flags counted only up to the step the env terminated the row. The legacy
columns keep counting while a finished row is driven on for the rest of its
batch, so they charge e.g. an off-road AFTER the goal. Rows without the columns
are an error under ``--term``, never a silent fall-back to the legacy ones.

    python -m rl.diffusion_ft.tools.select_analysis --splits_json S \\
        --rows DIR/*/rows.jsonl --out DIR/select_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("solved", "offroad", "overlap", "reached")


def load_rows(paths, term: bool = False):
    """{(arm, ckpt): {split: {scene: {repeat: row}}}}, last write wins. ``term``
    swaps each row's ``*_term`` flags in for the legacy ones (see module doc)."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    seeds = set()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if term:
                if "solved_term" not in r:
                    raise SystemExit(f"--term: {p} has rows without solved_term "
                                     f"(written before the runner recorded it)")
                r = {**r, **{m: r[f"{m}_term"] for m in METRICS}}
            seeds.add(int(r["seed"]))
            out[(r["arm"], r.get("ckpt", ""))][r["split"]][int(r["scene_id"])][int(r["repeat"])] = r
    if len(seeds) > 1:
        raise SystemExit(f"rows mix sampler seeds {sorted(seeds)} -- arms would not be paired")
    return out


def matrix(pol, split, scenes, repeats, key="solved"):
    """[n_scenes, repeats] float (NaN where a row is missing)."""
    m = np.full((len(scenes), repeats), np.nan)
    d = pol.get(split, {})
    for i, s in enumerate(scenes):
        for r, row in d.get(s, {}).items():
            if r < repeats:
                m[i, r] = float(row[key])
    return m


def complete_scenes(pols, split, repeats):
    """Scenes with every repeat present in EVERY given policy (so all are paired)."""
    sets = []
    for pol in pols:
        d = pol.get(split, {})
        sets.append({s for s, reps in d.items() if all(r in reps for r in range(repeats))})
    return sorted(set.intersection(*sets)) if sets else []


def weighted(per_split, w):
    return float(sum(w[k] * per_split[k] for k in w))


def bootstrap_delta(diff_by_split, w, n_boot=2000, seed=0):
    """Stratified scene bootstrap of sum_k w_k mean(diff_k). diff: {split: [n]}."""
    rng = np.random.default_rng(seed)
    tot = np.zeros(n_boot)
    for k, wk in w.items():
        d = diff_by_split[k]
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        tot += wk * d[idx].mean(axis=1)
    return [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))]


def any_cross(a, b):
    """Mean over r != r' of any(a[:, r], b[:, r']) -> [n]. Independent episodes.

    Needs at least 2 repeats: with one draw per policy there is no pair of
    INDEPENDENT episodes to combine, and averaging an empty list would report a
    silent NaN rather than "not measurable at this repeat count"."""
    R = a.shape[1]
    if R < 2:
        return None
    vals = [np.maximum(a[:, r], b[:, q]) for r in range(R) for q in range(R) if r != q]
    return np.mean(vals, axis=0)


def analyze(rows, spec, *, repeats, base_key=("base", ""), n_boot=2000, alt_of=None):
    pop = spec["population"]
    # The two-stratum weights are the splits file's own `strata` (failure +
    # success for the held-out scan; one weight-1 stratum for an unlabelled
    # shard slice, build_shard_splits.py).
    strata2 = tuple(pop["strata"])
    w2 = {k: float(pop["strata"][k]["weight"]) for k in strata2}
    w3 = ({k: float(v["weight"]) for k, v in pop["strata3"].items()}
          if "strata3" in pop else None)
    if base_key not in rows:
        raise SystemExit(f"no base rows ({base_key}); the analysis is relative to it")
    base = rows[base_key]
    splits = [k for k in (w3 or w2) if k in base]
    if w3 and set(splits) != set(w3):
        print(f"[select-analysis] marginal3 disabled: base rows lack "
              f"{sorted(set(w3) - set(splits))}")
        w3 = None
    if set(strata2) - set(splits):
        raise SystemExit(f"base rows lack {sorted(set(strata2) - set(splits))}")

    def per_policy(pol_list, name):
        """Stats for one policy, or a seed-mean over several (paired by scene)."""
        res = {"splits": {}}
        diffs, rec_d, brk_d, n_used = {}, {}, {}, {}
        for sp in splits:
            scenes = complete_scenes([base] + pol_list, sp, repeats)
            n_used[sp] = len(scenes)
            if not scenes:
                continue
            b = matrix(base, sp, scenes, repeats)
            arm_m = {m: np.mean([matrix(p, sp, scenes, repeats, m) for p in pol_list], axis=0)
                     for m in METRICS}
            a = arm_m["solved"]
            rec = np.mean([((matrix(p, sp, scenes, repeats) > .5) & (b < .5)).mean(1)
                           for p in pol_list], axis=0)
            brk = np.mean([((matrix(p, sp, scenes, repeats) < .5) & (b > .5)).mean(1)
                           for p in pol_list], axis=0)
            sw = np.mean([matrix(p, sp, scenes, repeats, "n_switch").sum()
                          / max(matrix(p, sp, scenes, repeats, "n_replans").sum(), 1.0)
                          for p in pol_list])
            diffs[sp], rec_d[sp], brk_d[sp] = a.mean(1) - b.mean(1), rec, brk
            res["splits"][sp] = {
                "n_scenes": len(scenes),
                **{f"{m}_rate": float(arm_m[m].mean()) for m in METRICS},
                "base_solved_rate": float(b.mean()),
                "delta": float(diffs[sp].mean()),
                "recovered": float(rec.mean()), "broken": float(brk.mean()),
                "switch_rate": float(sw)}
        for tag, w in (("marginal2", w2), ("marginal3", w3)):
            if not w or any(k not in diffs for k in w):
                continue
            s = res["splits"]
            res[tag] = {
                "solved": weighted({k: s[k]["solved_rate"] for k in w}, w),
                "base": weighted({k: s[k]["base_solved_rate"] for k in w}, w),
                "delta": weighted({k: s[k]["delta"] for k in w}, w),
                "ci95": bootstrap_delta(diffs, w, n_boot=n_boot),
                "recovered": weighted({k: s[k]["recovered"] for k in w}, w),
                "broken": weighted({k: s[k]["broken"] for k in w}, w)}
        return res

    out = {"repeats": repeats, "weights2": w2, "weights3": w3, "policies": {},
           "ckpt_mean": {}, "oracle": {}}
    by_arm = defaultdict(list)
    for (arm, ckpt), pol in rows.items():
        if (arm, ckpt) == base_key:
            continue
        label = arm if not ckpt else f"{arm}@{Path(ckpt).parent.name}/{Path(ckpt).name}"
        out["policies"][label] = per_policy([pol], label)
        if ckpt:
            by_arm[arm].append(pol)
    for arm, pols in by_arm.items():
        if len(pols) > 1:
            out["ckpt_mean"][arm] = {"n_ckpts": len(pols), **per_policy(pols, arm)}

    # ---- oracle: any candidate succeeds (episode level) -----------------------
    def oracle(alt_pols, tag, same_seed):
        per = {}
        for sp in splits:
            scenes = complete_scenes([base] + alt_pols, sp, repeats)
            if not scenes:
                return None
            b = matrix(base, sp, scenes, repeats)
            if alt_pols:
                vals = [np.maximum(b, matrix(p, sp, scenes, repeats)).mean(1) if same_seed
                        else any_cross(b, matrix(p, sp, scenes, repeats)) for p in alt_pols]
                if any(x is None for x in vals):
                    return None
                v = np.mean(vals, axis=0)
            else:
                v = any_cross(b, b)
                if v is None:
                    return None
            per[sp] = (v, b.mean(1))
        o = {"splits": {sp: {"any_rate": float(v.mean()), "gain": float((v - bm).mean())}
                        for sp, (v, bm) in per.items()}}
        for mt, w in (("marginal2", w2), ("marginal3", w3)):
            if w:
                o[mt] = {"any": weighted({k: per[k][0].mean() for k in w}, w),
                         "gain": weighted({k: (per[k][0] - per[k][1]).mean() for k in w}, w),
                         "ci95": bootstrap_delta({k: per[k][0] - per[k][1] for k in w}, w,
                                                 n_boot=n_boot)}
        return o

    alt_of = dict(alt_of or {})
    out["oracle"]["base_x2_independent"] = oracle([], "base2", False)
    if "ft" in by_arm:
        out["oracle"]["base_or_ft_same_seed"] = oracle(by_arm["ft"], "ft", True)
        out["oracle"]["base_or_ft_independent"] = oracle(by_arm["ft"], "ft", False)
        # Selection gap: how much of the same-seed oracle the selectors capture.
        # ONLY for arms whose alternative IS the fine-tune -- this oracle is
        # any(base, ft), so subtracting an arm that selects over a different
        # candidate set (e.g. the repair bank) from it reports a gap against a
        # ceiling that arm was never offered. `alt_of` comes from the arm meta;
        # arms whose alt is unknown are left without a gap rather than given a
        # wrong one.
        orc = out["oracle"]["base_or_ft_same_seed"]
        for arm, sm in out["ckpt_mean"].items():
            if (alt_of.get(arm) == "ft" and orc
                    and "marginal3" in sm and "marginal3" in orc):
                sm["selection_gap3"] = orc["marginal3"]["gain"] - sm["marginal3"]["delta"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_json", required=True, help="three-stratum splits file")
    p.add_argument("--rows", nargs="+", required=True, help="rows.jsonl files or globs")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--term", action="store_true",
                   help="score the termination-aware *_term flags (see module doc)")
    a = p.parse_args()
    paths = sorted({f for g in a.rows for f in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no rows match {a.rows}")
    rows = load_rows(paths, term=a.term)
    spec = json.loads(Path(a.splits_json).read_text())
    # arm -> its alternative, read from the runner's own meta. Needed so the
    # selection gap is only reported against an oracle the arm was offered.
    alt_of = {}
    for pth in paths:
        for m in sorted(Path(pth).parent.glob("meta_shard*.json")):
            try:
                for arm in json.loads(m.read_text()).get("arms", []):
                    alt_of.setdefault(arm["name"], arm.get("alt"))
            except Exception:
                continue
    res = analyze(rows, spec, repeats=a.repeats, n_boot=a.n_boot, alt_of=alt_of)
    res["rows"] = paths
    res["flags"] = "term" if a.term else "legacy"
    Path(a.out).write_text(json.dumps(res, indent=1))

    def line(label, r):
        m3, m2 = r.get("marginal3"), r.get("marginal2")
        s = f"  {label:<40}"
        if m2:
            s += f" m2 d {m2['delta']:+.4f} [{m2['ci95'][0]:+.4f},{m2['ci95'][1]:+.4f}]"
        if m3:
            s += (f"  m3 d {m3['delta']:+.4f} [{m3['ci95'][0]:+.4f},{m3['ci95'][1]:+.4f}]"
                  f"  rec {m3['recovered']:.4f} brk {m3['broken']:.4f}")
        return s
    print("[select-analysis] per policy (vs base, paired):")
    for k, r in res["policies"].items():
        print(line(k, r))
    if res["ckpt_mean"]:
        print("[select-analysis] checkpoint means (a SEED mean only if the ckpts are "
              "seeds of one config):")
        for k, r in res["ckpt_mean"].items():
            print(line(f"{k} (x{r['n_ckpts']})", r)
                  + (f"  selection_gap3 {r['selection_gap3']:+.4f}" if "selection_gap3" in r else ""))
    print("[select-analysis] oracles (episode-level upper bounds):")
    for k, o in res["oracle"].items():
        if o and o.get("marginal3"):
            print(f"  {k:<32} m3 gain {o['marginal3']['gain']:+.4f}")
        elif o is None:
            print(f"  {k:<32} not measurable at repeats={res['repeats']} (needs >= 2)")
    print(f"[select-analysis] wrote -> {a.out}")


if __name__ == "__main__":
    main()
