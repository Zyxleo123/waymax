"""Paired B-minus-A between the fine-tune arms of two ``select_eval`` runs.

``select_analysis`` scores every arm against the base INSIDE one run. Two runs
submitted separately with the same sampler seed, chunking and shards draw the
same base episodes (common random numbers), so their fine-tune arms are paired
per (scene, repeat) across runs too -- this reads that pairing directly instead
of differencing two independently bootstrapped deltas, which overstates the
error. Each side is the seed mean over its checkpoints (``ckpt_mean`` in
``select_analysis``); the CI is the same stratified scene bootstrap.

The pairing is checked, not assumed: the two runs' ``base`` rows must agree on
nearly every episode (``--max_base_mismatch``), or the runs did not share draws
and the comparison is refused.

    python -m rl.diffusion_ft.tools.compare_select --splits_json S \\
        --a 'EVAL_A/tasks/*/rows.jsonl' --b 'EVAL_B/tasks/*/rows.jsonl' \\
        --label_a point --label_b segment --out OUT.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from rl.diffusion_ft.tools.select_analysis import (
    METRICS, bootstrap_delta, complete_scenes, load_rows, matrix, weighted,
)


def _rows(pattern, term=False):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no rows match {pattern}")
    return load_rows(paths, term=term), paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--a", required=True, help="rows glob of the reference run")
    p.add_argument("--b", required=True, help="rows glob of the candidate run")
    p.add_argument("--label_a", default="a")
    p.add_argument("--label_b", default="b")
    p.add_argument("--arm", default="ft")
    p.add_argument("--arm_a", default="", help="arm read from run A (default --arm)")
    p.add_argument("--arm_b", default="", help="arm read from run B (default --arm). "
                   "Different arms, or A == B, compare two arms of one run, e.g. a "
                   "distilled `ft` against the checkpoint-free bank arm it imitates")
    p.add_argument("--ckpt_a", default="", help="regex: keep only run A checkpoints whose "
                   "path matches (e.g. one arm/iteration of a run that scored several)")
    p.add_argument("--ckpt_b", default="", help="regex, same for run B")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--max_base_mismatch", type=float, default=0.002)
    p.add_argument("--out", required=True)
    p.add_argument("--term", action="store_true",
                   help="score the termination-aware *_term flags (select_analysis --term)")
    a = p.parse_args()

    ra, pa = _rows(a.a, a.term)
    rb, pb = _rows(a.b, a.term)
    base_a, base_b = ra.get(("base", "")), rb.get(("base", ""))
    if base_a is None or base_b is None:
        raise SystemExit("both runs need a `base` arm to prove they share draws")
    seed = {int(r["seed"]) for pol in (base_a, base_b) for d in pol.values()
            for reps in d.values() for r in reps.values()}
    if len(seed) != 1:
        raise SystemExit(f"runs used different sampler seeds {sorted(seed)}; not paired")

    # ---- are the two runs' base episodes the same draws? --------------------
    n_cmp = n_diff = 0
    for sp, d in base_b.items():
        for sc, reps in d.items():
            for r, row in reps.items():
                other = base_a.get(sp, {}).get(sc, {}).get(r)
                if other is not None:
                    n_cmp += 1
                    n_diff += int(bool(other["solved"]) != bool(row["solved"]))
    frac = n_diff / max(n_cmp, 1)
    print(f"[compare] base episodes compared {n_cmp}, solved differs on {n_diff} "
          f"({100 * frac:.3f}%)")
    if n_cmp == 0 or frac > a.max_base_mismatch:
        raise SystemExit(f"base rows disagree on {100 * frac:.3f}% > "
                         f"{100 * a.max_base_mismatch:.2f}%: the runs did not share draws")

    arm_a, arm_b = a.arm_a or a.arm, a.arm_b or a.arm
    pols = {}
    for lab, rows, want, rx in ((a.label_a, ra, arm_a, a.ckpt_a),
                                (a.label_b, rb, arm_b, a.ckpt_b)):
        # A checkpoint-free arm (base, a bank) is one "checkpoint" keyed by name.
        pols[lab] = {(ck or f"<{want}>"): pol for (arm, ck), pol in rows.items()
                     if arm == want and (not rx or not ck or re.search(rx, ck))}
        if not pols[lab]:
            raise SystemExit(f"{lab}: no `{want}` rows" + (f" with a checkpoint matching "
                                                           f"/{rx}/" if rx else ""))
        print(f"[compare] {lab}: {len(pols[lab])} checkpoint(s): "
              + ", ".join(f"{Path(c).parent.name}/{Path(c).name}" for c in sorted(pols[lab])))

    spec = json.loads(Path(a.splits_json).read_text())
    pop = spec["population"]
    w2 = {k: float(pop["strata"][k]["weight"]) for k in ("failure_heldout", "success_heldout")}
    w3 = ({k: float(v["weight"]) for k, v in pop["strata3"].items()}
          if "strata3" in pop else None)

    la, lb = a.label_a, a.label_b
    everything = list(pols[la].values()) + list(pols[lb].values())
    res = {"labels": [la, lb], "arm": a.arm, "arms": [arm_a, arm_b], "repeats": a.repeats,
           "ckpt_filters": [a.ckpt_a, a.ckpt_b],
           "base_mismatch": {"compared": n_cmp, "differ": n_diff},
           "splits": {}, "per_ckpt": {}, "rows": {la: pa, lb: pb},
           "flags": "term" if a.term else "legacy"}
    diffs = {}
    for sp in (w3 or w2):
        scenes = complete_scenes(everything + [base_b], sp, a.repeats)
        if not scenes:
            print(f"[compare] {sp}: no scene complete in both runs")
            continue
        mean = {lab: {m: np.mean([matrix(q, sp, scenes, a.repeats, m)
                                  for q in pols[lab].values()], axis=0) for m in METRICS}
                for lab in (la, lb)}
        base = matrix(base_b, sp, scenes, a.repeats)
        diffs[sp] = mean[lb]["solved"].mean(1) - mean[la]["solved"].mean(1)
        res["splits"][sp] = {
            "n_scenes": len(scenes), "base_solved": float(base.mean()),
            **{f"{lab}_{m}": float(mean[lab][m].mean()) for lab in (la, lb) for m in METRICS},
            "delta_solved": float(diffs[sp].mean()),
            # scene-level: how many scenes each side wins on (seed-mean solved rate)
            "scenes_b_better": int((diffs[sp] > 1e-9).sum()),
            "scenes_a_better": int((diffs[sp] < -1e-9).sum())}
        for lab in (la, lb):
            for ck, q in pols[lab].items():
                key = f"{lab}:{Path(ck).parent.name}"
                res["per_ckpt"].setdefault(key, {})[sp] = float(
                    matrix(q, sp, scenes, a.repeats).mean())
    for tag, w in (("marginal2", w2), ("marginal3", w3)):
        if not w or any(k not in diffs for k in w):
            continue
        s = res["splits"]
        res[tag] = {f"{la}_solved": weighted({k: s[k][f"{la}_solved"] for k in w}, w),
                    f"{lb}_solved": weighted({k: s[k][f"{lb}_solved"] for k in w}, w),
                    "delta": weighted({k: s[k]["delta_solved"] for k in w}, w),
                    "ci95": bootstrap_delta(diffs, w, n_boot=a.n_boot)}
        # per-split bootstrap CI (unit weight on one stratum)
        for k in w:
            s[k]["ci95"] = bootstrap_delta({k: diffs[k]}, {k: 1.0}, n_boot=a.n_boot)
    for key, by in res["per_ckpt"].items():
        if w3 and all(k in by for k in w3):
            by["marginal3"] = weighted(by, w3)

    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[compare] {lb} `{arm_b}` - {la} `{arm_a}`, seed means, paired per (scene, repeat):")
    for sp, s in res["splits"].items():
        ci = s.get("ci95", [float("nan")] * 2)
        print(f"  {sp:<22} n={s['n_scenes']:<5} {la} {s[f'{la}_solved']:.3f}  "
              f"{lb} {s[f'{lb}_solved']:.3f}  d {s['delta_solved']:+.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}]  offroad {s[f'{la}_offroad']:.3f}->"
              f"{s[f'{lb}_offroad']:.3f}  overlap {s[f'{la}_overlap']:.3f}->"
              f"{s[f'{lb}_overlap']:.3f}  reached {s[f'{la}_reached']:.3f}->"
              f"{s[f'{lb}_reached']:.3f}")
    for tag in ("marginal2", "marginal3"):
        if tag in res:
            r = res[tag]
            print(f"  {tag:<22} d {r['delta']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]")
    print("[compare] per checkpoint (solved; seed spread):")
    for key, by in sorted(res["per_ckpt"].items()):
        print(f"  {key:<24} " + "  ".join(f"{k} {v:.3f}" for k, v in by.items()))
    print(f"[compare] wrote -> {a.out}")


if __name__ == "__main__":
    main()
