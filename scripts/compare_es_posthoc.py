"""Paired comparison of Diffusion-ES arms with / without the post-hoc repair.

Reads every per-scene JSON (``<tfrecord>.scenario_<idx>.json``) under each arm's
directory (all shard groups), and reports per arm: success, off-road, overlap,
goal-reached rates over the scenes EVERY arm finished; and per requested pair the
paired difference in success with a scene-bootstrap 95% CI and the discordant
counts (b better / a better, i.e. McNemar's table). Also, when the source run is
given, the reference rate of that same subset from the run it was rebuilt from.

    python scripts/compare_es_posthoc.py --root <run root> \
        --arms cv,cv_posthoc,wm_mf,wm_mf_posthoc \
        --pairs cv_posthoc:cv,wm_mf_posthoc:wm_mf,wm_mf:cv,wm_mf_posthoc:cv_posthoc \
        --out <root>/compare.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"^(?P<tf>.+)\.scenario_(?P<idx>\d+)\.json$")
KEYS = ("success", "offroad", "overlap", "goal_reached")
# Each view reads one evaluation rollout of the same executed trajectories: "log" the
# primary (log-replayed non-ego agents), "idm" the --also_eval_sim_agents idm rollout.
# Goal reaching depends on the ego alone, so both views share it.
VIEWS = {"log": {k: k for k in KEYS},
         "idm": {"success": "idm_success", "offroad": "idm_offroad", "overlap": "idm_overlap",
                 "goal_reached": "goal_reached"}}


def load_arm(d: Path) -> dict:
    rows = {}
    for f in d.rglob("*.json"):
        m = NAME_RE.match(f.name)
        if m is None:
            continue
        r = json.loads(f.read_text())
        rows[(m["tf"], int(m["idx"]))] = {
            view: ({k: bool(r[src]) for k, src in keys.items()}
                   if all(src in r for src in keys.values()) else None)
            for view, keys in VIEWS.items()}
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="<root>/<arm>/... holds each arm's scene JSONs")
    p.add_argument("--arms", required=True)
    p.add_argument("--pairs", default="")
    p.add_argument("--expect", type=int, default=0, help="scenes each arm should have (0 = no check)")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    data = {arm: load_arm(Path(a.root) / arm) for arm in arms}
    counts = {arm: len(v) for arm, v in data.items()}
    common = sorted(set.intersection(*(set(v) for v in data.values()))) if data else []
    print(f"[compare] scenes per arm {counts}; common {len(common)}")
    rep = {"root": a.root, "counts": counts, "n_common": len(common),
           "complete": bool(a.expect) and all(c == a.expect for c in counts.values()),
           "views": {}}
    if not common:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
        print("[compare] no scene finished by every arm; nothing to compare")
        return
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(common), size=(a.n_boot, len(common)))
    n = len(common)
    for view in VIEWS:
        have = [arm for arm in arms if all(data[arm][s][view] is not None for s in common)]
        if not have:
            continue
        mat = {arm: {k: np.array([data[arm][s][view][k] for s in common]) for k in KEYS} for arm in have}
        rv = rep.setdefault("views", {}).setdefault(view, {"arms": {}, "pairs": {}, "missing": sorted(set(arms) - set(have))})
        print(f"[compare] ===== view {view} (non-ego agents in the evaluation rollout), {n} scenes =====")
        print(f"[compare] {'arm':16s} {'success':>15s} {'collisions':>15s} {'offroad':>15s} {'goal not reached':>17s}")
        for arm in have:
            c = {"success": int(mat[arm]["success"].sum()), "collisions": int(mat[arm]["overlap"].sum()),
                 "offroad": int(mat[arm]["offroad"].sum()), "goal_not_reached": int((~mat[arm]["goal_reached"]).sum())}
            rv["arms"][arm] = {"counts": c, "rates": {k: v / n for k, v in c.items()}}
            print(f"[compare] {arm:16s} " + " ".join(f"{v:>6d} ({100 * v / n:5.1f}%)" for v in c.values()))
        for pair in [x for x in a.pairs.split(",") if x]:
            b, ref = pair.split(":")
            if b not in mat or ref not in mat:
                continue
            sb, sa = mat[b]["success"].astype(float), mat[ref]["success"].astype(float)
            d = sb - sa
            boot = d[idx].mean(axis=1)
            out = {"delta_success": float(d.mean()),
                   "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                   "b_better": int(((sb == 1) & (sa == 0)).sum()),
                   "a_better": int(((sb == 0) & (sa == 1)).sum())}
            for k in ("offroad", "overlap", "goal_reached"):
                out[f"delta_{k}"] = float(mat[b][k].mean() - mat[ref][k].mean())
            rv["pairs"][f"{b}_vs_{ref}"] = out
            print(f"[compare] {view}: {b} vs {ref}: success {out['delta_success']:+.4f} "
                  f"[{out['ci95'][0]:+.4f}, {out['ci95'][1]:+.4f}]  b>a {out['b_better']} a>b {out['a_better']}  "
                  f"offroad {out['delta_offroad']:+.4f} overlap {out['delta_overlap']:+.4f}")
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[compare] -> {a.out}")


if __name__ == "__main__":
    main()
