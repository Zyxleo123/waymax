"""What the population search did, read back from a select run (CPU, no re-roll).

Three questions, in the order they have to be answered:

1. **Plumbing.** Does ``es_p1`` (one member, no smoothing, same bank) reproduce
   ``b_cv`` row for row? If not, nothing below is attributable to the search.
2. **Outcome.** Per arm and split: solved rate, and PAIRED against the reference
   arm (``b_cv`` by default, the deployable bank): recovered / broken / delta.
   ``select_summary.json`` (tools/select_analysis.py) has the corpus-level m2/m3.
3. **Mechanism**, over the replans where the search ran (base predicted to
   violate), from the ``es_diag`` rows ``proposal_select`` writes:

   * ``raw_any`` / ``proj_any`` / ``gen0_any`` -- a feasible member exists among
     the raw samples / after projection / after projection + smoothing. The gap
     ``proj_any - raw_any`` is what projecting the population buys; ``proj_any``
     is "correct right off the bat";
   * ``retention`` -- of the members the projection made feasible, the share
     still feasible after smoothing (sum n_edit_kept / sum n_edit_feas);
   * ``survival`` -- how much of each edit the smoothing kept, 1 = all, 0 =
     snapped back to the raw sample (p10/p50/p90 over replans);
   * ``best_*`` -- where the winner came from: which generation, whether its
     lineage was projected, whether it descends from the base proposal itself.

Usage (CPU):
  python -m rl.diffusion_ft.tools.es_analysis --select_dir <dir> [--ref b_cv]
      [--identity es_p1:b_cv] [--out <dir>/es_summary.json]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(select_dir: Path):
    rows = []
    for f in sorted(glob.glob(str(select_dir / "tasks" / "*" / "rows.jsonl"))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue            # torn last line from a killed task
    # a requeued task can re-append a chunk: keep the first row per key
    seen, out = set(), []
    for r in rows:
        k = (r["arm"], int(r["scene_id"]), int(r["repeat"]))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def load_meta(select_dir: Path):
    es, cols = {}, None
    for m in sorted(glob.glob(str(select_dir / "tasks" / "*" / "meta_shard*.json"))):
        d = json.loads(Path(m).read_text())
        es.update(d.get("es", {}))
        cols = cols or d.get("es_diag_cols")
    return es, cols


def identity(rows, a: str, b: str):
    by = defaultdict(dict)
    for r in rows:
        if r["arm"] in (a, b):
            by[(int(r["scene_id"]), int(r["repeat"]))][r["arm"]] = r
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    if not pairs:
        return {"n": 0}
    out = {"n": len(pairs)}
    for k in ("solved", "offroad", "overlap", "reached", "n_switch"):
        out[f"agree_{k}"] = float(np.mean([x[k] == y[k] for x, y in pairs]))
    out["pass"] = bool(out["agree_solved"] >= 0.99 and out["agree_n_switch"] >= 0.99)
    return out


def outcomes(rows, ref: str):
    by = defaultdict(dict)
    for r in rows:
        by[(r["split"], int(r["scene_id"]), int(r["repeat"]))][r["arm"]] = bool(r["solved"])
    arms = sorted({r["arm"] for r in rows})
    out = {}
    for split in sorted({r["split"] for r in rows}):
        keys = [k for k in by if k[0] == split]
        for arm in arms:
            s = [by[k][arm] for k in keys if arm in by[k]]
            if not s:
                continue
            o = {"n": len(s), "solved": float(np.mean(s))}
            pr = [(by[k][arm], by[k][ref]) for k in keys if arm in by[k] and ref in by[k]]
            if pr and arm != ref:
                x = np.asarray(pr)
                o.update({f"vs_{ref}_recovered": float(np.mean(x[:, 0] & ~x[:, 1])),
                          f"vs_{ref}_broken": float(np.mean(~x[:, 0] & x[:, 1])),
                          f"vs_{ref}_delta": float(np.mean(x[:, 0]) - np.mean(x[:, 1]))})
            out.setdefault(split, {})[arm] = o
    return out


def _pct(v, qs=(10, 50, 90)):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return {f"p{q}": (float(np.percentile(v, q)) if len(v) else None) for q in qs}


def mechanism(select_dir: Path, es_cfgs: dict, cols):
    """Pool every es arm's per-replan diagnostics by (arm, split)."""
    if not cols:
        return {}
    ci = {c: i for i, c in enumerate(cols)}
    acc = defaultdict(list)
    for f in sorted(glob.glob(str(select_dir / "tasks" / "*" / "feat" / "*.npz"))):
        split = Path(f).stem.rsplit("_s", 1)[0]
        z = np.load(f)
        for k in z.files:
            if not k.endswith("/es_diag"):
                continue
            arm = k.split("/")[0]
            d = z[k].reshape(-1, len(cols))
            acc[(arm, split)].append(d[d[:, ci["trig"]] > 0.5])
    out = {}
    arm_cfg = {}
    for (arm, split), parts in sorted(acc.items()):
        d = np.concatenate(parts, 0) if parts else np.zeros((0, len(cols)))
        col = {c: d[:, i] for c, i in ci.items()}
        o = {"n_trig_replans": int(len(d))}
        if len(d):
            o.update({
                "raw_any": float(np.mean(col["n_raw_feas"] > 0)),
                "proj_any": float(np.mean(col["n_proj_feas"] > 0)),
                "gen0_any": float(np.mean(col["n_gen0_feas"] > 0)),
                "final_any": float(np.mean(col["n_final_feas"] > 0)),
                "raw_feas_mean": float(np.mean(col["n_raw_feas"])),
                "proj_feas_mean": float(np.mean(col["n_proj_feas"])),
                "gen0_feas_mean": float(np.mean(col["n_gen0_feas"])),
                "edited_mean": float(np.mean(col["n_edit"])),
                "retention": (float(col["n_edit_kept"].sum() / col["n_edit_feas"].sum())
                              if col["n_edit_feas"].sum() > 0 else None),
                "survival": _pct(col["survival"]),
                "smooth_disp_m": _pct(col["smooth_disp_m"]),
                "best_feas": float(np.mean(col["best_feas"] > 0.5)),
                "best_gen_hist": {int(g): int(c) for g, c in
                                  zip(*np.unique(col["best_gen"][col["best_feas"] > 0.5],
                                                 return_counts=True))},
                "best_edited": float(np.mean(col["best_edited"][col["best_feas"] > 0.5] > 0.5))
                if (col["best_feas"] > 0.5).any() else None,
                "best_line0": float(np.mean(col["best_line0"][col["best_feas"] > 0.5] > 0.5))
                if (col["best_feas"] > 0.5).any() else None,
                "best_dev_m": _pct(col["best_dev_m"][col["best_feas"] > 0.5]),
                "wall_s_mean": float(np.nanmean(col["wall_s"])),
            })
        out.setdefault(arm, {})[split] = o
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select_dir", type=Path, required=True)
    ap.add_argument("--ref", default="b_cv", help="arm every other arm is paired against")
    ap.add_argument("--identity", default="es_p1:b_cv",
                    help="'a:b' arms that must agree row for row ('' = skip)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = load_rows(a.select_dir)
    if not rows:
        raise SystemExit(f"no rows under {a.select_dir}/tasks/*/rows.jsonl")
    es_cfgs, cols = load_meta(a.select_dir)
    res = {"select_dir": str(a.select_dir), "ref": a.ref, "es": es_cfgs}
    if a.identity:
        x, y = a.identity.split(":")
        res["identity"] = {"arms": [x, y], **identity(rows, x, y)}
    res["outcomes"] = outcomes(rows, a.ref)
    res["mechanism"] = mechanism(a.select_dir, es_cfgs, cols)
    out = a.out or (a.select_dir / "es_summary.json")
    out.write_text(json.dumps(res, indent=1))

    if "identity" in res:
        i = res["identity"]
        print(f"[es] identity {i['arms'][0]} == {i['arms'][1]}: n={i['n']}  "
              + ("  ".join(f"{k[6:]} {v:.4f}" for k, v in i.items() if k.startswith("agree_"))
                 + f"  -> {'PASS' if i.get('pass') else 'FAIL'}" if i["n"] else "no paired rows"))
    for split, arms in res["outcomes"].items():
        print(f"\n[es] {split}")
        for arm, o in arms.items():
            s = f"  {arm:<12} solved {o['solved']:.3f}  (n={o['n']})"
            if f"vs_{a.ref}_delta" in o:
                s += (f"  vs {a.ref}: d {o[f'vs_{a.ref}_delta']:+.3f} rec "
                      f"{o[f'vs_{a.ref}_recovered']:.3f} brk {o[f'vs_{a.ref}_broken']:.3f}")
            print(s)
    for arm, splits in res["mechanism"].items():
        for split, o in splits.items():
            if not o.get("n_trig_replans"):
                continue
            sv, sd = o["survival"], o["smooth_disp_m"]
            print(f"\n[es] {arm} / {split}: {o['n_trig_replans']} triggered replans, "
                  f"{o['wall_s_mean']:.2f} s each")
            print(f"     feasible member exists: raw {o['raw_any']:.3f} -> projected "
                  f"{o['proj_any']:.3f} -> smoothed {o['gen0_any']:.3f} -> final {o['final_any']:.3f}")
            print(f"     mean feasible members: raw {o['raw_feas_mean']:.2f} proj "
                  f"{o['proj_feas_mean']:.2f} smoothed {o['gen0_feas_mean']:.2f}  "
                  f"(edited {o['edited_mean']:.2f})")
            ret = o["retention"]
            print(f"     retention {'-' if ret is None else f'{ret:.3f}'}  survival p10/50/90 "
                  + "/".join("-" if sv[q] is None else f"{sv[q]:.2f}" for q in ("p10", "p50", "p90"))
                  + "  smoothing moved (m) p50/p90 "
                  + "/".join("-" if sd[q] is None else f"{sd[q]:.2f}" for q in ("p50", "p90")))
            print(f"     winner feasible {o['best_feas']:.3f}; gen {o['best_gen_hist']}; "
                  f"projected lineage {o['best_edited']}; from base proposal {o['best_line0']}")
    print(f"\n[es] wrote {out}")


if __name__ == "__main__":
    main()
