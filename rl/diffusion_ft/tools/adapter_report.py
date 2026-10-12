"""Results table for the gated tail adapter pilot (rl/diffusion_ft/train_adapter.py).

Four policies on the SAME held-out scenes and sampler draws (seed 4242, 4
repeats, 4 shards x chunk 32 -- the configuration the reference repair run used):

* ``base``     -- the base diffusion policy (this eval's own base arm);
* ``teacher``  -- base + the existing repair optimizer (``o_lpA3``), read from the
  reference select run; usable only if that run's base rows match this eval's
  episode for episode (checked, as ``compare_select`` does);
* ``adp_open`` -- the adapter held fully on (gate forced to 1, strength 1);
* ``adp_gate`` -- the learned soft gate (strength 1).

Per split and corpus-weighted (strata3): solved rate, base failures recovered and
base successes broken (counts with denominators), off-road / collision / reached
rates, progress and pace columns, the gate's distribution on base-success vs
base-failure episodes, and -- from the training job -- latency, added parameters,
the strength-0 identity error and the offline strength sweep. Headline outcomes
are term-aware (``*_term``); legacy flags are kept alongside.

    python -m rl.diffusion_ft.tools.adapter_report --eval EVAL_DIR --ref REF_SELECT_DIR \\
        --train TRAIN_DIR --sweep SWEEP_DIR --splits_json S --out OUT_PREFIX
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("failure_heldout", "intermittent_heldout", "success_heldout")
FLAGS = ("solved", "offroad", "overlap", "reached")


def load(pattern):
    """{arm: {split: {(scene, repeat): row}}}; one sampler seed."""
    out = defaultdict(lambda: defaultdict(dict))
    seeds = set()
    paths = sorted(glob.glob(pattern))
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seeds.add(int(r["seed"]))
            out[r["arm"]][r["split"]][(int(r["scene_id"]), int(r["repeat"]))] = r
    if len(seeds) > 1:
        raise SystemExit(f"{pattern}: rows mix sampler seeds {sorted(seeds)}")
    return out, paths


def flag(r, m, term):
    return bool(r[f"{m}_term"] if term else r[m])


def base_match(a, b):
    """Share of shared (split, scene, repeat) base episodes whose outcome fields differ."""
    n = bad = 0
    for s in SPLITS:
        for k in set(a.get(s, {})) & set(b.get(s, {})):
            n += 1
            ra, rb = a[s][k], b[s][k]
            bad += any(ra[f] != rb[f] for f in ("solved", "offroad", "overlap", "reached",
                                                 "solved_term", "n_replans"))
    return n, bad


def policy_stats(pol, base, keys_by_split, term, w3):
    out = {"splits": {}}
    corpus, base_corpus = 0.0, 0.0
    for s in SPLITS:
        keys = keys_by_split[s]
        if not keys:
            continue
        P = np.array([[flag(pol[s][k], m, term) for m in FLAGS] for k in keys])
        Bm = np.array([[flag(base[s][k], m, term) for m in FLAGS] for k in keys])
        ps, bs = P[:, 0], Bm[:, 0]
        d = {"n": int(len(keys)), "solved": float(ps.mean()),
             "offroad": float(P[:, 1].mean()), "collision": float(P[:, 2].mean()),
             "reached": float(P[:, 3].mean()),
             "recovered": int((~bs & ps).sum()), "base_failures": int((~bs).sum()),
             "broken": int((bs & ~ps).sum()), "base_successes": int(bs.sum())}
        for c in ("trip_m", "goal_d_min", "pace_moving", "n_switch"):
            v = [pol[s][k].get(c) for k in keys]
            v = np.array([x for x in v if x is not None], float)
            if len(v):
                d[c] = float(np.mean(v))
        out["splits"][s] = d
        if w3:
            corpus += w3[s] * d["solved"]
            base_corpus += w3[s] * float(bs.mean())
    if w3:
        out["m3"] = corpus
        out["m3_delta"] = corpus - base_corpus
        # Corpus-weighted broken rate among base successes (the plan's <= 1% target).
        out["broken_rate_weighted"] = sum(
            w3[s] * out["splits"][s]["broken"] / max(out["splits"][s]["base_successes"], 1)
            for s in out["splits"]) / max(sum(w3[s] for s in out["splits"]), 1e-9)
    tot_b = sum(d["base_successes"] for d in out["splits"].values())
    tot_k = sum(d["broken"] for d in out["splits"].values())
    out["broken_total"] = f"{tot_k}/{tot_b}"
    out["broken_frac_pooled"] = tot_k / max(tot_b, 1)
    return out


def replan_distribution(eval_dir, arm, base_rows, term, field="gate", thresh=0.5):
    """A per-replan adapter diagnostic of ``arm`` (``gate`` or ``shift_m``, alive
    replans only), split by the base arm's outcome on the same (scene, repeat); the
    first replan is the exactly paired draw."""
    acc = defaultdict(list)
    first = defaultdict(list)
    for f in sorted(glob.glob(str(Path(eval_dir) / "tasks" / "*" / "feat" / "*.npz"))):
        z = np.load(f)
        split = Path(f).name.rsplit("_s", 1)[0]
        sids = z["scene_ids"]
        for key in z.files:
            if not key.startswith(f"{arm}/r") or not key.endswith(f"/{field}"):
                continue
            r = int(key.split("/")[1][1:])
            g = z[key]
            alive = z[f"{arm}/r{r}/alive"]
            for j, sid in enumerate(sids):
                row = base_rows.get(split, {}).get((int(sid), r))
                if row is None:
                    continue
                tag = "base_success" if flag(row, "solved", term) else "base_failure"
                v = g[j][alive[j]]
                acc[(split, tag)].extend(v.tolist())
                if len(v):
                    first[(split, tag)].append(float(v[0]))
    out = {}
    for (split, tag), v in sorted(acc.items()):
        v = np.asarray(v)
        f1 = np.asarray(first[(split, tag)])
        out.setdefault(split, {})[tag] = {
            "replans": int(len(v)), "mean": float(v.mean()),
            "open_frac": float((v > thresh).mean()),
            "q10_50_90": [float(q) for q in np.quantile(v, [0.1, 0.5, 0.9])],
            "first_replan_mean": float(f1.mean()) if len(f1) else None,
            "first_replan_open_frac": float((f1 > thresh).mean()) if len(f1) else None}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True, help="adapter eval select dir (tasks/*/rows.jsonl)")
    p.add_argument("--ref", default="", help="reference select dir holding base + the teacher arm")
    p.add_argument("--ref_arm", default="o_lpA3")
    p.add_argument("--train", default="", help="train_adapter out dir")
    p.add_argument("--sweep", default="", help="strength-sweep select dir")
    p.add_argument("--splits_json", required=True)
    p.add_argument("--arms", default="adp_open,adp_gate")
    p.add_argument("--max_base_mismatch", type=float, default=0.002)
    p.add_argument("--out", required=True, help="output prefix: <out>.json and <out>.md")
    a = p.parse_args()

    spec = json.loads(Path(a.splits_json).read_text())
    w3 = {k: float(v["weight"]) for k, v in spec["population"]["strata3"].items()}
    ev, ev_paths = load(str(Path(a.eval) / "tasks" / "*" / "rows.jsonl"))
    if "base" not in ev:
        raise SystemExit(f"{a.eval}: no base arm rows")
    report = {"eval": a.eval, "eval_files": len(ev_paths), "ref": a.ref, "term": {}, "legacy": {}}
    policies = {"base": ev["base"]}
    for arm in [x for x in a.arms.split(",") if x]:
        if arm in ev:
            policies[arm] = ev[arm]
    if a.ref:
        rf, _ = load(str(Path(a.ref) / "tasks" / "*" / "rows.jsonl"))
        n, bad = base_match(ev["base"], rf.get("base", {}))
        report["ref_base_match"] = {"shared": n, "mismatched": bad,
                                    "rate": bad / max(n, 1)}
        if a.ref_arm in rf and n and bad / n <= a.max_base_mismatch:
            policies["teacher"] = rf[a.ref_arm]
        else:
            report["ref_note"] = (f"teacher arm NOT paired: {bad}/{n} base episodes differ "
                                  f"(max {a.max_base_mismatch})")
    keys = {s: sorted(set.intersection(*[set(pol.get(s, {})) for pol in policies.values()]))
            for s in SPLITS}
    report["episodes"] = {s: len(v) for s, v in keys.items()}
    for term in (True, False):
        tag = "term" if term else "legacy"
        report[tag] = {nm: policy_stats(pol, ev["base"], keys, term, w3)
                       for nm, pol in policies.items()}
    if "adp_gate" in ev:
        report["gate"] = replan_distribution(a.eval, "adp_gate", ev["base"], True)
    # How far each adapter arm moved the base draw (m, max over the scored window);
    # 'open_frac' here is the share of replans moved by more than 10 cm.
    report["shift_m"] = {arm: replan_distribution(a.eval, arm, ev["base"], True, "shift_m", 0.1)
                         for arm in ("adp_open", "adp_gate") if arm in ev}
    # Ride quality (tools/viz_repair.py ride_<split>.json): per-episode peaks over all
    # episodes; the teacher's from the reference run's own ride job.
    ride = {}
    for s in SPLITS:
        for src, arms in ((Path(a.eval), {"base": "base", "adp_open": "adp_open", "adp_gate": "adp_gate"}),
                          (Path(a.ref) if a.ref else None, {"teacher": a.ref_arm})):
            f = src / f"ride_{s}.json" if src else None
            if f is None or not f.exists():
                continue
            d = json.loads(f.read_text())["arms"]
            for nm, arm in arms.items():
                if arm in d and (nm != "teacher" or "teacher" in policies):
                    ride.setdefault(s, {})[nm] = {
                        m: [d[arm]["all"][m]["p50"], d[arm]["all"][m]["p90"]]
                        for m in ("peak_a_lat", "peak_j_lat", "peak_a_lon", "peak_j_lon")
                        if d[arm]["all"].get(m)}
    if ride:
        report["ride"] = ride
    if a.sweep and glob.glob(str(Path(a.sweep) / "tasks" / "*" / "rows.jsonl")):
        sw, _ = load(str(Path(a.sweep) / "tasks" / "*" / "rows.jsonl"))
        sk = {s: sorted(set.intersection(*[set(pol.get(s, {})) for pol in sw.values()]))
              for s in SPLITS}
        report["sweep"] = {nm: policy_stats(pol, sw["base"], sk, True, None)
                           for nm, pol in sorted(sw.items())}
    if a.train:
        t = Path(a.train)
        for nm in ("preflight", "verify", "bench", "summary"):
            f = t / f"{nm}.json"
            if f.exists():
                report[nm] = json.loads(f.read_text())

    Path(a.out + ".json").write_text(json.dumps(report, indent=1))
    Path(a.out + ".md").write_text(markdown(report))
    print(markdown(report))


def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def markdown(r):
    L = []
    T = r["term"]
    bench = r.get("bench", {})
    ms = bench.get("ms", {})
    prm = bench.get("params", {})
    L.append("| policy | failure | intermittent | success | m3 (delta vs base) | recovered | broken | off-road / collision (failure) | latency ms | added params |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    names = {"base": "base", "teacher": "base + repair optimizer (o_lpA3)",
             "adp_open": "adapter fully on", "adp_gate": "learned soft gate"}
    for nm in ("base", "teacher", "adp_open", "adp_gate"):
        if nm not in T:
            continue
        s = T[nm]["splits"]
        rec = sum(d["recovered"] for d in s.values())
        bf = sum(d["base_failures"] for d in s.values())
        lat = (ms.get("base", {}).get("median") if nm == "base" else
               ms.get("adapter", {}).get("median") if nm.startswith("adp") else None)
        fh = s.get("failure_heldout", {})
        L.append(f"| {names[nm]} | {_f(s.get('failure_heldout', {}).get('solved'))} "
                 f"| {_f(s.get('intermittent_heldout', {}).get('solved'))} "
                 f"| {_f(s.get('success_heldout', {}).get('solved'), 4)} "
                 f"| {_f(T[nm].get('m3'), 4)} ({T[nm].get('m3_delta', 0):+.4f}) "
                 f"| {rec}/{bf} | {T[nm]['broken_total']} "
                 f"| {_f(fh.get('offroad'))} / {_f(fh.get('collision'))} "
                 f"| {_f(lat, 1)} | {prm.get('adapter_total', '-') if nm.startswith('adp') else 0} |")
    L.append("")
    L.append(f"Episodes per split: {r['episodes']}. Term-aware outcomes; recovered = base "
             f"failed, policy solved; broken = base solved, policy failed (all splits pooled).")
    if "ref_base_match" in r:
        m = r["ref_base_match"]
        L.append(f"Teacher rows paired via the reference run: base mismatch {m['mismatched']}/{m['shared']}"
                 + (f" -- {r['ref_note']}" if "ref_note" in r else ""))
    for nm in ("adp_open", "adp_gate", "teacher"):
        if nm in T:
            per = "; ".join(f"{s.split('_')[0]} rec {d['recovered']}/{d['base_failures']} "
                            f"brk {d['broken']}/{d['base_successes']}"
                            for s, d in T[nm]["splits"].items())
            L.append(f"- {nm}: {per}")
    if "gate" in r:
        L.append("")
        L.append("Gate (learned arm, alive replans), base-success vs base-failure episodes:")
        for s, d in r["gate"].items():
            L.append(f"- {s}: " + "; ".join(
                f"{tag} mean {v['mean']:.3f} open {v['open_frac']:.3f} (first replan "
                f"{_f(v['first_replan_mean'])}, n={v['replans']})" for tag, v in d.items()))
    if "ride" in r:
        L.append("")
        L.append("Ride: per-episode peak lateral accel / lateral jerk / longitudinal accel, "
                 "p50 / p90 over all episodes (m/s^2, m/s^3; tools/viz_repair.py):")
        for s, d in r["ride"].items():
            L.append(f"- {s}: " + "; ".join(
                f"{nm} {v['peak_a_lat'][0]:.2f}/{v['peak_a_lat'][1]:.2f}, "
                f"{v['peak_j_lat'][0]:.1f}/{v['peak_j_lat'][1]:.1f}, "
                f"{v['peak_a_lon'][0]:.2f}/{v['peak_a_lon'][1]:.2f}"
                for nm, v in d.items() if all(k in v for k in ("peak_a_lat", "peak_j_lat", "peak_a_lon"))))
    if "verify" in r:
        v = r["verify"]
        L.append("")
        L.append(f"Strength 0 (trained adapter) vs the base late chain: max |dx| "
                 f"{v['strength0_max_err']:.2e} normalized; vs the stored deployed draws "
                 f"{v.get('strength0_vs_stored_max', float('nan')):.2e} "
                 f"({'PASS' if v['strength0_pass'] else 'FAIL'}, {v['rows']} rows).")
        L.append("Offline strengths (validation scenes): " + "; ".join(
            f"s={k}: useful {_f(d.get('rec_useful'))}, shift {_f(d.get('rec_shift_m'))} m, "
            f"keep rms {_f(d.get('keep_rms_m'))} m" for k, d in v["strengths"].items()))
    if "sweep" in r:
        L.append("Closed-loop strengths (capped subset, 1 repeat): " + "; ".join(
            f"{nm} solved " + ", ".join(f"{s.split('_')[0]} {d['solved']:.3f}"
                                               f" (brk {d['broken']}/{d['base_successes']})"
                                               for s, d in st["splits"].items())
            for nm, st in r["sweep"].items()))
    if prm:
        L.append(f"Added parameters: {prm.get('adapter_total')} ({prm.get('adapters')} adapters + "
                 f"{prm.get('gate')} gate) = {100 * prm.get('added_fraction', 0):.3f}% of the "
                 f"{prm.get('base_denoiser')}-parameter denoiser; latency batch {bench.get('batch')} "
                 f"on {bench.get('device')}.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
