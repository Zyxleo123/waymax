"""Would a HARDNESS GATE turn the safety fine-tune into a corpus-level win?

The fine-tune is a small net loss when it drives every scene: on the verified
splits it lifts failure_heldout from ~0.02 to ~0.30 but costs ~0.014 of
success_heldout, and success scenes are 98.5% of the corpus. No training knob
moved that by more than the ~0.03 seed spread (see the stopping sweep). A gate
sidesteps the budget instead of fighting it: deploy the fine-tuned policy only on
scenes an a-priori risk score flags, and the frozen base everywhere else.

    delta(t) = sum_k w_k * mean_{s in k} 1[risk_s >= t] * (ft_s - base_s)

over the two held-out strata k (weights from the splits file's `population`
block). Scenes below the threshold run the base, so they contribute exactly zero
-- which is also why a gated comparison is far less noisy than an ungated one.

Inputs are per-scene rows from `train_safety --verify_only` (one file for the base,
one per fine-tuned checkpoint), all scored at the SAME verify seed so every
scene's (ft, base) pair shares its sampler draws, and `runs/hardness/risk.npz`
(scenario_risk for all 482,153 corpus scenes, indexed identically to the
sampler's glob-relative scene ids; checked at load).

What it reports, all written to JSON so the figure can be redrawn on CPU:

* the gated delta as a function of the CORPUS gate fraction (the share of all
  scenes that would run the fine-tune), per checkpoint and for the seed-mean;
* a stratified paired bootstrap CI on the seed-mean curve;
* the ungated delta (gate 100%) -- a consistency check against the training
  curve's marginal at the same iteration;
* the oracle ceiling (gate = exactly the failure stratum);
* a CROSS-VALIDATED operating point. Picking the best threshold on the same scenes
  it is scored on is optimistic, so the held-out scenes are split in half many
  times: choose the gate fraction on one half, score it on the other.

Usage::

    python -m rl.diffusion_ft.tools.gate_analysis --splits_json J \\
        --base_rows base/rows.jsonl --ft_rows ft_s0/rows.jsonl ft_s1/rows.jsonl \\
        --risk runs/hardness/risk.npz --out gate_curve.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

STRATA = ("failure_heldout", "success_heldout")


def load_rows(path, split_names):
    """{split: {scene_id: solved_frac}} plus the ckpt field, checked consistent."""
    out = {k: {} for k in split_names}
    ckpts, seeds = set(), set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] in out:
            out[r["split"]][int(r["scene_id"])] = float(r["solved_frac"])
            ckpts.add(r.get("ckpt", "")); seeds.add(int(r["verify_seed"]))
    if len(ckpts) > 1 or len(seeds) > 1:
        raise SystemExit(f"{path}: mixed ckpt/seed rows {ckpts} {seeds} -- refusing to pair")
    return out, (ckpts.pop() if ckpts else ""), (seeds.pop() if seeds else None)


def scene_risk(spec, risk_path, ids):
    """Risk for glob-relative scene ids, verifying the index alignment first."""
    R = np.load(risk_path, allow_pickle=True)
    paths = sorted(glob.glob(spec["shard_glob"]))
    counts = np.array([len(np.load(Path(spec["index_dir"]) / (Path(p).name + ".offsets.npy")))
                       for p in paths])
    cum = np.concatenate([[0], np.cumsum(counts)])
    if int(cum[-1]) != len(R["scenario_risk"]):
        raise SystemExit(f"corpus has {cum[-1]} scenes, risk table {len(R['scenario_risk'])}")
    ids = np.asarray(ids, np.int64)
    shard = np.searchsorted(cum, ids, side="right") - 1
    local = ids - cum[shard]
    names = np.array([Path(paths[s]).name for s in shard])
    # The risk table is written in sorted-shard, ascending-idx order, i.e. row ==
    # global id. Check it on these exact scenes rather than trusting it.
    if not (np.all(R["shard"][ids] == names) and np.all(R["scenario_idx"][ids] == local)):
        raise SystemExit("risk.npz row order does not match the sampler's scene ids")
    return R["scenario_risk"][ids].astype(np.float64)


def gated_delta(risk, diff, w, thr):
    """delta for each threshold. risk/diff: {stratum: [n]}; thr: [T] -> [T]."""
    tot = np.zeros(len(thr))
    for k in STRATA:
        m = risk[k][None, :] >= thr[:, None]                  # [T, n]
        tot += w[k] * (m * diff[k][None, :]).mean(axis=1)
    return tot


def gate_frac(risk, w, thr):
    return sum(w[k] * (risk[k][None, :] >= thr[:, None]).mean(axis=1) for k in STRATA)


def thresholds_for(risk, w, fracs):
    """Risk threshold giving each target corpus gate fraction (weighted strata)."""
    cand = np.unique(np.concatenate([risk[k] for k in STRATA]))[::-1]
    gf = gate_frac(risk, w, cand)                            # increasing as t falls
    out = []
    for f in fracs:
        i = int(np.argmin(np.abs(gf - f)))
        out.append(cand[i])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--base_rows", required=True)
    ap.add_argument("--ft_rows", nargs="+", required=True)
    ap.add_argument("--risk", default="runs/hardness/risk.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_cv", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    spec = json.loads(Path(a.splits_json).read_text())
    w = {k: float(spec["population"]["strata"][k]["weight"]) for k in STRATA}

    base, base_ckpt, vseed = load_rows(a.base_rows, STRATA)
    if base_ckpt:
        raise SystemExit(f"--base_rows were scored with ckpt={base_ckpt!r}, not the frozen base")
    fts = []
    for p in a.ft_rows:
        rows, ck, s = load_rows(p, STRATA)
        if not ck:
            raise SystemExit(f"{p} has no ckpt -- that is the base, not a fine-tune")
        if s != vseed:
            raise SystemExit(f"{p} verify_seed {s} != base {vseed}: the pair would not share draws")
        fts.append((p, ck, rows))

    ids, risk, b, f = {}, {}, {}, {}
    for k in STRATA:
        want = [int(i) for i in spec["splits"][k]["scene_ids"]]
        have = [i for i in want if i in base[k] and all(i in r[k] for _, _, r in fts)]
        if len(have) < len(want):
            print(f"[gate] {k}: {len(want) - len(have)} of {len(want)} scenes missing a row "
                  f"-- scoring the {len(have)} complete ones")
        ids[k] = np.asarray(have)
        risk[k] = scene_risk(spec, a.risk, have)
        b[k] = np.asarray([base[k][i] for i in have])
        f[k] = np.stack([[r[k][i] for i in have] for _, _, r in fts])   # [S, n]

    fracs = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20,
                      0.30, 0.50, 0.75, 1.0])
    thr = thresholds_for(risk, w, fracs)
    thr[-1] = -np.inf                                          # 100% = ungated
    real_frac = gate_frac(risk, w, thr)

    diff_mean = {k: f[k].mean(axis=0) - b[k] for k in STRATA}
    curve_mean = gated_delta(risk, diff_mean, w, thr)
    curves_seed = [gated_delta(risk, {k: f[k][j] - b[k] for k in STRATA}, w, thr)
                   for j in range(len(fts))]

    # Stratified paired bootstrap over scenes, on the seed-mean differences.
    boot = np.empty((a.n_boot, len(thr)))
    for i in range(a.n_boot):
        rr, dd = {}, {}
        for k in STRATA:
            idx = rng.integers(0, len(ids[k]), len(ids[k]))
            rr[k], dd[k] = risk[k][idx], diff_mean[k][idx]
        boot[i] = gated_delta(rr, dd, w, thr)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    p_pos = (boot > 0).mean(axis=0)

    # Cross-validated operating point: pick the gate fraction on half A, score
    # it on half B. Thresholds are re-derived on A so B never informs them.
    cv_delta, cv_frac = [], []
    for _ in range(a.n_cv):
        A, B = {}, {}
        for k in STRATA:
            perm = rng.permutation(len(ids[k])); h = len(perm) // 2
            A[k], B[k] = perm[:h], perm[h:]
        rA = {k: risk[k][A[k]] for k in STRATA}; dA = {k: diff_mean[k][A[k]] for k in STRATA}
        tA = thresholds_for(rA, w, fracs[:-1])
        best = tA[int(np.argmax(gated_delta(rA, dA, w, tA)))]
        rB = {k: risk[k][B[k]] for k in STRATA}; dB = {k: diff_mean[k][B[k]] for k in STRATA}
        cv_delta.append(float(gated_delta(rB, dB, w, np.array([best]))[0]))
        cv_frac.append(float(gate_frac(rB, w, np.array([best]))[0]))
    cv_delta = np.asarray(cv_delta)

    gain_fail = float(diff_mean["failure_heldout"].mean())
    oracle = w["failure_heldout"] * gain_fail
    base_marg = sum(w[k] * b[k].mean() for k in STRATA)

    res = {
        "splits_json": a.splits_json, "verify_seed": vseed, "weights": w,
        "n": {k: int(len(ids[k])) for k in STRATA},
        "base": {k: float(b[k].mean()) for k in STRATA} | {"marginal": float(base_marg)},
        "ft": [{"rows": p, "ckpt": ck,
                **{k: float(f[k][j].mean()) for k in STRATA}}
               for j, (p, ck, _) in enumerate(fts)],
        "auc_failure_vs_success": float(
            (np.argsort(np.argsort(np.concatenate([risk["failure_heldout"], risk["success_heldout"]])))
             [:len(ids["failure_heldout"])] + 1).sum() - len(ids["failure_heldout"])
            * (len(ids["failure_heldout"]) + 1) / 2) / (len(ids["failure_heldout"]) * len(ids["success_heldout"])),
        "curve": {"target_frac": fracs.tolist(), "gate_frac": real_frac.tolist(),
                  "threshold": [None if not np.isfinite(t) else float(t) for t in thr],
                  "delta_seed_mean": curve_mean.tolist(),
                  "ci95_lo": lo.tolist(), "ci95_hi": hi.tolist(), "p_delta_pos": p_pos.tolist(),
                  "delta_per_seed": [c.tolist() for c in curves_seed]},
        "ungated_delta": float(curve_mean[-1]),
        "oracle_ceiling": float(oracle),
        "cv": {"n": a.n_cv, "delta_mean": float(cv_delta.mean()),
               "delta_p05": float(np.percentile(cv_delta, 5)),
               "delta_p95": float(np.percentile(cv_delta, 95)),
               "frac_positive": float((cv_delta > 0).mean()),
               "chosen_gate_frac_median": float(np.median(cv_frac))},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))

    print(f"[gate] base marginal {base_marg:.4f}  failure {res['base']['failure_heldout']:.3f} "
          f"success {res['base']['success_heldout']:.3f}  (verify seed {vseed})")
    for r in res["ft"]:
        print(f"[gate] ft {Path(r['ckpt']).parent.name}/{Path(r['ckpt']).name}: "
              f"failure {r['failure_heldout']:.3f} success {r['success_heldout']:.3f}")
    print(f"[gate] risk AUC on these labels: {res['auc_failure_vs_success']:.3f}")
    print(f"\n{'gate%':>6} {'delta':>9} {'95% CI':>21} {'P>0':>5}   per-seed")
    for i in range(len(thr)):
        print(f"{100*real_frac[i]:6.1f} {curve_mean[i]:+9.4f} [{lo[i]:+.4f},{hi[i]:+.4f}] "
              f"{p_pos[i]:5.2f}   " + " ".join(f"{c[i]:+.4f}" for c in curves_seed))
    print(f"\n[gate] ungated (100%) delta {res['ungated_delta']:+.4f}   "
          f"oracle ceiling {oracle:+.4f}")
    print(f"[gate] CROSS-VALIDATED operating point: delta {cv_delta.mean():+.4f} "
          f"(90% [{res['cv']['delta_p05']:+.4f}, {res['cv']['delta_p95']:+.4f}]), "
          f"positive in {100*res['cv']['frac_positive']:.0f}% of {a.n_cv} splits, "
          f"median gate {100*res['cv']['chosen_gate_frac_median']:.1f}% of corpus")
    print(f"[gate] wrote {a.out}")


if __name__ == "__main__":
    main()
