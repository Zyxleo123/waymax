"""Fit a learned proposal selector from per-decision counterfactual labels (CPU).

Input: ``counterfactual --mode labels`` rows -- at one replan of a base episode,
the alternative candidate was executed instead and the base resumed;
``label = solved(switch) - solved(base)`` in {-1, 0, +1}, next to the geometric
features of both candidates (``proposal_select.FEAT_COLS``).

Model: 3-class logistic regression (L2, full-batch gradient descent, numpy), on
standardized features. Decision: switch iff ``p(+1) - p(-1) > theta``, with
``theta >= 0`` (a switch has to be expected to help), scored by scene-GROUPED
cross-validation -- rows of one scene are strongly correlated, and a row-level
split would leak the scene.

**THETA IS NOT DECIDABLE HERE, and the 09-14 run is why.** The objective below is
a sum of per-decision effects, and a switch whose expected effect is zero is FREE
under it -- nothing prices switching often. It chose ``theta=0``, which switches on
63% of decisions; in closed loop that arm broke 0.0049 of successes for no gain
over the two-line ``cons`` rule (which switches on 0.12%). So the fit now reports
the whole ``theta`` curve and picks its default under a **switch-rate cap**
(``--max_switch_rate``), and the deployable value is chosen by sweeping
``ft:learned:20:<theta>`` arms in ``proposal_select``, where the objective is the
episode outcome. Treat ``theta`` in this file as a starting point, not an answer.

Weighting is the population's, not the sample's: labels come from a stratified
pool (failure_train is ~1.5% of the corpus but a large share of the rows), so
each row carries ``w_stratum / (n_scenes_stratum * rows_of_its_scene)``. The
objective and every reported number are then per-scene averages at corpus
proportions. The unweighted fit would learn to switch wherever failures are
common, which is exactly the hardness-gate mistake.

Reported (all out-of-fold): the weighted mean single-decision effect of the
learned rule, of ``cons`` evaluated on the same rows, and of always-switch. A
single-decision effect is not the episode delta of a selector that may switch
several times -- ``proposal_select`` with a ``learned`` arm measures that.

    python -m rl.diffusion_ft.tools.fit_selector --labels DIR/*/labels.jsonl \\
        --splits_json S --out selector.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

STRATUM_OF = {"failure_train": "failure_heldout", "success_train": "success_heldout",
              "failure_heldout": "failure_heldout", "success_heldout": "success_heldout"}
FEATURE_NAMES = ("clear_p", "clear_a", "d_clear", "edge_p", "edge_a", "agent_p", "agent_a",
                 "prog_p", "prog_a", "d_prog", "replan_frac")


def featurize(F, replan, max_replans=8):
    """[N, 8] FEAT_COLS (clear_p, edge_p, agent_p, prog_p, clear_a, edge_a, agent_a,
    prog_a) + replan index -> [N, 11]. Agent gaps are capped at 10 m: past that
    "no neighbour" (1e3) and "far neighbour" mean the same thing."""
    F = np.asarray(F, np.float64)
    cp, ep, ap, pp, ca, ea, aa, pa = (F[:, i] for i in range(8))
    ap, aa = np.minimum(ap, 10.0), np.minimum(aa, 10.0)
    cp, ca = np.minimum(cp, 10.0), np.minimum(ca, 10.0)
    ep, ea = np.minimum(ep, 10.0), np.minimum(ea, 10.0)
    r = np.asarray(replan, np.float64) / float(max_replans)
    return np.stack([cp, ca, ca - cp, ep, ea, ap, aa, pp, pa, pa - pp, r], axis=1)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_logreg(X, y3, w, *, l2=1e-2, iters=3000, lr=0.5):
    """Weighted multinomial logistic regression. y3 in {0,1,2} = label+1."""
    n, d = X.shape
    W = np.zeros((3, d)); b = np.zeros(3)
    Y = np.eye(3)[y3]
    ws = w / max(w.sum(), 1e-12)
    for _ in range(iters):
        P = _softmax(X @ W.T + b)
        G = (P - Y) * ws[:, None]
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b


def predict_score(model, X_raw):
    """p(+1) - p(-1) for raw (featurized, unstandardized) rows."""
    X = (X_raw - np.asarray(model["mean"])) / np.asarray(model["std"])
    P = _softmax(X @ np.asarray(model["W"]).T + np.asarray(model["b"]))
    return P[:, 2] - P[:, 0]


def learned_switch(model, feat_B8, replan, theta=None):
    """[B] bool switch decision for the runtime selector (proposal_select).

    ``theta`` overrides the fitted threshold: the fit's objective cannot price
    over-switching, so the deployable threshold is swept in closed loop."""
    X = featurize(feat_B8, np.full(len(feat_B8), replan), model.get("max_replans", 8))
    t = float(model["theta"] if theta is None else theta)
    return predict_score(model, X) > t


def row_weights(scenes, strata, w_stratum):
    """w_stratum / (n_scenes_in_stratum * rows_of_scene), per row."""
    scenes = np.asarray(scenes); strata = np.asarray(strata)
    w = np.zeros(len(scenes))
    for s in np.unique(strata):
        m = strata == s
        u, inv, cnt = np.unique(scenes[m], return_inverse=True, return_counts=True)
        w[m] = w_stratum[s] / (len(u) * cnt[inv])
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--tol_m", type=float, default=1.0, help="cons rule tolerance, for comparison")
    p.add_argument("--max_switch_rate", type=float, default=0.01,
                   help="cap on the population-weighted share of decisions the default "
                        "theta may switch on. The per-decision objective cannot price "
                        "over-switching, so this is what keeps the default sane; the real "
                        "choice is a closed-loop sweep of ft:learned:W:theta arms.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from rl.diffusion_ft.proposal_select import select_switch

    paths = sorted({f for g in a.labels for f in glob.glob(g)})
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["scene_id"], r["replan"], r["split"])] = r      # last write wins
    rows = list(rows.values())
    if not rows:
        raise SystemExit(f"no label rows in {a.labels}")
    pop = json.loads(Path(a.splits_json).read_text())["population"]["strata"]
    w_stratum = {k: float(v["weight"]) for k, v in pop.items()}
    strata = np.array([STRATUM_OF[r["split"]] for r in rows])
    missing = set(w_stratum) - set(strata)
    if missing:
        raise SystemExit(f"labels lack strata {sorted(missing)}: a one-stratum fit has no "
                         f"do-no-harm signal")
    F = np.array([r["feat"] for r in rows]); rep = np.array([r["replan"] for r in rows])
    y = np.array([r["label"] for r in rows]); scenes = np.array([r["scene_id"] for r in rows])
    X = featurize(F, rep)
    w = row_weights(scenes, strata, w_stratum)

    rng = np.random.default_rng(a.seed)
    uscene = np.unique(scenes)
    fold_of = dict(zip(uscene.tolist(), rng.integers(0, a.folds, uscene.size).tolist()))
    fold = np.array([fold_of[s] for s in scenes.tolist()])
    oof = np.zeros(len(rows))
    for k in range(a.folds):
        tr, te = fold != k, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        W, b = fit_logreg((X[tr] - mu) / sd, y[tr] + 1, w[tr], l2=a.l2)
        oof[te] = predict_score({"mean": mu, "std": sd, "W": W, "b": b}, X[te])

    def effect(sw, m=None):
        m = np.ones(len(rows), bool) if m is None else m
        return float(np.sum(w[m] * y[m] * sw[m]))

    thetas = np.linspace(0.0, 0.9, 91)
    curve = [effect(oof > t) for t in thetas]
    rate = [float(np.sum(w * (oof > t))) for t in thetas]
    ok = [i for i, r in enumerate(rate) if r <= float(a.max_switch_rate)]
    if not ok:
        raise SystemExit(f"no theta in [0, 0.9] switches on <= {a.max_switch_rate} of "
                         f"decisions (min {min(rate):.4f}) -- the score is degenerate")
    theta = float(thetas[max(ok, key=lambda i: curve[i])])
    theta_free = float(thetas[int(np.argmax(curve))])
    cons = select_switch("cons", clear_p=F[:, 0], clear_a=F[:, 4], prog_p=F[:, 3],
                         prog_a=F[:, 7], tol_m=a.tol_m, margin_m=0.3, delta_m=0.2)

    mu, sd = X.mean(0), X.std(0) + 1e-6
    W, b = fit_logreg((X - mu) / sd, y + 1, w, l2=a.l2)
    per_stratum = {s: {"rows": int((strata == s).sum()),
                       "plus": int(((strata == s) & (y > 0)).sum()),
                       "minus": int(((strata == s) & (y < 0)).sum()),
                       "learned_oof_effect": effect(oof > theta, strata == s),
                       "cons_effect": effect(cons, strata == s),
                       "always_effect": effect(np.ones(len(rows), bool), strata == s)}
                   for s in w_stratum}
    out = {"feature_names": list(FEATURE_NAMES), "max_replans": 8,
           "mean": mu.tolist(), "std": sd.tolist(), "W": W.tolist(), "b": b.tolist(),
           "theta": theta, "theta_unconstrained": theta_free,
           "max_switch_rate": a.max_switch_rate, "l2": a.l2,
           "cv": {"folds": a.folds, "thetas": thetas.tolist(), "effect_curve": curve,
                  "switch_rate_curve": rate,
                  "learned_oof_effect": effect(oof > theta),
                  "learned_switch_rate": float(np.sum(w * (oof > theta))),
                  "cons_effect": effect(cons), "cons_switch_rate": float(np.sum(w * cons)),
                  "always_effect": effect(np.ones(len(rows), bool))},
           "per_stratum": per_stratum, "n_rows": len(rows), "n_scenes": int(uscene.size),
           "labels": paths, "splits_json": a.splits_json,
           "note": "effects are population-weighted mean single-decision solve-rate changes"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    cv = out["cv"]
    print(f"[fit_selector] {len(rows)} decisions / {uscene.size} scenes; "
          f"theta={theta:.2f} (capped at switch rate {a.max_switch_rate}); "
          f"uncapped argmax would be {theta_free:.2f} -- do not deploy that without a "
          f"closed-loop sweep, it cannot price over-switching")
    print(f"  single-decision effect (OOF): learned {cv['learned_oof_effect']:+.5f} "
          f"(switch {cv['learned_switch_rate']:.4f})  cons {cv['cons_effect']:+.5f} "
          f"(switch {cv['cons_switch_rate']:.4f})  always {cv['always_effect']:+.5f}")
    for s, v in per_stratum.items():
        print(f"  {s}: rows {v['rows']} +1 {v['plus']} -1 {v['minus']}  learned "
              f"{v['learned_oof_effect']:+.5f} cons {v['cons_effect']:+.5f}")
    print(f"[fit_selector] wrote -> {a.out}")


if __name__ == "__main__":
    main()
