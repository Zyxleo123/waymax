"""Rank the candidate hardness heuristics against the closed-loop failure labels.

Answers one question before any GPU time is spent: do any of these cheap,
a-priori descriptors actually separate the scenes the diffusion policy fails on
from the ones it does not?  If nothing beats chance, reweighting the training
distribution by them cannot help either, and we stop here instead of after four
training runs.

Everything is scored two ways:

  ungated       -- all evaluated scenarios
  expert-clean  -- only scenarios where log replay itself is clean

The gated numbers are the real ones.  Ungated, "the expert also crashes here"
leaks into every feature that correlates with tight geometry, and a heuristic
can look predictive purely by rediscovering dirty ground truth.

Cross-validation groups by shard, never by scenario: shards are geographic, and
a random split would let the model memorise a neighbourhood and report an
optimistic AUC that will not survive the 846 unevaluated shards.

Writes ranking.json (summary + gate verdict) and features.jsonl (one row per
feature, so the figures can be redrawn without recomputing anything).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from train_diffusion.hardness.descriptors import FAMILY_OF, FEATURE_NAMES


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), NaN scores dropped."""
    ok = np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    # average ranks within ties so a constant feature scores exactly 0.5
    sorted_scores = score[order]
    start = 0
    for idx in range(1, sorted_scores.size + 1):
        if idx == sorted_scores.size or sorted_scores[idx] != sorted_scores[start]:
            if idx - start > 1:
                ranks[order[start:idx]] = ranks[order[start:idx]].mean()
            start = idx
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def decile_lift(y_true: np.ndarray, score: np.ndarray, top: bool = True) -> dict:
    """Failure rate in the extreme decile of ``score`` over the base rate."""
    ok = np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    if y_true.size < 100 or y_true.sum() == 0:
        return {"lift": float("nan"), "decile_rate": float("nan"), "base_rate": float("nan")}
    k = max(1, y_true.size // 10)
    order = np.argsort(score, kind="mergesort")
    sel = order[-k:] if top else order[:k]
    base = float(y_true.mean())
    rate = float(y_true[sel].mean())
    return {"lift": rate / base if base > 0 else float("nan"), "decile_rate": rate, "base_rate": base}


def _prep(x: np.ndarray) -> np.ndarray:
    """Median-impute and standardise; drop non-finite columns to a constant."""
    x = np.array(x, dtype=np.float64, copy=True)
    for col in range(x.shape[1]):
        column = x[:, col]
        finite = np.isfinite(column)
        column[~finite] = np.median(column[finite]) if finite.any() else 0.0
        spread = column.std()
        x[:, col] = (column - column.mean()) / spread if spread > 1e-12 else 0.0
    return x


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """L2-regularised logistic regression (no sklearn in this env)."""
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    target = y.astype(np.float64)

    def objective(w):
        z = x @ w
        # log(1+exp(z)) computed stably
        loss = np.mean(np.logaddexp(0.0, z) - target * z)
        prob = 1.0 / (1.0 + np.exp(-z))
        grad = x.T @ (prob - target) / x.shape[0]
        penalty = l2 * np.concatenate([w[:-1], [0.0]]) / x.shape[0]
        return loss + 0.5 * l2 * np.sum(w[:-1] ** 2) / x.shape[0], grad + penalty

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500})
    return result.x


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))]) @ w


def grouped_cv_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = 5,
                   l2: float = 1.0, seed: int = 0) -> dict:
    """Held-out AUC with folds split on whole shards."""
    unique = np.unique(groups)
    if unique.size < n_folds or y.sum() < n_folds:
        return {"auc": float("nan"), "n_folds": 0}
    rng = np.random.default_rng(seed)
    assignment = {g: i % n_folds for i, g in enumerate(rng.permutation(unique))}
    fold_of = np.array([assignment[g] for g in groups])

    scores = np.full(y.size, np.nan)
    for fold in range(n_folds):
        test = fold_of == fold
        train = ~test
        if y[train].sum() == 0 or y[test].sum() == 0:
            continue
        x_prep = _prep(x)
        scores[test] = _predict(x_prep[test], _fit_logistic(x_prep[train], y[train], l2))
    valid = np.isfinite(scores)
    if valid.sum() < 100:
        return {"auc": float("nan"), "n_folds": n_folds}
    return {
        "auc": auc(y[valid], scores[valid]),
        "n_folds": n_folds,
        **{f"top_decile_{k}": v for k, v in decile_lift(y[valid], scores[valid]).items()},
    }


def load(labels_path: str, desc_dir: str):
    labels = np.load(labels_path, allow_pickle=True)
    key_to_row = {
        (str(s), int(i)): r
        for r, (s, i) in enumerate(zip(labels["shard"], labels["scenario_idx"]))
    }
    n_feat = len(FEATURE_NAMES)
    features = np.full((labels["scenario_idx"].size, n_feat), np.nan, dtype=np.float64)
    covered = np.zeros(labels["scenario_idx"].size, dtype=bool)

    files = sorted(glob.glob(os.path.join(desc_dir, "*.desc.npz")))
    if not files:
        raise SystemExit(f"no descriptor shards under {desc_dir}")
    for path in files:
        with np.load(path, allow_pickle=True) as handle:
            shard = str(handle["shard"])
            names = [str(v) for v in handle["feature_names"]]
            if names != list(FEATURE_NAMES):
                raise SystemExit(f"{path}: feature_names disagree with descriptors.py")
            for scenario_idx, row in zip(handle["scenario_indices"], handle["scenario_feat"]):
                target = key_to_row.get((shard, int(scenario_idx)))
                if target is not None:
                    features[target] = row
                    covered[target] = True
    return labels, features, covered, len(files)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--desc_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--gate_auc", type=float, default=0.60,
                        help="below this held-out AUC the heuristics are not worth training on")
    args = parser.parse_args(argv)

    labels, features, covered, n_shards = load(args.labels, args.desc_dir)
    failed = ~labels["pol_success"]
    shard_of = labels["shard"]

    masks = {
        "ungated": covered,
        "expert_clean": covered & labels["exp_labeled"] & labels["exp_success"],
    }

    report = {
        "n_descriptor_shards": n_shards,
        "n_scenarios_with_descriptors": int(covered.sum()),
        "slices": {},
    }
    rows = []

    for slice_name, mask in masks.items():
        if mask.sum() < 200 or failed[mask].sum() < 20:
            report["slices"][slice_name] = {"skipped": "too few labelled scenarios",
                                            "n": int(mask.sum()), "n_fail": int(failed[mask].sum())}
            continue
        y = failed[mask]
        groups = shard_of[mask]
        x = features[mask]

        univariate = {}
        for col, name in enumerate(FEATURE_NAMES):
            score = x[:, col]
            area = auc(y, score)
            # a feature that predicts failure when LOW is as useful as one that
            # predicts it when HIGH -- report the oriented strength either way
            oriented = area if not np.isnan(area) and area >= 0.5 else (1.0 - area if not np.isnan(area) else np.nan)
            entry = {
                "family": FAMILY_OF[name],
                "auc": area,
                "auc_oriented": oriented,
                "direction": "high=risky" if (not np.isnan(area) and area >= 0.5) else "low=risky",
                "coverage": float(np.isfinite(score).mean()),
                **decile_lift(y, score, top=(not np.isnan(area) and area >= 0.5)),
            }
            univariate[name] = entry
            rows.append({"slice": slice_name, "feature": name, **entry})

        families = sorted(set(FAMILY_OF.values()))
        multivariate = {}
        for family in families:
            cols = [i for i, n in enumerate(FEATURE_NAMES) if FAMILY_OF[n] == family]
            multivariate[family] = grouped_cv_auc(x[:, cols], y, groups, args.n_folds, args.l2)
        multivariate["ALL"] = grouped_cv_auc(x, y, groups, args.n_folds, args.l2)

        ranked = sorted(
            (n for n in univariate if not np.isnan(univariate[n]["auc_oriented"])),
            key=lambda n: univariate[n]["auc_oriented"], reverse=True,
        )
        report["slices"][slice_name] = {
            "n": int(mask.sum()),
            "n_fail": int(y.sum()),
            "base_rate": float(y.mean()),
            "top_features": [
                {"feature": n, **{k: univariate[n][k] for k in ("family", "auc_oriented", "direction", "lift")}}
                for n in ranked[:12]
            ],
            "univariate": univariate,
            "multivariate": multivariate,
        }

    gated = report["slices"].get("expert_clean", {})
    best = gated.get("multivariate", {}).get("ALL", {}).get("auc", float("nan"))
    report["gate"] = {
        "metric": "expert_clean / ALL / grouped-CV AUC",
        "value": best,
        "threshold": args.gate_auc,
        "passed": bool(np.isfinite(best) and best >= args.gate_auc),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(Path(args.out_dir) / "ranking.json", "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=float)
    with open(Path(args.out_dir) / "features.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=float) + "\n")

    for slice_name, block in report["slices"].items():
        if "skipped" in block:
            print(f"\n=== {slice_name}: skipped ({block['skipped']}) ===")
            continue
        print(f"\n=== {slice_name}: n={block['n']} fail={block['n_fail']} base={block['base_rate']:.3%} ===")
        print(f"{'feature':26s} {'family':13s} {'AUC':>6s} {'lift':>6s}  dir")
        for entry in block["top_features"]:
            print(f"{entry['feature']:26s} {entry['family']:13s} "
                  f"{entry['auc_oriented']:6.3f} {entry['lift']:6.2f}  {entry['direction']}")
        print("  multivariate (grouped-CV held-out AUC):")
        for family, res in sorted(block["multivariate"].items()):
            print(f"    {family:13s} {res.get('auc', float('nan')):6.3f}")

    verdict = report["gate"]
    print(f"\nGATE {'PASSED' if verdict['passed'] else 'FAILED'}: "
          f"{verdict['metric']} = {verdict['value']:.3f} (threshold {verdict['threshold']})")
    if not verdict["passed"]:
        # Exit 0 on a failed gate: a non-zero exit would strand every dependent
        # job in DependencyNeverSatisfied with no explanation of why.
        print("The cheap descriptors do not separate policy failures well enough to "
              "justify reweighting on them. Next lever: K-sample disagreement (family F).")


if __name__ == "__main__":
    main()
