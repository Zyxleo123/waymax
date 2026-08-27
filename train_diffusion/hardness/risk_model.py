"""Fit the failure-risk model and score every scenario in the dataset.

Fits on the expert-clean slice only -- the configuration whose held-out AUC was
actually measured (0.860, shard-grouped CV).  Fitting on all 73,920 labelled
scenarios would be 8x the data, but ~22.6% of its positives are scenes where the
logged expert itself crashes, and the model would happily learn to rank those
first; see runs/hardness/labels.json.

Scoring covers all 1000 shards, not just the 154 that were evaluated.  That is
the entire reason for using cheap a-priori descriptors instead of the labels
directly: 3,135 failure labels cannot reweight 2.3M training examples, but a
model fit on them can score every one.

The dirty-scene gate has to generalise too, and log replay only covers 21
shards.  ``road_clearance_min < 0`` -- the expert's own bounding box leaving the
drivable area under waymax's exact offroad test -- stands in for it: validated
at recall 1.00 against the log-replay offroad labels (4/4, with 3 false
positives in 515 explained by the eval truncating at reached_timestep).  Scenes
flagged by it are excluded from training rather than upweighted.

Writes risk.npz (per scenario and per anchor) and risk.json (fit diagnostics).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from train_diffusion.hardness.descriptors import FEATURE_NAMES
from train_diffusion.hardness.rank import _fit_logistic, auc, grouped_cv_auc


class Standardizer:
    """Median-impute + z-score with statistics frozen from the fit set."""

    def __init__(self, x: np.ndarray):
        self.median = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = self._impute(x)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std = np.where(self.std > 1e-12, self.std, 1.0)

    def _impute(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=np.float64, copy=True)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.take(self.median, np.nonzero(bad)[1])
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (self._impute(x) - self.mean) / self.std


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def load_labelled(labels_path: str, desc_dir: str):
    labels = np.load(labels_path, allow_pickle=True)
    index = {(str(s), int(i)): r for r, (s, i) in enumerate(zip(labels["shard"], labels["scenario_idx"]))}
    features = np.full((labels["scenario_idx"].size, len(FEATURE_NAMES)), np.nan)
    covered = np.zeros(labels["scenario_idx"].size, dtype=bool)
    for path in sorted(glob.glob(os.path.join(desc_dir, "*.desc.npz"))):
        with np.load(path, allow_pickle=True) as handle:
            shard = str(handle["shard"])
            for scenario_idx, row in zip(handle["scenario_indices"], handle["scenario_feat"]):
                target = index.get((shard, int(scenario_idx)))
                if target is not None:
                    features[target] = row
                    covered[target] = True
    return labels, features, covered


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--fit_desc_dir", required=True, help="descriptors for the labelled shards")
    parser.add_argument("--score_desc_dir", required=True, help="descriptors for every shard to score")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--fit_slice", choices=("expert_clean", "all"), default="expert_clean")
    parser.add_argument("--dirty_clearance", type=float, default=0.0,
                        help="expert scenes with road_clearance_min below this are excluded, not upweighted")
    args = parser.parse_args(argv)

    labels, features, covered = load_labelled(args.labels, args.fit_desc_dir)
    failed = ~labels["pol_success"]
    if args.fit_slice == "expert_clean":
        fit_mask = covered & labels["exp_labeled"] & labels["exp_success"]
    else:
        fit_mask = covered & ~(labels["exp_labeled"] & ~labels["exp_success"])
    if fit_mask.sum() < 500 or failed[fit_mask].sum() < 50:
        raise SystemExit(f"fit slice too small: n={int(fit_mask.sum())} pos={int(failed[fit_mask].sum())}")

    x_fit = features[fit_mask]
    y_fit = failed[fit_mask]
    groups = labels["shard"][fit_mask]

    # Re-measure held-out AUC with the exact fit configuration before shipping it.
    cv = grouped_cv_auc(x_fit, y_fit, groups, n_folds=5, l2=args.l2)

    scaler = Standardizer(x_fit)
    weights = _fit_logistic(scaler(x_fit), y_fit, args.l2)
    in_sample = auc(y_fit, _sigmoid(scaler(x_fit) @ weights[:-1] + weights[-1]))

    clearance_col = FEATURE_NAMES.index("road_clearance_min")
    scored_shards, scen_idx, scen_risk, anchor_risk, dirty = [], [], [], [], []
    anchors = None
    for path in sorted(glob.glob(os.path.join(args.score_desc_dir, "*.desc.npz"))):
        with np.load(path, allow_pickle=True) as handle:
            shard = str(handle["shard"])
            anchors = handle["anchors"] if anchors is None else anchors
            feats = handle["scenario_feat"].astype(np.float64)
            afeat = handle["anchor_feat"].astype(np.float64)
            risk = _sigmoid(scaler(feats) @ weights[:-1] + weights[-1])
            flat = afeat.reshape(-1, afeat.shape[-1])
            arisk = _sigmoid(scaler(flat) @ weights[:-1] + weights[-1]).reshape(afeat.shape[:2])
            n = feats.shape[0]
            scored_shards.append(np.full(n, shard, dtype=object))
            scen_idx.append(handle["scenario_indices"])
            scen_risk.append(risk)
            anchor_risk.append(arisk)
            clearance = feats[:, clearance_col]
            # NaN clearance == no road-edge points in the scene; treat as clean.
            dirty.append(np.isfinite(clearance) & (clearance < args.dirty_clearance))

    if not scen_idx:
        raise SystemExit(f"no descriptor shards under {args.score_desc_dir}")

    shard_arr = np.concatenate(scored_shards).astype(np.str_)
    risk_arr = np.concatenate(scen_risk)
    dirty_arr = np.concatenate(dirty)
    keep = ~dirty_arr
    anchor_arr = np.concatenate(anchor_risk)
    # Anchor windows are 5s against a full ~8s episode, so anchor risk averages
    # about half the scenario-level risk. The sampler normalises by a mean before
    # clipping, and using the scenario-level one there would push far more
    # examples onto the weight floor than intended -- store both.
    mean_anchor = float(anchor_arr[keep].mean()) if keep.any() else 0.0
    # Decile edges over the KEPT population: the sampler must not need global
    # statistics of its own to turn a risk into a weight.
    edges = np.quantile(risk_arr[keep], np.linspace(0.0, 1.0, 11)) if keep.any() else np.zeros(11)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        Path(args.out_dir) / "risk.npz",
        shard=shard_arr,
        scenario_idx=np.concatenate(scen_idx),
        scenario_risk=risk_arr.astype(np.float32),
        anchor_risk=anchor_arr.astype(np.float32),
        anchors=np.asarray(anchors, dtype=np.int32),
        dirty=dirty_arr,
        decile_edges=edges.astype(np.float64),
        mean_risk=np.asarray(float(risk_arr[keep].mean()) if keep.any() else 0.0),
        mean_anchor_risk=np.asarray(mean_anchor),
        feature_names=np.asarray(FEATURE_NAMES),
        coef=weights.astype(np.float64),
        scaler_median=scaler.median,
        scaler_mean=scaler.mean,
        scaler_std=scaler.std,
    )

    order = np.argsort(-np.abs(weights[:-1]))
    report = {
        "fit_slice": args.fit_slice,
        "n_fit": int(fit_mask.sum()),
        "n_fit_positives": int(y_fit.sum()),
        "fit_base_rate": float(y_fit.mean()),
        "grouped_cv": cv,
        "in_sample_auc": in_sample,
        "n_scored": int(risk_arr.size),
        "n_scored_shards": int(np.unique(shard_arr).size),
        "n_dirty_excluded": int(dirty_arr.sum()),
        "dirty_rate": float(dirty_arr.mean()),
        "risk_quantiles": {q: float(np.quantile(risk_arr[keep], float(q))) for q in ("0.1", "0.5", "0.9", "0.99")},
        "mean_risk": float(risk_arr[keep].mean()),
        "mean_anchor_risk": mean_anchor,
        "top_coefficients": [
            {"feature": FEATURE_NAMES[i], "coef": float(weights[i])} for i in order[:12]
        ],
    }
    with open(Path(args.out_dir) / "risk.json", "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=float)
    print(json.dumps(report, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
