"""Offline comparison of candidate pickers on held-out recorded ES replans.

Pickers per replan group: the discriminator's top-1, the ES reward's pick
(`cand_selected`), the GT-closest oracle (lowest ADE to the logged expert) and a
random candidate (expectation over the group). For each: top-1 ADE to GT, Waymax
overlap / offroad / unsafe rate within the label horizon (<= 5 s), goal gain,
route progress and average speed. Ranking quality of the scores: per-group
Spearman correlation with -ADE, pairwise accuracy on the safety+progress preference
pairs, and safety AUROC on groups with both safe and unsafe candidates.
95% confidence intervals bootstrap over scenarios.

Subsets: filtered test replans (training filters), and the scenes whose first bad
replan was "ES picked unsafe although a safe candidate existed" (285 in the decision
report) restricted to the test split, at all their clean replans and at that first
bad replan.

  python -m es_discriminator.real_es.evaluate --predictions .../test_predictions.npz --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .dataset import (
    DEFAULT_INVALID, DEFAULT_RUN, Bank, cand_batch, es_unsafe_scenes, invalid_scene_keys, load_bank,
    training_mask,
)

PICK_METRICS = ("ade", "overlap", "offroad", "unsafe", "goal_gain", "progress", "speed")
PICKERS = ("discriminator", "es_reward", "gt_oracle", "random")
EXPECTED_ES_UNSAFE_SCENES = 285


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis (ties share their mean rank)."""
    n = x.shape[-1]
    order = np.argsort(x, axis=-1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(n, dtype=np.float64)[None].repeat(len(x), 0), axis=-1)
    out = np.empty_like(ranks)
    for i in range(len(x)):
        xs = x[i][order[i]]
        r = ranks[i][order[i]]
        _, start, counts = np.unique(xs, return_index=True, return_counts=True)
        for s, c in zip(start, counts):
            if c > 1:
                r[s:s + c] = r[s:s + c].mean()
        out[i][order[i]] = r
    return out


def preference_matrix(unsafe: np.ndarray, progress: np.ndarray, margin: float) -> np.ndarray:
    """[K,K] True where candidate i should beat j: safe over unsafe; among safe, >= margin more progress."""
    safe = ~unsafe
    better = safe[:, None] & unsafe[None, :]
    p = np.nan_to_num(progress, nan=-np.inf)
    both = safe[:, None] & safe[None, :]
    with np.errstate(invalid="ignore"):
        better |= both & ((p[:, None] - p[None, :]) >= margin)
    return better


def per_group_values(bank: Bank, gidx: np.ndarray, scores: np.ndarray, *, progress_key: str = "goal_gain",
                     margin: float = 2.0) -> dict:
    """Per-group metric values for every picker plus ranking statistics."""
    c = cand_batch(bank, gidx, ("ade", "selected", "overlap_step", "offroad_step", "unsafe",
                                "goal_gain", "progress", "speed"))
    vals = {
        "ade": c["ade"].astype(np.float64),
        "overlap": (c["overlap_step"] > 0).astype(np.float64),
        "offroad": (c["offroad_step"] > 0).astype(np.float64),
        "unsafe": c["unsafe"].astype(np.float64),
        "goal_gain": c["goal_gain"].astype(np.float64),
        "progress": c["progress"].astype(np.float64),
        "speed": c["speed"].astype(np.float64),
    }
    n = len(gidx)
    ar = np.arange(n)
    ade_inf = np.where(np.isfinite(vals["ade"]), vals["ade"], np.inf)
    picks = {
        "discriminator": np.argmax(scores, axis=1),
        "es_reward": np.argmax(c["selected"], axis=1),
        "gt_oracle": np.argmin(ade_inf, axis=1),
    }
    out = {p: {} for p in PICKERS}
    for m, v in vals.items():
        for p, idx in picks.items():
            out[p][m] = v[ar, idx]
        out["random"][m] = np.nanmean(v, axis=1)

    # Ranking statistics.
    finite = np.isfinite(vals["ade"])
    sp = np.full(n, np.nan)
    ok = finite.sum(1) > 2
    if ok.any():
        rs = _ranks(scores[ok])
        ra = _ranks(-np.where(finite[ok], vals["ade"][ok], np.nanmax(vals["ade"][ok]) + 1))
        rs -= rs.mean(1, keepdims=True)
        ra -= ra.mean(1, keepdims=True)
        den = np.sqrt((rs ** 2).sum(1) * (ra ** 2).sum(1))
        sp[ok] = np.where(den > 0, (rs * ra).sum(1) / np.maximum(den, 1e-12), np.nan)
    pair_acc = np.full(n, np.nan)
    auroc = np.full(n, np.nan)
    unsafe = c["unsafe"].astype(bool)
    prog = c[progress_key].astype(np.float64)
    for i in range(n):
        pref = preference_matrix(unsafe[i], prog[i], margin)
        if pref.any():
            d = scores[i][:, None] - scores[i][None, :]
            pair_acc[i] = float((d[pref] > 0).mean() + 0.5 * (d[pref] == 0).mean())
        u = unsafe[i]
        if u.any() and (~u).any():
            r = _ranks(scores[i][None])[0]
            n_pos, n_neg = (~u).sum(), u.sum()
            auroc[i] = (r[~u].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    out["ranking"] = {"spearman_score_vs_neg_ade": sp, "pairwise_accuracy": pair_acc, "safety_auroc": auroc}
    return out


def bootstrap(values: np.ndarray, scen: np.ndarray, *, n_boot: int = 1000, seed: int = 0) -> dict:
    ok = np.isfinite(values)
    v, s = values[ok], scen[ok]
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    uniq, inv = np.unique(s, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1)
    return {"mean": float(v.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(len(v))}


def summarize_groups(bank: Bank, gidx: np.ndarray, scores: np.ndarray, *, n_boot: int = 1000, **kw) -> dict:
    if len(gidx) == 0:
        return {"n_groups": 0, "n_scenes": 0}
    vals = per_group_values(bank, gidx, scores, **kw)
    scen = bank.groups["scen"][gidx]
    out = {"n_groups": int(len(gidx)), "n_scenes": int(len(np.unique(scen)))}
    for p in (*PICKERS, "ranking"):
        out[p] = {m: bootstrap(v, scen, n_boot=n_boot) for m, v in vals[p].items()}
    out["discriminator_minus_es"] = {
        m: bootstrap(vals["discriminator"][m] - vals["es_reward"][m], scen, n_boot=n_boot) for m in PICK_METRICS}
    return out


def quick_metrics(bank: Bank, gidx: np.ndarray, scores: np.ndarray, **kw) -> dict[str, float]:
    """Point estimates for per-epoch validation logging / checkpoint selection."""
    vals = per_group_values(bank, gidx, scores, **kw)
    m = {f"{p}/{k}": float(np.nanmean(vals[p][k])) for p in ("discriminator", "es_reward", "gt_oracle")
         for k in ("ade", "unsafe", "goal_gain")}
    m["ade_gap_to_oracle"] = float(np.nanmean(vals["discriminator"]["ade"] - vals["gt_oracle"]["ade"]))
    for k, v in vals["ranking"].items():
        m[f"ranking/{k}"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return m


def full_report(bank: Bank, gidx: np.ndarray, scores: np.ndarray, *, invalid: set,
                n_boot: int = 1000, **kw) -> dict:
    """All subsets for the groups in `gidx` (scored by `scores`), which should be one split."""
    pos = {int(g): i for i, g in enumerate(gidx)}
    clean, filt = training_mask(bank, invalid)
    in_set = np.zeros(bank.num_groups, bool)
    in_set[gidx] = True
    rep = {"filters": filt}
    sel = np.flatnonzero(in_set & clean)
    rep["filtered"] = summarize_groups(bank, sel, scores[[pos[g] for g in sel]], n_boot=n_boot, **kw)

    scene_mask, first_bad = es_unsafe_scenes(bank, invalid)
    total = int(scene_mask.sum())
    rep["es_unsafe_scenes_total"] = total
    full_bank = len(bank.files) == 300
    rep["es_unsafe_scenes_matches_report"] = (total == EXPECTED_ES_UNSAFE_SCENES) if full_bank else None
    if full_bank and total != EXPECTED_ES_UNSAFE_SCENES:
        raise RuntimeError(f"ES-unsafe scene rule gives {total}, decision report has {EXPECTED_ES_UNSAFE_SCENES}")
    gs = bank.groups
    pose_ok = gs["group_valid"] & ~(gs["anchor_overlap"] | gs["anchor_offroad"]) & np.isfinite(gs["min_ade"])
    in_scenes = scene_mask[gs["scen"]]
    sel_all = np.flatnonzero(in_set & in_scenes & pose_ok)
    rep["es_unsafe_scenes_all_replans"] = summarize_groups(
        bank, sel_all, scores[[pos[g] for g in sel_all]], n_boot=n_boot, **kw)
    fb = first_bad[scene_mask]
    fb = np.array([g for g in fb if in_set[g] and pose_ok[g]], dtype=np.int64)
    rep["es_unsafe_scenes_first_bad_replan"] = summarize_groups(
        bank, fb, scores[[pos[g] for g in fb]], n_boot=n_boot, **kw)
    rep["es_unsafe_scenes_in_split"] = int(len(np.unique(gs["scen"][np.flatnonzero(in_set & in_scenes)])))
    return rep


def _fmt(b: dict | None, pct: bool = False) -> str:
    if not b or b.get("mean") is None:
        return "-"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.2f}")
    return f"{f(b['mean'])} [{f(b['lo'])}, {f(b['hi'])}]"


def to_markdown(rep: dict, title: str) -> str:
    L = [f"# {title}", "", f"Filters (groups remaining): `{json.dumps(rep['filters'])}`", ""]
    pct = {"overlap", "offroad", "unsafe"}
    for name in ("filtered", "es_unsafe_scenes_all_replans", "es_unsafe_scenes_first_bad_replan"):
        r = rep[name]
        L += [f"## {name} ({r.get('n_groups', 0)} replans, {r.get('n_scenes', 0)} scenes)", ""]
        if not r.get("n_groups"):
            L += ["(none in this split)", ""]
            continue
        L += ["| picker | ADE m | overlap % | offroad % | unsafe % | goal gain m | progress m | speed m/s |",
              "|---|---|---|---|---|---|---|---|"]
        for p in (*PICKERS, "discriminator_minus_es"):
            L.append(f"| {p} | " + " | ".join(_fmt(r[p][m], m in pct) for m in PICK_METRICS) + " |")
        rk = r["ranking"]
        L += ["", f"Ranking: Spearman(score, -ADE) {_fmt(rk['spearman_score_vs_neg_ade'])}, "
                  f"pairwise accuracy {_fmt(rk['pairwise_accuracy'])}, safety AUROC {_fmt(rk['safety_auroc'])}", ""]
    L.append(f"ES-unsafe scenes: {rep['es_unsafe_scenes_total']} total "
             f"(report: {EXPECTED_ES_UNSAFE_SCENES}), {rep['es_unsafe_scenes_in_split']} in this split")
    return "\n".join(L) + "\n"


def save_predictions(path: str, bank: Bank, gidx: np.ndarray, scores: np.ndarray) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, group_index=gidx, scen_id=bank.groups["scen_id"][gidx],
                        group_t=bank.groups["group_t"][gidx], scores=scores.astype(np.float32),
                        files=np.asarray([os.path.basename(f) for f in bank.files]))
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="*_predictions.npz written by the trainer")
    p.add_argument("--data_dir", default=DEFAULT_RUN)
    p.add_argument("--invalid_at_start", default=DEFAULT_INVALID)
    p.add_argument("--progress_key", default="goal_gain")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default="Offline evaluation")
    a = p.parse_args()
    pred = np.load(a.predictions)
    bank = load_bank(a.data_dir)
    key = {(s, int(t)): g for g, (s, t) in enumerate(zip(bank.groups["scen_id"], bank.groups["group_t"]))}
    gidx = np.array([key[(s, int(t))] for s, t in zip(pred["scen_id"], pred["group_t"])])
    rep = full_report(bank, gidx, pred["scores"], invalid=invalid_scene_keys(a.invalid_at_start),
                      progress_key=a.progress_key)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out_dir, "eval.json"), "w"), indent=1)
    md = to_markdown(rep, a.title)
    open(os.path.join(a.out_dir, "eval.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
