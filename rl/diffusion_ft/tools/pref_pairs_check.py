"""What the Diffusion-DPO pairs are, and whether they are BC's edit rows.

Reads `proposal_select --dump_bc` rows that carry `rejected` (submit_pref_dpo.sh
step 1) and, optionally, the bank-BC collection they re-run
(bank_bc/20260928_121223_39589, same arm and sampler seed). CPU only.

  pairs        switched rows of solved episodes (the trainer's default pool),
               per split and per scene
  edit         per-pair peak and mean |target - rejected| over covered steps, m,
               and the per-model-step profile (where along the plan the edit is)
  same_rows    matched on (split, scene_id, repeat, replan): does `target` agree
               with the reference collection bit for bit, and is `switched` the
               same? If yes, DPO and the BC runs trained on the same edits and
               differ only in the objective.

    python -m rl.diffusion_ft.tools.pref_pairs_check --data 'DIR/collect/tasks/*/bc_data/b_l24_r20/*.npz' \\
        --ref 'BCD/collect/tasks/*/bc_data/b_l24_r20/*.npz' --ego_range 100 --out OUT.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

KEYS = ("target", "valid", "switched", "split", "scene_id", "repeat", "replan", "ep_solved")


def _load(pattern, extra=()):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no rows match {pattern}")
    parts = {k: [] for k in KEYS + tuple(extra)}
    for f in files:
        z = np.load(f)
        for k in parts:
            if k not in z.files:
                raise SystemExit(f"{f} has no '{k}'")
            parts[k].append(z[k])
    return {k: np.concatenate(v, 0) for k, v in parts.items()}, len(files)


def _q(x):
    return {f"p{int(q * 100)}": float(np.quantile(x, q)) for q in (0.1, 0.5, 0.9)} | {
        "max": float(np.max(x)) if len(x) else 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--ref", default="", help="reference collection glob (optional)")
    p.add_argument("--ego_range", type=float, default=100.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    R, nf = _load(a.data, extra=("rejected", "rt_err_m"))
    n = len(R["target"])
    sw = R["switched"].astype(bool)
    pair = sw & R["ep_solved"].astype(bool)
    v = R["valid"].astype(bool)
    d = np.linalg.norm((R["target"] - R["rejected"])[..., :2], axis=-1) * a.ego_range
    kept_gap = float(np.abs(R["target"] - R["rejected"])[~sw].max()) if (~sw).any() else 0.0
    res = {"rows": n, "files": nf, "switched": int(sw.sum()), "pairs_solved": int(pair.sum()),
           "pairs_all": int(sw.sum()), "kept_rows_target_minus_rejected_max": kept_gap,
           "pairs_by_split": {s: int((R["split"][pair] == s).sum())
                              for s in np.unique(R["split"])},
           "pair_scenes": int(len(np.unique(R["scene_id"][pair]))),
           "rows_by_split": {s: int((R["split"] == s).sum()) for s in np.unique(R["split"])},
           "episodes_solved_by_split": {
               s: float(R["ep_solved"][(R["split"] == s) & (R["replan"] == 0)].mean())
               for s in np.unique(R["split"])}}
    if pair.any():
        dp, vp = d[pair], v[pair]
        res["edit_peak_m"] = _q(np.where(vp, dp, 0).max(1))
        res["edit_mean_m"] = _q((dp * vp).sum(1) / np.maximum(vp.sum(1), 1))
        res["edit_step_profile_m"] = [float(x) for x in
                                      (dp * vp).sum(0) / np.maximum(vp.sum(0), 1)]
        res["rejected_round_trip_m"] = _q(R["rt_err_m"][pair])
        per_scene = np.unique(R["scene_id"][pair], return_counts=True)[1]
        res["pairs_per_scene"] = _q(per_scene)
    print(json.dumps({k: v for k, v in res.items() if k != "edit_step_profile_m"}, indent=1))

    if a.ref:
        F, _ = _load(a.ref)
        key = lambda S: list(zip(S["split"].tolist(), S["scene_id"].tolist(),
                                 S["repeat"].tolist(), S["replan"].tolist()))
        ref_ix = {k: i for i, k in enumerate(key(F))}
        got = [(i, ref_ix.get(k)) for i, k in enumerate(key(R))]
        both = np.array([(i, j) for i, j in got if j is not None], int).reshape(-1, 2)
        same = {"rows_new": n, "rows_ref": len(F["target"]), "matched": int(len(both))}
        if len(both):
            ti, tj = both[:, 0], both[:, 1]
            eq = np.all(R["target"][ti] == F["target"][tj], axis=(1, 2))
            same.update({
                "target_identical": float(eq.mean()),
                "switched_identical": float((R["switched"][ti] == F["switched"][tj]).mean()),
                "ep_solved_identical": float((R["ep_solved"][ti] == F["ep_solved"][tj]).mean()),
                "target_max_abs_diff": float(np.abs(R["target"][ti] - F["target"][tj]).max()),
                "ref_edit_rows_solved": int((F["switched"] & F["ep_solved"]).sum())})
        res["same_rows"] = same
        print("same_rows:", json.dumps(same, indent=1))
        if same.get("target_identical", 0) < 1.0 or same["matched"] != n:
            print("NOTE: the recollection is NOT the reference collection row for row -- "
                  "DPO vs BC then differs in data as well as objective")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[pairs] wrote {a.out}")


if __name__ == "__main__":
    main()
