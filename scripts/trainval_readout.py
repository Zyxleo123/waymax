"""Log-replay readout of the ES arms over WOMD train shards 0-1 + val shards 0-1, per split and pooled.

Train 0-1 come from the train2 run root; val 0-1 are group_0 of the val10all run (same code and configs).
Per slice: every scene, then the expert-clean scenes (the logged human succeeds under the same evaluator),
then by the disc's training split (fail1500's 5 shard-1 scenes may be in-sample for both scorers).
Reuses scripts/compare_es_disc.py's loader and summarizer.

    python scripts/trainval_readout.py --train_root <awm_train2_...> --val_root <awm_val10all_...> --out <stem>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_es_disc import KEYS, load_arm, summarize  # noqa: E402

ARMS = ["awm_sc_posthoc", "mf_disc_posthoc", "expert"]
PAIRS = ["awm_sc_posthoc:mf_disc_posthoc", "awm_sc_posthoc:expert", "mf_disc_posthoc:expert"]
NAMES = {"awm_sc_posthoc": "ES + repair + AnchorWM + learned reward",
         "mf_disc_posthoc": "ES + repair + MF + disc", "expert": "expert (logged human)"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", required=True)
    p.add_argument("--val_root", required=True)
    p.add_argument("--val_group", default="group_0", help="val10all group holding val shards 0-1")
    p.add_argument("--split_json", default="")
    p.add_argument("--expect", default="train=934,val=595")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    expect = {k: int(v) for k, v in (x.split("=") for x in a.expect.split(","))}
    src = {"train": lambda arm: Path(a.train_root) / "log" / arm,
           "val": lambda arm: Path(a.val_root) / "log" / arm / a.val_group}
    rows = {sp: {arm: load_arm(f(arm), "log") for arm in ARMS} for sp, f in src.items()}
    counts = {sp: {arm: len(r) for arm, r in d.items()} for sp, d in rows.items()}
    print("[readout] scenes per split/arm:", counts)
    split = json.loads(Path(a.split_json).read_text())["assignment"] if a.split_json else {}

    rep = {"counts": counts, "expect": expect,
           "complete": all(c == expect[sp] for sp, d in counts.items() for c in d.values()), "slices": {}}
    md = [f"# ES best arms, WOMD train 0-1 + val 0-1 (log replay)\n",
          f"train root `{a.train_root}`  \nval root `{a.val_root}` ({a.val_group})\n",
          "scenes per arm: " + "; ".join(f"{sp} " + ", ".join(f"{k} {v}/{expect[sp]}" for k, v in d.items())
                                         for sp, d in counts.items()) + "\n",
          "arms: " + "; ".join(f"`{k}` = {v}" for k, v in NAMES.items()) + "\n"]
    common = {sp: sorted(set.intersection(*(set(r) for r in d.values()))) for sp, d in rows.items()}
    keyed = [(sp, s) for sp in ("train", "val") for s in common[sp]]

    def mat_for(sel):
        return {arm: {k: np.array([bool(rows[sp][arm][s][k]) for sp, s in sel]) for k in KEYS} for arm in ARMS}

    for tag, sel in (("train", [x for x in keyed if x[0] == "train"]),
                     ("val", [x for x in keyed if x[0] == "val"]), ("all", keyed)):
        if not sel:
            continue
        mat = mat_for(sel)
        clean = mat["expert"]["success"]
        out, lines = summarize(mat, ARMS, PAIRS, a.n_boot, tag)
        rep["slices"][tag] = out
        md += [f"\n## {tag}: {len(sel)} scenes\n"] + lines
        if clean.any():
            sub = {arm: {k: v[clean] for k, v in d.items()} for arm, d in mat.items()}
            out_c, lines_c = summarize(sub, ARMS[:2], PAIRS[:1], a.n_boot, f"{tag}@expert_clean")
            rep["slices"][f"{tag}@expert_clean"] = out_c
            md += [f"\n### {tag}, expert-clean: {int(clean.sum())} scenes (expert fails {int((~clean).sum())})\n"] + lines_c
        if split:
            lab = np.array([split.get(rows[sp]["expert"][s].get("scenario_id") or "", "unseen") for sp, s in sel])
            seen = {str(k): int((lab == k).sum()) for k in sorted(set(lab))}
            rep["slices"][tag]["disc_split_labels"] = seen
            md.append(f"\ndisc/learned-reward fine-tune split labels of these scenes: {seen}\n")
    Path(a.out + ".json").write_text(json.dumps(rep, indent=1) + "\n")
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print(f"[readout] -> {a.out}.json, {a.out}.md")


if __name__ == "__main__":
    main()
