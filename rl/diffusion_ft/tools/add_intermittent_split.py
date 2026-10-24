"""Add the scenes the base solves only SOMETIMES as a third eval stratum.

`rebuild_verified_splits` keeps a failure only if the base solved it in 0 of N
verification rollouts and a held-out success only if it solved it in N of N.
Everything in between was dropped from both pools, so the marginal rate covers
96.5% of the expert-clean corpus and says nothing about the rest. That rest is
the group most likely to benefit from extra sampling plus selection: a successful
behaviour already exists inside the base's own sampling distribution, and the
question is only whether a selector can find it. Measured from the seed-777 rows:
223 of the 628 failure candidates and 54 of the 2564 verified success candidates
are intermittent -- a stratum about twice the size of the verified-failure one.

This writes a NEW splits file (the verified one is left untouched, so every
existing run keeps its identity):

* ``intermittent_heldout``: expert-clean scenes with ``0 < solved_frac < 1``.
  None of them is in ``failure_train`` or ``success_train`` (rebuild excluded
  both kinds), so the stratum is held out from every existing fine-tune by
  construction. Checked here rather than assumed.
* ``population.strata`` is copied UNCHANGED, so the trainer's two-stratum
  marginal stays bit-for-bit comparable with every earlier run.
* ``population.strata3`` adds the third stratum. Its size is counted exactly on
  the failure candidates (all verified) and extrapolated on the success
  candidates (a sample was verified), the same way the success stratum is.

CPU only; reads saved rows. Usage (submitted, never inline)::

    python -m rl.diffusion_ft.tools.add_intermittent_split \\
        --rows .../verify_rows.jsonl --in_json .../eval_splits_verified.json \\
        --source_json .../eval_splits.json --out_json .../eval_splits_3strata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.diffusion_ft.tools.rebuild_verified_splits import read_rows


def intermittent_strata(rows: dict[int, dict], fail_src: list[int],
                        succ_src: list[int], strata2: dict) -> tuple[list[int], dict, dict]:
    """``(scene_ids, strata3, counts)`` from verification rows.

    Pure function so the arithmetic is testable without files. ``strata2`` is
    the existing two-stratum block; its ``n`` values are reused verbatim so the
    three-stratum weights differ from the two-stratum ones only by the added
    stratum, never by a re-count of the other two.
    """
    def mid(r):
        return r.get("clean_frac", 1.0) >= 1.0 and 0.0 < float(r["solved_frac"]) < 1.0

    fail_mid = [s for s in fail_src if s in rows and mid(rows[s])]
    succ_verified = [s for s in succ_src if s in rows]
    succ_mid = [s for s in succ_verified if mid(rows[s])]
    # Failure candidates are fully enumerated; success candidates were sampled,
    # so their intermittent share is scaled to the whole success source.
    frac_succ_mid = (len(succ_mid) / len(succ_verified)) if succ_verified else 0.0
    n_int = float(len(fail_mid)) + len(succ_src) * frac_succ_mid
    n_fail = float(strata2["failure_heldout"]["n"])
    n_succ = float(strata2["success_heldout"]["n"])
    denom = n_fail + n_succ + n_int
    strata3 = {
        "failure_heldout": {"n": n_fail, "weight": n_fail / denom},
        "success_heldout": {"n": n_succ, "weight": n_succ / denom},
        "intermittent_heldout": {"n": n_int, "weight": n_int / denom},
    }
    counts = {"fail_source_intermittent": len(fail_mid),
              "success_source_verified": len(succ_verified),
              "success_source_intermittent": len(succ_mid),
              "success_intermittent_fraction": frac_succ_mid}
    return sorted(set(fail_mid) | set(succ_mid)), strata3, counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, help="the verify rows the splits were built from")
    p.add_argument("--in_json", required=True, help="eval_splits_verified.json")
    p.add_argument("--source_json", required=True,
                   help="the UNverified splits file rebuild read (its failure/success "
                        "candidate lists define the corpus)")
    p.add_argument("--out_json", required=True)
    a = p.parse_args()

    rows = read_rows(Path(a.rows))
    spec = json.loads(Path(a.in_json).read_text())
    src = json.loads(Path(a.source_json).read_text())["splits"]
    fail_src = sorted({int(i) for n in ("failure_heldout", "failure_train")
                       for i in src.get(n, {}).get("scene_ids", [])})
    succ_src = sorted({int(i) for n in ("success_heldout", "success_train")
                       for i in src.get(n, {}).get("scene_ids", [])})
    pop = spec.get("population") or {}
    if set(pop.get("strata", {})) != {"failure_heldout", "success_heldout"}:
        raise SystemExit("[intermittent] input has no two-stratum population block; "
                         "run rebuild_verified_splits first")

    ids, strata3, counts = intermittent_strata(rows, fail_src, succ_src, pop["strata"])
    # Held out from every training pool by construction -- verify, do not assume.
    trained = set()
    for n in ("failure_train", "success_train"):
        trained |= {int(i) for i in spec["splits"][n]["scene_ids"]}
    evald = set()
    for n in ("failure_heldout", "success_heldout"):
        evald |= {int(i) for i in spec["splits"][n]["scene_ids"]}
    clash_t, clash_e = trained & set(ids), evald & set(ids)
    if clash_t or clash_e:
        raise SystemExit(f"[intermittent] {len(clash_t)} scene(s) in a training pool and "
                         f"{len(clash_e)} in another held-out split -- refusing")

    corpus = float(pop.get("corpus_expert_clean", len(fail_src) + len(succ_src)))
    covered = sum(v["n"] for v in strata3.values())
    out = dict(spec)
    out["splits"] = dict(spec["splits"])
    out["splits"]["intermittent_heldout"] = {
        "criteria": "verified: 0 < base solved_frac < 1 over N rollouts of the eval "
                    "path, expert-clean; in no training pool",
        "scene_ids": ids}
    out["counts"] = {k: len(v["scene_ids"]) for k, v in out["splits"].items()}
    out["population"] = dict(pop)
    out["population"]["strata3"] = strata3
    out["population"]["coverage3"] = covered / corpus if corpus else 0.0
    out["population"]["intermittent_counts"] = counts
    out["population"]["note3"] = (
        "strata3 adds intermittent_heldout; strata (two-stratum) is unchanged so the "
        "trainer's logged marginal stays comparable with runs on the verified file")
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    print(f"[intermittent] {len(ids)} scenes  ({json.dumps(counts)})")
    print("[intermittent] strata3: " + "  ".join(
        f"{k} w={v['weight']:.5f} (n={v['n']:.0f})" for k, v in strata3.items())
        + f"  coverage3={out['population']['coverage3']:.4f}")
    print(f"[intermittent] wrote -> {a.out_json}")


if __name__ == "__main__":
    main()
