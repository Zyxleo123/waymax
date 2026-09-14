"""Summarize a repair search (CPU): what fraction of base failures has a VERIFIED
repair, of which kind, how far back it had to rewind.

Reads ``counterfactual --mode repair`` rows. Read ``id_consistent`` first: it is
the fraction of searched failures where the identity branch reproduced the base
episode exactly. Below 1.0 the branching engine is not replaying the original
episode, and every repair count here is unreliable.

    python -m rl.diffusion_ft.tools.repair_summary --rows 'DIR/*/repairs.jsonl' --out S.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(rows):
    todo = [r for r in rows if r["todo"]]
    out = {"scenes": len(rows), "base_solved": sum(r["base_solved"] for r in rows),
           "failures_searched": len(todo),
           "id_consistent": (sum(r["id_consistent"] for r in todo) / len(todo)) if todo else None,
           "repaired": sum(r["repaired"] for r in todo),
           "repaired_rate": (sum(r["repaired"] for r in todo) / len(todo)) if todo else None}
    by_type = defaultdict(lambda: [0, 0])
    for r in todo:
        by_type[r["fail_type"]][0] += 1
        by_type[r["fail_type"]][1] += int(r["repaired"])
    out["by_fail_type"] = {k: {"n": n, "repaired": m, "rate": m / n} for k, (n, m) in by_type.items()}
    out["chosen_family"] = dict(Counter(r["cand"].split("h")[0].rstrip("0123456789.-s")
                                        for r in todo if r["repaired"]))
    out["chosen_cand"] = dict(Counter(r["cand"] for r in todo if r["repaired"]))
    out["rewind"] = dict(Counter(str(r["rewind"]) for r in todo if r["repaired"]))
    any_succ = Counter()
    for r in todo:
        for c in r["success_by_cand"]:
            any_succ[c] += 1
    out["scenes_any_success_by_cand"] = dict(any_succ)
    out["targets"] = sum(r.get("n_targets", 0) for r in todo if r["repaired"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = {}
    for f in sorted({f for g in a.rows for f in glob.glob(g)}):
        for line in Path(f).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["scene_id"], r["split"])] = r
    s = summarize(list(rows.values()))
    Path(a.out).write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
