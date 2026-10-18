"""Expert log replay on every scene of a splits file -> which ones are solvable.

The held-out strata were built on ``exp_clean`` scenes only
(``scan_womd_headroom.py``): the logged human, replayed through the same env,
reaches the goal and is never off-road or in overlap. An unlabelled shard slice
(``build_shard_splits.py``) has no such gate, so some of its "failures" are
scenes no driver could solve -- dirty ground truth, not policy error.

This replays the expert on each scene with the env built EXACTLY as
``proposal_select`` builds it (``SafetyFineTuner`` from the same flags), so the
label and the policy's ``solved`` come from the same flags in the same env. Per
scene it records the expert's three flags, ``exp_clean`` (the scan's definition)
and ``log_clean`` (the runner's own reset gate, ``_gate_expert_clean``: the log
never leaves the road edge after the replayed prefix; a ``log_clean`` = False
row is failed for every arm before it starts).

Then it writes a copy of the splits file with each labelled split restricted
to ``exp_clean & log_clean`` scenes, under the SAME split name, so existing
``rows.jsonl`` re-analyze against it unchanged (``select_analysis`` keeps only
scenes the splits file lists).

Resumable: scenes already in ``--expert_rows`` are skipped.

    python -m rl.diffusion_ft.tools.expert_label_splits <train_safety flags> \\
        --label_splits val10 --expert_rows OUT/expert_rows.jsonl \\
        --clean_splits_out OUT/val10_expclean_splits.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main():
    from rl.diffusion_ft.train_safety import SafetyFineTuner, build_argparser

    p = build_argparser()
    p.description = __doc__
    p.add_argument("--label_splits", required=True, help="comma-separated split names")
    p.add_argument("--label_chunk", type=int, default=32)
    p.add_argument("--expert_rows", required=True, help="per-scene JSONL (appended)")
    p.add_argument("--clean_splits_out", required=True)
    args = p.parse_args()
    names = [n.strip() for n in args.label_splits.split(",") if n.strip()]
    args.splits = ""              # stream chunks; do not preload eval states
    args.probe_per_split = 0
    t = SafetyFineTuner(args)
    spec = t._read_splits_spec(args.splits_json)

    out = Path(args.expert_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn last line from a killed task
            have[(r["split"], int(r["scene_id"]))] = r

    for name in names:
        ids = t._split_ids(spec, name)
        todo = [i for i in ids if (name, i) not in have]
        print(f"[expert] {name}: {len(ids)} scenes, {len(todo)} to do", flush=True)
        for start in range(0, len(todo), int(args.label_chunk)):
            sub = todo[start:start + int(args.label_chunk)]
            state = t.sampler.load_explicit(sub)
            t.env.reset(state, seed=args.seed)
            log_clean = np.asarray(t._gate_expert_clean(t.env.sim_state)) > 0.5
            exp = t.env.expert_replay_episode()
            reached = np.asarray(exp["reached_any_b"], bool)
            offroad = np.asarray(exp["offroad_any_b"], bool)
            overlap = np.asarray(exp["overlap_any_b"], bool)
            with out.open("a") as f:
                for j, sid in enumerate(sub):
                    r = {"split": name, "scene_id": int(sid),
                         "exp_reached": bool(reached[j]), "exp_offroad": bool(offroad[j]),
                         "exp_overlap": bool(overlap[j]),
                         "exp_clean": bool(reached[j] and not offroad[j] and not overlap[j]),
                         "log_clean": bool(log_clean[j])}
                    f.write(json.dumps(r) + "\n")
                    have[(name, int(sid))] = r
            done = start + len(sub)
            print(f"[expert] {name} {done}/{len(todo)}  exp_clean "
                  f"{np.mean(reached & ~offroad & ~overlap):.3f}  log_clean "
                  f"{log_clean.mean():.3f}", flush=True)

    clean = json.loads(json.dumps(spec))
    summary = {}
    for name in names:
        ids = t._split_ids(spec, name)
        rows = [have[(name, i)] for i in ids]
        keep = [r["scene_id"] for r in rows if r["exp_clean"] and r["log_clean"]]
        summary[name] = {
            "n": len(ids), "kept": len(keep),
            "exp_not_reached": sum(not r["exp_reached"] for r in rows),
            "exp_offroad": sum(r["exp_offroad"] for r in rows),
            "exp_overlap": sum(r["exp_overlap"] for r in rows),
            "log_dirty": sum(not r["log_clean"] for r in rows),
            "exp_clean_but_log_dirty": sum(r["exp_clean"] and not r["log_clean"] for r in rows)}
        clean["splits"][name] = {
            **spec["splits"][name], "scene_ids": keep,
            "criteria": spec["splits"][name].get("criteria", "") +
                        "; restricted to expert-solvable (exp_clean & log_clean)"}
        clean.setdefault("counts", {})[name] = len(keep)
        strata = clean.get("population", {}).get("strata", {})
        if name in strata:
            strata[name]["n"] = float(len(keep))
        print(f"[expert] {name}: kept {len(keep)}/{len(ids)}  {summary[name]}", flush=True)
    clean["expert_filter"] = {"source_splits_json": str(args.splits_json),
                              "expert_rows": str(out), "summary": summary}
    Path(args.clean_splits_out).write_text(json.dumps(clean, indent=1))
    print(f"[expert] wrote -> {args.clean_splits_out}")


if __name__ == "__main__":
    main()
