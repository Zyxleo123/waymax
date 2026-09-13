"""Turn a headroom scan into a training pool: hard, winnable, WOMD.

Stage B of the hard-WOMD selection. Reads the per-scene rows written by
`scan_womd_headroom.py` and applies two filters, in this order:

  1. **Expert-clean gate.** Drop every scene whose logged human is itself
     off-road, in contact, or never reaches the goal. This is not hygiene, it is
     the whole reason the scan rolls an expert arm: selecting on policy failure
     selects just as hard for dirty ground truth, and 22.6% of the diffusion
     policy's failures on the curated set were unwinnable scenes. A pool full of
     those trains the policy to imitate the impossible and caps the achievable
     solve rate at something below 1 for reasons the reward cannot fix.

  2. **Headroom filter.** Keep the scenes the base policy does NOT solve
     deterministically. That is the definition of headroom: on a uniform WOMD
     draw the base already solves ~95-98%, which is why the first WOMD re-run
     could not be read.

The output pool is a list of global scene ids against a specific `--shard_glob`.
Those ids are positions in that glob's concatenated shard order, so the pool is
only meaningful against the same glob -- which is recorded in the JSON and
re-checked by the trainer.

Reads only saved artifacts, no model and no GPU, so the summary re-renders
cheaply (a `cpu` job).

Usage:
    python -m rl.diffusion_ft.tools.select_hard_womd \
        --scan_dir OUT --out_json OUT/hard_womd.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan_dir", required=True, help="Dir holding rows_task*.jsonl.")
    p.add_argument("--out_json", default="", help="Default: <scan_dir>/hard_womd.json")
    p.add_argument("--shard_glob", required=True,
                   help="Recorded in the pool; the ids only mean anything against it.")
    p.add_argument("--index_dir", default="")
    p.add_argument("--require_expert_clean", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Drop scenes whose logged human is offroad/in contact or "
                        "never reaches. Turning this OFF admits unwinnable scenes.")
    p.add_argument("--include_marginal", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Also admit scenes the base SOLVES but with off-road steps "
                        "-- near-misses. Widens a thin pool at the cost of a "
                        "weaker definition of headroom.")
    p.add_argument("--min_pool", type=int, default=64,
                   help="Fail loudly below this: a pool smaller than a few "
                        "batches is memorization wearing a corpus's clothes.")
    return p.parse_args()


def load_rows(scan_dir: Path) -> list[dict]:
    rows: dict[int, dict] = {}
    files = sorted(globmod.glob(str(scan_dir / "rows_task*.jsonl")))
    if not files:
        raise SystemExit(f"no rows_task*.jsonl under {scan_dir} -- did the scan run?")
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a killed task
                # Last write wins: a resubmitted task may re-roll a scene whose
                # row was written but not counted.
                rows[int(r["global_id"])] = r
    print(f"[select] {len(rows)} unique scenes from {len(files)} task files")
    return list(rows.values())


def main() -> None:
    args = _parse_args()
    scan_dir = Path(args.scan_dir)
    out_json = Path(args.out_json) if args.out_json else scan_dir / "hard_womd.json"
    rows = load_rows(scan_dir)
    n_scanned = len(rows)

    expert_dirty = [r for r in rows if not r.get("exp_clean", False)]
    pool_src = ([r for r in rows if r.get("exp_clean", False)]
                if args.require_expert_clean else rows)

    unsolved = [r for r in pool_src if not r["det_solve"]]
    marginal = [r for r in pool_src if r["det_solve"] and r.get("det_off_steps", 0) > 0]
    keep = unsolved + (marginal if args.include_marginal else [])
    ids = sorted(int(r["global_id"]) for r in keep)

    # Why the base fails, over the KEPT scenes. These call for different things:
    # never-reached is a competence/route problem, reached-but-dirty is exactly
    # what the entry_clean ordering was designed to price.
    def _mode(r: dict) -> str:
        if not r["det_reached"]:
            return "never_reached"
        if r["det_collided"] and r["det_offroad"]:
            return "reached_offroad+collided"
        if r["det_collided"]:
            return "reached_collided"
        if r["det_offroad"]:
            return "reached_offroad"
        return "solved_marginal"
    modes = Counter(_mode(r) for r in keep)

    det_solve = sum(r["det_solve"] for r in rows) / max(n_scanned, 1)
    sto_rows = [r for r in rows if "sto_solve" in r]
    sto_solve = (sum(r["sto_solve"] for r in sto_rows) / len(sto_rows)) if sto_rows else float("nan")

    lines = [
        "# Hard-WOMD pool",
        "",
        f"- scanned: **{n_scanned}** scenes",
        f"- base deterministic solve over the scan: **{det_solve:.3f}**",
        f"- base stochastic solve (collection policy, sigma="
        f"{rows[0].get('sigma_floor', float('nan'))}): **{sto_solve:.3f}**"
        + ("" if sto_rows else "  _(not rolled)_"),
        f"- expert-dirty (log itself offroad / in contact / never reaches): "
        f"**{len(expert_dirty)}** ({len(expert_dirty) / max(n_scanned, 1):.1%})",
        f"- expert-clean and unsolved by the base: **{len(unsolved)}**",
        f"- solved-but-marginal (off-road steps, still counted a solve): "
        f"**{len(marginal)}**"
        + ("  _(included)_" if args.include_marginal else "  _(excluded)_"),
        "",
        f"## Pool: {len(ids)} scenes",
        "",
        "| failure mode | scenes |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {v} |" for k, v in modes.most_common()]
    # The conditional expert-dirty rate is the number the uniform 0/64 read could
    # not give: dirtiness among HARD scenes is what selection actually exposes.
    hard_all = [r for r in rows if not r["det_solve"]]
    if hard_all:
        cond = sum(1 for r in hard_all if not r.get("exp_clean", False)) / len(hard_all)
        lines += ["", f"Expert-dirty rate among base failures: **{cond:.1%}** "
                      f"({sum(1 for r in hard_all if not r.get('exp_clean', False))}"
                      f"/{len(hard_all)}). A uniform draw measured 0/64; this is the "
                      f"rate that matters when selecting on difficulty."]
    summary = "\n".join(lines) + "\n"
    (scan_dir / "hard_womd_summary.md").write_text(summary)
    print(summary)

    if len(ids) < int(args.min_pool):
        raise SystemExit(
            f"FATAL: pool has {len(ids)} scenes, below --min_pool {args.min_pool}. "
            f"Scan more scenes (--n_scenes) or widen with --include_marginal; do "
            f"NOT train on a pool this thin -- it is an overfit target, not a corpus.")

    out_json.write_text(json.dumps({
        "shard_glob": args.shard_glob,
        "index_dir": args.index_dir,
        "n_scanned": n_scanned,
        "criteria": {
            "require_expert_clean": bool(args.require_expert_clean),
            "include_marginal": bool(args.include_marginal),
            "base_det_solve_over_scan": det_solve,
        },
        "failure_modes": dict(modes),
        "scene_ids": ids,
    }, indent=2) + "\n")
    print(f"[select] wrote {len(ids)} scene ids -> {out_json}")
    print(f"[select] summary -> {scan_dir / 'hard_womd_summary.md'}")


if __name__ == "__main__":
    main()
