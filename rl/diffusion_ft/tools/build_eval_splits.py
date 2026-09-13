"""Turn one headroom scan into fixed eval splits: failure / success / train.

Reads the per-scene rows `scan_womd_headroom.py` writes (base deterministic
outcome + expert-clean gate) and partitions the EXPERT-CLEAN scenes -- the only
ones a policy can fairly be scored on -- by whether the base policy already
solves them:

  failure  = exp_clean AND NOT det_solve   -- the last few percent to nail
  success  = exp_clean AND     det_solve   -- the do-no-harm guard: a good loss
                                             must leave these alone

Each is split into a fixed HELD-OUT slice (for eval, so "fixed the failures
without breaking the successes" is measured on scenes never trained on) and a
TRAIN remainder (for a future fine-tune to actually train on). The partition is
a deterministic shuffle under --seed, so it is stable across reruns. Scene ids
are positions in --shard_glob's concatenated shard order, recorded in the JSON
and re-checked by whatever loads it.

CPU only -- reads saved rows, no model, no GPU.

Usage:
    python -m rl.diffusion_ft.tools.build_eval_splits \
        --scan_dir SCAN --shard_glob 'GLOB' --out_json SCAN/eval_splits.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from rl.diffusion_ft.tools.select_hard_womd import load_rows


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


def _partition(ids: list[int], n_heldout: int, rng: np.random.Generator):
    """Deterministic shuffle, then first ``n_heldout`` held out, rest train."""
    ids = sorted(int(i) for i in ids)
    perm = rng.permutation(len(ids))
    shuffled = [ids[i] for i in perm]
    n = min(n_heldout, len(shuffled))
    return sorted(shuffled[:n]), sorted(shuffled[n:])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan_dir", required=True)
    p.add_argument("--shard_glob", required=True,
                   help="the glob the scan ran against; recorded for re-checking")
    p.add_argument("--index_dir", default="/zfsauton/scratch/yixiz/waymax_rs/womd_offsets")
    p.add_argument("--out_json", default="")
    p.add_argument("--heldout_failure", type=int, default=64,
                   help="failure scenes held out for eval (rest go to train)")
    p.add_argument("--heldout_success", type=int, default=64,
                   help="success scenes held out for eval (rest go to train)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scan_dir = Path(args.scan_dir)
    out_json = Path(args.out_json) if args.out_json else scan_dir / "eval_splits.json"
    rows = load_rows(scan_dir)
    n_scanned = len(rows)
    rng = np.random.default_rng(args.seed)

    clean = [r for r in rows if r.get("exp_clean", False)]
    failure = [r for r in clean if not r["det_solve"]]
    success = [r for r in clean if r["det_solve"]]
    modes = Counter(_mode(r) for r in failure)

    f_hold, f_train = _partition([r["global_id"] for r in failure],
                                 args.heldout_failure, rng)
    s_hold, s_train = _partition([r["global_id"] for r in success],
                                 args.heldout_success, rng)

    base_solve = sum(r["det_solve"] for r in rows) / max(n_scanned, 1)
    clean_frac = len(clean) / max(n_scanned, 1)
    splits = {
        "failure_heldout": {"criteria": "exp_clean & ~det_solve",
                            "scene_ids": f_hold},
        "success_heldout": {"criteria": "exp_clean & det_solve",
                            "scene_ids": s_hold},
        "failure_train": {"criteria": "exp_clean & ~det_solve", "scene_ids": f_train},
        "success_train": {"criteria": "exp_clean & det_solve", "scene_ids": s_train},
    }
    payload = {
        "shard_glob": args.shard_glob,
        "index_dir": args.index_dir,
        "seed": args.seed,
        "n_scanned": n_scanned,
        "base_det_solve_over_scan": base_solve,
        "expert_clean_frac": clean_frac,
        "failure_modes": dict(modes),
        "counts": {k: len(v["scene_ids"]) for k, v in splits.items()},
        "splits": splits,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"[splits] scanned {n_scanned}  expert-clean {len(clean)} ({clean_frac:.1%})  "
          f"base solve {base_solve:.3f}")
    print(f"[splits] failure {len(failure)}  -> heldout {len(f_hold)}  train {len(f_train)}")
    print(f"[splits] success {len(success)}  -> heldout {len(s_hold)}  train {len(s_train)}")
    print(f"[splits] failure modes: {dict(modes)}")
    if len(f_hold) < 8:
        print(f"[splits] WARNING: only {len(f_hold)} failure scenes held out -- the "
              f"failure eval rate will be coarse. Scan more scenes.")
    print(f"[splits] wrote {out_json}")


if __name__ == "__main__":
    main()
