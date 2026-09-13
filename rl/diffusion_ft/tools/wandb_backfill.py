"""Push an already-finished (or still-running) safety fine-tune log to W&B.

``train_safety.py`` only grew a ``--wandb`` flag on 2026-09-02, so every run
before that exists solely as ``train_log.jsonl`` -- including the 500-iteration
job 40438. The rows are all there; nothing needs recomputing. This reads them and
replays them into a W&B run, which is why it is a CPU job: it touches no
checkpoint, no corpus and no GPU.

Idempotent by design. The run id is persisted to ``wandb_run_id.txt`` in the
out_dir, the same file the trainer writes, so:

* re-running this against a longer log continues the same W&B run and skips rows
  already pushed (it never rewinds the step axis, which W&B refuses anyway);
* a *resumed* training job with ``--wandb`` picks up the same id and continues
  the identical curve, instead of opening a second chart at the iteration the
  wall-clock limit happened to fall on.

That last point is the reason to backfill a run that is still going: it puts the
history in place under the id the trainer will later adopt.

Usage:
    python -m rl.diffusion_ft.tools.wandb_backfill --out_dir DIR [--project P] [--name N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_rows(log_path: Path) -> list[dict]:
    """Rows in iteration order, tolerating a truncated final line.

    A run killed mid-write (a time limit, a preemption) leaves a partial last
    line. Dropping it is right; refusing to read the other 500 rows is not.
    """
    rows, bad = [], 0
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[backfill] skipped {bad} unparseable line(s) (truncated write?)")
    return sorted(rows, key=lambda r: int(r.get("iter", 0)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="a train_safety out_dir")
    p.add_argument("--project", default="diffusion_ft_safety")
    p.add_argument("--entity", default=None)
    p.add_argument("--name", default=None, help="defaults to the out_dir basename")
    p.add_argument("--group", default=None)
    p.add_argument("--dry_run", type=int, default=0,
                   help="1 = parse and report what would be pushed, contact nothing")
    args = p.parse_args()

    out = Path(args.out_dir)
    log_path = out / "train_log.jsonl"
    if not log_path.exists():
        raise SystemExit(f"[backfill] no train_log.jsonl in {out}")
    rows = load_rows(log_path)
    if not rows:
        raise SystemExit(f"[backfill] {log_path} has no rows")

    # `meta.json` is written next to the checkpoint and carries the settings the
    # run was launched with -- the only record of them once the sbatch log rolls.
    meta_path = out / "latest" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    keys = sorted({k for r in rows for k in r})
    print(f"[backfill] {log_path}: {len(rows)} rows, iters "
          f"{rows[0].get('iter')}..{rows[-1].get('iter')}, {len(keys)} keys")
    print(f"[backfill] meta: {meta if meta else '(none -- pre-dates meta.json)'}")
    if args.dry_run:
        print(f"[backfill] dry run; keys = {keys}")
        return

    import wandb

    id_path = out / "wandb_run_id.txt"
    run_id = id_path.read_text().strip() if id_path.exists() else wandb.util.generate_id()
    run = wandb.init(project=args.project, entity=args.entity,
                     name=args.name or out.name, group=args.group,
                     id=run_id, resume="allow",
                     config={**meta, "out_dir": str(out), "backfilled": True})
    id_path.write_text(run_id)

    # W&B rejects a step that moves backwards, so a re-run must start above what
    # is already there rather than replaying from zero.
    start = int(getattr(run, "step", 0) or 0) if getattr(run, "resumed", False) else -1
    pushed = 0
    for r in rows:
        it = int(r.get("iter", 0))
        if it <= start:
            continue
        wandb.log(r, step=it)
        pushed += 1
    wandb.finish()
    print(f"[backfill] pushed {pushed}/{len(rows)} rows to {args.project}/{run.name} "
          f"(id {run_id}); {len(rows) - pushed} already present")
    print(f"[backfill] run id saved to {id_path} -- a resumed trainer with --wandb "
          f"will continue this same run")


if __name__ == "__main__":
    main()
