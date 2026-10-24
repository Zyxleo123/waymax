"""Per-scene expert-replay winnability ceiling for the DPPO reward.

Answers a single question before we spend GPU on RL: *is this scene winnable at
all under the training reward?* It builds the exact same env as
``train_dppo.py`` (:class:`DiffusionWaymaxEnv` + :class:`RewardConfig`) and rolls
the ego's *logged* trajectory via :meth:`expert_replay_episode` -- the reward the
best-possible (human-demonstration) policy would collect. If that ceiling is as
low as what DPPO already reaches, the scene has no headroom and RL cannot help;
if it is high and clean, the gap is real and worth optimizing.

CPU-only (no model forward on the expert path). Submitted, not run inline.

    python -m rl.diffusion_ft.winnability \
        --tfrecord /data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord \
        --indices 0-31 --out runs/dft_winnability/failures_0_31.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # expert replay needs no GPU

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.train_dppo import load_batch
from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint


def parse_indices(spec: str) -> list[int]:
    """Accept ``0-31`` ranges and ``0,3,7`` lists (mixed)."""
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfrecord", type=str,
                   default="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--indices", type=str, default="0-31",
                   help="e.g. '0-31' or '0,1,2,5'")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/dft_winnability/ceiling.json")
    args = p.parse_args(argv)

    indices = parse_indices(args.indices)
    state = load_batch(args.tfrecord, indices)
    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])

    # Same construction as DPPOTrainer._build_for.
    env = DiffusionWaymaxEnv(
        load_diffusion_checkpoint(args.ckpt), reward_config=RewardConfig(),
        max_num_objects=n_obj, use_ema=args.use_ema,
        replan_interval_steps=args.replan_interval_steps, seed=args.seed,
    )
    env.reset(state)
    ep = env.expert_replay_episode()

    ret = np.asarray(ep["episode_return_b"], dtype=float)
    outer = np.asarray(ep["outer_reward_b"], dtype=float)
    offroad = np.asarray(ep["offroad_any_b"], dtype=bool)
    overlap = np.asarray(ep["overlap_any_b"], dtype=bool)
    reached = np.asarray(ep["reached_any_b"], dtype=bool)
    clean = reached & (~offroad) & (~overlap)

    rows = []
    print(f"[winnability] tfrecord={args.tfrecord}")
    print(f"[winnability] {'idx':>4} {'ceiling':>9} {'reached':>7} {'offroad':>7} "
          f"{'overlap':>7} {'clean':>5}")
    for k, i in enumerate(indices):
        row = {"index": int(i), "expert_ceiling": float(ret[k]),
               "outer_reward": float(outer[k]), "reached": bool(reached[k]),
               "offroad": bool(offroad[k]), "overlap": bool(overlap[k]),
               "clean_win": bool(clean[k])}
        rows.append(row)
        print(f"[winnability] {i:>4} {ret[k]:>9.3f} {str(bool(reached[k])):>7} "
              f"{str(bool(offroad[k])):>7} {str(bool(overlap[k])):>7} "
              f"{str(bool(clean[k])):>5}")

    win = [r["index"] for r in rows if r["clean_win"]]
    print(f"[winnability] clean-winnable {len(win)}/{len(indices)}: {win}")
    print(f"[winnability] ceiling mean={ret.mean():.3f} "
          f"min={ret.min():.3f} max={ret.max():.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"tfrecord": args.tfrecord, "indices": indices,
                   "clean_winnable": win, "rows": rows}, f, indent=1)
    print(f"[winnability] wrote {out_path}")


if __name__ == "__main__":
    main()
