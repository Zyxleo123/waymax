"""How much of the DPPO collision rate is the SDC's fault, and how much is the sim?

Every non-ego agent in this environment replays its logged trajectory and is
completely non-reactive (``controlled_object=ObjectType.SDC`` on a
``PlanningAgentEnvironment``), and the collision metric is Waymax ``overlap`` --
symmetric bounding-box contact with **no fault attribution**. So an SDC that
deviates from the recorded path can be driven into by a car on rails and charged
for it.

That predicts a specific, measurable thing: collision should rise steeply with
*deviation from the log*, independent of driving quality. This probe replays the
human demonstration itself -- known-good driving, 0.031 collision unperturbed --
under a controlled perturbation, and reports collision vs perturbation size.

  * ``lateral``  shift the whole path d metres perpendicular to its heading.
  * ``delay``    make the SDC k world steps late (at time t it is where the log
                 was at t-k), i.e. correct path, wrong schedule.

d=0 / k=0 is the built-in control: it must reproduce the unperturbed expert
numbers, or the harness is wrong rather than the sim.

Reading it: if a 1 m offset -- well inside normal tracking error -- takes
collision from ~0.03 to ~0.5, then most of the 0.67 collision rate every DPPO run
sits at is exogenous, no reward can remove it, and `solve` (which ANDs
``~collided``) has been floored by the simulator the whole time. If collision
stays low out to several metres, the collisions are real driving errors and the
reward is still the place to look.

CPU-only (expert replay does no model forward). Submitted, not run inline:
    sbatch rl/diffusion_ft/slurm/probe_logreplay_collision.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.train_dppo import load_batch
from rl.diffusion_ft.winnability import parse_indices


def perturb_lateral(log_bt5: jnp.ndarray, metres: float) -> jnp.ndarray:
    """Shift the path ``metres`` perpendicular to its own heading.

    Layout is ``x, y, yaw, vel_x, vel_y``; the left-normal of heading ``yaw`` is
    ``(-sin yaw, cos yaw)``. Yaw and velocity are left alone: a pure lateral
    offset keeps the same heading and speed profile, so the only thing that
    changes is *where* the car is relative to agents that cannot see it.
    """
    yaw = log_bt5[..., 2]
    dx = -jnp.sin(yaw) * metres
    dy = jnp.cos(yaw) * metres
    return log_bt5.at[..., 0].add(dx).at[..., 1].add(dy)


def perturb_delay(log_bt5: jnp.ndarray, steps: int) -> jnp.ndarray:
    """Make the SDC ``steps`` world steps late: at time t it is where the log was
    at t-steps (clamped at the start, so it simply lingers at its initial pose).

    The path is exactly the human's -- only the schedule is wrong. This is the
    perturbation a slightly-too-slow policy actually produces.
    """
    if steps <= 0:
        return log_bt5
    t = log_bt5.shape[1]
    idx = jnp.clip(jnp.arange(t) - int(steps), 0, t - 1)
    return log_bt5[:, idx, :]


def run_case(env, state, log_bt5, label):
    """Reset and replay one perturbed trajectory; return aggregate rates."""
    env.reset(state)
    ep = env.expert_replay_episode(log_override_bt5=log_bt5)
    offroad = np.asarray(ep["offroad_any_b"], dtype=bool)
    overlap = np.asarray(ep["overlap_any_b"], dtype=bool)
    reached = np.asarray(ep["reached_any_b"], dtype=bool)
    clean = reached & (~offroad) & (~overlap)
    return {
        "case": label,
        "n": int(offroad.size),
        "collided": float(overlap.mean()),
        "offroad": float(offroad.mean()),
        "reached": float(reached.mean()),
        "solve": float(clean.mean()),
        "collided_idx": [int(i) for i in np.where(overlap)[0]],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfrecord", type=str,
                   default="/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--indices", type=str, default="0-31")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--lateral", type=str, default="0,0.25,0.5,1.0,1.5,2.0,3.0",
                   help="lateral offsets in metres (0 is the control)")
    p.add_argument("--delay", type=str, default="0,1,2,3,5",
                   help="delays in world steps (0 is the control)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default="runs/dft_logreplay_probe/collision_vs_deviation.json")
    args = p.parse_args(argv)

    indices = parse_indices(args.indices)
    state = load_batch(args.tfrecord, indices)
    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])

    # Same env construction as winnability.py / DPPOTrainer._build_for. The reward
    # config is irrelevant here (we read metric flags, not reward), so leave it
    # at the default rather than implying the numbers depend on a recipe.
    env = DiffusionWaymaxEnv(
        load_diffusion_checkpoint(args.ckpt), reward_config=RewardConfig(),
        max_num_objects=n_obj, use_ema=True,
        replan_interval_steps=args.replan_interval_steps, seed=args.seed,
    )
    env.reset(state)
    base_log = env.ego_log_world_trajectory()

    rows = []
    print(f"[probe] tfrecord={args.tfrecord} scenes={len(indices)}")
    print(f"[probe] every non-ego agent is log-replay (non-reactive); "
          f"collision metric is symmetric overlap, no fault attribution\n")
    print(f"[probe] {'case':>16} {'collided':>9} {'offroad':>8} {'reached':>8} {'solve':>7}")
    print("[probe] " + "-" * 52)

    for m in [float(x) for x in args.lateral.split(",") if x.strip()]:
        r = run_case(env, state, perturb_lateral(base_log, m), f"lateral {m:g}m")
        rows.append(r)
        print(f"[probe] {r['case']:>16} {r['collided']:9.3f} {r['offroad']:8.3f} "
              f"{r['reached']:8.3f} {r['solve']:7.3f}")

    for k in [int(x) for x in args.delay.split(",") if x.strip()]:
        r = run_case(env, state, perturb_delay(base_log, k), f"delay {k}step")
        rows.append(r)
        print(f"[probe] {r['case']:>16} {r['collided']:9.3f} {r['offroad']:8.3f} "
              f"{r['reached']:8.3f} {r['solve']:7.3f}")

    # ---- verdict ---------------------------------------------------------- #
    ctrl = next(r for r in rows if r["case"] == "lateral 0m")
    lat1 = next((r for r in rows if r["case"] == "lateral 1m"), None)
    print()
    if ctrl["collided"] > 0.15:
        print(f"[probe] WARNING: the d=0 control already collides "
              f"{ctrl['collided']:.3f}; expected ~0.03. The harness, not the sim, "
              f"is suspect -- do not read the rest.")
    elif ctrl["n"] < 16:
        print(f"[probe] n={ctrl['n']} is too small for a verdict: the unperturbed "
              f"expert collides on ~1 scene in 32, so a subset this size usually "
              f"has no baseline collisions to lift from. Re-run on the full set.")
    elif lat1 is not None:
        lift = lat1["collided"] - ctrl["collided"]
        print(f"[probe] control (on the log) collides {ctrl['collided']:.3f}; "
              f"1 m off the log collides {lat1['collided']:.3f} (+{lift:.3f})")
        if lift > 0.25:
            print("[probe] VERDICT: collision is dominated by deviation from the log, "
                  "not by driving quality. No reward term can remove it, and `solve` "
                  "is floored by the simulator. Fix the sim (reactive agents) or the "
                  "metric (at-fault-only collision) before any further reward work.")
        else:
            print("[probe] VERDICT: a 1 m deviation does NOT explain the collision "
                  "rate. The collisions DPPO incurs are real driving errors; the "
                  "reward/policy remains the place to look.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"tfrecord": args.tfrecord, "indices": indices, "rows": rows}, f, indent=1)
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
