"""Which raw-WOMD scenes does the base diffusion policy actually fail?

Motivation (2026-09-02). The WOMD re-run of the `entry_clean` DPPO experiment
smoke-tested green and then reported `eval_solve_deterministic = 0.979` at
iteration 0: the base checkpoint already solves ~95-98% of *uniformly drawn* WOMD
scenes, so there is nothing for the reward to fix and no way to read a result.
The ScenarioMax failure set had headroom because it was SELECTED for it -- but
that corpus carries the degenerate timestamp column that invalidated every DPPO
run taken on it (see data/preprocess.py::ego_source_time_seconds).

So: keep WOMD (real timestamps, the corpus the checkpoint was pretrained on) and
recover the headroom by selecting on THIS policy's own measured failures.

This is the measurement stage. It rolls, per scene:

  * the **base policy, deterministically** -- the deployment trajectory, and the
    thing whose solve rate we want headroom in;
  * the **base policy, stochastically** at `--sigma_floor` -- what DPPO actually
    collects with, recorded per scene so the collection/deployment gap can be
    read per scene rather than as a batch mean;
  * the **expert log replay** -- the control, and a gate rather than a curiosity.

Why the expert control is not optional: selecting on policy failure selects just
as hard for *dirty ground truth*. 22.6% of the diffusion policy's failures on the
curated set were scenes where the logged human is itself off-road or in contact
-- unwinnable, and a training pool full of them teaches nothing. WOMD measured
0/64 expert-dirty on a UNIFORM draw, but that rate says nothing about the
conditional rate among hard scenes, which is the only rate that matters here.
`select_hard_womd.py` gates on this column; this script's job is to produce it.

Per-scene rows, not batch means: the whole point is to keep the identity of each
scene so a pool can be built from it.

Output: one JSONL row per scene in `--out_dir/rows_task<K>.jsonl`. Resumable --
rows already written are skipped, so a task that hits the time limit is fixed by
resubmitting. Array-shardable via `--task_id/--n_tasks` (contiguous slices of one
seeded, sorted id list, so the split is reproducible and the tasks never collide
on a file).

Usage (submitted, never inline -- see slurm/scan_womd_headroom.sbatch):
    python -m rl.diffusion_ft.tools.scan_womd_headroom \
        --shard_glob '/.../training_tfexample.tfrecord-*-of-01000' \
        --out_dir OUT --n_scenes 4096 --batch 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.scene_sampler import (
    DEFAULT_INDEX_DIR, ShardIndex, _state_from_bytes, build_ds_cfg,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard_glob", required=True)
    p.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--n_scenes", type=int, default=4096,
                   help="Total scenes to scan across ALL array tasks.")
    p.add_argument("--batch", type=int, default=32,
                   help="Scenes per rollout batch. The env/actor are built once "
                        "per object count, so keep this constant.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the scene draw. Same seed = same scan set, which "
                        "is what makes the scan resumable and extendable.")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--n_tasks", type=int, default=1)
    # Rollout shape: must match how the policy is run in training, or the
    # measured failures are of a different policy than the one being fine-tuned.
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--sigma_floor", type=float, default=0.03,
                   help="Exploration floor for the STOCHASTIC arm -- the policy "
                        "DPPO collects with. The deterministic arm ignores it.")
    p.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=True,
                   help="Also roll the stochastic (collection) policy per scene.")
    p.add_argument("--num_paths", type=int, default=45,
                   help="Raw WOMD tf_example route dims (ScenarioMax is 10x300).")
    p.add_argument("--num_points_per_path", type=int, default=800)
    # Reward config. `solve` (reached & !collided & !offroad) is reward-
    # independent, but TERMINATION is not: an absorbing offroad would stop the
    # episode and hide everything after it. Non-absorbing matches `entry_clean`,
    # the recipe the pool is being built for.
    p.add_argument("--route_reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--terminate_on_offroad", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--terminate_on_collision", action=argparse.BooleanOptionalAction,
                   default=False)
    return p.parse_args()


def scan_ids(total: int, n_scenes: int, seed: int) -> np.ndarray:
    """The scan set: `n_scenes` distinct global ids, seeded and sorted.

    Sorted so an array task's contiguous slice is a stable, reproducible subset:
    re-running with a larger `--n_scenes` redraws, so extend a scan by scanning a
    second seed rather than by growing this number.
    """
    rng = np.random.default_rng(int(seed))
    n = min(int(n_scenes), int(total))
    return np.sort(rng.choice(int(total), size=n, replace=False)).astype(np.int64)


def done_ids(out_path: Path) -> set[int]:
    """Global ids already written, so a resubmit skips them."""
    if not out_path.exists():
        return set()
    got: set[int] = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                got.add(int(json.loads(line)["global_id"]))
            except Exception:
                # A torn last line from a killed task: ignore it and let the
                # scene be re-rolled rather than dropping the whole file.
                continue
    return got


def rollout_per_scene(env, actor, state, *, deterministic: bool, seed: int,
                      replan: int) -> dict[str, np.ndarray]:
    """One batched episode, keeping PER-SCENE outcome flags (not batch means).

    Mirrors `train_dppo.Trainer.evaluate`, which collapses to means -- here the
    scene identity is the entire product.
    """
    features = env.reset(state)
    horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
    n_steps = (horizon - env.timestep) // replan
    rng = jax.random.PRNGKey(int(seed))
    B = env.batch_size
    prev_done = np.zeros(B, dtype=bool)
    ret = np.zeros(B, dtype=np.float64)
    reached = np.zeros(B, dtype=bool)
    collided = np.zeros(B, dtype=bool)
    offroad = np.zeros(B, dtype=bool)
    off_steps = np.zeros(B)
    live_steps = np.zeros(B)

    for _ in range(int(n_steps)):
        cond1, cond2 = env.compute_condition(features)
        mask = features["inst_valid"]
        rng, krng = jax.random.split(rng)
        traj_ncT = actor.act(cond1, cond2, mask, rng=krng, deterministic=deterministic)
        world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
        out = env.step(trajectory_world_bt5=world)
        live = ~prev_done
        comps = out.components
        zb = np.zeros(B, dtype=bool)
        ret += np.asarray(out.reward_b) * live
        reached |= (np.asarray(comps.get("reached_flag", zb)).astype(bool) & live)
        collided |= (np.asarray(comps.get("collision_flag", zb)).astype(bool) & live)
        offroad |= (np.asarray(comps.get("offroad_flag", zb)).astype(bool) & live)
        ps = out.per_step_components
        if ps and ps.get("offroad_flag") is not None:
            off_steps += np.asarray(ps["offroad_flag"]).astype(bool).sum(axis=-1) * live
            live_steps += int(np.asarray(ps["offroad_flag"]).shape[-1]) * live
        prev_done = prev_done | np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
        features = out.features
        if prev_done.all():
            break

    return {
        "return": ret,
        "reached": reached,
        "collided": collided,
        "offroad": offroad,
        "solve": reached & ~collided & ~offroad,
        "off_steps": off_steps,
        "live_steps": live_steps,
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rows_task{int(args.task_id):03d}.jsonl"

    index = ShardIndex(args.shard_glob, index_dir=(args.index_dir or None))
    ids = scan_ids(index.total, args.n_scenes, args.seed)
    mine = np.array_split(ids, max(int(args.n_tasks), 1))[int(args.task_id)]
    already = done_ids(out_path)
    todo = [int(i) for i in mine.tolist() if int(i) not in already]

    print(f"host={os.environ.get('SLURMD_NODENAME', 'unknown')} "
          f"task={args.task_id}/{args.n_tasks}")
    print(f"[scan] corpus: {len(index.paths)} shards, {index.total} scenes")
    print(f"[scan] scan set {ids.size}, mine {mine.size}, already done "
          f"{len(already)}, to do {len(todo)}")
    print(f"[scan] sigma_floor={args.sigma_floor} routes="
          f"{args.num_paths}x{args.num_points_per_path} batch={args.batch}")
    if not todo:
        print("[scan] nothing to do -- this task is already complete.")
        return

    ds_cfg = build_ds_cfg(
        batch_dims=(int(args.batch),), include_sdc_paths=bool(args.route_reward),
        num_paths=int(args.num_paths), num_points_per_path=int(args.num_points_per_path),
        max_num_rg_points=30000,
    )
    reward_cfg = RewardConfig(
        route_reward=bool(args.route_reward),
        terminate_on_offroad=bool(args.terminate_on_offroad),
        terminate_on_collision=bool(args.terminate_on_collision),
        off_route_threshold_m=(0.0 if args.route_reward else None),
    )

    env = actor = None
    n_written = 0
    t0 = time.time()
    # Drop the tail: the env/actor are sized to a fixed batch, and rebuilding for
    # a short final batch costs a full recompile for a handful of scenes. The
    # dropped ids stay in the scan set, so a resubmit with a different --batch
    # picks them up.
    for start in range(0, len(todo) - int(args.batch) + 1, int(args.batch)):
        chunk = todo[start:start + int(args.batch)]
        picks = [index.to_pick(i) for i in chunk]
        state = _state_from_bytes([index.read(p) for p in picks], ds_cfg)

        if env is None:
            n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])
            # Loaded lazily, after the first batch: the env must adopt the
            # batch's actual object count (it asserts on it), and paying the
            # checkpoint restore before knowing that count buys nothing.
            ckpt = load_diffusion_checkpoint(args.ckpt)
            env = DiffusionWaymaxEnv(
                ckpt, reward_config=reward_cfg, max_num_objects=n_obj,
                use_ema=True, replan_interval_steps=int(args.replan_interval_steps),
                seed=int(args.seed),
            )
            actor = DiffusionRLActor(
                env.policy.diffusion, k_trainable=int(args.k_trainable),
                prefix_len=int(args.prefix_len),
                sigma_sample_floor=float(args.sigma_floor),
                sigma_logprob_floor=float(args.sigma_floor),
                seed=int(args.seed),
            )
            print(f"[scan] built env/actor for {n_obj} objects "
                  f"({time.time() - t0:.1f}s)")

        det = rollout_per_scene(env, actor, state, deterministic=True,
                                seed=args.seed + 778, replan=args.replan_interval_steps)
        sto = (rollout_per_scene(env, actor, state, deterministic=False,
                                 seed=args.seed + 1777, replan=args.replan_interval_steps)
               if args.stochastic else None)
        # The control. Same env, same reward config, the logged human's own poses.
        env.reset(state)
        exp = env.expert_replay_episode()

        with out_path.open("a") as f:
            for b, gid in enumerate(chunk):
                shard_id, rec = picks[b]
                row = {
                    "global_id": int(gid),
                    "shard": Path(index.paths[shard_id]).name,
                    "shard_id": int(shard_id),
                    "record": int(rec),
                    "det_solve": bool(det["solve"][b]),
                    "det_reached": bool(det["reached"][b]),
                    "det_collided": bool(det["collided"][b]),
                    "det_offroad": bool(det["offroad"][b]),
                    "det_return": float(det["return"][b]),
                    "det_off_steps": float(det["off_steps"][b]),
                    "det_live_steps": float(det["live_steps"][b]),
                    # The expert control: `expert_clean` is the gate
                    # select_hard_womd.py applies.
                    "exp_reached": bool(np.asarray(exp["reached_any_b"])[b]),
                    "exp_offroad": bool(np.asarray(exp["offroad_any_b"])[b]),
                    "exp_overlap": bool(np.asarray(exp["overlap_any_b"])[b]),
                    "exp_return": float(np.asarray(exp["episode_return_b"])[b]),
                    "sigma_floor": float(args.sigma_floor),
                }
                row["exp_clean"] = bool(row["exp_reached"] and not row["exp_offroad"]
                                        and not row["exp_overlap"])
                if sto is not None:
                    row.update({
                        "sto_solve": bool(sto["solve"][b]),
                        "sto_reached": bool(sto["reached"][b]),
                        "sto_collided": bool(sto["collided"][b]),
                        "sto_offroad": bool(sto["offroad"][b]),
                        "sto_return": float(sto["return"][b]),
                        "sto_off_steps": float(sto["off_steps"][b]),
                    })
                f.write(json.dumps(row) + "\n")
                n_written += 1

        done_n = start + len(chunk)
        print(f"[scan] {done_n}/{len(todo)} scenes  "
              f"det_solve={float(det['solve'].mean()):.3f}  "
              f"exp_clean={float((np.asarray(exp['reached_any_b']) & ~np.asarray(exp['offroad_any_b']) & ~np.asarray(exp['overlap_any_b'])).mean()):.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    print(f"[scan] wrote {n_written} rows -> {out_path}")


if __name__ == "__main__":
    main()
