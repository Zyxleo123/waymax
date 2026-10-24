"""Roll out DPPO checkpoints and record *how they drive*, not just what they score.

Motivation: across the 500-iteration DPPO run the shaped eval return climbed
(0.64 -> 4.81) while the solve rate stayed pinned at 0 (one lucky scene out of
128 at the last eval). The reward makes several very different behaviours look
like that, and the trainer's logs cannot tell them apart:

    progress = prev_dist - goal_dist          (telescopes to init_dist - final_dist)
    offroad  = -5, charged ONCE, does NOT terminate
    collision = -10, charged once, terminates
    goal     = +10 within 3 m, terminates

Read as metres, a return of 4.81 is "closed ~4.8 m more than it started" if the
episode was clean, ~9.8 m if it paid one offroad, ~19.8 m if it also crashed. So
the two live hypotheses are:

  H1 (reward hacking) -- offroad does not terminate and costs 5 once, while
     progress is unbounded, so cutting the corner across a median strictly pays.
     Signature: reached/offroad rates BOTH climb, distance closed climbs, offroad
     step fraction climbs, detour ratio falls toward 1 (a straight line to goal).
  H2 (just barely moving) -- the policy is nowhere near the 3 m goal threshold and
     the entire 250-iteration gain is a few metres of extra creep on scenes it was
     always going to fail. Signature: reached stays ~0, distance closed small and
     roughly unchanged in relative terms, offroad rate flat and already ~1.

These prescribe opposite fixes -- H1 needs the reward repaired (terminate on
offroad, or charge it per step), H2 needs more optimization or an easier task --
so guessing is expensive. This script measures which one it is.

H1 was confirmed and the V-Max parity reward was built to fix it. That run
(job 34866) then collected exactly ONE outer replan step in 80% of iterations:
all 32 worlds absorbed inside the first 10 world steps, already at iteration 0
with the frozen base policy, so the episode degenerated to a one-step bandit and
the solve rate never moved. Under independent per-world rates measured here (base
absorbs at replan 1 in 29% of rollouts) all-32-in-2s is ~1e-17, so the suspicion
is the terminating condition itself, not the driving. Hence H3:

  H3 (termination attribution) -- WHICH flag absorbs the batch, and at which
     world step? The script therefore takes the full reward config on the command
     line (defaults reproduce the legacy bare ``RewardConfig()`` exactly, so old
     artifacts stay comparable) and records, per world: the replan and world step
     at which it absorbed, the flag(s) true at that step, and the per-world-step
     offroad/collision/goal rate over the first prefix. Run it twice -- legacy vs
     parity -- and diff the attribution table. The expert log replay is the
     control: if the *human demonstration* also absorbs in replan 1 under the
     parity config, the metric/threshold is wrong, not the policy.

For each arm (a saved DPPO actor, or the un-fine-tuned base policy) it rolls the
same fixed-seed episodes the trainer evaluates on and writes, per scene:

  * the decomposed return (progress / offroad / collision / goal), so a rising
    return can be attributed to a term rather than assumed to be competence;
  * the outcome flags (reached / collided / offroad) *separately*, which the
    trainer collapses into a single solve bit -- the breakdown is the whole
    diagnosis (reached-but-offroad is hacking, never-reached is not);
  * goal distance closed, min goal distance ever reached, and where the episode
    ended, distinguishing "stops short" from "overshoots" from "drives away";
  * the executed path with a per-world-step offroad/collision mask, so the
    companion viz job can draw the excursions on the roadgraph.

GPU job. Writes artifacts only -- every figure is drawn from these files by
`rl.diffusion_ft.tests.viz_drive` on the CPU partition, so plots re-render
without re-running the policy.

Usage:
    python -m rl.diffusion_ft.tests.diagnose_drive \
        --tfrecord .../failures.tfrecord --indices 0,1,...,31 \
        --arms "base=,dppo500=/path/to/ckpt_0250" --seeds 4 --out_dir OUT
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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
from rl.diffusion_ft.train_dppo import load_batch

# Road-edge element ids (waymax MapElementIds): the boundary the offroad metric
# is defined against. Kept as literals so the viz artifact is self-describing.
ROAD_EDGE_IDS = (14, 15, 16)
LANE_IDS = (1, 2, 3)


# --------------------------------------------------------------------------- #
def _ego_rows(state):
    """Ego ``x, y, yaw, vel_x, vel_y`` from the *simulated* trajectory, [B, T, 5]."""
    is_sdc_bn = jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32)
    ego_b = jnp.argmax(is_sdc_bn, axis=1)
    sim = state.sim_trajectory

    def row(f):
        return jnp.take_along_axis(jnp.asarray(f), ego_b[:, None, None], axis=1)[:, 0]

    return np.asarray(
        jnp.stack([row(sim.x), row(sim.y), row(sim.yaw), row(sim.vel_x), row(sim.vel_y)],
                  axis=-1)
    )


def _log_ego_rows(state):
    """Same layout from the untouched *log* trajectory (the human demonstration)."""
    is_sdc_bn = jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32)
    ego_b = jnp.argmax(is_sdc_bn, axis=1)
    log = state.log_trajectory

    def row(f):
        return jnp.take_along_axis(jnp.asarray(f), ego_b[:, None, None], axis=1)[:, 0]

    return np.asarray(
        jnp.stack([row(log.x), row(log.y), row(log.yaw), row(log.vel_x), row(log.vel_y)],
                  axis=-1)
    )


# --------------------------------------------------------------------------- #
def _cause_label(off: bool, col: bool, rch: bool, trunc: bool, absorbed: bool) -> str:
    """Human-readable reason a world stopped being stepped.

    Multiple flags can be true on the same world step (offroad *and* collision is
    common when cutting a corner into parked traffic), so the label is a joined
    set rather than a priority pick -- collapsing it would be exactly the kind of
    attribution error this diagnosis exists to avoid.
    """
    if not absorbed:
        return "alive"
    if trunc:
        return "truncated"
    parts = [n for n, f in (("offroad", off), ("collision", col), ("goal", rch)) if f]
    return "+".join(parts) if parts else "unknown"


def rollout(env, actor, state, *, deterministic: bool, seed: int, replan: int,
            max_replans: int | None):
    """One batched fixed-seed episode, recorded at world-step resolution.

    Mirrors ``DPPOTrainer.evaluate`` exactly on the quantities it shares (same
    live-masking of the return, same OR-accumulation of the flags over live
    steps), and additionally keeps the per-world-step components and the path.
    """
    features = env.reset(state)
    # Waymax reset leaves state.timestep at init_steps - 1, which is where the
    # reward's init_goal_dist is measured -- anchor every "initial" quantity there
    # rather than at env._start_timestep (init_steps), one step later.
    start_step = env.timestep
    horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
    n_steps = max_replans or ((horizon - env.timestep) // replan)
    rng = jax.random.PRNGKey(int(seed))
    B = env.batch_size

    prev_done = np.zeros(B, dtype=bool)
    n_steps_run = 0
    ret = np.zeros(B, dtype=np.float64)
    reached = np.zeros(B, dtype=bool)
    collided = np.zeros(B, dtype=bool)
    offroad = np.zeros(B, dtype=bool)
    comp_sums = {k: np.zeros(B) for k in
                 ("progress", "offroad", "collision", "goal_bonus", "action_penalty")}
    # World-step-resolution flags over the whole episode, laid out on the
    # simulator's own time axis so they index the path directly.
    off_step = np.zeros((B, horizon), dtype=bool)
    col_step = np.zeros((B, horizon), dtype=bool)
    rch_step = np.zeros((B, horizon), dtype=bool)
    live_step = np.zeros((B, horizon), dtype=bool)
    done_at = np.full(B, horizon - 1, dtype=np.int32)   # world step the episode ended
    n_replans_live = np.zeros(B, dtype=np.int32)

    # --- H3 termination attribution ------------------------------------- #
    # Which condition absorbed each world, and where. `cfg` decides which flags
    # actually terminate, so the same rollout code attributes correctly under the
    # legacy config (collision + goal) and the parity one (+ offroad).
    cfg = env._reward_cfg
    absorb_replan = np.zeros(B, dtype=np.int32)     # 1-based; 0 == never absorbed
    absorb_step = np.full(B, -1, dtype=np.int32)    # world step it absorbed at
    absorb_off = np.zeros(B, dtype=bool)
    absorb_col = np.zeros(B, dtype=bool)
    absorb_rch = np.zeros(B, dtype=bool)
    absorb_trunc = np.zeros(B, dtype=bool)
    rows_b = np.arange(B)

    for replan_i in range(int(n_steps)):
        t_before = env.timestep
        cond1, cond2 = env.compute_condition(features)
        mask = features["inst_valid"]
        rng, krng = jax.random.split(rng)
        traj_ncT = actor.act(cond1, cond2, mask, rng=krng, deterministic=deterministic)
        world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
        out = env.step(trajectory_world_bt5=world)

        live = ~prev_done
        ret += np.asarray(out.reward_b) * live
        comps = out.components
        zb = np.zeros(B, dtype=bool)
        for k in comp_sums:
            if k in comps:
                comp_sums[k] += np.asarray(comps[k]) * live
        reached |= (np.asarray(comps.get("reached_flag", zb)).astype(bool) & live)
        collided |= (np.asarray(comps.get("collision_flag", zb)).astype(bool) & live)
        offroad |= (np.asarray(comps.get("offroad_flag", zb)).astype(bool) & live)
        n_replans_live += live.astype(np.int32)

        ps = out.per_step_components
        if ps:
            off_k = np.asarray(ps["offroad_flag"]).astype(bool)      # [B, n_exec]
            col_k = np.asarray(ps["collision_flag"]).astype(bool)
            rch_k = np.asarray(ps["reached_flag"]).astype(bool)
            n_exec = int(off_k.shape[-1])
            sl = slice(t_before + 1, t_before + 1 + n_exec)
            off_step[:, sl] = off_k & live[:, None]
            col_step[:, sl] = col_k & live[:, None]
            rch_step[:, sl] = rch_k & live[:, None]
            live_step[:, sl] = live[:, None]

            # First *terminating* world step inside this prefix, per world. The
            # flags are raw (reward_step does not mask them by `active`), so a
            # world that died in an earlier prefix is excluded via `live`.
            term_k = rch_k.copy()
            if cfg.terminate_on_collision:
                term_k = term_k | col_k
            if cfg.terminate_on_offroad:
                term_k = term_k | off_k
            first_k = np.argmax(term_k, axis=1)
            newly_term = live & term_k.any(axis=1) & (absorb_replan == 0)
            absorb_replan = np.where(newly_term, replan_i + 1, absorb_replan)
            absorb_step = np.where(newly_term, t_before + 1 + first_k, absorb_step)
            absorb_off |= newly_term & off_k[rows_b, first_k] & cfg.terminate_on_offroad
            absorb_col |= newly_term & col_k[rows_b, first_k] & cfg.terminate_on_collision
            absorb_rch |= newly_term & rch_k[rows_b, first_k]

        newly_done = (~prev_done) & (np.asarray(out.terminated_b) | np.asarray(out.truncated_b))
        # A world that ended without any terminating flag ran out of horizon.
        newly_trunc = newly_done & (absorb_replan == 0)
        absorb_trunc |= newly_trunc
        absorb_replan = np.where(newly_trunc, replan_i + 1, absorb_replan)
        absorb_step = np.where(newly_trunc, env.timestep, absorb_step)
        done_at = np.where(newly_done, env.timestep, done_at).astype(np.int32)
        prev_done = prev_done | np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
        features = out.features
        n_steps_run = replan_i + 1
        if prev_done.all():
            break

    path = _ego_rows(env.sim_state)                     # [B, T, 5]
    goal_xy = np.asarray(env._goal_xy_b2)               # [B, 2]
    dist_t = np.linalg.norm(path[:, :, :2] - goal_xy[:, None, :], axis=-1)  # [B, T]
    solved = reached & ~collided & ~offroad
    return {
        "return": ret, "reached": reached, "collided": collided, "offroad": offroad,
        "solved": solved, "comp_sums": comp_sums, "path": path, "goal_xy": goal_xy,
        "dist_t": dist_t, "off_step": off_step, "col_step": col_step,
        "rch_step": rch_step,
        "live_step": live_step, "done_at": done_at, "n_replans": n_replans_live,
        "log_path": _log_ego_rows(env.sim_state),
        "start_step": int(start_step),
        "absorb_replan": absorb_replan, "absorb_step": absorb_step,
        "absorb_off": absorb_off, "absorb_col": absorb_col,
        "absorb_rch": absorb_rch, "absorb_trunc": absorb_trunc,
        "n_steps_batch": int(n_steps_run),
    }


# --------------------------------------------------------------------------- #
def summarize(scene_idx, arm, seed, r, i):
    """One JSONL row per (arm, seed, scene): every number the viz needs."""
    d0 = float(r["dist_t"][i, r["start_step"]])
    end = int(r["done_at"][i])
    dend = float(r["dist_t"][i, end])
    live = r["live_step"][i]
    # Min distance over the *live* part only: a terminated world keeps being
    # stepped by the batched env, so the post-terminal path is not the policy's.
    live_idx = np.where(live)[0]
    dmin = float(r["dist_t"][i, live_idx].min()) if live_idx.size else d0
    xy = r["path"][i, r["start_step"]:end + 1, :2]
    pathlen = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum()) if xy.shape[0] > 1 else 0.0
    speed = np.linalg.norm(r["path"][i, r["start_step"]:end + 1, 3:5], axis=-1)
    n_live_steps = int(live.sum())
    return {
        "arm": arm, "seed": int(seed), "scene": int(scene_idx),
        "return": float(r["return"][i]),
        "reached": bool(r["reached"][i]), "collided": bool(r["collided"][i]),
        "offroad": bool(r["offroad"][i]), "solved": bool(r["solved"][i]),
        **{f"r_{k}": float(v[i]) for k, v in r["comp_sums"].items()},
        "init_goal_dist": d0, "final_goal_dist": dend, "min_goal_dist": dmin,
        "dist_closed": d0 - dend, "frac_closed": (d0 - dend) / max(d0, 1e-6),
        "path_len": pathlen,
        # A beeline scores path_len ~= dist_closed; a policy that follows the road
        # network must travel further than the straight line. <1 means it also
        # stopped short or backtracked.
        "detour_ratio": pathlen / max(d0 - dend, 1e-6),
        "mean_speed": float(speed.mean()) if speed.size else 0.0,
        "max_speed": float(speed.max()) if speed.size else 0.0,
        "n_live_steps": n_live_steps,
        "n_offroad_steps": int(r["off_step"][i].sum()),
        "frac_offroad_steps": float(r["off_step"][i].sum()) / max(n_live_steps, 1),
        "first_offroad_step": (int(np.argmax(r["off_step"][i])) if r["off_step"][i].any() else -1),
        "first_collision_step": (int(np.argmax(r["col_step"][i])) if r["col_step"][i].any() else -1),
        "done_at": end, "n_replans": int(r["n_replans"][i]),
        # --- H3 termination attribution --- #
        "absorb_replan": int(r["absorb_replan"][i]),
        "absorb_step": int(r["absorb_step"][i]),
        # World steps since the episode anchor, not an absolute sim timestep: 1
        # means "the very first world step of the episode already tripped it".
        # For a replan-1 absorption this is also the offset into the first prefix.
        "absorb_step_since_start": (int(r["absorb_step"][i]) - r["start_step"]
                                    if r["absorb_step"][i] >= 0 else -1),
        "absorb_cause": _cause_label(
            bool(r["absorb_off"][i]), bool(r["absorb_col"][i]), bool(r["absorb_rch"][i]),
            bool(r["absorb_trunc"][i]), bool(r["absorb_replan"][i] > 0)),
        "absorbed_in_replan1": bool(r["absorb_replan"][i] == 1),
        "n_steps_batch": int(r["n_steps_batch"]),
    }


# --------------------------------------------------------------------------- #
def first_prefix_rates(r, replan: int) -> dict[str, np.ndarray]:
    """Per-world-step flag rate over the FIRST prefix only, [replan] each.

    Every world is live for the first prefix by construction, so a plain mean over
    the batch is the rate. This is the curve that answers H3 directly: an offroad
    rate that is already ~1 at world step 1 is a metric/threshold problem, one
    that ramps over 10 steps is the policy actually driving off the road.
    """
    st = r["start_step"]
    sl = slice(st + 1, st + 1 + replan)
    return {
        "offroad": r["off_step"][:, sl].mean(axis=0),
        "collision": r["col_step"][:, sl].mean(axis=0),
        "goal": r["rch_step"][:, sl].mean(axis=0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH,
                   help="base diffusion checkpoint (frozen tokenizer + denoiser)")
    p.add_argument("--tfrecord", type=str, required=True)
    p.add_argument("--indices", type=str, required=True)
    p.add_argument("--arms", type=str, required=True,
                   help="comma-separated name=path; an empty path means the "
                        "un-fine-tuned base policy (no actor state loaded)")
    p.add_argument("--seeds", type=int, default=4,
                   help="fixed eval seeds per arm; matches train_dppo's eval_seeds")
    p.add_argument("--stochastic", action="store_true",
                   help="also roll the stochastic (exploration-floor) policy")
    p.add_argument("--expert", action="store_true", default=True,
                   help="also replay the human log as a reference arm")
    p.add_argument("--no-expert", dest="expert", action="store_false")
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--sigma_floor", type=float, default=0.03)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--max_replans", type=int, default=0)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--n_path_scenes", type=int, default=32,
                   help="how many scenes to dump paths+roadgraph for (viz)")
    # --- reward config ---------------------------------------------------- #
    # Defaults reproduce the legacy bare `RewardConfig()` this script used to
    # hardcode, so a launch that sets none of these is byte-identical to the
    # 20260819 artifacts. The parity recipe is turned on explicitly, and mirrors
    # train_dppo._build_for term for term -- a diagnosis run against a different
    # env than the trainer's would measure the wrong thing.
    p.add_argument("--reward_progress", type=float, default=1.0)
    p.add_argument("--reward_collision", type=float, default=-10.0)
    p.add_argument("--reward_offroad", type=float, default=-5.0)
    p.add_argument("--reward_goal_bonus", type=float, default=10.0)
    p.add_argument("--reward_off_route", type=float, default=-0.2)
    p.add_argument("--route_reward", action="store_true", default=False)
    p.add_argument("--no-route_reward", dest="route_reward", action="store_false")
    p.add_argument("--terminate_offroad", action="store_true", default=False)
    p.add_argument("--no-terminate_offroad", dest="terminate_offroad", action="store_false")
    # The alternative to terminating: charge `--reward_offroad` on EVERY off-road
    # world step, so a noisy rollout still yields a full-length episode of signal
    # while a long excursion still costs more than a short one. Intended with
    # --no-terminate_offroad; the cost scales with dwell, so use a coefficient
    # ~5x smaller than the terminating variant's.
    p.add_argument("--offroad_per_step", action="store_true", default=False)
    p.add_argument("--no-offroad_per_step", dest="offroad_per_step",
                   action="store_false")
    # Same treatment for collision. Needed because a cheap terminal collision is
    # an ESCAPE from the remaining per-step offroad bill: with offroad at -0.2 a
    # full 80-step excursion runs to -16 while crashing costs -1.0 once and
    # freezes the world, so ending early strictly pays. Pair with
    # --no-terminate_collision.
    p.add_argument("--collision_per_step", action="store_true", default=False)
    p.add_argument("--no-collision_per_step", dest="collision_per_step",
                   action="store_false")
    # Hardcoded True in the trainer; exposed here only so the attribution table
    # can be read with collision termination taken out of the picture.
    p.add_argument("--terminate_collision", action="store_true", default=True)
    p.add_argument("--no-terminate_collision", dest="terminate_collision",
                   action="store_false")
    p.add_argument("--goal_shaping_coef", type=float, default=0.0)
    p.add_argument("--goal_shaping_normalize", action="store_true", default=True)
    p.add_argument("--no-goal_shaping_normalize", dest="goal_shaping_normalize",
                   action="store_false")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="only used as the shaping discount (must match the run's)")
    p.add_argument("--num_paths", type=int, default=10)
    p.add_argument("--num_points_per_path", type=int, default=300)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    arms = []
    for spec in args.arms.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, _, path = spec.partition("=")
        arms.append((name.strip(), path.strip()))
    if not arms:
        raise SystemExit("--arms parsed to nothing")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] out={out_dir} scenes={len(indices)} arms={[a for a, _ in arms]}")

    ckpt = load_diffusion_checkpoint(args.ckpt)
    state = load_batch(args.tfrecord, indices,
                       include_sdc_paths=args.route_reward,
                       num_paths=args.num_paths,
                       num_points_per_path=args.num_points_per_path)
    if args.route_reward:
        sp = state.sdc_paths
        if sp is None:
            raise SystemExit(
                "--route_reward needs sdc_paths but the loader returned None "
                "(check num_paths/num_points_per_path match the tfrecord).")
        v = np.asarray(sp.valid).reshape(len(indices), -1).any(axis=1)
        print(f"[diag] route: {int(v.sum())}/{len(indices)} scenes have a valid "
              f"sdc_path (num_paths={args.num_paths}x{args.num_points_per_path})")
        if not v.all():
            print(f"[diag] WARNING: no valid route for scenes "
                  f"{[indices[i] for i in np.where(~v)[0]]}")
    reward_cfg = RewardConfig(
        progress=args.reward_progress,
        collision=args.reward_collision,
        offroad=args.reward_offroad,
        goal_bonus=args.reward_goal_bonus,
        terminate_on_collision=args.terminate_collision,
        terminate_on_offroad=args.terminate_offroad,
        offroad_per_step=args.offroad_per_step,
        collision_per_step=args.collision_per_step,
        route_reward=args.route_reward,
        progression_indicator=args.route_reward,
        off_route_threshold_m=(0.0 if args.route_reward else None),
        off_route_penalty=args.reward_off_route,
    )
    print(f"[diag] reward: route={int(args.route_reward)} "
          f"term_offroad={int(args.terminate_offroad)} "
          f"offroad_per_step={int(args.offroad_per_step)} "
          f"term_collision={int(args.terminate_collision)} "
          f"collision_per_step={int(args.collision_per_step)} "
          f"progress={args.reward_progress} offroad={args.reward_offroad} "
          f"collision={args.reward_collision} goal={args.reward_goal_bonus} "
          f"off_route={args.reward_off_route} shaping={args.goal_shaping_coef}")
    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])
    env = DiffusionWaymaxEnv(
        ckpt, reward_config=reward_cfg, max_num_objects=n_obj,
        use_ema=args.use_ema, replan_interval_steps=args.replan_interval_steps, seed=0,
        goal_shaping_coef=args.goal_shaping_coef,
        goal_shaping_discount=args.gamma,
        goal_shaping_normalize=args.goal_shaping_normalize,
    )
    actor = DiffusionRLActor(
        env.policy.diffusion, k_trainable=args.k_trainable, prefix_len=args.prefix_len,
        sigma_sample_floor=args.sigma_floor, sigma_logprob_floor=args.sigma_floor, seed=0,
    )
    max_replans = args.max_replans or None

    rows: list[dict] = []
    paths: dict[str, np.ndarray] = {}
    curves: dict[str, list[dict]] = {}

    for arm, path in arms:
        t0 = time.time()
        if path:
            pkl = Path(path) / "actor_value.pkl" if Path(path).is_dir() else Path(path)
            with open(pkl, "rb") as f:
                sd = pickle.load(f)
            actor.load_state_dict(jax.tree_util.tree_map(jnp.asarray, sd["actor"]))
            print(f"[diag] arm={arm}: loaded {pkl}")
        else:
            # Re-seat the pretrained weights so arm order cannot leak a previous
            # arm's fine-tuned denoiser into the "base" measurement.
            actor.reset_trainable_to_frozen()
            print(f"[diag] arm={arm}: base policy (no actor state loaded)")

        modes = [("det", True)] + ([("sto", False)] if args.stochastic else [])
        for mode, det in modes:
            for s in range(args.seeds):
                # Same fixed seeds train_dppo.evaluate uses, so a row here is
                # directly comparable to the eval curve in train_log.jsonl.
                seed = (778 + 2 * s) if det else (1777 + 2 * s)
                r = rollout(env, actor, state, deterministic=det, seed=seed,
                            replan=args.replan_interval_steps, max_replans=max_replans)
                for i, sc in enumerate(indices):
                    row = summarize(sc, f"{arm}/{mode}", seed, r, i)
                    rows.append(row)
                curves.setdefault(f"{arm}/{mode}", []).append(
                    first_prefix_rates(r, args.replan_interval_steps))
                if s == 0 and det:
                    key = f"{arm}"
                    paths[f"path_{key}"] = r["path"].astype(np.float32)
                    paths[f"off_{key}"] = r["off_step"]
                    paths[f"col_{key}"] = r["col_step"]
                    paths[f"rch_{key}"] = r["rch_step"]
                    paths[f"live_{key}"] = r["live_step"]
                    paths[f"done_{key}"] = r["done_at"]
                print(f"[diag] {arm}/{mode} seed={seed} "
                      f"ret={r['return'].mean():+.2f} reached={r['reached'].mean():.2f} "
                      f"off={r['offroad'].mean():.2f} col={r['collided'].mean():.2f} "
                      f"solved={r['solved'].mean():.2f} "
                      f"nstep={r['n_steps_batch']} "
                      f"absorb@1={np.mean(r['absorb_replan'] == 1):.2f}")
        print(f"[diag] arm={arm} done in {time.time()-t0:.0f}s")

    if args.expert:
        # The human log through the identical executor and reward: the ceiling
        # this reward can score, and the check for scenes whose ground truth is
        # itself dirty (a scene the expert cannot solve is not the policy's fault).
        env.reset(state)
        ex = env.expert_replay_episode()
        exp_path = _ego_rows(env.sim_state)
        goal_xy = np.asarray(env._goal_xy_b2)
        dist_t = np.linalg.norm(exp_path[:, :, :2] - goal_xy[:, None, :], axis=-1)
        reached = np.asarray(ex["reached_any_b"]).astype(bool)
        coll = np.asarray(ex["overlap_any_b"]).astype(bool)
        off = np.asarray(ex["offroad_any_b"]).astype(bool)
        ep_ret = np.asarray(ex["episode_return_b"])
        st = int(env._start_timestep) - 1   # the reset anchor (see rollout)
        # Attribution for the control arm, from the same per-world-step flags the
        # policy arms use. If the human log absorbs in replan 1 too, the config is
        # unsatisfiable and no amount of policy improvement can fix it.
        ex_off = np.asarray(ex["per_step_offroad_bk"]).astype(bool)     # [B, K]
        ex_col = np.asarray(ex["per_step_collision_bk"]).astype(bool)
        ex_rch = np.asarray(ex["per_step_reached_bk"]).astype(bool)
        ex_absorb = np.asarray(ex["absorb_replan_b"]).astype(int)
        ex_term = ex_rch.copy()
        if reward_cfg.terminate_on_collision:
            ex_term = ex_term | ex_col
        if reward_cfg.terminate_on_offroad:
            ex_term = ex_term | ex_off
        # A zero-width flag axis (no prefix executed) must not reach argmax.
        ex_first = (np.argmax(ex_term, axis=1) if ex_term.shape[-1]
                    else np.zeros(len(indices), dtype=int))
        ex_any = (ex_term.any(axis=1) if ex_term.shape[-1]
                  else np.zeros(len(indices), dtype=bool))
        ex_rows = np.arange(len(indices))
        curves["expert/det"] = [{
            "offroad": ex_off[:, :args.replan_interval_steps].mean(axis=0),
            "collision": ex_col[:, :args.replan_interval_steps].mean(axis=0),
            "goal": ex_rch[:, :args.replan_interval_steps].mean(axis=0),
        }]
        for i, sc in enumerate(indices):
            rows.append({
                "arm": "expert/det", "seed": -1, "scene": int(sc),
                "return": float(ep_ret[i]), "reached": bool(reached[i]),
                "collided": bool(coll[i]), "offroad": bool(off[i]),
                "solved": bool(reached[i] and not coll[i] and not off[i]),
                "init_goal_dist": float(dist_t[i, st]),
                "final_goal_dist": float(dist_t[i, -1]),
                "min_goal_dist": float(dist_t[i, st:].min()),
                "dist_closed": float(dist_t[i, st] - dist_t[i, -1]),
                "frac_closed": float((dist_t[i, st] - dist_t[i, -1]) / max(dist_t[i, st], 1e-6)),
                "absorb_replan": int(ex_absorb[i]),
                "absorb_step_since_start": (int(ex_first[i]) + 1 if ex_any[i] else -1),
                "absorb_cause": _cause_label(
                    bool(ex_any[i] and ex_off[i, ex_first[i]]
                         and reward_cfg.terminate_on_offroad),
                    bool(ex_any[i] and ex_col[i, ex_first[i]]
                         and reward_cfg.terminate_on_collision),
                    bool(ex_any[i] and ex_rch[i, ex_first[i]]),
                    bool(ex_absorb[i] > 0 and not ex_any[i]),
                    bool(ex_absorb[i] > 0)),
                "absorbed_in_replan1": bool(ex_absorb[i] == 1),
                "n_replans": int(ex["n_replans"]),
            })
        paths["path_expert"] = exp_path.astype(np.float32)
        print(f"[diag] expert replay: reached={reached.mean():.2f} off={off.mean():.2f} "
              f"col={coll.mean():.2f} clean={np.mean(reached & ~coll & ~off):.2f}")

    # Roadgraph + goal, once, for the scenes we draw. Subsampled to keep the npz
    # small: only road edges and lane centers, which is what "offroad" means.
    rg = env.sim_state.roadgraph_points
    rg_x = np.asarray(rg.x); rg_y = np.asarray(rg.y)
    rg_t = np.asarray(rg.types); rg_v = np.asarray(rg.valid)
    npz = {
        "indices": np.asarray(indices, dtype=np.int32),
        "goal_xy": np.asarray(env._goal_xy_b2, dtype=np.float32),
        "log_path": _log_ego_rows(env.sim_state).astype(np.float32),
        "start_step": np.int32(env._start_timestep - 1),
        "rg_x": rg_x.astype(np.float32), "rg_y": rg_y.astype(np.float32),
        "rg_types": rg_t.astype(np.int16), "rg_valid": rg_v,
        **paths,
    }
    np.savez_compressed(out_dir / "rollouts.npz", **npz)
    with open(out_dir / "rollouts.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # A compact console verdict so the .out file alone answers the question.
    print("\n[diag] === per-arm summary (deterministic) ===")
    # `nrep` and `offFr` are what a NON-terminating offroad config is read off:
    # offroad-ep (`offR`) is a per-episode OR and saturates, so it cannot tell a
    # 1-step clip from a 24-step excursion. `nrep` is the per-episode replan count
    # (the signal the actor actually gets, unlike the batch-level n_steps_batch)
    # and `offFr` the fraction of live world steps spent off-road.
    hdr = (f"{'arm':>16} {'ret':>8} {'prog':>8} {'off':>7} {'col':>7} {'goal':>6} "
           f"{'reach':>6} {'offR':>6} {'colR':>6} {'solve':>6} {'closed':>7} {'detour':>7} "
           f"{'nrep':>6} {'offFr':>6}")
    print(hdr)
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for arm, rr in by_arm.items():
        g = lambda k: np.mean([x[k] for x in rr if k in x]) if any(k in x for x in rr) else float("nan")
        print(f"{arm:>16} {g('return'):>8.2f} {g('r_progress'):>8.2f} "
              f"{g('r_offroad'):>7.2f} {g('r_collision'):>7.2f} {g('r_goal_bonus'):>6.2f} "
              f"{g('reached'):>6.2f} {g('offroad'):>6.2f} {g('collided'):>6.2f} "
              f"{g('solved'):>6.2f} {g('frac_closed'):>7.2f} {g('detour_ratio'):>7.2f} "
              f"{g('n_replans'):>6.2f} {g('frac_offroad_steps'):>6.2f}")
    # ----------------------------------------------------------------- #
    # H3: what absorbs the batch, and when. This is the section the parity
    # degeneracy is read off -- keep it last so it is the tail of the .out.
    print("\n[diag] === termination attribution ===")
    causes = ("offroad", "collision", "goal", "offroad+collision",
              "offroad+goal", "collision+goal", "offroad+collision+goal",
              "truncated", "unknown", "alive")
    print(f"{'arm':>16} {'n':>5} {'absorb@1':>9} {'med step':>9} "
          + " ".join(f"{c[:11]:>12}" for c in causes))
    attribution: dict[str, dict] = {}
    for arm, rr in by_arm.items():
        have = [x for x in rr if "absorb_cause" in x]
        if not have:
            continue
        n = len(have)
        a1 = float(np.mean([x["absorbed_in_replan1"] for x in have]))
        steps = [x["absorb_step_since_start"] for x in have
                 if x["absorb_step_since_start"] > 0]
        med = float(np.median(steps)) if steps else float("nan")
        frac = {c: float(np.mean([x["absorb_cause"] == c for x in have])) for c in causes}
        gm = lambda k: float(np.mean([x[k] for x in have if k in x])) \
            if any(k in x for x in have) else float("nan")
        attribution[arm] = {"n": n, "absorb_at_replan1": a1,
                            "median_absorb_step_since_start": med, "cause_frac": frac,
                            "mean_n_replans": gm("n_replans"),
                            "mean_frac_offroad_steps": gm("frac_offroad_steps"),
                            "mean_n_offroad_steps": gm("n_offroad_steps"),
                            "mean_r_offroad": gm("r_offroad")}
        print(f"{arm:>16} {n:>5} {a1:>9.2f} {med:>9.1f} "
              + " ".join(f"{frac[c]:>12.2f}" for c in causes))

    print("\n[diag] === first-prefix flag rate, per world step (all worlds live) ===")
    curve_json: dict[str, dict] = {}
    for arm, cl in curves.items():
        avg = {k: np.mean([c[k] for c in cl], axis=0) for k in ("offroad", "collision", "goal")}
        curve_json[arm] = {k: [float(x) for x in v] for k, v in avg.items()}
        print(f"  {arm}")
        for k, v in avg.items():
            print(f"    {k:>10}: " + " ".join(f"{x:5.2f}" for x in v))

    with open(out_dir / "term_attribution.json", "w") as f:
        json.dump({"attribution": attribution, "first_prefix_rates": curve_json,
                   "reward": {
                       "route_reward": bool(args.route_reward),
                       "terminate_offroad": bool(args.terminate_offroad),
                       "offroad_per_step": bool(args.offroad_per_step),
                       "collision_per_step": bool(args.collision_per_step),
                       "terminate_collision": bool(args.terminate_collision),
                       "terminate_collision": bool(args.terminate_collision),
                       "progress": args.reward_progress,
                       "offroad": args.reward_offroad,
                       "collision": args.reward_collision,
                       "goal_bonus": args.reward_goal_bonus,
                       "off_route": args.reward_off_route,
                       "goal_shaping_coef": args.goal_shaping_coef,
                   }}, f, indent=2)

    print(f"\n[diag] wrote {out_dir}/rollouts.jsonl ({len(rows)} rows), rollouts.npz "
          f"and term_attribution.json")


if __name__ == "__main__":
    main()
