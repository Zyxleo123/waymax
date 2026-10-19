"""Mechanics gates for reactive NPCs and the safety loss's neighbour source.

Two separate claims get confused easily, so they are asserted separately here:

**The simulator claim.** `DiffusionWaymaxEnv(sim_agents=...)` hands every non-ego
object to a waypoint-following policy that keeps it on its own logged path and
modulates only its speed against whatever box is ahead of it, the ego included
(`rl/diffusion_ft/sim_agents.py`). The gates below check that this (a) leaves
`sim_agents="none"` exactly as it was, (b) actually moves the NPCs, (c) moves
them ALONG their route and never AHEAD of their recording, and (d) does not touch
the ego, which is still driven by the planning action and must track its
commanded pose identically under both.

(c) is the gate with teeth, and it is the one that failed on the first run (job
58075). Waymax's stock `IDMRoutePolicy` accelerates any unobstructed car toward
`desired_vel=30` m/s at `max_accel=2.0`, so the NPCs raced 50+ m ahead of their
recording, ran out of logged route, and were extrapolated straight past its end
-- which is not reactivity and measured strictly worse than rails. `--reactive`
selects which model to gate; `idm_brake` is the fix and the default.

**The loss claim.** `safety_objective.candidate_safety(..., neighbor_state=...)`
reads the other agents' boxes from a state's `sim_trajectory` instead of its
`log_trajectory`. Under log replay those two are the SAME motion, so the switch
must be a no-op there -- that equality is the gate that says the plumbing is
addressing the right rows and timesteps, and it is why it is checked under
`sim_agents="none"` where the answer is known in advance. Under IDM they differ,
and the difference is the whole point.

The silent-failure mode this is really guarding is narrow and nasty:
`sim_trajectory` is filled INVALID at reset and written one world step at a time,
so a neighbour window that runs past `state.timestep` reads `x = -1,
valid = False`, `safety_terms` masks those pairs out, and the agent barrier goes
quietly to **zero** -- indistinguishable from an empty road. `neighbor_window`
raises on that, and `candidate_safety` reports `n_neighbors` so a training run
can log the number that would have gone to 0.

Needs the checkpoint (the env owns a policy) but runs no denoiser: every rollout
here is `expert_replay_episode`, which feeds logged poses as actions. CPU,
minutes. Submitted, not run inline:
    sbatch rl/diffusion_ft/slurm/test_reactive_npc.sbatch
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv, _splice_sim_into_log_batched
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.safety_objective import (
    SafetyObjectiveConfig, candidate_safety, neighbor_window)
from rl.diffusion_ft.train_dppo import load_batch
from rl.diffusion_ft.winnability import parse_indices

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def build_env(ckpt, n_obj, sim_agents, replan):
    return DiffusionWaymaxEnv(
        ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
        use_ema=True, replan_interval_steps=replan, seed=0,
        sim_agents=sim_agents,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfrecord", type=str,
                   default="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--indices", type=str, default="0-7")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    # Which reactive model to gate against log replay. `idm_brake` is the one
    # meant for use; `idm` is waymax stock, kept so its failure stays
    # reproducible (see rl/diffusion_ft/sim_agents.py).
    p.add_argument("--reactive", type=str, default="idm_brake",
                   choices=("idm", "idm_brake"))
    args = p.parse_args(argv)
    arm_r = args.reactive

    indices = parse_indices(args.indices)
    state = load_batch(args.tfrecord, indices)
    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])
    ckpt = load_diffusion_checkpoint(args.ckpt)
    print(f"[gate] scenes={len(indices)} objects={n_obj} reactive={arm_r}\n",
          flush=True)

    # ---- roll the SAME expert trajectory under both agent models ---------- #
    # Same ego action sequence by construction: the ego is driven from the log
    # in both arms, so every difference below belongs to the NPCs.
    ends, eps = {}, {}
    for arm in ("none", arm_r):
        env = build_env(ckpt, n_obj, arm, args.replan_interval_steps)
        env.reset(state)
        eps[arm] = env.expert_replay_episode()
        ends[arm] = env.sim_state
        print(f"[gate] rolled sim_agents={arm}: "
              f"collided={float(np.mean(np.asarray(eps[arm]['overlap_any_b']))):.3f} "
              f"offroad={float(np.mean(np.asarray(eps[arm]['offroad_any_b']))):.3f} "
              f"reached={float(np.mean(np.asarray(eps[arm]['reached_any_b']))):.3f}",
              flush=True)
    print(flush=True)

    # ---- gate 1: log replay really is log replay -------------------------- #
    # Under sim_agents="none" every non-ego object's SIMULATED pose must equal
    # its LOGGED pose at every executed step. This is the definition of the
    # default arm, and it is what makes `neighbor_window`'s log read exact.
    end = ends["none"]
    t_end = int(end.timestep)
    lx, ly = np.asarray(end.log_trajectory.x), np.asarray(end.log_trajectory.y)
    sx, sy = np.asarray(end.sim_trajectory.x), np.asarray(end.sim_trajectory.y)
    lvalid = np.asarray(end.log_trajectory.valid)
    keep = (~np.asarray(end.object_metadata.is_sdc))[:, :, None] & lvalid
    sl = slice(0, t_end + 1)
    m = keep[:, :, sl]
    dev_none = np.abs(sx[:, :, sl] - lx[:, :, sl])[m]
    dev_none = np.maximum(dev_none, np.abs(sy[:, :, sl] - ly[:, :, sl])[m])
    check("log replay: NPC sim pose == log pose at every executed step",
          float(dev_none.max()) < 1e-3,
          f"max |sim-log| = {float(dev_none.max()):.2e} m over {int(m.sum())} poses")

    # ---- gate 1b: the observation splice is a no-op under log replay ------ #
    # `_splice_sim_into_log_batched` now feeds the preprocessor EVERY object's
    # executed motion, not just the ego's, so that a reactive NPC is visible to
    # the policy instead of only to the metrics. That change must not disturb a
    # single existing result, and under log replay it provably cannot: where
    # `sim_valid` holds the simulated pose IS the logged pose (gate 1 above), and
    # where it does not the log is kept. Asserted here rather than argued.
    spliced = _splice_sim_into_log_batched(end)
    dmax = 0.0
    for f in ("x", "y", "yaw", "vel_x", "vel_y"):
        dmax = max(dmax, float(np.abs(
            np.asarray(getattr(spliced.log_trajectory, f))
            - np.asarray(getattr(end.log_trajectory, f))).max()))
    check("observation splice is a no-op under sim_agents='none'",
          dmax < 1e-6, f"max |spliced - log| = {dmax:.2e} over all objects/steps")

    # And under a reactive model it must NOT be a no-op, or the policy still
    # cannot see the traffic and the whole point of the change is lost.
    spliced_r = _splice_sim_into_log_batched(ends[arm_r])
    dmax_r = max(float(np.abs(
        np.asarray(getattr(spliced_r.log_trajectory, f))
        - np.asarray(getattr(ends[arm_r].log_trajectory, f))).max())
        for f in ("x", "y"))
    check(f"observation splice DOES move under {arm_r} (NPCs are observable)",
          dmax_r > 0.5, f"max |spliced - log| = {dmax_r:.3f} m")

    # ---- gate 2: the agent model actually moves the NPCs ----------------- #
    # If this is ~0 the flag is wired but inert, which would make every reactive
    # result a restatement of the log-replay one.
    end_i = ends[arm_r]
    six, siy = np.asarray(end_i.sim_trajectory.x), np.asarray(end_i.sim_trajectory.y)
    d_idm = np.hypot(six[:, :, sl] - lx[:, :, sl], siy[:, :, sl] - ly[:, :, sl])
    d_idm_valid = d_idm[m]
    moved = float(np.mean(d_idm_valid > 0.5))
    check(f"{arm_r} moves NPCs off their logged schedule",
          float(np.percentile(d_idm_valid, 99)) > 0.5,
          f"p50={np.percentile(d_idm_valid, 50):.3f} "
          f"p99={np.percentile(d_idm_valid, 99):.3f} m; "
          f"{moved:.1%} of poses > 0.5 m off the log")

    # ---- gate 3: WHERE they moved to, split into the two components ------- #
    # A waypoint-following policy is supposed to change only the SCHEDULE, so
    # the deviation above has to decompose into ~0 across the path and whatever
    # along it. Both halves are measured against each agent's own logged
    # polyline, whose points are indexed by time -- which is what makes the
    # along-path half readable in STEPS and signed:
    #
    #   lateral  = distance from the simulated pose to the nearest point of the
    #              agent's own logged path. Nonzero means it left its route.
    #   schedule = (index of that nearest point) - t, in world steps. NEGATIVE
    #              is behind the recording (braking, which is the entire point of
    #              a reactive agent); POSITIVE is ahead of it, which no reactive
    #              model should ever produce and is the signature of an IDM
    #              free-road term pulling agents toward a `desired_vel` that has
    #              nothing to do with what the human drove.
    #
    # This is the gate that caught waymax's stock IDM (job 58075): `desired_vel`
    # 30 m/s and `max_accel` 2.0 accelerate any unobstructed car at ~2 m/s^2
    # toward 108 km/h, so the NPCs raced, ran out of logged route, and were then
    # extrapolated straight past its end.
    #
    # The schedule half needs one exclusion, and it is not cosmetic. A PARKED
    # agent's logged path is effectively a single point, so "the nearest
    # waypoint" is arbitrary among identical ones, `argmin` returns index 0, and
    # the offset reads -t for every step -- which is what produced the
    # meaningless p1 = -83 in job 59763's first reading. Agents whose logged path
    # is shorter than `MIN_PATH_M` carry no schedule information and are dropped
    # from that statistic only; they still count for the lateral one, where a
    # stationary agent's distance to its own point IS well defined.
    MIN_PATH_M = 2.0
    lat, sched, n_static, n_moving = [], [], 0, 0
    for b in range(lx.shape[0]):
        for i in range(lx.shape[1]):
            if not keep[b, i, sl].any():
                continue
            t_idx = np.where(lvalid[b, i])[0]
            if t_idx.size < 2:
                continue
            path = np.stack([lx[b, i][t_idx], ly[b, i][t_idx]], -1)
            steps = np.where(keep[b, i, sl])[0]
            pts = np.stack([six[b, i, sl], siy[b, i, sl]], -1)[keep[b, i, sl]]
            d = np.linalg.norm(pts[:, None, :] - path[None, :, :], axis=-1)
            j = d.argmin(axis=1)
            lat.append(d[np.arange(d.shape[0]), j])
            # Cumulative arclength along the agent's own valid logged path.
            s_path = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))])
            if s_path[-1] < MIN_PATH_M:
                n_static += 1
                continue
            n_moving += 1
            # Offset in METRES of arclength, not in waypoint index. The index is
            # ill-conditioned exactly where it matters: wherever the human
            # stopped, the path has a dense cluster of near-identical waypoints
            # and `argmin` picks among them arbitrarily, which showed up as a
            # spurious +-30..80 STEPS in jobs 59763/59778. Arclength is flat
            # across such a cluster, so every member maps to the same number.
            pos_in_path = np.searchsorted(t_idx, steps).clip(0, len(t_idx) - 1)
            sched.append(s_path[j] - s_path[pos_in_path])
    lat = np.concatenate(lat) if lat else np.zeros(1)
    sched = np.concatenate(sched) if sched else np.zeros(1)
    check(f"{arm_r} keeps NPCs on their own logged path",
          float(np.percentile(lat, 95)) < 1.0,
          f"lateral dist to own path: p50={np.percentile(lat, 50):.3f} "
          f"p95={np.percentile(lat, 95):.3f} p99={np.percentile(lat, 99):.3f} m")
    ahead = float(np.mean(sched > 1.0))
    check(f"{arm_r} NPCs never run AHEAD of their recording "
          f"(reactivity may only delay)",
          float(np.percentile(sched, 99)) <= 1.0,
          f"along-path offset in METRES (neg=braking, pos=racing): "
          f"p1={np.percentile(sched, 1):.2f} p50={np.percentile(sched, 50):.2f} "
          f"p99={np.percentile(sched, 99):.2f}; {ahead:.1%} of poses >1 m ahead "
          f"[{n_moving} moving agents; {n_static} near-stationary excluded, "
          f"path < {MIN_PATH_M:g} m]")

    # ---- gate 3b: the model invents no collisions on known-good driving --- #
    # The cheap proxy for the expensive finding. Jobs 59781/59812 scored 1000
    # `success_heldout` scenes the BASE policy solves cleanly and found its
    # solve rate fell 0.9968 -> 0.9357 under `idm_brake`, entirely through
    # overlap (0.0000 -> 0.0610), with `reached` and `offroad` unchanged and the
    # bank intervening on 0.1% of decisions -- i.e. the SIMULATOR invented a 6%
    # collision floor on good driving. Cause: stock IDM lets an NPC brake for
    # another NPC using generic spacing constants that real logged traffic
    # violates, so traffic stops where the human did not and the ego drives into
    # it.
    #
    # Here the ego is the human's own log, which is about as known-good as
    # driving gets, so ANY collision the reactive arm has that log replay does
    # not is the model inventing one. n=8, so this is a smoke alarm, not a rate.
    c_none = float(np.mean(np.asarray(eps["none"]["overlap_any_b"])))
    c_react = float(np.mean(np.asarray(eps[arm_r]["overlap_any_b"])))
    check(f"{arm_r} invents no collisions on the human's own log",
          c_react <= c_none + 1e-9,
          f"collided {c_none:.3f} (none) -> {c_react:.3f} ({arm_r}) "
          f"over {int(np.asarray(eps['none']['overlap_any_b']).size)} scenes")

    # ---- gate 4: the ego is untouched ------------------------------------- #
    # The planning action drives the ego under both arms, so its executed motion
    # must be identical. If this fails, the flag is changing the thing being
    # measured rather than the thing it is measured against, and no comparison
    # between the arms means anything.
    is_sdc = np.asarray(end.object_metadata.is_sdc)
    ego_i = np.argmax(is_sdc, axis=1)
    ego_none = np.stack([sx[np.arange(len(ego_i)), ego_i][:, sl],
                         sy[np.arange(len(ego_i)), ego_i][:, sl]], -1)
    ego_idm = np.stack([six[np.arange(len(ego_i)), ego_i][:, sl],
                        siy[np.arange(len(ego_i)), ego_i][:, sl]], -1)
    ego_dev = float(np.abs(ego_none - ego_idm).max())
    check("ego motion is identical under both agent models",
          ego_dev < 1e-3, f"max |none - idm| = {ego_dev:.2e} m")

    # ---- gate 5: neighbour source is a no-op under log replay ------------- #
    # `neighbor_state=` must select the same boxes the log read selects, when the
    # two sources hold the same motion. Any row/timestep misaddressing shows up
    # here as a mismatch, on data where the right answer is known.
    t0, w = 20, 10
    box_log, val_log = neighbor_window(end, t0, w)
    box_sim, val_sim = neighbor_window(end, t0, w, neighbor_state=end)
    same_valid = bool(np.array_equal(np.asarray(val_log), np.asarray(val_sim)))
    vmask = np.asarray(val_log)[..., None]
    dbox = float(np.abs(np.where(vmask, np.asarray(box_log) - np.asarray(box_sim),
                                 0.0)).max())
    check("neighbor_state is a no-op under sim_agents='none'",
          same_valid and dbox < 1e-3,
          f"valid identical={same_valid}, max |box| diff={dbox:.2e}")

    # ---- gate 6: under IDM the two sources genuinely differ --------------- #
    box_i_log, val_i_log = neighbor_window(end_i, t0, w)
    box_i_sim, val_i_sim = neighbor_window(end_i, t0, w, neighbor_state=end_i)
    # BOTH validities. Masking on the sim side alone compares a real coordinate
    # against the log's invalid sentinel (-1, -1) wherever an agent has dropped
    # out of the recording, which reads as a ~14 km "displacement" and made the
    # first run of this gate meaningless.
    vm = np.asarray(val_i_log) & np.asarray(val_i_sim)
    dpos = np.hypot(np.asarray(box_i_sim)[..., 0] - np.asarray(box_i_log)[..., 0],
                    np.asarray(box_i_sim)[..., 1] - np.asarray(box_i_log)[..., 1])[vm]
    check(f"under {arm_r} the log and sim neighbour sets differ",
          float(np.percentile(dpos, 99)) > 0.25,
          f"per-neighbour displacement p50={np.percentile(dpos, 50):.3f} "
          f"p99={np.percentile(dpos, 99):.3f} m")

    # ---- gate 7: the silent-zero guard fires ------------------------------ #
    # A window past `timestep` must RAISE, not read invalid placeholders and
    # report a clear road.
    raised = False
    try:
        neighbor_window(end_i, t_end, 5, neighbor_state=end_i)
    except ValueError:
        raised = True
    check("a neighbour window past the simulated horizon raises",
          raised, f"timestep={t_end}, asked for {t_end + 1}..{t_end + 5}")

    # ---- gate 8: candidate_safety differentiates under both sources ------- #
    # The point of the module is an analytic gradient. Swapping the neighbour
    # source must not break it, and `n_neighbors` must be non-zero -- a zero
    # there is the signature of the silent failure gate 7 guards against.
    cfg = SafetyObjectiveConfig()
    # The candidate under test is the ego's own logged window -- known-good
    # geometry, so a non-finite loss or a zero gradient here is the plumbing,
    # not the trajectory.
    env_c = build_env(ckpt, n_obj, "none", args.replan_interval_steps)
    env_c.reset(state)
    cand = jnp.asarray(env_c.ego_log_world_trajectory()[:, t0:t0 + w + 1, :])

    for label, nstate in (("log", None), ("sim", end_i)):  # noqa: B007
        ref = end if nstate is None else end_i

        def total(c, nstate=nstate, ref=ref):
            return candidate_safety(cfg, c, state=ref, t0=t0, n_steps=w,
                                    neighbor_state=nstate)["total"]

        val, g = jax.value_and_grad(total)(cand)
        terms = candidate_safety(cfg, cand, state=ref, t0=t0, n_steps=w,
                                 neighbor_state=nstate)
        gn = float(jnp.linalg.norm(g))
        nn = float(terms["n_neighbors"])
        check(f"candidate_safety[{label}]: finite loss, finite non-zero gradient, "
              f"non-empty neighbour set",
              bool(np.isfinite(float(val))) and np.isfinite(gn) and gn > 0 and nn > 0,
              f"total={float(val):.4f} |grad|={gn:.4e} n_neighbors={nn:.2f}")

    print()
    if FAILS:
        print(f"[gate] {len(FAILS)} FAILED: {', '.join(FAILS)}")
        raise SystemExit(1)
    print("[gate] all reactive-NPC gates passed")


if __name__ == "__main__":
    main()
