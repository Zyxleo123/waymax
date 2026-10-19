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

The 2026-08-30 run answered that: lateral is cheap (1 m costs one scene in 32,
0.031 -> 0.062) and DELAY is what is expensive (5 steps 0.125, 40 steps -- a car
that never moves -- 0.719). Non-reactivity does not punish being in the wrong
*place*; it punishes being there at the wrong *time*, which is precisely what a
slow policy is.

--sim_agents (added 2026-09-19) runs every case under BOTH agent models and
prints them paired:

  ``none``  every non-ego agent replays its log and cannot see the ego (what
            every result before 2026-09-19 used).
  ``idm``   waymax ``IDMRoutePolicy``: each agent stays on its own logged path
            and modulates only its speed by IDM against whatever box is ahead of
            it, the ego included. Longitudinal reactivity only -- a car brakes
            for the ego, it never steers around it.

Three things to read off the paired table, in order:

  1. **The d=0 / k=0 control under ``idm``.** This is the gate, not a curve
     point. Reactive agents brake and therefore fall BEHIND their logged
     schedule, so they end up where the recording never put them -- IDM can
     manufacture collisions on the human demonstration itself. If ``idm``'s
     control collides materially worse than ``none``'s ~0.03, reactive agents
     are a worse ground truth than the rails they replace and nothing
     downstream is worth doing. Read this first and stop here if it fails.
  2. **The delay curve's slope under ``idm``.** How much of the 0.719 is rails.
     Whatever flattens is simulator artifact that no reward, filter or
     fine-tune was ever able to remove.
  3. **The lateral curve under ``idm``.** The control for "IDM did not simply
     make every agent timid": lateral was already cheap, so it should stay
     roughly where it was. If lateral collision ALSO collapses, the agents are
     yielding to everything and the collision metric has gone slack.

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
from rl.diffusion_ft.sim_agents import SIM_AGENT_CHOICES
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
        "sim_agents": env.sim_agents,
        "n": int(offroad.size),
        "collided": float(overlap.mean()),
        "offroad": float(offroad.mean()),
        "reached": float(reached.mean()),
        "solve": float(clean.mean()),
        "collided_idx": [int(i) for i in np.where(overlap)[0]],
    }


def case_labels(lateral, delay) -> list[str]:
    """The case labels, in the order they are run, shared by both agent models.

    Built once so the two arms are guaranteed to be the same cases in the same
    order -- a paired table whose rows are assembled independently is one
    refactor away from silently comparing different perturbations.
    """
    return ([f"lateral {m:g}m" for m in lateral]
            + [f"delay {k}step" for k in delay])


def run_arm(args, ckpt, state, n_obj, sim_agents: str, lateral, delay) -> list[dict]:
    """Every case under one agent model. Returns rows in `case_labels` order."""
    # Same env construction as winnability.py / DPPOTrainer._build_for. The reward
    # config is irrelevant here (we read metric flags, not reward), so leave it
    # at the default rather than implying the numbers depend on a recipe.
    env = DiffusionWaymaxEnv(
        ckpt, reward_config=RewardConfig(),
        max_num_objects=n_obj, use_ema=True,
        replan_interval_steps=args.replan_interval_steps, seed=args.seed,
        sim_agents=sim_agents,
    )
    env.reset(state)
    # The unperturbed ego log, read back from the env. Identical across arms (it
    # comes from `log_trajectory`, which no agent model touches), but re-read per
    # arm so the perturbations are applied to the same object the arm drives.
    base_log = env.ego_log_world_trajectory()

    rows = []
    for m in lateral:
        rows.append(run_case(env, state, perturb_lateral(base_log, m),
                             f"lateral {m:g}m"))
    for k in delay:
        rows.append(run_case(env, state, perturb_delay(base_log, k),
                             f"delay {k}step"))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfrecord", type=str,
                   default="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--indices", type=str, default="0-31")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--lateral", type=str, default="0,0.25,0.5,1.0,1.5,2.0,3.0",
                   help="lateral offsets in metres (0 is the control)")
    # Reaches 40, unlike the original 0,1,2,3,5. The delay curve is the entire
    # reason reactive agents are interesting and it is FLAT out to 5 steps under
    # log replay (0.031 -> 0.125); the 0.719 lives at 40, where the ego is a car
    # that never moves. A sweep that stops at 5 cannot see whether reactivity
    # removes it -- job 58076 stopped at 5 and reported "+0.000 removed", which
    # was true and uninformative.
    p.add_argument("--delay", type=str, default="0,2,5,10,20,40",
                   help="delays in world steps (0 is the control)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim_agents", type=str, default="none,idm_brake",
                   help=f"comma list of agent models to run, each of "
                        f"{SIM_AGENT_CHOICES}. The default pairs log replay "
                        f"against the braking-only IDM; add 'idm' to include "
                        f"waymax stock, which races (see sim_agents.py).")
    p.add_argument("--out", type=str,
                   default="runs/dft_logreplay_probe/collision_vs_deviation.json")
    args = p.parse_args(argv)

    arms = [a.strip() for a in args.sim_agents.split(",") if a.strip()]
    bad = [a for a in arms if a not in SIM_AGENT_CHOICES]
    if bad:
        raise SystemExit(
            f"--sim_agents entries must be in {SIM_AGENT_CHOICES}, got {bad}")
    if not arms:
        raise SystemExit("--sim_agents is empty")

    lateral = [float(x) for x in args.lateral.split(",") if x.strip()]
    delay = [int(x) for x in args.delay.split(",") if x.strip()]

    indices = parse_indices(args.indices)
    state = load_batch(args.tfrecord, indices)
    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])
    # Loaded once and shared: the arms differ in the agent model, not the policy,
    # and a second copy is pure memory.
    ckpt = load_diffusion_checkpoint(args.ckpt)

    print(f"[probe] tfrecord={args.tfrecord} scenes={len(indices)} objects={n_obj}")
    print(f"[probe] agent models: {', '.join(arms)}")
    print(f"[probe] collision metric is symmetric overlap, no fault attribution\n")

    by_arm: dict[str, dict[str, dict]] = {}
    rows: list[dict] = []
    for arm in arms:
        arm_rows = run_arm(args, ckpt, state, n_obj, arm, lateral, delay)
        rows.extend(arm_rows)
        by_arm[arm] = {r["case"]: r for r in arm_rows}

    # ---- the paired table -------------------------------------------------- #
    # `collided` is the column the agent model moves; the other three are printed
    # for the first arm only when there are two, because a two-arm four-metric
    # table is unreadable and the full numbers are in the JSON either way.
    labels = case_labels(lateral, delay)
    if len(arms) == 1:
        arm = arms[0]
        print(f"[probe] {'case':>16} {'collided':>9} {'offroad':>8} "
              f"{'reached':>8} {'solve':>7}   (sim_agents={arm})")
        print("[probe] " + "-" * 62)
        for lab in labels:
            r = by_arm[arm][lab]
            print(f"[probe] {lab:>16} {r['collided']:9.3f} {r['offroad']:8.3f} "
                  f"{r['reached']:8.3f} {r['solve']:7.3f}")
    else:
        head = "".join(f"{a + ' coll':>12}" for a in arms)
        print(f"[probe] {'case':>16}{head}{'delta':>9}   "
              f"{'solve ' + arms[0]:>12}{'solve ' + arms[-1]:>12}")
        print("[probe] " + "-" * (16 + 12 * len(arms) + 9 + 24))
        for lab in labels:
            cols = "".join(f"{by_arm[a][lab]['collided']:12.3f}" for a in arms)
            d = by_arm[arms[-1]][lab]["collided"] - by_arm[arms[0]][lab]["collided"]
            print(f"[probe] {lab:>16}{cols}{d:9.3f}   "
                  f"{by_arm[arms[0]][lab]['solve']:12.3f}"
                  f"{by_arm[arms[-1]][lab]['solve']:12.3f}")

    # ---- verdict ---------------------------------------------------------- #
    base_arm = arms[0]
    ctrl = by_arm[base_arm]["lateral 0m"] if "lateral 0m" in by_arm[base_arm] else None
    lat1 = by_arm[base_arm].get("lateral 1m")
    print()
    if ctrl is None:
        print("[probe] no d=0 control in --lateral; add 0 to it or the table has "
              "no baseline to read against.")
    elif ctrl["collided"] > 0.15:
        print(f"[probe] WARNING: the d=0 control already collides "
              f"{ctrl['collided']:.3f} under sim_agents={base_arm}; expected ~0.03. "
              f"The harness, not the sim, is suspect -- do not read the rest.")
    elif ctrl["n"] < 16:
        print(f"[probe] n={ctrl['n']} is too small for a verdict: the unperturbed "
              f"expert collides on ~1 scene in 32, so a subset this size usually "
              f"has no baseline collisions to lift from. Re-run on the full set.")
    elif lat1 is not None:
        lift = lat1["collided"] - ctrl["collided"]
        print(f"[probe] [{base_arm}] control (on the log) collides "
              f"{ctrl['collided']:.3f}; 1 m off the log collides "
              f"{lat1['collided']:.3f} (+{lift:.3f})")
        if lift > 0.25:
            print("[probe] VERDICT: collision is dominated by deviation from the log, "
                  "not by driving quality. No reward term can remove it, and `solve` "
                  "is floored by the simulator. Fix the sim (reactive agents) or the "
                  "metric (at-fault-only collision) before any further reward work.")
        else:
            print("[probe] VERDICT: a 1 m deviation does NOT explain the collision "
                  "rate. The collisions DPPO incurs are real driving errors; the "
                  "reward/policy remains the place to look.")

    # ---- the reactive-agent gate, read BEFORE the curves ------------------- #
    # Three questions, in this order. Only the third is the one we WANT an answer
    # to; the first two decide whether the third means anything.
    if "none" in by_arm and "lateral 0m" in by_arm["none"]:
        n = by_arm["none"]["lateral 0m"]["n"]
        one = 1.0 / max(n, 1)          # one scene: the resolution of every rate
        for arm in arms:
            if arm == "none":
                continue
            print()
            print(f"[probe] === GATE: is '{arm}' a valid replacement for log "
                  f"replay?  (n={n} scenes, 1 scene = {one:.3f}) ===")

            # (1) Does it damage the demonstration? Reactive agents brake and so
            # fall BEHIND their logged schedule, ending up where the recording
            # never put them -- a reactive model can manufacture collisions on
            # the human log itself. Stated in scenes, because at n=32 a rate
            # difference of 0.031 IS one scene and is not by itself a result.
            c0, c1 = by_arm["none"]["lateral 0m"], by_arm[arm]["lateral 0m"]
            s0, s1 = c0["collided"] * n, c1["collided"] * n
            print(f"[probe]  (1) human log itself: {c0['collided']:.3f} -> "
                  f"{c1['collided']:.3f}  ({s0:.0f} -> {s1:.0f} scenes)")
            damaged = (s1 - s0) >= 2
            if damaged:
                print(f"[probe]      FAIL: it collides the demonstration it is "
                      f"supposed to reproduce. A worse ground truth than the "
                      f"rails it replaces; stop here.")
            elif s1 > s0:
                print(f"[probe]      MARGINAL: one extra scene is the "
                      f"resolution limit at this n, not a finding. Re-run wider "
                      f"before trusting either direction.")
            else:
                print(f"[probe]      PASS")

            # (2) Did it go slack? Lateral collision was already cheap under
            # rails, so it should stay roughly where it was. Rising means the
            # model ADDS collisions; collapsing means every agent is yielding to
            # everything and `overlap` no longer measures anything.
            lat = [l for l in labels if l.startswith("lateral ") and l != "lateral 0m"]
            if lat:
                d = [by_arm[arm][l]["collided"] - by_arm["none"][l]["collided"]
                     for l in lat]
                worst_l = lat[int(np.argmax(np.abs(d)))]
                print(f"[probe]  (2) lateral column, largest change at "
                      f"{worst_l}: {by_arm['none'][worst_l]['collided']:.3f} -> "
                      f"{by_arm[arm][worst_l]['collided']:.3f} "
                      f"({max(d, key=abs):+.3f})")
                if max(d) >= 2 * one:
                    print(f"[probe]      FAIL: it ADDS collisions where the ego "
                          f"is merely off-line. That is not reactivity.")
                elif min(d) <= -4 * one:
                    print(f"[probe]      FAIL: lateral collision collapsed too. "
                          f"The agents are yielding to everything and the "
                          f"collision metric has gone slack.")
                else:
                    print(f"[probe]      PASS")

            # (3) The measurement. Whatever the delay column loses is simulator
            # artifact that no reward, filter or fine-tune could ever have
            # removed -- but only if (1) and (2) passed.
            dly = [l for l in labels if l.startswith("delay ") and l != "delay 0step"]
            if dly:
                deep = dly[-1]
                d0, d1 = by_arm["none"][deep]["collided"], by_arm[arm][deep]["collided"]
                print(f"[probe]  (3) at {deep}: {d0:.3f} -> {d1:.3f} "
                      f"({d0 - d1:+.3f} removed by reactivity)")
                if d0 - d1 >= 4 * one:
                    print(f"[probe]      the rails were charging the ego "
                          f"{d0 - d1:.3f} for running late. That much of every "
                          f"collision number in this project is exogenous.")
                elif abs(d0 - d1) < 2 * one:
                    print(f"[probe]      reactivity removes NOTHING here. Either "
                          f"the agents are not braking for the ego, or the "
                          f"collisions at this delay are not rear-ends.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"tfrecord": args.tfrecord, "indices": indices,
                   "sim_agents": arms, "rows": rows}, f, indent=1)
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
