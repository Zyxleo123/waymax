"""Does the reward's ORDERING agree with the solve criterion?

Every DPPO reward change so far has been a magnitude change, and the runs read
the same way each time: the return climbs, the solve rate does not. Job 37016
(`penalty_up`) is the clean example -- eval return improved 10x (-24.4 -> -2.7)
while off-road-ever got *worse* (0.28 -> 0.50). That is not a coefficient that
needs another turn of the crank; it is the reward ranking trajectories in an
order that does not put the solved ones on top. No optimizer change (SAC
included) fixes it -- a better optimizer climbs the same wrong ranking faster.

This test is the gate that should run BEFORE any training launch. It is pure
reward algebra on synthetic episode archetypes: no simulator, no checkpoint,
CPU, seconds. It scores each archetype under each candidate recipe and checks
four ordering properties that the solve criterion (reached AND never off-road
AND never collided) implies but that the reward has to be *built* to have.

    P1  solve-is-argmax   the clean solving episode outscores every other
                          archetype, so the reward's maximum IS the objective.
    P2  avoidance > dwell avoiding an excursion pays more than shortening one
                          by a step. A per-step charge alone gives ratio 1.0:
                          the marginal value of the LAST off-road step equals
                          that of any interior one, so an optimizer buys
                          shortening (and stopping), never avoidance. This is
                          exactly the penalty_up signature.
    P3  excursions priced two 6-step excursions cost strictly more than one
                          12-step excursion of the same total dwell. Both the
                          charged-once form (the second excursion is free) and
                          the per-step form (cost depends only on dwell) score 0
                          here; only a rising-edge charge prices the decision.
    P4  promptness        reaching the goal sooner never scores worse than
                          reaching it later.

P4 catches a defect nothing has documented. `progression_indicator` pays
`progress` per world step in which the route fraction advanced *at all*, so the
progress collected is proportional to EPISODE LENGTH, and reaching the goal
terminates the episode and forfeits the rest of that stream. Under the parity
recipe actually launched (progress 0.2, 40 world steps, goal bonus 5.0) the
budget forgone is 0.2 * (40 - k) against a bonus of 5.0, so stepping into the
goal before world step ~15 is *punished*, and never reaching the goal at all
(+8.0) beats reaching it at step 10 (+7.0). The base policy's reached=0.00 is
the rational response.

The fix for P4 is not a bigger bonus, it is the telescoping form: waymax's
`sdc_progression` is a route fraction normalized to [0, 1] (see
`waymax/metrics/route.py`, FULL_PROGRESS_VALUE), NOT the arclength in metres
that `env.py` called it, so `progress * (s - prev_s)` sums to
`progress * (fraction covered)` regardless of how long it took. It needs a
coefficient sized against the goal bonus (~5.0), not the 0.2 that a per-step
indicator wants.

Recipes below are the real ones: `noescape` and `penalty_up` are jobs 36946 and
37016, `perstep` is the 2026-08-24 offroad_per_step recipe, and `entry_clean` is
the proposal. The legacy recipes are asserted to FAIL at least one property --
if they ever pass, this test has stopped discriminating and is worthless.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import itertools

import jax.numpy as jnp

from rl.diffusion_ft.reward import RewardConfig, init_reward_state, reward_step

# One episode = 8 replans x 5 world steps, the DPPO rollout shape.
HORIZON = 40
START_DIST = 50.0
GOAL_THRESH = 3.0


# --------------------------------------------------------------------------- #
def make_episode(*, reach_at=None, excursions=(), collide_at=None,
                 stall_frac=0.6, drive=True):
    """Build the per-world-step metric sequence for one episode archetype.

    Args:
      reach_at: world step at which the ego arrives at the goal, or None if it
        never gets there.
      excursions: ``((start, length), ...)`` off-road intervals, 1-based steps.
      collide_at: world step of first contact, or None.
      stall_frac: route fraction reached by a non-arriving episode.
      drive: False for an ego that never moves (no progress at all).

    Returns a list of ``(goal_dist, overlap, offroad, s)`` per world step. ``s``
    is the *normalized* route fraction, matching waymax `sdc_progression`.
    """
    off = [False] * (HORIZON + 1)
    for start, length in excursions:
        for t in range(start, start + length):
            if t <= HORIZON:
                off[t] = True
    steps = []
    for t in range(1, HORIZON + 1):
        if not drive:
            dist, s = START_DIST, 0.0
        elif reach_at is None:
            frac = min(t / HORIZON, 1.0) * stall_frac
            dist, s = START_DIST * (1.0 - frac), frac
        else:
            frac = min(t / reach_at, 1.0)
            # Arrive exactly inside the goal radius at `reach_at`.
            dist, s = START_DIST * (1.0 - frac), frac
        col = collide_at is not None and t >= collide_at
        steps.append((dist, 1.0 if col else 0.0, 1.0 if off[t] else 0.0, s))
    return steps


def score(cfg: RewardConfig, steps) -> float:
    """Episode return under `cfg`: the sum of per-step rewards (Gate 3 form)."""
    state = init_reward_state(init_goal_dist_b=jnp.array([START_DIST]),
                              init_s_b=jnp.array([0.0]))
    total = 0.0
    for dist, col, off, s in steps:
        reward_b, state, _, _ = reward_step(
            cfg, state,
            goal_dist_b=jnp.array([dist]),
            overlap_b=jnp.array([col]),
            offroad_b=jnp.array([off]),
            s_b=jnp.array([s]),
            lateral_b=jnp.array([2.0 if off else 0.0]),
        )
        total += float(reward_b[0])
    return total


# --------------------------------------------------------------------------- #
# The archetypes. A corner-cut both goes off-road AND arrives sooner (that is
# what makes it tempting), so `cut1_long` reaches earlier than `clean_solve`.
ARCHETYPES = {
    "clean_solve":  dict(reach_at=24),
    # The same clean solve, but prompt. Under an indicator progress term this
    # scores WORSE than never arriving at all, which is the whole of P1/P4.
    "clean_solve_fast": dict(reach_at=10),
    "cut_2step":    dict(reach_at=22, excursions=((8, 2),)),
    "cut_12step":   dict(reach_at=18, excursions=((4, 12),)),
    # Same total dwell as cut_12step, split in two. Both excursions must finish
    # before the episode absorbs at the goal or the dwell is not actually equal.
    "cut_2x6step":  dict(reach_at=18, excursions=((2, 6), (9, 6))),
    "clean_stall":  dict(reach_at=None),
    "dirty_stall":  dict(reach_at=None, excursions=((8, 12),)),
    # Boundary chatter: 6 one-step crossings, HALF the dwell of cut_12step but
    # 6 rising edges instead of 1. This is what a sigma-0.1 sampled rollout
    # actually looks like, and it is the entry charge's main exposure -- see the
    # `entry charge exposure` line printed below. Six re-crossings of a road edge
    # is genuinely worse driving than one clean excursion, so costing more is
    # intended; costing 6x more is a coefficient worth checking against measured
    # edge counts (`eval_off_entries_*` in train_log.jsonl) before a long run.
    "chatter":      dict(reach_at=24, excursions=tuple((6 + 2 * i, 1) for i in range(6))),
    "park":         dict(drive=False),
    "crash":        dict(reach_at=24, collide_at=10),
}
SOLVERS = ("clean_solve", "clean_solve_fast")

_PARITY = dict(route_reward=True, progression_indicator=True,
               off_route_threshold_m=0.0, off_route_penalty=-0.2,
               terminate_on_collision=True, goal_threshold_m=GOAL_THRESH,
               action_penalty=0.0)

RECIPES = {
    # Job 36946: offroad charged once, non-terminating.
    "noescape": RewardConfig(progress=0.2, offroad=-0.2, collision=-0.2,
                             goal_bonus=5.0, **_PARITY),
    # Job 37016: same shape, bigger magnitudes. Return went up 10x, offroad too.
    "penalty_up": RewardConfig(progress=0.2, offroad=-1.0, collision=-1.5,
                               goal_bonus=5.0, **_PARITY),
    # 2026-08-24: charge by dwell instead of once. Fixes farming, not ranking.
    "perstep": RewardConfig(progress=0.2, offroad=-0.2, collision=-0.2,
                            offroad_per_step=True, goal_bonus=5.0, **_PARITY),
    # Proposal: telescoping route progress (duration-independent, coefficient
    # sized against the bonus), a rising-edge charge that prices ENTERING the
    # off-road region, a small dwell charge that prices staying there, and a
    # goal bonus paid only to a clean arrival.
    "entry_clean": RewardConfig(
        progress=5.0, offroad=-0.1, offroad_entry=-1.0,
        collision=-0.1, collision_entry=-2.0,
        goal_bonus=5.0, clean_goal_bonus=True,
        **{**_PARITY, "progression_indicator": False},
    ),
}


# --------------------------------------------------------------------------- #
def properties(cfg: RewardConfig) -> dict[str, tuple[bool, str]]:
    """Evaluate P1-P4 for one recipe."""
    R = {name: score(cfg, make_episode(**kw)) for name, kw in ARCHETYPES.items()}

    # P1: the clean solver is the strict argmax.
    best = max(R, key=R.get)
    p1 = best in SOLVERS
    inversions = [(a, b) for a, b in itertools.product(SOLVERS, R)
                  if b not in SOLVERS and R[b] >= R[a]]
    p1_msg = (f"argmax={best} ({R[best]:+.2f}); "
              f"{len(inversions)} solver/non-solver inversion(s)"
              + (f" worst={inversions[0][1]}" if inversions else ""))
    p1 = p1 and not inversions

    # P2: what avoiding the last off-road step pays, vs shortening by one step.
    #     Measured on a single excursion so the entry charge is held fixed for
    #     the shortening comparison and released only by avoidance.
    r_off1 = score(cfg, make_episode(reach_at=24, excursions=((8, 1),)))
    r_off0 = score(cfg, make_episode(reach_at=24))
    r_off12 = score(cfg, make_episode(reach_at=24, excursions=((8, 12),)))
    r_off11 = score(cfg, make_episode(reach_at=24, excursions=((8, 11),)))
    avoid = r_off0 - r_off1                 # 1 off-road step -> none
    shorten = r_off11 - r_off12             # 12 off-road steps -> 11
    ratio = avoid / shorten if shorten > 1e-9 else float("inf")
    p2 = ratio >= 3.0
    p2_msg = f"avoid={avoid:+.3f} shorten/step={shorten:+.3f} ratio={ratio:.2f}"

    # P3: same total dwell, split into two excursions. Held at a common arrival
    #     step so the only difference between the two is the excursion count.
    r_one = score(cfg, make_episode(reach_at=24, excursions=((4, 12),)))
    r_two = score(cfg, make_episode(reach_at=24, excursions=((4, 6), (12, 6))))
    split_price = r_one - r_two
    p3 = split_price > 1e-6
    p3_msg = f"1x12 {r_one:+.2f} vs 2x6 {r_two:+.2f} -> 2nd excursion costs {split_price:+.3f}"

    # P4: dawdling must not pay. The residual spread under a telescoping progress
    #     term is discretization -- the ego absorbs 3 m short of the goal, so the
    #     route fraction banked depends slightly on where that lands -- hence a
    #     tolerance rather than strict monotonicity. What must not survive is a
    #     systematic gradient rewarding lateness, which is what an indicator
    #     progress term (progress per step *alive*) creates.
    curve = [(k, score(cfg, make_episode(reach_at=k))) for k in (8, 16, 24, 32)]
    late_gain = max(b - a for (_, a), (_, b) in itertools.combinations(curve, 2))
    p4 = late_gain <= 0.25
    p4_msg = ("price of dawdling " + f"{late_gain:+.2f}; R(reach@k) "
              + " ".join(f"{k}:{v:+.2f}" for k, v in curve))

    # Not a pass/fail property -- a number to read. The entry charge bills edges,
    # so its total scales with how often the trajectory re-crosses the boundary,
    # which under exploration noise is not the same thing as how badly it drives.
    r_chatter = score(cfg, make_episode(**ARCHETYPES["chatter"]))
    r_one12 = score(cfg, make_episode(reach_at=24, excursions=((4, 12),)))
    exposure = (f"6 edges/6 steps {r_chatter:+.2f} vs 1 edge/12 steps "
                f"{r_one12:+.2f} -> chatter costs {r_one12 - r_chatter:+.2f}")

    return {
        "P1 solve-is-argmax": (p1, p1_msg),
        "P2 avoidance>dwell": (p2, p2_msg),
        "P3 excursions priced": (p3, p3_msg),
        "P4 promptness": (p4, p4_msg),
    }, R, exposure


# --------------------------------------------------------------------------- #
def main() -> int:
    results = {}
    for name, cfg in RECIPES.items():
        props, R, exposure = properties(cfg)
        results[name] = props
        print(f"\n=== {name} ===")
        for label, kw in ARCHETYPES.items():
            print(f"    {label:<14} {score(cfg, make_episode(**kw)):+8.2f}")
        for label, (ok, msg) in props.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<22} {msg}")
        print(f"  [note] entry charge exposure   {exposure}")

    print("\n=== verdict ===")
    ok = True

    # The proposal must satisfy all four.
    for label, (passed, msg) in results["entry_clean"].items():
        if not passed:
            print(f"  entry_clean FAILS {label}: {msg}")
            ok = False
    if ok:
        print("  entry_clean: all four ordering properties hold")

    # And the legacy recipes must not -- otherwise the test does not
    # discriminate and its passing means nothing.
    for name in ("noescape", "penalty_up", "perstep"):
        failed = [l for l, (p, _) in results[name].items() if not p]
        if not failed:
            print(f"  ! {name} passes every property -- the test has stopped "
                  f"discriminating; do not trust a green run")
            ok = False
        else:
            print(f"  {name}: fails {', '.join(failed)} (as documented)")

    print(f"\nreward_ranking overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
