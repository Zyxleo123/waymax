"""`collision_per_step`: close the termination escape that `offroad_per_step` opened.

Charging offroad by dwell (``test_offroad_per_step``) fixed the one-step bandit
-- job 36466 collected all 8 replans at every iteration -- but created a second
farm. With offroad at -0.2/step an 80-step episode spent off-road runs to -16,
while a collision costs -1.0 ONCE and freezes the world (``RewardState.done``
zeroes every later step). Ending the episode is therefore an escape worth up to
+15, and the policy took it: over 300 iterations
``corr(mean_episode_return, n_alive) = -0.80``, episode length fell 8.0 -> 5.9
replans, train return "improved" -1.82 -> -0.78 purely by getting shorter, and
``eval_solve_deterministic`` stayed 0.000 in all 60 evals.

The fix is the collision twin of the offroad flag: charge collision on every
colliding step and stop terminating on it. With both per-step flags on and both
terminations off, the only reachable absorbing state is the goal, so no policy
can shorten its way out of a penalty.

This asserts:

* the default config is unchanged -- collision charged once, terminal (so every
  launch that does not set the flag is byte-identical to before);
* with the flag on, N colliding steps cost N x `collision` and nothing terminates;
* **the escape is closed**: under the job-36466 recipe crashing early beats
  staying alive, and under the per-step recipe it does not -- this is the whole
  point of the flag, stated as an inequality;
* the Gate 3 sum-to-return invariant still holds (components sum == reward).

No simulator and no checkpoint: pure reward algebra, CPU, seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from rl.diffusion_ft.reward import RewardConfig, init_reward_state, reward_step

ONE = jnp.array([1.0])
ZERO = jnp.array([0.0])

# One replan is 10 world steps and an episode is 8 replans (DFT_REPLAN=10), so a
# worst-case dwell is 80 charged steps. That horizon is what makes the escape
# profitable, so the economics check below uses it rather than a round number.
EPISODE_STEPS = 80


def _run(cfg: RewardConfig, col_seq, off_seq=None, *, goal_dist: float = 50.0):
    """Step the reward through fixed collision/offroad patterns; ego holds still."""
    if off_seq is None:
        off_seq = [False] * len(col_seq)
    state = init_reward_state(init_goal_dist_b=jnp.array([goal_dist]),
                              init_s_b=ZERO)
    total_collision, total, terminated, rewards = 0.0, 0.0, [], []
    for col, off in zip(col_seq, off_seq):
        reward_b, state, term_b, comps = reward_step(
            cfg, state,
            goal_dist_b=jnp.array([goal_dist]),
            overlap_b=(ONE if col else ZERO),
            offroad_b=(ONE if off else ZERO),
            s_b=ZERO, lateral_b=ZERO,
        )
        total_collision += float(comps["collision"][0])
        total += float(reward_b[0])
        terminated.append(bool(term_b[0]))
        rewards.append(float(reward_b[0]))
    return total_collision, total, terminated, rewards


def check_default_unchanged() -> bool:
    cfg = RewardConfig()
    if cfg.collision_per_step is not False:
        print("  default flipped: collision_per_step must default to False")
        return False
    col_total, _, terms, rewards = _run(cfg, [True] * 6)
    ok = (col_total == -10.0) and terms[0] and not any(terms[1:]) and all(
        r == 0.0 for r in rewards[1:])
    print(f"  6 colliding steps -> collision {col_total:+.1f} (want -10.0, "
          f"charged once), terminates at step "
          f"{terms.index(True) + 1 if any(terms) else None} (want 1)")
    return ok


def check_per_step_charges_dwell() -> bool:
    cfg = RewardConfig(collision=-0.2, collision_per_step=True,
                       terminate_on_collision=False)
    col_total, _, terms, _ = _run(cfg, [True] * 6)
    ok = abs(col_total - (-1.2)) < 1e-5 and not any(terms)
    print(f"  6 colliding steps -> collision {col_total:+.1f} (want -1.2 = "
          f"6 x -0.2), terminated={any(terms)} (want False)")
    return ok


def check_escape_closed() -> bool:
    """The measured bug, as an inequality.

    An agent stuck off-road for the rest of the episode chooses between paying
    the dwell bill and crashing now. Under job 36466's recipe crashing wins; the
    flag has to reverse that.
    """
    off_all = [True] * EPISODE_STEPS
    crash_at_5 = [False] * 4 + [True] * (EPISODE_STEPS - 4)

    # Job 36466: offroad dwell-charged, collision cheap and terminal.
    broken = RewardConfig(progress=0.0, offroad=-0.2, offroad_per_step=True,
                          terminate_on_offroad=False, collision=-1.0,
                          terminate_on_collision=True, route_reward=True,
                          progression_indicator=True, off_route_threshold_m=0.0,
                          off_route_penalty=0.0, goal_bonus=0.0)
    stay_broken = _run(broken, [False] * EPISODE_STEPS, off_all)[1]
    crash_broken = _run(broken, crash_at_5, off_all)[1]

    # The fix: collision dwell-charged too, nothing terminates but the goal.
    fixed = RewardConfig(progress=0.0, offroad=-0.2, offroad_per_step=True,
                         terminate_on_offroad=False, collision=-0.2,
                         collision_per_step=True, terminate_on_collision=False,
                         route_reward=True, progression_indicator=True,
                         off_route_threshold_m=0.0, off_route_penalty=0.0,
                         goal_bonus=0.0)
    stay_fixed = _run(fixed, [False] * EPISODE_STEPS, off_all)[1]
    crash_fixed = _run(fixed, crash_at_5, off_all)[1]

    ok = (crash_broken > stay_broken) and (crash_fixed <= stay_fixed)
    print(f"  job-36466 recipe: stay {stay_broken:+.1f} vs crash-at-5 "
          f"{crash_broken:+.1f} -> crashing pays {crash_broken - stay_broken:+.1f} "
          f"(want > 0, the bug)")
    print(f"  with collision_per_step: stay {stay_fixed:+.1f} vs crash-at-5 "
          f"{crash_fixed:+.1f} -> crashing pays {crash_fixed - stay_fixed:+.1f} "
          f"(want <= 0, the fix)")
    return ok


def check_no_absorbing_state_but_goal() -> bool:
    """With both per-step flags on, only reaching the goal can end an episode."""
    cfg = RewardConfig(offroad=-0.2, offroad_per_step=True,
                       terminate_on_offroad=False, collision=-0.2,
                       collision_per_step=True, terminate_on_collision=False)
    _, _, terms, _ = _run(cfg, [True] * 12, [True] * 12)
    _, _, goal_terms, _ = _run(cfg, [False] * 4, goal_dist=1.0)
    ok = (not any(terms)) and goal_terms[0]
    print(f"  12 steps colliding AND off-road -> terminated={any(terms)} "
          f"(want False); inside goal radius -> terminated={goal_terms[0]} "
          f"(want True)")
    return ok


def check_sum_to_return() -> bool:
    cfg = RewardConfig(collision=-0.2, collision_per_step=True,
                       terminate_on_collision=False)
    state = init_reward_state(init_goal_dist_b=jnp.array([50.0]), init_s_b=ZERO)
    reward_b, _, _, comps = reward_step(
        cfg, state, goal_dist_b=jnp.array([40.0]), overlap_b=ONE,
        offroad_b=ZERO, s_b=jnp.array([3.0]), lateral_b=ZERO,
    )
    parts = sum(float(comps[k][0]) for k in
                ("progress", "step_penalty", "action_penalty", "collision",
                 "offroad", "goal_bonus"))
    ok = abs(parts - float(reward_b[0])) < 1e-5
    print(f"  components sum {parts:+.4f} == reward {float(reward_b[0]):+.4f}")
    return ok


def main() -> int:
    checks = {
        "default unchanged (charged once, terminal)": check_default_unchanged,
        "per-step charges every colliding step": check_per_step_charges_dwell,
        "termination escape is closed": check_escape_closed,
        "only the goal absorbs": check_no_absorbing_state_but_goal,
        "components sum to the reward": check_sum_to_return,
    }
    results = {}
    for name, fn in checks.items():
        print(f"\n[{name}]")
        results[name] = bool(fn())
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    allp = all(results.values())
    print(f"\ncollision_per_step overall: {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
