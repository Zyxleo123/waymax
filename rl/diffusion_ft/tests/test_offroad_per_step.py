"""`offroad_per_step`: charge the offroad penalty by dwell, not once per episode.

Terminating on offroad (the V-Max parity recipe of job 34866) stopped the
corner-cut farming but collapsed collection to a one-step bandit: at every sigma
the exploration noise leaves the road on the first world step often enough that
the whole batch absorbs immediately, and the sweep in
``sigma_sweep/20260821_192513`` found no noise level that keeps the episode alive
AND leaves the candidate-KL gate meaningful (target_kl ~ 1/sigma^2).

The per-step charge is the other way out of that bind: offroad does not end the
episode, so a noisy rollout still yields its full replan budget of signal, but
each off-road step pays, so a long excursion costs strictly more than a clipped
corner and the excursion still does not pay for itself.

This asserts:

* the default config is unchanged -- offroad charged exactly once, non-terminal
  (so every launch that does not set the flag is byte-identical to before);
* with the flag on, N off-road steps cost N x `offroad` and nothing terminates;
* cost is monotone in dwell (a 2-step clip < a 24-step excursion);
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


def _run(cfg: RewardConfig, off_seq, *, goal_dist: float = 50.0):
    """Step the reward through a fixed offroad pattern; ego holds still."""
    state = init_reward_state(init_goal_dist_b=jnp.array([goal_dist]),
                             init_s_b=ZERO)
    total_offroad, terminated, rewards = 0.0, [], []
    for off in off_seq:
        reward_b, state, term_b, comps = reward_step(
            cfg, state,
            goal_dist_b=jnp.array([goal_dist]),
            overlap_b=ZERO,
            offroad_b=(ONE if off else ZERO),
            s_b=ZERO, lateral_b=ZERO,
        )
        total_offroad += float(comps["offroad"][0])
        terminated.append(bool(term_b[0]))
        rewards.append(float(reward_b[0]))
    return total_offroad, terminated, rewards


def check_default_unchanged() -> bool:
    cfg = RewardConfig()
    if cfg.offroad_per_step is not False:
        print("  default flipped: offroad_per_step must default to False")
        return False
    total, terms, _ = _run(cfg, [True] * 6)
    ok = (total == -5.0) and not any(terms)
    print(f"  6 offroad steps -> total {total:+.1f} (want -5.0, charged once), "
          f"terminated={any(terms)} (want False)")
    return ok


def check_per_step_charges_dwell() -> bool:
    cfg = RewardConfig(offroad=-0.2, offroad_per_step=True,
                       terminate_on_offroad=False)
    total, terms, _ = _run(cfg, [True] * 6)
    ok = abs(total - (-1.2)) < 1e-5 and not any(terms)
    print(f"  6 offroad steps -> total {total:+.1f} (want -1.2 = 6 x -0.2), "
          f"terminated={any(terms)} (want False)")
    return ok


def check_terminating_variant_ends_episode() -> bool:
    """The config this replaces: one charge, then the episode is frozen."""
    cfg = RewardConfig(offroad=-1.0, terminate_on_offroad=True)
    total, terms, rewards = _run(cfg, [True] * 6)
    ok = terms[0] and not any(terms[1:]) and total == -1.0 and all(
        r == 0.0 for r in rewards[1:])
    print(f"  terminates at step {terms.index(True) + 1 if any(terms) else None} "
          f"(want 1), total {total:+.1f} (want -1.0), post-terminal rewards "
          f"{rewards[1:]} (want all 0.0)")
    return ok


def check_monotone_in_dwell() -> bool:
    cfg = RewardConfig(offroad=-0.2, offroad_per_step=True,
                       terminate_on_offroad=False)
    clip = _run(cfg, [True, True] + [False] * 22)[0]
    excursion = _run(cfg, [True] * 24)[0]
    ok = clip > excursion
    print(f"  2-step clip {clip:+.1f} vs 24-step excursion {excursion:+.1f} "
          f"(want the clip to cost less)")
    return ok


def check_sum_to_return() -> bool:
    cfg = RewardConfig(offroad=-0.2, offroad_per_step=True,
                       terminate_on_offroad=False)
    state = init_reward_state(init_goal_dist_b=jnp.array([50.0]), init_s_b=ZERO)
    reward_b, _, _, comps = reward_step(
        cfg, state, goal_dist_b=jnp.array([40.0]), overlap_b=ZERO,
        offroad_b=ONE, s_b=jnp.array([3.0]), lateral_b=ZERO,
    )
    parts = sum(float(comps[k][0]) for k in
                ("progress", "step_penalty", "action_penalty", "collision",
                 "offroad", "goal_bonus"))
    ok = abs(parts - float(reward_b[0])) < 1e-5
    print(f"  components sum {parts:+.4f} == reward {float(reward_b[0]):+.4f}")
    return ok


def main() -> int:
    checks = {
        "default unchanged (charged once, non-terminal)": check_default_unchanged,
        "per-step charges every off-road step": check_per_step_charges_dwell,
        "terminating variant still ends the episode": check_terminating_variant_ends_episode,
        "cost monotone in dwell": check_monotone_in_dwell,
        "components sum to the reward": check_sum_to_return,
    }
    results = {}
    for name, fn in checks.items():
        print(f"\n[{name}]")
        results[name] = bool(fn())
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    allp = all(results.values())
    print(f"\noffroad_per_step overall: {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
