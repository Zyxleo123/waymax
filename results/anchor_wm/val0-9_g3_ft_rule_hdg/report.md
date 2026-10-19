# Closed loop: ft_rule_hdg scorer vs baselines

Generated 2026-10-19 04:59. Run: `/zfsauton/scratch/ishanj/anchor_wm/closed_loop_val/val0-9_es_anchorwm_ft_rule_hdg_20261019_014026_best_reward_scorer_offmargin_tiebreak`

**580 scenes done, target 10 shards** (partial); paired comparisons use the 580 scenes every run has.

## all (n = 580)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 99.3% | 0.3% | 0.3% | 100.0% | 0.12 m |
| MF+rule | 99.3% | 0.3% | 0.3% | 100.0% | 0.13 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.0 | +0.0 to +0.0 | 0 / 0 |

## shard 00000 (n = 230)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 99.6% | 0.0% | 0.4% | 100.0% | 0.12 m |
| MF+rule | 99.6% | 0.0% | 0.4% | 100.0% | 0.11 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.0 | +0.0 to +0.0 | 0 / 0 |

## shard 00001 (n = 205)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 99.0% | 0.5% | 0.5% | 100.0% | 0.13 m |
| MF+rule | 99.0% | 0.5% | 0.5% | 100.0% | 0.13 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.0 | +0.0 to +0.0 | 0 / 0 |

## shard 00002 (n = 145)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 99.3% | 0.7% | 0.0% | 100.0% | 0.12 m |
| MF+rule | 99.3% | 0.7% | 0.0% | 100.0% | 0.15 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.0 | +0.0 to +0.0 | 0 / 0 |
