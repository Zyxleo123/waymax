# Closed loop: ft_rule_hdg scorer vs baselines

Generated 2026-10-19 05:59. Run: `/zfsauton/scratch/ishanj/anchor_wm/closed_loop_val/val0-9_es_anchorwm_ft_rule_hdg_20261019_014026_best_reward_scorer_offmargin_tiebreak`

**1406 scenes done, target 10 shards** (partial); paired comparisons use the 1406 scenes every run has.

## all (n = 1406)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 98.7% | 0.4% | 0.9% | 99.9% | 0.13 m |
| MF+rule | 98.6% | 0.4% | 0.8% | 99.6% | 0.12 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.1 | -0.2 to +0.5 | 4 / 2 |

## shard 00000 (n = 286)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 99.7% | 0.0% | 0.3% | 100.0% | 0.13 m |
| MF+rule | 99.3% | 0.0% | 0.3% | 99.7% | 0.11 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.3 | +0.0 to +1.0 | 1 / 0 |

## shard 00001 (n = 309)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 98.7% | 1.0% | 0.3% | 100.0% | 0.13 m |
| MF+rule | 98.7% | 0.6% | 0.3% | 99.7% | 0.13 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.0 | -1.0 to +1.0 | 1 / 1 |

## shard 00002 (n = 306)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 99.3% | 0.3% | 0.3% | 100.0% | 0.14 m |
| MF+rule | 99.0% | 0.3% | 0.3% | 99.7% | 0.14 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.3 | +0.0 to +1.0 | 1 / 0 |

## shard 00003 (n = 225)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 97.3% | 0.4% | 1.8% | 99.6% | 0.13 m |
| MF+rule | 96.9% | 0.9% | 1.8% | 99.1% | 0.13 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.4 | +0.0 to +1.3 | 1 / 0 |

## shard 00004 (n = 165)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 98.8% | 0.0% | 1.2% | 100.0% | 0.11 m |
| MF+rule | 99.4% | 0.0% | 0.6% | 100.0% | 0.11 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | -0.6 | -1.8 to +0.0 | 0 / 1 |

## shard 00005 (n = 105)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 97.1% | 0.0% | 2.9% | 100.0% | 0.13 m |
| MF+rule | 97.1% | 0.0% | 2.9% | 100.0% | 0.12 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.0 | +0.0 to +0.0 | 0 / 0 |

## shard 00006 (n = 10)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 100.0% | 0.0% | 0.0% | 100.0% | 0.26 m |
| MF+rule | 100.0% | 0.0% | 0.0% | 100.0% | 0.12 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF+rule | +0.0 | +0.0 to +0.0 | 0 / 0 |
