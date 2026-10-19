# Closed loop: ft_rule_hdg scorer vs baselines

Generated 2026-10-19 05:43. Run: `/zfsauton/scratch/ishanj/anchor_wm/closed_loop/fail1500_es_anchorwm_ft_rule_hdg_20261019_014026_best_reward_scorer_offbinary_seed0`

**1480 scenes done, target 1500 scenes** (partial); paired comparisons use the 1480 scenes every run has.

## all (n = 1480)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 76.2% | 5.9% | 17.7% | 99.5% | 0.24 m |
| MF + rule | 74.9% | 5.9% | 19.5% | 99.7% | 0.14 m |
| MF + v3 | 74.5% | 7.1% | 18.4% | 99.6% | 0.21 m |
| A2 + rule | 74.1% | 5.2% | 19.6% | 98.6% | 0.14 m |
| A2 scorer | 70.2% | 8.3% | 18.0% | 95.4% | 0.32 m |
| MF + rule (margin) | 75.9% | 6.5% | 16.8% | 98.3% | 0.19 m |
| A2 + rule (margin) | 75.9% | 6.1% | 16.6% | 97.9% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +1.4 | -0.1 to +2.8 | 65 / 45 |
| MF + v3 | +1.7 | +0.5 to +2.8 | 52 / 27 |
| A2 + rule | +2.1 | +0.7 to +3.5 | 75 / 44 |
| A2 scorer | +6.0 | +4.5 to +7.5 | 115 / 26 |
| MF + rule (margin) | +0.3 | -1.1 to +1.6 | 55 / 51 |
| A2 + rule (margin) | +0.3 | -1.0 to +1.6 | 48 / 44 |

## train (n = 1044)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 76.7% | 6.1% | 17.0% | 99.5% | 0.25 m |
| MF + rule | 74.7% | 6.6% | 18.9% | 99.6% | 0.14 m |
| MF + v3 | 74.8% | 7.5% | 17.7% | 99.5% | 0.22 m |
| A2 + rule | 74.0% | 5.8% | 18.9% | 98.5% | 0.15 m |
| A2 scorer | 70.4% | 8.6% | 17.1% | 94.9% | 0.32 m |
| MF + rule (margin) | 75.6% | 7.3% | 16.2% | 98.1% | 0.20 m |
| A2 + rule (margin) | 75.7% | 6.9% | 15.9% | 97.6% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +2.0 | +0.4 to +3.6 | 48 / 27 |
| MF + v3 | +1.9 | +0.6 to +3.3 | 38 / 18 |
| A2 + rule | +2.7 | +1.1 to +4.3 | 54 / 26 |
| A2 scorer | +6.3 | +4.5 to +8.2 | 84 / 18 |
| MF + rule (margin) | +1.1 | -0.5 to +2.7 | 43 / 31 |
| A2 + rule (margin) | +1.1 | -0.4 to +2.6 | 39 / 28 |

## val (n = 218)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 73.9% | 5.0% | 20.6% | 99.1% | 0.24 m |
| MF + rule | 75.7% | 3.2% | 21.6% | 100.0% | 0.15 m |
| MF + v3 | 72.9% | 6.0% | 21.1% | 99.5% | 0.20 m |
| A2 + rule | 74.3% | 4.1% | 22.0% | 99.5% | 0.14 m |
| A2 scorer | 67.9% | 7.3% | 20.6% | 95.0% | 0.32 m |
| MF + rule (margin) | 76.1% | 4.6% | 19.7% | 98.6% | 0.17 m |
| A2 + rule (margin) | 75.7% | 4.6% | 19.7% | 99.1% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | -1.8 | -5.5 to +1.8 | 7 / 11 |
| MF + v3 | +0.9 | -1.8 to +3.7 | 6 / 4 |
| A2 + rule | -0.5 | -4.6 to +3.2 | 9 / 10 |
| A2 scorer | +6.0 | +2.3 to +9.6 | 16 / 3 |
| MF + rule (margin) | -2.3 | -6.0 to +0.9 | 5 / 10 |
| A2 + rule (margin) | -1.8 | -5.0 to +1.4 | 5 / 9 |

## test (n = 218)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 76.1% | 5.5% | 18.3% | 99.5% | 0.20 m |
| MF + rule | 74.8% | 5.0% | 20.2% | 99.5% | 0.13 m |
| MF + v3 | 74.8% | 6.4% | 18.8% | 100.0% | 0.19 m |
| A2 + rule | 74.3% | 3.2% | 20.6% | 98.2% | 0.12 m |
| A2 scorer | 71.6% | 7.8% | 19.3% | 98.2% | 0.28 m |
| MF + rule (margin) | 77.5% | 4.6% | 17.0% | 99.1% | 0.17 m |
| A2 + rule (margin) | 77.5% | 4.1% | 16.5% | 98.2% | 0.17 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +1.4 | -2.3 to +5.0 | 10 / 7 |
| MF + v3 | +1.4 | -1.8 to +4.6 | 8 / 5 |
| A2 + rule | +1.8 | -2.3 to +6.0 | 12 / 8 |
| A2 scorer | +4.6 | +0.5 to +8.7 | 15 / 5 |
| MF + rule (margin) | -1.4 | -5.0 to +2.3 | 7 / 10 |
| A2 + rule (margin) | -1.4 | -4.6 to +1.8 | 4 / 7 |
