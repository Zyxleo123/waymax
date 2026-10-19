# Closed loop: ft_rule_hdg scorer vs baselines

Generated 2026-10-19 04:42. Run: `/zfsauton/scratch/ishanj/anchor_wm/closed_loop/fail1500_es_anchorwm_ft_rule_hdg_20261019_014026_best_reward_scorer_offbinary_seed0`

**1090 scenes done, target 1500 scenes** (partial); paired comparisons use the 1090 scenes every run has.

## all (n = 1090)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 76.0% | 5.8% | 18.0% | 99.4% | 0.23 m |
| MF + rule | 74.6% | 6.0% | 19.7% | 99.6% | 0.14 m |
| MF + v3 | 74.1% | 6.9% | 19.0% | 99.4% | 0.21 m |
| A2 + rule | 73.9% | 5.3% | 19.6% | 98.3% | 0.14 m |
| A2 scorer | 70.3% | 8.2% | 18.2% | 95.8% | 0.32 m |
| MF + rule (margin) | 75.9% | 6.3% | 17.1% | 98.4% | 0.19 m |
| A2 + rule (margin) | 75.9% | 6.1% | 16.7% | 98.0% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +1.4 | -0.4 to +3.0 | 49 / 34 |
| MF + v3 | +1.8 | +0.5 to +3.3 | 42 / 22 |
| A2 + rule | +2.1 | +0.6 to +3.9 | 54 / 31 |
| A2 scorer | +5.7 | +3.9 to +7.5 | 83 / 21 |
| MF + rule (margin) | +0.1 | -1.7 to +1.7 | 41 / 40 |
| A2 + rule (margin) | +0.1 | -1.5 to +1.7 | 34 / 33 |

## train (n = 780)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 76.3% | 5.8% | 17.7% | 99.4% | 0.23 m |
| MF + rule | 74.2% | 6.3% | 19.7% | 99.5% | 0.14 m |
| MF + v3 | 74.4% | 7.1% | 18.6% | 99.4% | 0.21 m |
| A2 + rule | 73.6% | 5.8% | 19.4% | 98.3% | 0.15 m |
| A2 scorer | 70.3% | 7.9% | 17.9% | 95.3% | 0.32 m |
| MF + rule (margin) | 75.5% | 6.7% | 16.8% | 98.1% | 0.19 m |
| A2 + rule (margin) | 75.5% | 6.7% | 16.3% | 97.8% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +2.1 | +0.1 to +4.1 | 37 / 21 |
| MF + v3 | +1.9 | +0.1 to +3.7 | 31 / 16 |
| A2 + rule | +2.7 | +0.8 to +4.6 | 39 / 18 |
| A2 scorer | +6.0 | +3.8 to +8.2 | 62 / 15 |
| MF + rule (margin) | +0.8 | -1.2 to +2.7 | 32 / 26 |
| A2 + rule (margin) | +0.8 | -1.0 to +2.6 | 28 / 22 |

## val (n = 154)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 74.7% | 4.5% | 20.1% | 98.7% | 0.26 m |
| MF + rule | 76.0% | 3.9% | 20.8% | 100.0% | 0.15 m |
| MF + v3 | 74.0% | 5.2% | 20.8% | 99.4% | 0.21 m |
| A2 + rule | 75.3% | 3.9% | 21.4% | 99.4% | 0.14 m |
| A2 scorer | 68.2% | 7.8% | 19.5% | 94.8% | 0.31 m |
| MF + rule (margin) | 76.0% | 5.2% | 19.5% | 99.4% | 0.17 m |
| A2 + rule (margin) | 76.6% | 3.9% | 19.5% | 98.7% | 0.22 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | -1.3 | -5.8 to +3.2 | 5 / 7 |
| MF + v3 | +0.6 | -2.6 to +3.9 | 4 / 3 |
| A2 + rule | -0.6 | -5.2 to +3.9 | 6 / 7 |
| A2 scorer | +6.5 | +1.9 to +11.0 | 12 / 2 |
| MF + rule (margin) | -1.3 | -5.8 to +2.6 | 4 / 6 |
| A2 + rule (margin) | -1.9 | -5.8 to +1.9 | 3 / 6 |

## test (n = 156)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 75.6% | 7.1% | 17.3% | 100.0% | 0.21 m |
| MF + rule | 75.0% | 6.4% | 18.6% | 100.0% | 0.13 m |
| MF + v3 | 73.1% | 7.7% | 19.2% | 100.0% | 0.18 m |
| A2 + rule | 73.7% | 4.5% | 19.2% | 97.4% | 0.12 m |
| A2 scorer | 72.4% | 9.6% | 17.9% | 99.4% | 0.33 m |
| MF + rule (margin) | 77.6% | 5.8% | 16.0% | 99.4% | 0.19 m |
| A2 + rule (margin) | 76.9% | 5.1% | 16.0% | 98.1% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +0.6 | -3.8 to +5.1 | 7 / 6 |
| MF + v3 | +2.6 | -1.3 to +6.4 | 7 / 3 |
| A2 + rule | +1.9 | -3.2 to +6.4 | 9 / 6 |
| A2 scorer | +3.2 | -1.3 to +7.7 | 9 / 4 |
| MF + rule (margin) | -1.9 | -6.4 to +1.9 | 5 / 8 |
| A2 + rule (margin) | -1.3 | -5.1 to +1.9 | 3 / 5 |
