# Closed loop: ft_rule_hdg scorer vs baselines

Generated 2026-10-19 06:03. Run: `/zfsauton/scratch/ishanj/anchor_wm/closed_loop/fail1500_es_anchorwm_ft_rule_hdg_20261019_014026_best_reward_scorer_offbinary_seed0`

**1500 scenes done, target 1500 scenes** (final); paired comparisons use the 1500 scenes every run has.

## all (n = 1500)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 76.1% | 6.0% | 17.7% | 99.5% | 0.24 m |
| MF + rule | 74.7% | 6.0% | 19.5% | 99.7% | 0.14 m |
| MF + v3 | 74.5% | 7.1% | 18.3% | 99.6% | 0.21 m |
| A2 + rule | 74.0% | 5.3% | 19.6% | 98.6% | 0.14 m |
| A2 scorer | 70.2% | 8.4% | 17.9% | 95.5% | 0.32 m |
| MF + rule (margin) | 75.9% | 6.6% | 16.8% | 98.3% | 0.19 m |
| A2 + rule (margin) | 75.9% | 6.2% | 16.5% | 97.9% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +1.3 | -0.1 to +2.6 | 66 / 46 |
| MF + v3 | +1.5 | +0.4 to +2.7 | 52 / 29 |
| A2 + rule | +2.1 | +0.6 to +3.5 | 76 / 45 |
| A2 scorer | +5.9 | +4.3 to +7.4 | 115 / 27 |
| MF + rule (margin) | +0.2 | -1.1 to +1.5 | 56 / 53 |
| A2 + rule (margin) | +0.1 | -1.1 to +1.5 | 48 / 46 |

## train (n = 1058)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 76.7% | 6.1% | 16.9% | 99.5% | 0.25 m |
| MF + rule | 74.7% | 6.7% | 18.8% | 99.6% | 0.14 m |
| MF + v3 | 75.0% | 7.5% | 17.6% | 99.5% | 0.21 m |
| A2 + rule | 74.0% | 6.0% | 18.8% | 98.5% | 0.14 m |
| A2 scorer | 70.6% | 8.6% | 17.0% | 95.0% | 0.32 m |
| MF + rule (margin) | 75.6% | 7.4% | 16.1% | 98.1% | 0.20 m |
| A2 + rule (margin) | 75.8% | 6.9% | 15.8% | 97.6% | 0.21 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +2.1 | +0.5 to +3.8 | 49 / 27 |
| MF + v3 | +1.8 | +0.5 to +3.2 | 38 / 19 |
| A2 + rule | +2.7 | +1.1 to +4.4 | 55 / 26 |
| A2 scorer | +6.1 | +4.3 to +8.0 | 84 / 19 |
| MF + rule (margin) | +1.1 | -0.5 to +2.8 | 44 / 32 |
| A2 + rule (margin) | +0.9 | -0.7 to +2.5 | 39 / 29 |

## val (n = 221)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 73.3% | 5.0% | 21.3% | 99.1% | 0.24 m |
| MF + rule | 75.1% | 3.2% | 22.2% | 100.0% | 0.15 m |
| MF + v3 | 72.4% | 5.9% | 21.7% | 99.5% | 0.20 m |
| A2 + rule | 73.8% | 4.1% | 22.6% | 99.5% | 0.14 m |
| A2 scorer | 67.4% | 7.2% | 21.3% | 95.0% | 0.32 m |
| MF + rule (margin) | 75.6% | 4.5% | 20.4% | 98.6% | 0.17 m |
| A2 + rule (margin) | 75.1% | 4.5% | 20.4% | 99.1% | 0.20 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | -1.8 | -5.9 to +1.8 | 7 / 11 |
| MF + v3 | +0.9 | -1.8 to +3.6 | 6 / 4 |
| A2 + rule | -0.5 | -4.5 to +3.2 | 9 / 10 |
| A2 scorer | +5.9 | +2.3 to +10.0 | 16 / 3 |
| MF + rule (margin) | -2.3 | -5.9 to +0.9 | 5 / 10 |
| A2 + rule (margin) | -1.8 | -5.0 to +1.4 | 5 / 9 |

## test (n = 221)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 75.6% | 6.3% | 18.1% | 99.5% | 0.21 m |
| MF + rule | 74.7% | 5.4% | 19.9% | 99.5% | 0.13 m |
| MF + v3 | 74.7% | 6.8% | 18.6% | 100.0% | 0.20 m |
| A2 + rule | 74.2% | 3.6% | 20.4% | 98.2% | 0.12 m |
| A2 scorer | 71.0% | 8.6% | 19.0% | 98.2% | 0.28 m |
| MF + rule (margin) | 77.4% | 5.0% | 16.7% | 99.1% | 0.17 m |
| A2 + rule (margin) | 77.4% | 4.5% | 16.3% | 98.2% | 0.17 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +0.9 | -3.2 to +4.5 | 10 / 8 |
| MF + v3 | +0.9 | -2.7 to +4.1 | 8 / 6 |
| A2 + rule | +1.4 | -2.7 to +5.4 | 12 / 9 |
| A2 scorer | +4.5 | +0.5 to +8.6 | 15 / 5 |
| MF + rule (margin) | -1.8 | -5.4 to +1.8 | 7 / 11 |
| A2 + rule (margin) | -1.8 | -5.0 to +1.4 | 4 / 8 |
