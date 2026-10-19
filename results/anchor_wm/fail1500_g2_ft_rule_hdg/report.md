# G2 closed loop: ft_rule_hdg scorer vs baselines

Generated 2026-10-19 03:42. Run: `/zfsauton/scratch/ishanj/anchor_wm/closed_loop/fail1500_es_anchorwm_ft_rule_hdg_20261019_014026_best_reward_scorer_offbinary_seed0`

**470 / 1500 scenes done** (partial); paired comparisons use the 470 scenes every run has.

## all (n = 470)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 74.5% | 6.6% | 17.9% | 98.7% | 0.23 m |
| MF + rule | 72.8% | 6.6% | 20.4% | 99.4% | 0.13 m |
| MF + v3 | 73.0% | 7.0% | 19.4% | 98.9% | 0.21 m |
| A2 + rule | 72.6% | 5.5% | 19.8% | 97.9% | 0.14 m |
| A2 scorer | 67.9% | 9.1% | 18.5% | 94.9% | 0.32 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +1.7 | -1.1 to +4.3 | 24 / 16 |
| MF + v3 | +1.5 | -0.6 to +3.8 | 18 / 11 |
| A2 + rule | +1.9 | -0.6 to +4.5 | 25 / 16 |
| A2 scorer | +6.6 | +3.8 to +9.6 | 41 / 10 |

## train (n = 349)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 75.4% | 5.2% | 18.3% | 98.9% | 0.25 m |
| MF + rule | 72.2% | 6.3% | 21.2% | 99.1% | 0.14 m |
| MF + v3 | 73.4% | 6.3% | 19.8% | 98.9% | 0.22 m |
| A2 + rule | 72.2% | 5.4% | 20.1% | 97.7% | 0.15 m |
| A2 scorer | 68.5% | 7.7% | 18.9% | 94.6% | 0.33 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | +3.2 | +0.3 to +6.0 | 19 / 8 |
| MF + v3 | +2.0 | -0.6 to +4.6 | 14 / 7 |
| A2 + rule | +3.2 | +0.3 to +6.0 | 19 / 8 |
| A2 scorer | +6.9 | +3.4 to +10.3 | 32 / 8 |

## val (n = 60)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 71.7% | 6.7% | 20.0% | 96.7% | 0.22 m |
| MF + rule | 75.0% | 5.0% | 20.0% | 100.0% | 0.12 m |
| MF + v3 | 73.3% | 5.0% | 20.0% | 98.3% | 0.21 m |
| A2 + rule | 75.0% | 3.3% | 21.7% | 100.0% | 0.14 m |
| A2 scorer | 65.0% | 10.0% | 18.3% | 93.3% | 0.32 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | -3.3 | -11.7 to +5.0 | 2 / 4 |
| MF + v3 | -1.7 | -8.3 to +3.3 | 1 / 2 |
| A2 + rule | -3.3 | -11.7 to +5.0 | 2 / 4 |
| A2 scorer | +6.7 | +0.0 to +15.0 | 5 / 1 |

## test (n = 61)

| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |
|---|---|---|---|---|---|
| ft_rule_hdg scorer | 72.1% | 14.8% | 13.1% | 100.0% | 0.18 m |
| MF + rule | 73.8% | 9.8% | 16.4% | 100.0% | 0.12 m |
| MF + v3 | 70.5% | 13.1% | 16.4% | 100.0% | 0.13 m |
| A2 + rule | 72.1% | 8.2% | 16.4% | 96.7% | 0.11 m |
| A2 scorer | 67.2% | 16.4% | 16.4% | 98.4% | 0.33 m |

| ft_rule_hdg scorer minus | Δ success (pts) | 95% CI | Fixed / broken |
|---|---|---|---|
| MF + rule | -1.6 | -9.8 to +6.6 | 3 / 4 |
| MF + v3 | +1.6 | -4.9 to +8.2 | 3 / 2 |
| A2 + rule | +0.0 | -8.2 to +9.8 | 4 / 4 |
| A2 scorer | +4.9 | -1.6 to +11.5 | 4 / 1 |
