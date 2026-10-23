# AnchorWM scorer: hard-scene mining, DAgger pools and diagnostics (babel)

Helper scripts for the screen -> pool -> bank -> fine-tune loop. They read results under
`/data/user_data/mineuih/ishan/` and run on a babel compute node (`/data` is not mounted on the login node), with
`PYTHONPATH` at the repo root.

| Script | What it does |
|---|---|
| `build_finetune_set.py` | `mining/hard1500/finetune_set.json`: every valid hard scene of the PDM-Closed screens (overlap / offroad after t=20, moving ego, logged ego clean under log replay), annotated with Diffusion-ES + PDMS outcomes only |
| `build_hard1500.py` | the earlier fixed 1500-scene version of the same set, with train / val / test split and planner tiers |
| `make_pool_r1.py`, `make_pool_r2.py` | DAgger round 1 / 2 mining pools (PLR-style failure score over the previous bank, fresh screened scenes, random moving scenes) |
| `bank_stats.py` | failure statistics of ES banks: failed scenes, unsafe candidates, mixed groups, pre-failure and near-miss groups |
| `fixable.py`, `fixable_windows.py` | could a perfect scorer have avoided a closed-loop failure: does the ES pool hold a safe candidate 0-3 / 3-5 / 5-8 s before it |
| `eval_scorers_bank.py` | offline scorer metrics (finetune.validate) of several checkpoints on one bank's val split |
| `agg_multi.py` | closed-loop outcome table + paired comparison of runs (`name=run_dir`) |

Pipeline pieces in the repo root: `slurm_pdm_screen.sh` (CPU screen), `slurm_targets_baseline.sh` (ES+PDMS / log replay on
a target list), `slurm_collect_anchorwm_bank.sh` (`TARGETS=` in-loop bank collection), `anchor_wm/slurm/babel/finetune.sh`
(`BANKS=`, focus sampling via `EXTRA_ARGS="--focus_frac ..."`), `slurm_es_anchorwm_valfail1568.sh` (`CKPT=` / `SCORER_CKPT=`).
