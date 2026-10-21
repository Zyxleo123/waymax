# Diffusion-ES + world models + repair + learned rewards (branch `es_wm_anchor`)

Diffusion-ES (es_WM c0fded3) with mineuih's goal-conditioned diffusion policy. Each candidate plan
can be repaired against a world model's forecast before a learned reward ranks it. Every arm is
scored by es_WM's runner with Waymax metrics. This page covers the arms, the protocol, how to run
them, and the results so far (Babel, 2026-10-21).

## Components

| Part | What | Checkpoint (`$NEWROOT=/data/user_data/yixiz/autonlab`) |
|---|---|---|
| Policy | mineuih goal_cond_diffusion_policy, epoch 1000 (`latest` is a stale epoch-500 copy) | `$NEWROOT/zfsauton/scratch/mineuih/waymax_rs/new_checkpoints/train_diffusion/goal_cond_diffusion_policy/epoch_1000` |
| MF | Sagar's multi-future Gaussian WM, epoch 450 | `.../sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt` |
| AnchorWM | Ishan's joint WM + plan scorer, `ft_rule_hdg` (`anchor_wm/`, vendored from mini-hong/waymax_rs@ishan 5755663) | `.../ishanj/anchor_wm/finetune/ft_rule_hdg_20261019_014026/best.pt` |
| learned reward | Ishan's: the scorer head of the AnchorWM checkpoint | same file |
| disc | Sagar's WM-conditioned discriminator v3 | `.../sbellad/es_discriminator/safety_progress/safety_progress_v3_20261016_155456_86103/best.pt` |
| repair | o_am (no-rule argmin repair) of every candidate against the WM's modes, before ranking | in `planner/` |

Each learned scorer always runs its own WM internally: the learned reward uses AnchorWM, the disc a
frozen MF epoch 450. The planner's forecaster feeds only the repair. **Without repair, the WM
choice therefore changes nothing**, and "AnchorWM + learned reward" equals "MF + learned reward".

## Arms (`ES_ARMS_SEL`)

| Arm | Forecaster (repair) | Reward | Repair |
|---|---|---|---|
| `awm_sc_posthoc` | AnchorWM | learned reward | yes |
| `mf_disc_posthoc` | MF | disc | yes |
| `awm_disc_posthoc`, `mf_sc_posthoc` | AnchorWM / MF | disc / learned reward | yes (cross-over arms) |
| `awm_sc`, `mf_disc` | n/a | learned reward / disc | no |
| `expert` | n/a | n/a | the logged human through the same evaluator (the ceiling) |

Older arms (`cv*`, `wm_mf*`, `wmI_*`, `awm`, `awm_posthoc`, `diff*`, `gt`) use the handcrafted
reward, so they need lane graphs (see Caveats). Set `ES_MF_WM_CKPT` to the epoch-450 file above:
the default is `latest.pt`, which is a different model.

## Evaluation protocol

- **Views.** `log`: the other agents replay the log, non-reactive. Every `log` scene JSON also
  carries `idm_*` keys, which re-score the same executed ego path with evaluation-only stock IDM.
  The closed-loop `idm` view is stock Waymax IDM, and most of its drop comes from the NPC model
  itself: the human's own log falls from 98.70% to 92.08% on val 0-9. Use `idm_faithful`
  (branch `es_wm_idmfix`) for reactive results.
- **Success** = goal reached ∧ no overlap ∧ no off-road. The goal is the SDC's last valid logged
  position, reached within 2 m. Overlap and off-road are Waymax's `overlap` (all valid objects,
  pedestrians and cyclists included) and `offroad` (signed distance to the road edges).
- **Window** = t ∈ [10, goal step], inclusive (Ishan 4219b45 = mineuih f82488d). t=10 is the
  logged start pose every planner shares, and it counts. `*_oldwin` keys hold es_WM's original
  window (t ∈ [20, goal+20]).
- Planning starts at t=10 with 1 s of history. The world dt is the median (`ES_WORLD_DT_METHOD`).
  The ego state is set directly from the plan (`StateDynamics`, no kinematic model).
- Results on different GPUs differ at the mm level (XLA autotuning), the same size as run-to-run noise.

## Running

Submit from a compute node (`sbatch` must be on PATH). Everything lands under
`$NEWROOT/zfsauton/scratch/yixiz/waymax_rs/es_posthoc/runs/awm_<subset>_<stamp>/`.

```bash
cd ~/waymax_rs_es_anchor
E450=$NEWROOT/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt
ES_SUBSET=train2 ES_ARMS_SEL=awm_sc_posthoc,mf_disc_posthoc,expert ES_VIEWS=log ES_PACK=1 \
  ES_MF_WM_CKPT=$E450 ES_MEM_PER_CFG_GB=32 ES_TIME=12:00:00 \
  ES_PAIRS_SEL=awm_sc_posthoc:mf_disc_posthoc bash slurm/sim/submit_es_disc.sh
ES_SMOKE=1 bash slurm/sim/submit_es_disc.sh          # 5 scenes x 3 configs, one GPU job
```

- One array task per (view, arm, shard group), 8 CPU and an A6000. Peak RSS is 12-16 GB, so 32 GB
  is enough. A ~600-scene group takes about 6 h; the expert arm takes about 20 min.
- **Requeue-safe.** Finished scenes are skipped (`--start_from`), so after a preemption or time
  limit, resubmit the same `<root>/env` directory. A task requeued after it already finished shows a
  short `Elapsed` in sacct and does no work.
- Output is `<root>/log/<arm>/group_<g>/<tfrecord>.scenario_<idx>.json` (plus `.trajectory.npz`
  with per-step overlap and off-road). The compare job writes `<root>/compare.{json,md}`: rates,
  paired differences with a scene-bootstrap 95% CI, and discordant counts.
- `scripts/trainval_readout.py` pools train 0-1 and val 0-1 across two roots, with an
  expert-clean slice.

## Subsets (`ES_SUBSET`, under `es_posthoc/`)

| Subset | Scenes | Notes |
|---|---|---|
| `valfail1568` | 1568 | mineuih's new-policy failures over all 150 val shards (`womd_compact/valfail1568`); 2 groups |
| `val10all` | 2916 | every record of val shards 0-9; 5 groups × 2 shards (group_0 = shards 0-1, 595) |
| `train2` | 934 | every record of train shards 0-1 (455 + 479); 1 shard per group |
| `fail1500` | 1500 | 187 training shards. **In-sample for both learned scorers** (`--split_json` slices it) |

`val10all`, `valfail1568` and `train2` pad a shard's short last batch instead of dropping it
(`--pad_short_batches`).

## Results (log replay)

### valfail1568: 1568 scenes, the policy's own failures

| Arm | All 1568 | Expert-clean 1187 |
|---|---|---|
| AnchorWM + learned reward + repair | **77.17%** | 98.65% |
| MF + learned reward + repair | 77.17% | 98.65% |
| AnchorWM + disc + repair | 76.66% | 97.89% |
| MF + disc + repair | 76.40% | 97.98% |
| learned reward, no repair | 74.81% | 96.55% |
| disc, no repair | 72.39% | 93.26% |
| expert (logged human) | 75.70% | — |

- **Repair:** +2.36 [+1.34, +3.38] points on the learned reward and +4.02 [+2.87, +5.23] on the disc.
- **Without repair**, the learned reward beats the disc by +2.42 [+1.21, +3.64].
- **With repair**, every WM or scorer swap is inside the CI.
- **The human fails 381 of these scenes** (295 off-road, 87 overlap), mostly briefly (off-road
  median 0.4 s). 184 of them (11.7% of the set) already violate at t=10, so no planner can win them.
- Roots: `runs/awm_valfail1568_20261019_212050_533784` (`compare.md` for the four native arms,
  `compare6.md` for all six); expert in `runs/awm_valfail1568_20261020_144643_3257341`.

### val 0-9: 2916 scenes

Done for `awm_sc_posthoc`, `mf_disc_posthoc` and `expert` (2916/2916 each), root
`runs/awm_val10all_20261020_150640_3272023`; the table is pending. The expert alone scores 98.70%
(t ≥ 11: 99.18%; overlap 0.34%, off-road 0.96%).

### Train 0-1 + val 0-1: 1529 scenes

| Arm | Train 934 | Val 595 | All 1529 | Expert-clean 1519 | Non-trivial 1119 |
|---|---|---|---|---|---|
| AnchorWM + learned reward + repair | 99.25% | 99.16% | **99.22%** | 99.74% | 99.11% |
| MF + disc + repair | 99.14% | 99.16% | **99.15%** | 99.67% | 99.02% |
| expert (logged human) | 99.25% | 99.50% | 99.35% | — | 99.29% |

- The paired difference is +0.07 [−0.20, +0.39] (3 vs 2 discordant scenes): a tie.
- There is no train-vs-val gap.
- "Non-trivial" drops the 410 scenes (27%) where the SDC starts within 2 m of its goal.
- None of the 1529 scenario ids is in the disc's split file.
- Roots: `runs/awm_train2_20261020_232409_1678722` (`compare_trainval.md`); val from `val10all` group_0.

## Caveats

- **Training data of the learned scorers.** Both were fine-tuned on `fail1500_plus_womd_train_s02_11`,
  so fail1500 is partly in-sample. valfail1568, val 0-9 and train 0-1 are held out. The learned
  reward was trained on candidates from sbellad's policy and MF, so it sees a shifted
  distribution with mineuih's policy.
- **Lane graphs.** Babel's zips (`lane_graphs_babel`, rebuilt by `scripts/rebuild_lane_graphs.py`
  on `posthoc-opt-es`) cover only the val10, fail1500 and valfail1568 targets. A scene missing
  from an existing zip crashes the runner (`set_lane_graph(None)`). So `train2` points every config
  at an empty directory, and only learned-reward arms run there (`--lane_graph_optional`; the
  handcrafted scorer raises if it is called). Neither the learned-reward arms nor the evaluation
  read lane graphs: the red-light check is commented out.
- **The expert's result depends on the evaluator, not on WOMD.** On valfail1568 the window alone
  moves it from 75.70% (t ≥ 10) to 84.50% (t ≥ 11) or 78.00% (old window). Numbers from another
  evaluator, e.g. GPUDrive (per-agent rates, off-road as touching a road edge, no pedestrians or
  cyclists), are not comparable without rescoring.
