# waymax_rs — ES planning with learned world models

Closed-loop Waymax simulation of an ego planner (Diffusion-ES or Flow-Matching-ES)
whose candidate plans are scored against a forecast of the other agents:
constant velocity (CV), privileged ground truth (GT), or a learned world model (WM).

- [1. Setup](#1-setup)
- [2. Repository layout](#2-repository-layout)
- [3. Data and scene subsets](#3-data-and-scene-subsets)
- [4. Running simulations](#4-running-simulations)
- [5. Outputs](#5-outputs)
- [6. Experiment log](#6-experiment-log)
- [7. Training](#7-training)
- [8. World models](#8-world-models)
- [9. Analysis and evaluation](#9-analysis-and-evaluation)
- [10. Cluster notes](#10-cluster-notes)
- [11. Lane graphs and goal-reaching (original setup)](#11-lane-graphs-and-goal-reaching-original-setup)

---

## 1. Setup

```bash
cd /data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
source /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3/bin/activate /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs
export PYTHONPATH=$PWD:$PYTHONPATH
```

`env.yml` has the full environment (JAX/Flax, Waymax, PyTorch, ...). The Slurm
scripts activate the environment themselves.

---

## 2. Repository layout

```text
slurm/
  sim/
    run_es.sh                   the only simulation job body (all logic lives here)
    trainval.sh                 submit runs on WOMD train+val shards 00/01
    fail1500.sh                 submit runs on the fail1500 subset
    probe250.sh                 submit runs on the 250-scene probe subset (+ "build")
    _submit.sh                  shared submitter (comma lists, --dry_run, --exclude)
    fail1500_gt_filter.sh       re-score runs on the GT-filtered fail1500 scenes
    fail1500_gt_filter_reeval.sh  build the GT filter and re-evaluate
  train/
    det_wm.sh  mf_wm.sh         world-model training
    diffusion.sh  vla.sh        policy training
legacy/                         pre-reorganisation scripts, kept for reference (see §6)
  sim/  train/  analysis/  annotation_vla/  docs/

simulation/run_simulation.py    closed-loop simulator entry point
planner/                        Diffusion-ES and FM-ES planners
scores/                         collision (SAT) / offroad / goal scoring
model/world_model/              deterministic + Gaussian WM inference ("production" backend)
train_deterministic_world_model/  deterministic WM training/eval code
multiFuture_GaussianWorldModel/   multi-future Gaussian WM (training, inference, evaluation)
train_diffusion/  train_vla/    policy training code
scripts/                        subset builders, filters, analysis tools
viz/                            better Waymax visualisation (import viz as visualization)
```

---

## 3. Data and scene subsets

| Item | Path |
|---|---|
| WOMD tf_example | `/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/{training,validation}` |
| Lane graphs | `/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs` |
| Sim-state cache (WM training) | `/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz` |

The old WOMD copy at `/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd` was deleted on 2026-10-02.
Several Python scripts still default to it — always pass the new path
(`--tfrecord_dir`, `--tf_root`, `--tfrecord_path`) when calling them directly.

**Subsets** (`--subset` in the simulation scripts):

| Subset | Scenes | Where | Notes |
|---|---|---|---|
| `trainval` | all scenes of training shards 00000+00001 and validation shards 00000+00001 | symlinks created per run under the exp root | both splits run inside one job |
| `fail1500` | 1500 scenes where FM-ES failed (offroad 658, overlap 679, goal 139, both 24), 187 shards | `/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0` | main benchmark; GT-filtered variant n=1102 |
| `probe250` | 250 scenes sampled from fail1500 | `/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/fm_hparam_probe_n250_seed0` | FM hyper-parameter probe; `num_worlds` defaults to 1 |

Subset tfrecords are symlinks into the WOMD folder. Every job checks at start
that all shards are readable on its node and aborts otherwise (a node without
the `/data/user_data/yixiz/autonlab/zfsauton/datasets` mount used to finish with 0 scenes and print `DONE`).

---

## 4. Running simulations

Submit from **rhea or lov4** (sbatch is denied on lov1). Each entry script takes
named flags; comma-separated values expand into one job per combination, and
duplicates (same run name) are submitted once.

```bash
bash slurm/sim/fail1500.sh --planner fm --forecast cv,wm_det,wm_mf --it 6 --rs 40
bash slurm/sim/fail1500.sh --planner diffusion --forecast wm_mf --mf_collision gaussian_uniform,gaussian_learned --M 4,8,16
bash slurm/sim/trainval.sh --planner diffusion --forecast cv,gt,wm_det,wm_mf
bash slurm/sim/probe250.sh --planner fm --forecast gt --t_start 0.2,0.5,0.8
```

Always preview first — `--dry_run` prints every job's run name, output dir and
exact `run_simulation.py` command without submitting:

```bash
bash slurm/sim/fail1500.sh --planner fm --forecast wm_mf --it 6 --rs 40 --dry_run
```

### Flags

| Flag | Values / default | Meaning |
|---|---|---|
| `--planner` | `diffusion` \| `fm` | Diffusion-ES or FM-ES |
| `--forecast` | `cv` \| `gt` \| `wm_det` \| `wm_mf` \| `wm_gau` \| `wm_gau_mean` | other-agent forecast (`wm` = `wm_det`) |
| `--mf_mode` | `expected_independent` (default) \| `top1` \| `worst_case_independent` | multi-future WM scoring |
| `--mf_collision` | `mean_uniform` \| `mean_learned` \| `gaussian_uniform` \| `gaussian_learned` | MF collision ablation (fail1500 only) |
| `--M` | 8 | samples per mode for `gaussian_*` |
| `--mf_seed` | = `--seed` | Gaussian sampling seed (combined with WOMD scenario id + timestep) |
| `--ablation_subdir` | — | e.g. `_smoke_n10` for small ablation test runs |
| `--seed` | 0 | planner seed |
| `--num_scenarios` | 100000 (all) | cap scenes, e.g. 10 for a smoke test |
| `--num_worlds` | 5 (probe250: 1) | scenes simulated in parallel |
| `--it` / `--pop` / `--elite` | 4 / 64 / 8 | ES iterations / population / elites |
| `--rs` | 20 | FM resample ODE steps |
| `--t_start` | 0.2 | FM resample start time (written into names verbatim, e.g. `0.20`) |
| `--fm_steps` / `--solver` | 50 / `euler` | FM full-sample ODE steps / solver |
| `--resample_timesteps` | 5 | Diffusion resample timesteps |
| `--wm_refresh` | 10 | WM refresh interval (steps) |
| `--wm_ckpt` | see §8 | override the WM checkpoint |
| `--num_collision_worlds` | 8 | `wm_gau` Monte-Carlo samples |
| `--tag_suffix` | — | probe250: appended to the run tag (Phase-2 config name) |
| `--run_suffix` | — | appended to any run name |
| `--probe_n` / `--probe_seed` | 250 / 0 | probe subset to use |
| `--dry_run` | — | print jobs, submit nothing |
| `--exclude` | — | `sbatch --exclude`, e.g. `gpu26,gpu31` |

Fixed in all runs: ego replans every 10 steps, WM device `auto`, collision
sample seed 0. A crashed run resumes into the same `simulation_*` folder (up to
3 retries) and skips finished scenes.

You can also run one job directly: `sbatch slurm/sim/run_es.sh --subset ... --planner ... --forecast ...`,
or inspect it with `bash slurm/sim/run_es.sh ... --print_cmd`.

---

## 5. Outputs

Run names and folders are the same as before the reorganisation, so existing
results remain comparable.

| Subset / planner / forecast | Output dir |
|---|---|
| fail1500, FM-ES, any forecast | `Flow_matching_sim/es_fm_<forecast>_fail1500_it<N>_rs<R>_p<P>_e<E>/` |
| fail1500, Diffusion-ES, cv/gt | `immediate_exp/es_diffusion_<forecast>_fail1500/` |
| fail1500, Diffusion-ES, WM | `world_model_sim/es_diffusion_<wm tag>_fail1500/` |
| fail1500, MF collision ablation | `{immediate_exp \| Flow_matching_sim}/mf_multiSampling_collision_ablation/[<subdir>/]<variant>/seed_<S>/` (Diffusion vs FM never mixed) |
| trainval, cv/gt | `immediate_exp/es_{diffusion,fm}_<forecast>/` |
| trainval, WM | `world_model_sim/es_{diffusion,fm}_<wm tag>_r<refresh>/` |
| probe250 | `Flow_matching_sim/fm_hparam_probe_runs/<tag>/` |

All roots are under `/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/`. WM tags carry the checkpoint
epoch and mode, e.g. `wm_mf_epoch0450_expected_independent`, `wm_det_epoch0500`,
`wm_gau_epoch0370_k8`. Non-default ES/FM budgets are appended to the name
(e.g. `_t0.3`, `_n30`, `_it6_rt5_p64_e8`).

Inside each run folder: `simulation_<planner>_<split>_<tag>_<stamp>/` with one
JSON per scene (`success`, `overlap`, `offroad`, `goal_reached`, `goal_xy`, ...)
and `summary.json`. Logs go to `<root>/logs/` (ablation: `<variant>/seed_<S>/logs/`);
Slurm stdout is `<log dir>/<run name>_<jobid>.out`.

---

## 6. Experiment log

Every experiment run so far, with the command that reproduces it. Success rates
are on all 1500 fail1500 scenes unless noted. The original script for each row
is in `legacy/` (column "was").

### fail1500 — forecast comparison

Success % (overlap / offroad failure counts). Commands are relative to `slurm/sim/`.

| Forecast | Diffusion-ES (it4) | FM-ES it4/rs20 | FM-ES it6/rs40 |
|---|---|---|---|
| GT | 52.27 (183 / 414) | 51.13 (200 / 412) | 52.20 (191 / 404) |
| CV | 46.20 (264 / 435) | 44.93 (275 / 439) | 46.00 (270 / 425) |
| det WM (`wm_det`) | 47.33 (255 / 424) | 45.20 (275 / 435) | 46.20 (270 / 423) |
| MF WM expected_independent | 50.47 (205 / 415) | 48.20 (232 / 425) | 48.87 (226 / 415) |
| MF WM top1 | 47.60 (251 / 426) | | |
| MF WM worst_case_independent | 44.33 (268 / 462) | | |
| Gaussian WM K=8 (`wm_gau`) | 47.20 (230 / 434) | | |
| Gaussian WM mean (`wm_gau_mean`) | 48.00 (249 / 427) | | |

FM-ES GT budget sweep: it4/rs5/p64 49.27, it6/rs5/p64 49.60, it4/rs5/p128/e16 51.33, it6/rs5/p128/e16 51.20.

| Runs | Command | Was |
|---|---|---|
| Diffusion-ES GT, CV, det WM | `fail1500.sh --planner diffusion --forecast gt,cv,wm_det` | `legacy/sim/slurm_es_failure_subset_1500.sh es {gt,cv,wm_det}` |
| Diffusion-ES MF WM | `fail1500.sh --planner diffusion --forecast wm_mf --mf_mode expected_independent,top1,worst_case_independent` | same, `es wm_mf <mode>` |
| Diffusion-ES Gaussian WM | `fail1500.sh --planner diffusion --forecast wm_gau,wm_gau_mean` | same, `es wm_gau{,_mean}` |
| FM-ES it4/rs20 | `fail1500.sh --planner fm --forecast gt,cv,wm_det,wm_mf` | `legacy/sim/slurm_submit_es_fm_fail1500{,_cv_wm}.sh` |
| FM-ES it6/rs40 | `fail1500.sh --planner fm --forecast gt,cv,wm_det,wm_mf --it 6 --rs 40` | `legacy/sim/slurm_submit_es_fm_fail1500_{it6_rs40,cv_wm_it6_rs40}.sh`, `slurm_es_fm_fail1500_it6_rs40.sh` |
| FM-ES rs5 sweep | `fail1500.sh --planner fm --forecast gt --rs 5 --it 4,6` and `... --pop 128 --elite 16` | `legacy/sim/slurm_submit_es_fm_fail1500_rs5_p128.sh`, `slurm_es_fm_fail1500_{it4,it6}_rs5_p128.sh` |

Note: the Diffusion-ES det-WM run lives in `immediate_exp/es_diffusion_wm10_fail1500`
(older naming); a fresh `--forecast wm_det` run writes `world_model_sim/es_diffusion_wm_det_epoch0500_fail1500`.

Findings: the MF WM is the best non-GT forecast for both planners. FM-ES at
it6/rs40 vs it4/rs20 is significantly better for CV and det WM (McNemar p ≈ 0.003),
and is not significantly behind Diffusion-ES (p = 0.88 CV, 0.23 det WM, 0.08 MF WM).

### fail1500 — MF collision ablation (Diffusion-ES, it4)

```bash
bash slurm/sim/fail1500.sh --planner diffusion --forecast wm_mf \
  --mf_collision mean_uniform,gaussian_uniform,gaussian_learned --M 8
# full grid: --mf_collision mean_uniform,mean_learned,gaussian_uniform,gaussian_learned --M 4,8,16
# FM-ES:     same with --planner fm
# smoke:     add --num_scenarios 10 --ablation_subdir _smoke_n10
```

| Variant | Success | Overlap | Offroad | Failed only on goal |
|---|---|---|---|---|
| expected_independent (= `mean_learned`) | 50.47 | 205 | 415 | 135 |
| `mean_uniform` | 50.87 | 202 | 415 | 134 |
| `gaussian_uniform_M08` | 45.67 | 200 | 435 | 199 |
| `gaussian_learned_M08` | 46.13 | 187 | 438 | 199 |

M = 4 and 16 have not been run yet.

Uniform vs learned mode weights: no difference (learned probabilities are near
uniform). Gaussian sampling with the predicted σ makes the planner too
conservative (many more missed goals, barely fewer collisions).
Was: `legacy/sim/slurm_submit_mf_collision_ablation.sh`.

### fail1500 — data-move re-check (2026-10-03)

Diffusion-ES GT rerun after the WOMD move, written to `immediate_exp/_womd_path_recheck/`:
52.20 vs 52.27, 1499/1500 scenes with the same outcome. Was: `legacy/sim/slurm_resubmit_after_womd_move.sh`
(which also has the scene-by-scene `compare` mode).

### probe250 — FM hyper-parameter probe (GT forecast)

```bash
bash slurm/sim/probe250.sh build                     # once (subset already exists)
bash slurm/sim/probe250.sh --planner diffusion --forecast gt                                  # Diff reference
bash slurm/sim/probe250.sh --planner fm --forecast gt --t_start 0.05,0.15,0.20,0.40,0.5,0.6,0.8   # Phase 1
bash slurm/sim/probe250.sh --planner fm --forecast gt --t_start 0.20 --it 6 --rs 20 --tag_suffix it6_rs20   # Phase 2
bash slurm/sim/probe250.sh --planner fm --forecast gt --t_start 0.20 --it 4 --rs 40 --tag_suffix it4_rs40
bash slurm/sim/probe250.sh --planner fm --forecast gt --t_start 0.20 --it 6 --rs 40 --tag_suffix it6_rs40
python scripts/summarize_fm_hparam_probe.py
```

Phase 1 picked t_start 0.2; Phase 2 picked it6/rs40 (used on fail1500 above).
Was: `legacy/sim/slurm_fm_hparam_probe.sh {diff, fm <t>, phase2 <cfg>, sweep*}`.

### trainval — shards 00+01

```bash
bash slurm/sim/trainval.sh --planner diffusion --forecast cv,gt,wm_det
bash slurm/sim/trainval.sh --planner diffusion --forecast wm --wm_refresh 10,20,50
bash slurm/sim/trainval.sh --planner diffusion --forecast wm_mf --mf_mode expected_independent,top1,worst_case_independent
bash slurm/sim/trainval.sh --planner diffusion --forecast wm_gau,wm_gau_mean
bash slurm/sim/trainval.sh --planner fm --forecast cv,gt
```

Was: `legacy/sim/slurm_es_baseline.sh` (Diffusion) and `legacy/sim/slurm_es_fm_baseline.sh` (FM).
FM-ES with a WM on trainval is new (`--planner fm --forecast wm_mf`) and writes to `world_model_sim/`.

### About `legacy/`

The legacy simulation scripts still work unchanged (they were only re-pointed to
call each other under `legacy/sim/`), e.g. `bash legacy/sim/slurm_submit_mf_collision_ablation.sh es`.
Their trainval scripts use the deleted WOMD path; use `slurm/sim/trainval.sh` instead.
The legacy train/annotation `*_submit.sh` scripts call their job script by bare
file name, so run them from inside their folder; `slurm/train/` replaces them.

61 legacy configurations (every row in this section) were checked to be identical (same `run_simulation.py`
arguments, run name and output dir) to what the legacy scripts produced; the new
scripts additionally pass `--seed 0` / `--fm_solver euler` explicitly where the
old ones relied on those defaults.

---

## 7. Training

All training scripts submit themselves when run with `bash`.

```bash
bash slurm/train/mf_wm.sh                       # multi-future Gaussian WM
NUM_MODES=8 EPOCHS=300 bash slurm/train/mf_wm.sh
bash slurm/train/det_wm.sh                      # deterministic WM
bash slurm/train/diffusion.sh pretrain [--subgoal] [--tag T] [--resume_path P]
bash slurm/train/diffusion.sh train --pretrained_ckpt P [--tag T]
bash slurm/train/vla.sh pretrain [--subgoal] [--tag T]
bash slurm/train/vla.sh finetune --scene_tokenizer_ckpt P [--subgoal] [--tag T]
```

WM hyper-parameters are environment variables (`EPOCHS`, `STEPS_PER_EPOCH`,
`VAL_STEPS`, `BATCH_SIZE`, `ANCHOR_STEPS`, `HISTORY_STEPS`, `FUTURE_STEPS`,
`HIDDEN_DIM`, `MAX_RANGE`, `NUM_MODES`, `SIGMA_EPS`, `CACHE_DIR`, `SAVE_DIR`, `WANDB_*`).
Checkpoints: `/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/{ckpts,ckpts_multifuture_gaussian}/`.
Policy training logs go to `<repo>/logs/`.

Quick WM smoke test before a long run:

```bash
cd multiFuture_GaussianWorldModel   # or train_deterministic_world_model
python train_world_model.py --cache_dir /data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
  --save_dir /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/smoke --device cuda \
  --epochs 1 --steps_per_epoch 10 --val_steps 5 --batch_size 4 --anchor_steps 20,30,40 --amp
```

---

## 8. World models

The WM replaces CV/GT at the same interface: it forecasts other agents' future
`[x, y, yaw, length, width]` and the existing scorer computes collision/offroad/goal.

| Forecast | Model | Default checkpoint | Scoring |
|---|---|---|---|
| `wm_det` | deterministic WM (`model/world_model`, "production" backend) | `world_model/ckpts/world_model_20260912_170741/epoch_0500.pt` | SAT on one predicted future |
| `wm_mf` | multi-future Gaussian WM, K=6 modes (`multiFuture_GaussianWorldModel`) | `world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt` | per-agent risk over modes (see `--mf_mode`, `--mf_collision`) |
| `wm_gau`, `wm_gau_mean` | single-future Gaussian WM (training code removed; inference kept) | `world_model/ckpts_gaussian/gaussian_world_model_20260916_194054/epoch_0370.pt` | K=8 Monte-Carlo / mean only |

Inputs: 2 s of history (10 samples at 0.2 s) for every agent in the ego frame,
map polylines and traffic lights. Output: 25 future samples (5 s), interpolated
to the 10 Hz scorer timestep. Static objects stay fixed.

`expected_independent`: P_i = Σ_k p_ik·c_ik per agent, plan score = Π_i (1 − P_i).
`--mf_collision` methods replace the per-mode collision c_ik with either the
mean trajectory (`mean_*`) or the hit fraction over M Gaussian samples (`gaussian_*`),
and the mode weights with learned probabilities or uniform 1/K. Samples are
drawn once per WM prediction and shared by all ego candidates; the RNG is seeded
per scene from the WOMD `scenario/id`, so results do not depend on batch composition.

Background and the original V1 design notes: `legacy/docs/WORLD_MODEL_INSTRUCTIONS.md`;
change history: `train_deterministic_world_model/WORLD_MODEL_CHANGELOG.md`.

---

## 9. Analysis and evaluation

| Tool | Command |
|---|---|
| MF WM ADE/FDE per anchor vs CV | `sbatch multiFuture_GaussianWorldModel/evaluation/slurm_evaluate_per_anchor.sh` |
| MF collision-method validation | `sbatch multiFuture_GaussianWorldModel/evaluation/slurm_validate_mf_collision_methods.sh` |
| Mode-probability analysis | `python multiFuture_GaussianWorldModel/evaluation/evaluate_mode_probs.py` |
| GT-filtered fail1500 re-scoring | `bash slurm/sim/fail1500_gt_filter.sh` |
| FM probe summary | `python scripts/summarize_fm_hparam_probe.py` |

One-off investigations from September (CV vs WM forecast comparison, WM overlap
gap diagnosis, collider analysis, forecast visualisation, Diff-OK/FM-fail viewer
export) are in `legacy/analysis/`; they still run with `sbatch legacy/analysis/<script>.sh`
because they `cd` to the repo root.

Visualisation: `import viz as visualization` instead of `from waymax import visualization`.

---

## 10. Cluster notes

- Submit from **rhea or lov4**; `sbatch` is denied on lov1.
- Never use `sbatch --export=...` (breaks on this cluster); all scripts here
  pass settings as arguments after the script name or as inherited environment variables.
- If a node cannot see `/data/user_data/yixiz/autonlab/zfsauton/datasets`, the job aborts within seconds and
  names the node; resubmit with `--exclude <node>`.
- Do not write scratch files to `/tmp`.

---

## 11. Lane graphs and goal-reaching (original setup)

The scorer needs pre-extracted lane graphs. `scripts/extract_roadgraph.py`
requires the `waymo-open-dataset` protos:

```bash
cd waymax_rs
git clone https://github.com/waymo-research/waymo-open-dataset.git
pip install protobuf grpcio-tools
cd waymo-open-dataset
python -m grpc_tools.protoc -I=src --python_out=src \
  src/waymo_open_dataset/*.proto src/waymo_open_dataset/protos/*.proto
cd ../scripts
python extract_roadgraph.py --split <training|validation> --output_dir /data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs
```

Goal-reaching experiment:

```bash
python scripts/main_goal_reaching.py --checkpoint_path /path_to_ckpt/epoch_0100/ --split <training|validation> \
  --exp_dir /path_to_save_results/exp/ --lane_graph_dir /data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs \
  --num_worlds <N> --num_scenarios <N> --use_es
```
