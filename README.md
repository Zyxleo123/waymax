# waymax_rs — ES planning with learned world models

Closed-loop Waymax simulation of an ego planner (Diffusion-ES or Flow-Matching-ES)
whose candidate plans are scored against a forecast of the other agents:
constant velocity (CV), privileged ground truth (GT), or the multi-future Gaussian
world model (WM). A learned discriminator (`es_discriminator/`) can replace the
handcrafted ES reward.

- [1. Setup](#1-setup)
- [2. Repository layout](#2-repository-layout)
- [3. Data and scene subsets](#3-data-and-scene-subsets)
- [4. Running simulations](#4-running-simulations)
- [5. Outputs](#5-outputs)
- [6. Experiment log](#6-experiment-log)
- [7. Training](#7-training)
- [8. World model](#8-world-model)
- [9. Cluster notes](#9-cluster-notes)
- [10. Lane graphs and goal-reaching (original setup)](#10-lane-graphs-and-goal-reaching-original-setup)

---

## 1. Setup

```bash
cd /zfsauton2/home/sbellad/baseline/waymax_rs
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
export PYTHONPATH=$PWD:$PYTHONPATH
```

`env.yml` has the full environment (JAX/Flax, Waymax, PyTorch, ...). The Slurm
scripts activate the environment themselves.

---

## 2. Repository layout

```text
slurm/
  sim/
    submit.sh                   every simulation submission (job grids + named experiments)
    run_es.sh                   job: one Diffusion-ES / FM-ES simulation
    run_policy.sh               job: one plain diffusion / FM simulation (no ES)
  train/
    mf_wm.sh                    multi-future Gaussian WM training
    diffusion.sh  vla.sh        policy training
    discriminator_data.sh       discriminator data: collect ES candidates, labels, WM caches
    train_wm_discriminator.sh   discriminator training + offline evaluation

simulation/run_simulation.py    closed-loop simulator entry point
planner/                        Diffusion-ES and FM-ES planners
scores/                         collision (SAT) / offroad / goal scoring
multiFuture_GaussianWorldModel/ multi-future Gaussian WM (training, data, inference)
es_discriminator/               learned ES reward (see es_discriminator/README.md)
  analyse_discriminator/        architecture ablations A0-A8 (+ their scripts and tests)
train_diffusion/  train_vla/    policy training code
model/                          diffusion / VLA networks
scripts/                        data preparation (training cache, fail1500 subset, annotation)
viz/                            better Waymax visualisation (import viz as visualization)
```

---

## 3. Data and scene subsets

| Item | Path |
|---|---|
| WOMD tf_example | `/zfsauton/datasets/womd/tf_example/{training,validation}` |
| Lane graphs | `/zfsauton/scratch/mineuih/waymax_rs/lane_graphs` |
| Sim-state cache (WM training) | `/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz` |

The old WOMD copy at `/zfsauton/scratch/eshau/womd` was deleted on 2026-10-02.
Several Python scripts still default to it — always pass the new path
(`--tfrecord_dir`, `--tf_root`, `--tfrecord_path`) when calling them directly.

**Subsets** (`--subset` in the simulation scripts):

| Subset | Scenes | Where | Notes |
|---|---|---|---|
| `trainval` | all scenes of training shards 00000+00001 and validation shards 00000+00001 | symlinks created per run under the exp root | both splits run inside one job |
| `fail1500` | 1500 scenes where FM-ES failed (offroad 658, overlap 679, goal 139, both 24), 187 shards | `/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0` | main benchmark (built by `scripts/build_failure_subset_for_es.py`) |
| `val10` | WOMD validation shards 00000-00009 (2916 scenes; 2895 run) | `/zfsauton/scratch/sbellad/womd_val_shards00_09` | final evaluation |

Subset tfrecords are symlinks into the WOMD folder. Every job checks at start
that all shards are readable on its node and aborts otherwise (a node without
the `/zfsauton/datasets` mount used to finish with 0 scenes and print `DONE`).

---

## 4. Running simulations

Submit from **rhea or lov4** (sbatch is denied on lov1). Everything goes through
`slurm/sim/submit.sh`; comma-separated values expand into one job per combination,
and duplicates (same run name) are submitted once.

```bash
bash slurm/sim/submit.sh es --subset fail1500 --planner fm --forecast cv,wm_mf --it 6 --rs 40
bash slurm/sim/submit.sh es --subset fail1500 --planner diffusion --forecast wm_mf --mf_collision gaussian_uniform,gaussian_learned --M 4,8,16
bash slurm/sim/submit.sh es --subset trainval --planner diffusion --forecast cv,gt,wm_mf
bash slurm/sim/submit.sh policy --subset val10 --planner diffusion,fm --other_agents log,idm
```

Named experiments submit the exact job set behind a reported result
(`bash slurm/sim/submit.sh --help` lists them):

| Name | Jobs |
|---|---|
| `val10_all [--other_agents log,idm]` | plain diffusion / FM, Diffusion-ES and FM-ES (it6/rs40) with gt, cv, wm_mf on val10 |
| `idm_fail1500` | the same planners on fail1500 with IDM other agents |
| `final_val10 [--exp_root R]` | paper runs on val10 -> `final_runs/` (diffusion only, CV, WM, WM + discriminator trained on fail1500) |
| `disc_v3` | discriminator v3 (A0): val10 log + IDM -> `final_runs/`, and trainval |
| `dtfix` | fail1500 after the world-dt fix -> `final_runs_dtfix/`: WM, WM + discriminator v3 |
| `disc_v4` | discriminator v4 (A4) on fail1500 -> `final_runs_dtfix/` |

Always preview first — `--dry_run` prints every job's run name, output dir and
exact `run_simulation.py` command without submitting:

```bash
bash slurm/sim/submit.sh disc_v4 --dry_run
```

### Flags (`submit.sh es`, `run_es.sh`)

| Flag | Values / default | Meaning |
|---|---|---|
| `--subset` | `trainval` \| `fail1500` \| `val10` | scenes |
| `--planner` | `diffusion` \| `fm` | Diffusion-ES or FM-ES |
| `--forecast` | `cv` \| `gt` \| `wm_mf` | other-agent forecast |
| `--other_agents` | `log` (default) \| `idm` | log replay or reactive IDM agents (names get `IDM_`) |
| `--mf_mode` | `expected_independent` (default) \| `top1` \| `worst_case_independent` | multi-future WM scoring |
| `--mf_collision` | `mean_uniform` \| `mean_learned` \| `gaussian_uniform` \| `gaussian_learned` | MF collision ablation (fail1500 only) |
| `--M` | 8 | samples per mode for `gaussian_*` |
| `--mf_seed` | = `--seed` | Gaussian sampling seed (combined with WOMD scenario id + timestep) |
| `--ablation_subdir` | — | e.g. `_smoke_n10` for small ablation test runs |
| `--disc_ckpt` | — | Diffusion-ES with a trained discriminator as the complete ES reward |
| `--seed` | 0 | planner seed |
| `--num_scenarios` | 100000 (all) | cap scenes, e.g. 10 for a smoke test |
| `--num_worlds` | 5 | scenes simulated in parallel |
| `--it` / `--pop` / `--elite` | 4 / 64 / 8 | ES iterations / population / elites |
| `--rs` | 20 | FM resample ODE steps |
| `--t_start` | 0.2 | FM resample start time (written into names verbatim, e.g. `0.20`) |
| `--fm_steps` / `--solver` | 50 / `euler` | FM full-sample ODE steps / solver |
| `--resample_timesteps` | 5 | Diffusion resample timesteps |
| `--wm_ckpt` | see §8 | override the WM checkpoint |
| `--run_suffix` | — | appended to any run name |
| `--exp_root` | — | put the run under `<root>/<run name>` |
| `--dry_run` | — | print jobs, submit nothing |
| `--exclude` | — | `sbatch --exclude`, e.g. `gpu26,gpu31` |

Fixed in all runs: ego replans every 10 steps, WM device `auto`. A crashed run
resumes into the same `simulation_*` folder (up to 3 retries) and skips finished scenes.

You can also run one job directly: `sbatch slurm/sim/run_es.sh --subset ... --planner ... --forecast ...`,
or inspect it with `bash slurm/sim/run_es.sh ... --print_cmd`.

---

## 5. Outputs

| Subset / planner / forecast | Output dir |
|---|---|
| fail1500, FM-ES, any forecast | `Flow_matching_sim/es_fm_<forecast>_fail1500_it<N>_rs<R>_p<P>_e<E>/` |
| fail1500, Diffusion-ES, cv/gt | `immediate_exp/es_diffusion_<forecast>_fail1500/` |
| fail1500, Diffusion-ES, WM | `world_model_sim/es_diffusion_<wm tag>_fail1500/` |
| fail1500, IDM | `{Diffusion_ES_sim \| Flow_matching_sim}/es_*_IDM_*` |
| fail1500, MF collision ablation | `{immediate_exp \| Flow_matching_sim}/mf_multiSampling_collision_ablation/[<subdir>/]<variant>/seed_<S>/` (Diffusion vs FM never mixed) |
| trainval, cv/gt | `immediate_exp/es_{diffusion,fm}_<forecast>/` |
| trainval, WM | `world_model_sim/es_{diffusion,fm}_<wm tag>_r10/` |
| val10 | `{Diffusion_ES_sim \| Flow_matching_sim}/es_*_val10*/` |
| discriminator reward | `es_discriminator/closed_loop/es_diffusion_[IDM_]discriminator*/` (or `--exp_root`) |
| plain diffusion / FM | `{diffusion_sim \| Flow_matching_sim}/{diffusion,fm}_only[_IDM]_<subset>_seed<S>/` |

All roots are under `/zfsauton/scratch/sbellad/`. WM tags carry the checkpoint
epoch and mode, e.g. `wm_mf_epoch0450_expected_independent`. Non-default ES/FM
budgets are appended to the name (e.g. `_t0.3`, `_n30`, `_it6_rt5_p64_e8`).

Inside each run folder: `simulation_<planner>_<split>_<tag>_<stamp>/` with one
JSON per scene (`success`, `overlap`, `offroad`, `goal_reached`, `goal_xy`, ...)
and `summary.json`. Logs go to `<root>/logs/` (ablation: `<variant>/seed_<S>/logs/`);
Slurm stdout is `<log dir>/<run name>_<jobid>.out`.

---

## 6. Experiment log

Success rates are on all 1500 fail1500 scenes unless noted. Rows marked "removed"
used the deterministic or single-future Gaussian WM, which are no longer in the repo.

### fail1500 — forecast comparison

Success % (overlap / offroad failure counts).

| Forecast | Diffusion-ES (it4) | FM-ES it4/rs20 | FM-ES it6/rs40 |
|---|---|---|---|
| GT | 52.27 (183 / 414) | 51.13 (200 / 412) | 52.20 (191 / 404) |
| CV | 46.20 (264 / 435) | 44.93 (275 / 439) | 46.00 (270 / 425) |
| det WM (removed) | 47.33 (255 / 424) | 45.20 (275 / 435) | 46.20 (270 / 423) |
| MF WM expected_independent | 50.47 (205 / 415) | 48.20 (232 / 425) | 48.87 (226 / 415) |
| MF WM top1 | 47.60 (251 / 426) | | |
| MF WM worst_case_independent | 44.33 (268 / 462) | | |
| Gaussian WM K=8 (removed) | 47.20 (230 / 434) | | |
| Gaussian WM mean (removed) | 48.00 (249 / 427) | | |

FM-ES GT budget sweep: it4/rs5/p64 49.27, it6/rs5/p64 49.60, it4/rs5/p128/e16 51.33, it6/rs5/p128/e16 51.20.

| Runs | Command (`bash slurm/sim/submit.sh es --subset fail1500 ...`) |
|---|---|
| Diffusion-ES GT, CV | `--planner diffusion --forecast gt,cv` |
| Diffusion-ES MF WM | `--planner diffusion --forecast wm_mf --mf_mode expected_independent,top1,worst_case_independent` |
| FM-ES it4/rs20 | `--planner fm --forecast gt,cv,wm_mf` |
| FM-ES it6/rs40 | `--planner fm --forecast gt,cv,wm_mf --it 6 --rs 40` |
| FM-ES rs5 sweep | `--planner fm --forecast gt --rs 5 --it 4,6` and `... --pop 128 --elite 16` |

Findings: the MF WM is the best non-GT forecast for both planners. FM-ES at
it6/rs40 vs it4/rs20 is significantly better for CV and det WM (McNemar p ≈ 0.003),
and is not significantly behind Diffusion-ES (p = 0.88 CV, 0.23 det WM, 0.08 MF WM).

### fail1500 — MF collision ablation (Diffusion-ES, it4)

```bash
bash slurm/sim/submit.sh es --subset fail1500 --planner diffusion --forecast wm_mf \
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

### fail1500 — data-move re-check (2026-10-03)

Diffusion-ES GT rerun after the WOMD move, written to `immediate_exp/_womd_path_recheck/`:
52.20 vs 52.27, 1499/1500 scenes with the same outcome.

### FM hyper-parameter probe (250 fail1500 scenes, GT forecast)

Phase 1 picked t_start 0.2; Phase 2 picked it6/rs40 (used on fail1500 above).
Results: `/zfsauton/scratch/sbellad/Flow_matching_sim/fm_hparam_probe_runs/`.

### trainval — shards 00+01

```bash
bash slurm/sim/submit.sh es --subset trainval --planner diffusion --forecast cv,gt
bash slurm/sim/submit.sh es --subset trainval --planner diffusion --forecast wm_mf --mf_mode expected_independent,top1,worst_case_independent
bash slurm/sim/submit.sh es --subset trainval --planner fm --forecast cv,gt
```

FM-ES with a WM on trainval (`--planner fm --forecast wm_mf`) writes to `world_model_sim/`.

---

## 7. Training

```bash
bash slurm/train/mf_wm.sh                       # multi-future Gaussian WM (submits itself)
NUM_MODES=8 EPOCHS=300 bash slurm/train/mf_wm.sh
bash slurm/train/diffusion.sh pretrain [--subgoal] [--tag T] [--resume_path P]
bash slurm/train/diffusion.sh train --pretrained_ckpt P [--tag T]
bash slurm/train/vla.sh pretrain [--subgoal] [--tag T]
bash slurm/train/vla.sh finetune --scene_tokenizer_ckpt P [--subgoal] [--tag T]

# discriminator (details: es_discriminator/README.md)
sbatch slurm/train/discriminator_data.sh --source fail1500
sbatch --array=2-11%2 --gres=gpu:a6000:1 --time=16:00:00 slurm/train/discriminator_data.sh --source womd
sbatch slurm/train/train_wm_discriminator.sh --data fail1500_plus_womd_train_s02_11
```

WM hyper-parameters are environment variables (`EPOCHS`, `STEPS_PER_EPOCH`,
`VAL_STEPS`, `BATCH_SIZE`, `ANCHOR_STEPS`, `HISTORY_STEPS`, `FUTURE_STEPS`,
`HIDDEN_DIM`, `MAX_RANGE`, `NUM_MODES`, `SIGMA_EPS`, `CACHE_DIR`, `SAVE_DIR`, `WANDB_*`).
Checkpoints: `/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/`.
Policy training logs go to `<repo>/logs/`.

Quick WM smoke test before a long run:

```bash
cd multiFuture_GaussianWorldModel
python train_world_model.py --cache_dir /zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
  --save_dir /zfsauton/scratch/sbellad/world_model/smoke --device cuda \
  --epochs 1 --steps_per_epoch 10 --val_steps 5 --batch_size 4 --anchor_steps 20,30,40 --amp
```

---

## 8. World model

The WM replaces CV/GT at the same interface: it forecasts other agents' future
`[x, y, yaw, length, width]` and the existing scorer computes collision/offroad/goal.

| Forecast | Model | Default checkpoint | Scoring |
|---|---|---|---|
| `wm_mf` | multi-future Gaussian WM, K=6 modes (`multiFuture_GaussianWorldModel`) | `world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt` | per-agent risk over modes (see `--mf_mode`, `--mf_collision`) |

Inputs: 2 s of history (10 samples at 0.2 s) for every agent in the ego frame,
map polylines and traffic lights. Output: 25 future samples (5 s), interpolated
to the 10 Hz scorer timestep. Static objects stay fixed.

`expected_independent`: P_i = Σ_k p_ik·c_ik per agent, plan score = Π_i (1 − P_i).
`--mf_collision` methods replace the per-mode collision c_ik with either the
mean trajectory (`mean_*`) or the hit fraction over M Gaussian samples (`gaussian_*`),
and the mode weights with learned probabilities or uniform 1/K. Samples are
drawn once per WM prediction and shared by all ego candidates; the RNG is seeded
per scene from the WOMD `scenario/id`, so results do not depend on batch composition.

Visualisation: `import viz as visualization` instead of `from waymax import visualization`.

---

## 9. Cluster notes

- Submit from **rhea or lov4**; `sbatch` is denied on lov1.
- Never use `sbatch --export=...` (breaks on this cluster); all scripts here
  pass settings as arguments after the script name or as inherited environment variables.
- If a node cannot see `/zfsauton/datasets`, the job aborts within seconds and
  names the node; resubmit with `--exclude <node>`.
- Do not write scratch files to `/tmp`.

---

## 10. Lane graphs and goal-reaching (original setup)

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
python extract_roadgraph.py --split <training|validation> --output_dir /zfsauton/scratch/mineuih/waymax_rs/lane_graphs
```

Goal-reaching experiment:

```bash
python scripts/main_goal_reaching.py --checkpoint_path /path_to_ckpt/epoch_0100/ --split <training|validation> \
  --exp_dir /path_to_save_results/exp/ --lane_graph_dir /zfsauton/scratch/mineuih/waymax_rs/lane_graphs \
  --num_worlds <N> --num_scenarios <N> --use_es
```
