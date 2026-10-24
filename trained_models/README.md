# Trained models used for evaluation

Copies of every checkpoint the evaluations rely on, each with a `config.json` holding the full
model and training settings (read from the checkpoint itself). Files are byte-identical copies of
the originals on scratch (`source_checkpoint` in each `config.json`). The checkpoints are not
tracked by git (see `.gitignore`); the READMEs and `config.json` files are.

```text
trained_models/
  world_models/
    mf_wm_old_20260916/          epoch_0450.pt   multi-future Gaussian WM used everywhere so far
    mf_wm_new_mapfix_20261017/   epoch_0500.pt   same model retrained with map_selection=nearest_even
    ishan_mf_wm_20261005/        epoch_0450.pt   Ishan's agent-frame multi-future WM
  discriminators/
    safety_progress_v3_20261016_155456_86103/   A0, the original discriminator (v3)
    A4_v4/  A5/  A8/             ablations of A0
                                 each: best.pt + config.json, training_config.json, eval_*.json/md, history.json
  diffusion/
    minuei_goal_cond_diffusion/  epoch_1000/ (Orbax checkpoint) + config.json
../trainval_shards/              WOMD shards of the trainval benchmark (see the end of this file)
```

---

## World models

All three predict, for up to 128 agents, a 6-mode Gaussian mixture over 25 future steps
(0.2 s apart, 5 s) from 10 history steps (0.2 s apart, 2 s), map polylines (64 segments x 32
points) and 16 traffic lights, within 50 m. Each mode is a mean trajectory (x, y, yaw) with a
per-step 2-D covariance and a mode probability. Training: the sim-state cache
`/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz` (WOMD), anchors at steps 20/30/40,
10 % validation split, 500 epochs x 2000 steps, batch 16, lr 2e-4, weight decay 1e-4, AMP,
seed 0, Gaussian NLL + mode classification (weight 1.0) + yaw (weight 0.5), collision-focus
weighting (agents within 20 m, x2).

| Model | Params | Epoch | Val minADE@6 | Val top-1 ADE / FDE | Val NLL | Loads with this repo |
|---|---|---|---|---|---|---|
| `mf_wm_old_20260916` | 4.83 M | 450 | 0.30 m* | 0.77 m* / – | – | yes |
| `mf_wm_new_mapfix_20261017` | 4.83 M | 500 | 0.33 m | 0.82 m / 1.97 m | -11.27 | yes |
| `ishan_mf_wm_20261005` | 3.41 M | 450 (run to 500) | 0.23 m† | 0.53 m / 1.35 m† | -11.82† | no (needs Ishan's code) |

\* progress-bar values at epoch 450 (the W&B summary of this run was not saved).
† W&B summary at the end of the run (epoch 500), not of the saved epoch-450 file.

### `mf_wm_old_20260916` (multi-future Gaussian WM, original)

- **Source:** `/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt`
  (W&B `world Model` / run `p0xludxp`).
- **Used by:** every `--forecast wm_mf` simulation (Diffusion-ES / FM-ES planner WM), and the
  discriminators: their cached WM inputs were produced by this checkpoint and the deploy scorer
  runs it online (`es_discriminator/real_es/wm_cache.py`, `DEFAULT_CKPT`).
- **Map input:** `map_selection=legacy`.
- **Load:** `simulation/run_simulation.py --other_agent_forecast wm --world_model_ckpt trained_models/world_models/mf_wm_old_20260916/epoch_0450.pt`
  or `slurm/sim/run_es.sh --forecast wm_mf --wm_ckpt <same path>`.

### `mf_wm_new_mapfix_20261017` (multi-future Gaussian WM, map fix)

- **Source:** `/zfsauton/scratch/sbellad/world_model/mf_gaussian_old_mapFix_retry/mf_gaussian_old_mapFix_retry_20261017_022450/epoch_0500.pt`
  (W&B run `s87c6uv9`, `slurm/train/mf_wm.sh` with `MAP_SELECTION=nearest_even`).
- **Difference from the old model:** only the map input. `nearest_even` keeps the polylines
  nearest the ego with evenly spaced points instead of the legacy selection. Same architecture
  and hyper-parameters otherwise.
- **Status:** trained to the final epoch; not yet used in any simulation or discriminator.
  Its mode probabilities are much flatter than Ishan's (val top-1 confidence 0.29 vs 0.71, mode
  entropy 1.62 vs 0.67).
- **Load:** as the old model, with `--wm_ckpt trained_models/world_models/mf_wm_new_mapfix_20261017/epoch_0500.pt`.

### `ishan_mf_wm_20261005` (Ishan's multi-future WM)

- **Source:** `/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt`.
- **Difference:** decodes each agent's future in its own frame (`agent_frame_decoding=true`) with a
  direct decoder (`decoder_type=direct`) instead of the latent future-dynamics transformer; 3.41 M
  parameters.
- **Compatibility:** this repo's `multiFuture_GaussianWorldModel` code does not have those options;
  loading fails with missing `future_time_embedding` / `future_dynamics.*` weights. Use Ishan's
  version of the WM code to run it.

---

## Discriminators (learned Diffusion-ES reward)

Score every Diffusion-ES candidate from the scene, the candidate trajectory, the frozen old
multi-future WM prediction (6 modes, 32 agents) and 256 road-edge tokens; the planner uses the
score as the complete ES reward (`--disc_ckpt`). All four share data and training: combined set
`fail1500_plus_womd_train_s02_11` (fail1500 + WOMD training shards 02-11; 50,400 replans x 320
candidates; 33,041 train / 7,194 val / 7,552 test replans), Waymax safety + progress pairwise
ranking loss (80 % safety pairs, `max_pairs` 512, progress = goal gain with a 2 m margin),
64 candidates per replan per step, 16 replans per batch, 30 epochs, lr 3e-4, weight decay 1e-4,
200 warm-up steps, hidden 128, 4 heads, dropout 0.1, seed 0. The checkpoint is selected on the
validation top-1 unsafe rate. Architecture flags: `es_discriminator/analyse_discriminator/ablations.py`.

| Model | Architecture | Params | Best epoch |
|---|---|---|---|
| `safety_progress_v3_20261016_155456_86103` (A0) | original (discriminator v3) | 552,169 | 26 |
| `A4_v4` | A0 without goal fusion, map tokens, agent tokens (discriminator v4) | 500,073 | 19 |
| `A5` | A4 with goal + mean traffic-light conditioning instead of scene cross-attention | 450,665 | 28 |
| `A8` | A4 with mean-only (no max) time pooling of the WM / road-edge attention | 466,793 | 22 |

Held-out test replans (identical for all models; 95 % intervals in each `eval_test.md`):

| Model | Top-1 unsafe | Collision | Off-road | Goal gain | Pairwise acc | Safety AUROC | Unsafe, 49 hard replans |
|---|---|---|---|---|---|---|---|
| A0 (v3) | 2.6 % | 1.8 % | 0.8 % | 23.54 m | 0.949 | 0.937 | 24.5 % |
| `A4_v4` | 2.7 % | 1.9 % | 0.8 % | 23.52 m | 0.944 | 0.939 | 20.4 % |
| `A5` | 2.7 % | 1.9 % | 0.8 % | 23.55 m | 0.948 | 0.933 | 28.6 % |
| `A8` | 2.7 % | 1.9 % | 0.8 % | 23.51 m | 0.944 | 0.930 | 24.5 % |

"Hard replans" = the first bad replan of the test scenes where the handcrafted ES reward picked an
unsafe plan (the ES reward is 100 % unsafe there). Closed loop on fail1500 (log replay, after the
world-dt fix, `final_runs_dtfix/`): A0 (v3) 71.8 % success (1077 / 1500; 149 collisions, 273 off-road),
A4_v4 71.7 % (1076 / 1500; 153 collisions, 270 off-road). Paired difference -0.07 points
(95 % interval -1.2 to +1.1; 37 vs 36 scenes where only one succeeds; McNemar p = 1.0).
A5 and A8 have not been run in closed loop.

- **Sources:** `/zfsauton/scratch/sbellad/es_discriminator/safety_progress/safety_progress_v3_{20261016_155456_86103, A4_20261017_212234_87831, A5_20261018_154026_88420, A8_20261018_165128_88423}/`.
- **Files:** `best.pt` (state dict + `model_kwargs`, data config, candidate features, training
  args), `config.json` (summary: flags, parameters, training settings, test metrics),
  `training_config.json` (the trainer's own config), `eval_{val,test}.{json,md}`, `history.json`
  (per-epoch metrics), `result.json`, `deploy_check.json`.
- **Deploy check:** online vs offline scores differ by about 0.006-0.010 (tolerance 0.001) for all
  four; this comes from the WM inputs (the WM-free ablation A7 passes), not from these models.
- **Load:**

  ```python
  from es_discriminator.real_es.deploy import DiscriminatorScorer
  scorer = DiscriminatorScorer("trained_models/discriminators/A4_v4/best.pt")   # also loads the old WM
  ```

  or in simulation: `slurm/sim/run_es.sh --subset fail1500 --planner diffusion --forecast wm_mf --disc_ckpt trained_models/discriminators/A4_v4/best.pt`.

---

## Diffusion policy

### `minuei_goal_cond_diffusion` (Minuei's goal-conditioned diffusion policy)

- **Source:** `/zfsauton/scratch/mineuih/waymax_rs/new_checkpoints/train_diffusion/goal_cond_diffusion_policy/epoch_1000`
  (Orbax / OCDBT checkpoint directory, 749 MB, epoch 1000; resumed from
  `train_diffusion_goal_20261014_232111`).
- **Model:** diffusion policy predicting 25 ego steps at 0.2 s (`predict_type=v`, 100 denoising
  steps), conditioned on the scene (128 objects, 128 map segments x 128 points, 16 traffic lights,
  30 m range) and on the goal (`goal_conditioned=true`; the goal is dropped with probability 0.3
  during training). hidden 512, condition dim 256.
- **Training:** WOMD training shards (`tfrecord@1000`), 1000 epochs x 1000 steps, batch 64, lr 1e-4,
  weight decay 1e-4, 2000 warm-up steps, grad clip 0.5, EMA every 4 steps, seed 0.
  All settings: `config.json` (`training_metadata`, identical to `epoch_1000/metadata.json`).
- **Compatibility:** this repo's `train_diffusion/utils/infer.py` cannot build it (it expects
  metadata keys such as `inst_dim`, and only knows `subgoal_conditioned`, not `goal_conditioned`);
  use Minuei's version of `train_diffusion`. Not used in our simulations so far, which use
  `/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500`.

---

## Trainval shards (`../trainval_shards/`)

The four WOMD tfrecords of the `trainval` simulation benchmark, copied from
`/zfsauton/datasets/womd/tf_example` (byte-identical, 3.9 GB):

```text
trainval_shards/training/training_tfexample.tfrecord-0000{0,1}-of-01000
trainval_shards/validation/validation_tfexample.tfrecord-0000{0,1}-of-00150
```

`slurm/sim/run_es.sh --subset trainval` still links the originals; to run from this copy, call
`simulation/run_simulation.py --tfrecord_dir trainval_shards --split {training,validation}` directly.
