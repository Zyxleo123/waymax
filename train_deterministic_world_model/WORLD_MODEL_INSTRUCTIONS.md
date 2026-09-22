# Latent Multi-Agent World Model for FM-ES

This package adds a learned world-model forecast mode (`wm`) between the existing
constant-velocity (`cv`) and privileged future-GT (`gt`) modes.

The intended experiment is:

```text
FM-ES + CV  <  FM-ES + learned world model  <  FM-ES + Future GT
```

with the largest expected improvement in **dynamic-agent overlap / collision**.
The existing hand-written/static FM-ES score functions are kept intact. The
network does **not** separately predict collision, off-road, goal, etc. Instead,
it predicts the physical future of surrounding agents, and the existing scorer
computes those metrics from the decoded trajectories.

---

## 1. Architecture implemented

```text
past agent histories + map polylines + traffic lights
                    |
                    v
        per-agent temporal encoder (GRU)
                    |
                    v
       multi-agent/map interaction transformer
                    |
                    v
        interaction-aware latent scene z_t
                    |
                    v
      latent multi-step dynamics transformer
                    |
          z_{t+1}, ..., z_{t+T}
                    |
                    v
          physical trajectory decoder
                    |
                    v
future [x, y, yaw] for every dynamic surrounding agent
                    |
                    v
existing FM-ES collision / offroad / goal / ... scorer
```

### Inputs

The model uses only information available at or before the current timestep:

- last 10 sampled states for every tracked object (default: 2.0 s history)
- position and velocity in the current ego frame
- relative yaw (sin/cos)
- object length / width
- object type
- map-polyline features already used by the existing project
- current traffic-light features already used by the existing project

### Output

The decoder predicts 25 future samples by default (5.0 s at 0.2 s spacing):

- future xy displacement from each agent's current location
- future yaw change

At simulation time those predictions are converted back to world coordinates,
interpolated to the world/scorer timestep, and returned in the exact shape the
current scorer expects:

```text
[num_objects, num_time_steps, 5]
[x, y, yaw, length, width]
```

Static/non-dynamic objects remain fixed. V1 is deterministic. A later V2 can
return multiple probabilistic futures through the collision-ensemble interface.

---

## 2. Important files

New files:

```text
model/world_model/config.py
model/world_model/network.py
model/world_model/inference.py
data/world_model_cache_loader.py
train_world_model.py
evaluate_world_model.py
slurm_train_world_model.sh
WORLD_MODEL_INSTRUCTIONS.md
```

Modified integration files:

```text
scores/helpers.py
scores/scorer.py
scores/scorer_lane_graph.py
planner/fm_es_planner.py
planner/diffusion_es_planner.py
simulation/run_simulation.py
slurm_es_fm_baseline.sh
slurm_es_failure_subset.sh
```

---

## 3. Environment

Use the same Waymax environment as the baseline project.

Example on the current cluster:

```bash
cd /zfsauton2/home/sbellad/baseline/waymax_rs
source /zfsauton/scratch/sbellad/miniconda3/bin/activate \
  /zfsauton/scratch/sbellad/conda_envs/waymax_rs
export PYTHONPATH=$PWD:$PYTHONPATH
```

The existing `env.yml` already contains PyTorch, Waymax, NumPy, JAX, Flax, etc.

---

## 4. Data

The training loader uses the same cached Waymax simulator-state `.npz` shards
already used by the VLA/diffusion code.

Default/example cache directory used elsewhere in this repository:

```bash
/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz
```

The world-model loader reads logged future states **only as supervised training
targets**. The model inputs are restricted to history/current state + map +
current traffic-light state.

---

## 5. First smoke test

Before launching a long job, train only a few steps:

```bash
python train_world_model.py \
  --cache_dir /zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
  --save_dir /zfsauton/scratch/sbellad/world_model_ckpts/smoke \
  --device cuda \
  --epochs 1 \
  --steps_per_epoch 10 \
  --val_steps 5 \
  --batch_size 4 \
  --anchor_steps 20,30,40 \
  --amp
```

Expected output includes training/validation loss plus ADE/FDE in meters.
A checkpoint is written to:

```text
/zfsauton/scratch/sbellad/world_model_ckpts/smoke/latest.pt
```

---

## 6. Recommended full training command

```bash
python train_world_model.py \
  --cache_dir /zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
  --save_dir /zfsauton/scratch/sbellad/world_model_ckpts/v1 \
  --device cuda \
  --epochs 50 \
  --batch_size 16 \
  --anchor_steps 20,30,40 \
  --history_steps 10 \
  --history_stride 2 \
  --future_steps 25 \
  --future_stride 2 \
  --hidden_dim 256 \
  --num_heads 4 \
  --num_scene_layers 3 \
  --num_future_layers 2 \
  --max_range 50 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --collision_focus_radius_m 20 \
  --collision_focus_weight 2 \
  --amp
```

Interpretation of the defaults:

```text
history: 10 samples * 0.2 s = about 2 seconds
future:  25 samples * 0.2 s = 5 seconds
```

Nearby agents receive extra loss weight because the current CV -> GT gap is
mostly overlap with dynamic objects.

### SLURM alternative

```bash
CACHE_DIR=/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
SAVE_DIR=/zfsauton/scratch/sbellad/world_model_ckpts/v1 \
sbatch slurm_train_world_model.sh
```

---

## 7. Resume training

```bash
python train_world_model.py \
  --cache_dir /zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
  --save_dir /zfsauton/scratch/sbellad/world_model_ckpts/v1 \
  --resume /zfsauton/scratch/sbellad/world_model_ckpts/v1/latest.pt \
  --device cuda \
  --epochs 80 \
  --batch_size 16 \
  --anchor_steps 20,30,40 \
  --history_steps 10 \
  --history_stride 2 \
  --future_steps 25 \
  --future_stride 2 \
  --hidden_dim 256 \
  --num_heads 4 \
  --num_scene_layers 3 \
  --num_future_layers 2 \
  --max_range 50 \
  --amp
```

When resuming, shape/model arguments must match the checkpoint configuration.

---

## 8. Offline sanity check: world model vs constant velocity

Do this **before** spending time on a full FM-ES closed-loop run.

```bash
python evaluate_world_model.py \
  --cache_dir /zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
  --world_model_ckpt /zfsauton/scratch/sbellad/world_model_ckpts/v1/latest.pt \
  --device cuda \
  --batch_size 16 \
  --max_batches 100 \
  --anchor_steps 20,30,40
```

It prints:

```text
World Model: ADE=...  FDE=...
Constant Vel: ADE=...  FDE=...
```

The first gate is that the learned model should improve over CV on future-motion
prediction. If it cannot beat CV offline, do not expect it to close the FM-ES
CV -> GT overlap gap.

---

## 9. Run FM-ES with the learned world model

The new forecast mode is:

```text
--other_agent_forecast wm
```

and requires:

```text
--world_model_ckpt /path/to/latest.pt
```

Example:

```bash
python simulation/run_simulation.py \
  --planner es_fm \
  --fm_ckpt_path /zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500 \
  --world_model_ckpt /zfsauton/scratch/sbellad/world_model_ckpts/v1/latest.pt \
  --world_model_device cpu \
  --other_agent_forecast wm \
  --tfrecord_dir /zfsauton/scratch/eshau/womd/tf_example \
  --lane_graph_dir /zfsauton/scratch/mineuih/waymax_rs/lane_graphs \
  --exp_dir /zfsauton/scratch/sbellad/immediate_exp/es_fm_wm \
  --split validation \
  --num_worlds 5 \
  --num_scenarios 1000 \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations 4 \
  --resample_t_start 0.2 \
  --visualize_mode none
```

### Device note

FM-ES/JAX and the world model/PyTorch can compete for GPU memory.
The safest first run is:

```text
--world_model_device cpu
```

If two GPUs are allocated, putting the world model on the other GPU is better,
for example:

```text
--world_model_device cuda:1
```

If you deliberately place both JAX and Torch on one GPU, disable aggressive JAX
preallocation first and monitor memory carefully:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

---

## 10. Run the existing baseline SLURM script in WM mode

`slurm_es_fm_baseline.sh` now accepts `cv`, `wm`, or `gt`.

```bash
WORLD_MODEL_CKPT=/zfsauton/scratch/sbellad/world_model_ckpts/v1/latest.pt \
WORLD_MODEL_DEVICE=cpu \
sbatch --job-name=es_fm_wm slurm_es_fm_baseline.sh wm
```

The original commands still work:

```bash
sbatch --job-name=es_fm_cv slurm_es_fm_baseline.sh cv
sbatch --job-name=es_fm_gt slurm_es_fm_baseline.sh gt
```

---

## 11. Run on the shared 1500 hard/failure subset

This is the experiment most directly comparable to the 1500-case numbers already
reported.

```bash
WORLD_MODEL_CKPT=/zfsauton/scratch/sbellad/world_model_ckpts/v1/latest.pt \
WORLD_MODEL_DEVICE=cpu \
sbatch --job-name=es_fm_wm_fail1500 \
  slurm_es_failure_subset.sh es_fm wm
```

For a clean three-way comparison, run the same exact subset with:

```bash
sbatch --job-name=es_fm_cv_fail1500 slurm_es_failure_subset.sh es_fm cv

WORLD_MODEL_CKPT=/zfsauton/scratch/sbellad/world_model_ckpts/v1/latest.pt \
WORLD_MODEL_DEVICE=cpu \
sbatch --job-name=es_fm_wm_fail1500 slurm_es_failure_subset.sh es_fm wm

sbatch --job-name=es_fm_gt_fail1500 slurm_es_failure_subset.sh es_fm gt
```

The result to look for is not merely overall success. Break out at least:

```text
overlap / dynamic collision
offroad
goal
success rate
```

The strongest result would be a clear reduction in overlap failures from CV to
WM while offroad/goal stay approximately unchanged.

---

## 12. Why this is a latent world model rather than one network per metric

The latent model predicts the **future scene state**:

```text
history -> z_t -> future latent sequence -> future agent states
```

The existing scorer still determines:

```text
collision / overlap
road feasibility
goal progress
etc.
```

This is important because CV vs GT differs in the forecast of other agents. The
world model is therefore inserted at the same interface as CV/GT rather than
replacing trustworthy geometric scoring logic with learned metric heads.

---

## 13. V1 limitations / recommended next experiments

This package intentionally implements the simplest scientifically clean first
version.

1. **V1 is deterministic.** It predicts one most-likely future per agent.
2. It is **not yet ego-candidate-conditioned**. All FM-ES ego candidates are
   evaluated against the same predicted surrounding world, matching the current
   CV/GT experimental structure.
3. It predicts dynamic-agent motion; it does not generate pixels or BEV frames.
4. Current validity is repeated over the horizon; there is no learned existence
   probability head yet.

Once V1 shows `WM > CV`, the best next upgrades are:

- multimodal K-future decoder with probabilities
- collision-risk aggregation over K predicted worlds
- ego-candidate/action-conditioned rollouts
- explicit validity/existence prediction
- heavier weighting or mining of turning/braking/yielding failure cases

Do **not** add these before establishing that the drop-in learned future forecast
closes a measurable part of the CV -> GT gap.
