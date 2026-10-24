#!/bin/bash
#SBATCH --job-name=vmh_train
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#
# Stage C: hardness-balanced fine-tune of the pretrained V-Max SAC (warm-started
# from model_best) with the refit risk table driving scenario sampling. All non-
# hardness overrides reproduce the original raw-WOMD run's .hydra/overrides.yaml;
# only init_checkpoint, hardness_*, total_timesteps, and the run name change.
#
# a6000 (48G) matches the original run (gpu24). vmax is imported via CWD (not
# pip-installed), so run from V-Max/.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/vmax_hardness"
# shellcheck source=/dev/null
source "$ROOT/rl/vmax_hardness/slurm/_preamble.sh" "$@"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

# Gate: the refit risk table and the warm-start checkpoint must exist.
if [[ ! -f "$VMH_RISK_OUT/risk.npz" ]]; then
  echo "GATE: $VMH_RISK_OUT/risk.npz missing -- Stage B did not refit the risk model." ; exit 0
fi
if [[ ! -f "$VMH_BEST_CKPT" ]]; then
  echo "GATE: warm-start checkpoint missing: $VMH_BEST_CKPT." ; exit 0
fi

echo "host=${SLURMD_NODENAME:-unknown} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "fine-tune: init=$VMH_BEST_CKPT risk=$VMH_RISK_OUT/risk.npz alpha=$VMH_ALPHA steps=$VMH_TOTAL_TIMESTEPS"
echo "save -> $VMH_SAVE_ROOT/$VMH_NAME_RUN"

cd "$ROOT/V-Max"
"$VMH_TRAIN_PYTHON" -m vmax.scripts.training.train \
  algorithm=sac \
  network/encoder=lq \
  path_dataset=/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord@1000 \
  path_dataset_eval=/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/validation/validation_tfexample.tfrecord@150 \
  waymo_dataset=false \
  num_sdc_paths="$VMH_NUM_PATHS" \
  num_points_per_sdc_path="$VMH_NUM_POINTS_PER_PATH" \
  max_num_objects="$VMH_MAX_NUM_OBJECTS" \
  total_timesteps="$VMH_TOTAL_TIMESTEPS" \
  num_envs=16 \
  num_episode_per_epoch=4 \
  scenario_length=80 \
  num_scenario_per_eval=256 \
  log_freq=5 \
  save_freq=2400 \
  eval_freq=50 \
  seed=0 \
  name_run="$VMH_NAME_RUN" \
  name_exp="$VMH_SAVE_ROOT" \
  observation_config.objects.num_closest_objects=16 \
  observation_config.roadgraphs.element_types="[road_edge]" \
  algorithm.network.policy.layer_sizes="[256,64,32]" \
  algorithm.network.value.layer_sizes="[256,64,32]" \
  reward_config.off_route=-0.2 \
  reward_config.progression=0.2 \
  reward_config.overlap=-1.0 \
  reward_config.offroad=-1.0 \
  reward_config.red_light=-1.0 \
  init_checkpoint="$VMH_BEST_CKPT" \
  hardness_risk_table="$VMH_RISK_OUT/risk.npz" \
  hardness_index_dir="$VMH_INDEX_DIR" \
  hardness_alpha="$VMH_ALPHA" \
  hardness_weight_clip="$VMH_WEIGHT_CLIP" \
  hardness_exclude_dirty="$VMH_EXCLUDE_DIRTY" \
  wandb="$VMH_WANDB" \
  wandb_project="$VMH_WANDB_PROJECT" \
  wandb_name="$VMH_NAME_RUN" \
  wandb_mode=online
