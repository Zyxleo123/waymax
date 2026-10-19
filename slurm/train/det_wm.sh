#!/bin/bash
#SBATCH --job-name=det_wm
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs/train_deterministic_world_model
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/world_model_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/world_model_%j.err

# Train the deterministic world model (train_deterministic_world_model/).
#   bash slurm/train/det_wm.sh                      # submits itself
#   EPOCHS=300 BATCH_SIZE=32 bash slurm/train/det_wm.sh
# Hyper-parameters are environment variables (inherited by sbatch; no --export).
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model
  exec sbatch "$(readlink -f "$0")" "$@"
fi

REPO="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
ROOT="${REPO}/train_deterministic_world_model"
CACHE_DIR="${CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
SCRATCH_WM="${SCRATCH_WM:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model}"
SAVE_DIR="${SAVE_DIR:-${SCRATCH_WM}/ckpts}"
DEVICE="${WORLD_MODEL_TRAIN_DEVICE:-cuda}"

set --
set +u
# shellcheck source=/dev/null
source /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3/bin/activate /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
mkdir -p "${SCRATCH_WM}" "${SAVE_DIR}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo "host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-}  python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Each "epoch" is a capped SGD chunk + validation (a full pass is ~80k batches).
python train_world_model.py \
  --cache_dir "${CACHE_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS:-500}" \
  --steps_per_epoch "${STEPS_PER_EPOCH:-2000}" \
  --val_steps "${VAL_STEPS:-100}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --anchor_steps "${ANCHOR_STEPS:-20,30,40}" \
  --history_steps "${HISTORY_STEPS:-10}" \
  --history_stride "${HISTORY_STRIDE:-2}" \
  --future_steps "${FUTURE_STEPS:-25}" \
  --future_stride "${FUTURE_STRIDE:-2}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --max_range "${MAX_RANGE:-50}" \
  --wandb_project "${WANDB_PROJECT:-world Model}" \
  --wandb_entity "${WANDB_ENTITY:-scbellad2025-carnegie-mellon-university}" \
  --wandb_name "${WANDB_NAME:-world_model}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --amp
