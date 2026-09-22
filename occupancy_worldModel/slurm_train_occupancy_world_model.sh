#!/bin/bash
#SBATCH --job-name=occ_wm
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs/occupancy_worldModel
#SBATCH --output=/zfsauton/scratch/sbellad/world_model/occupancy_world_model_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/world_model/occupancy_world_model_%j.err

set -euo pipefail

ROOT="${OCC_WM_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs/occupancy_worldModel}"
CACHE_DIR="${CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
SCRATCH_WM="${SCRATCH_WM:-/zfsauton/scratch/sbellad/world_model}"
SAVE_DIR="${SAVE_DIR:-${SCRATCH_WM}/ckpts_occupancy}"
DEVICE="${WORLD_MODEL_TRAIN_DEVICE:-cuda}"

CONDA_ROOT="${CONDA_ROOT:-/zfsauton/scratch/sbellad/miniconda3}"
WAYMAX_ENV="${WAYMAX_ENV:-/zfsauton/scratch/sbellad/conda_envs/waymax_rs}"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

mkdir -p "${SCRATCH_WM}" "${SAVE_DIR}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-}  python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

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
  --occ_x_min "${OCC_X_MIN:--10}" \
  --occ_x_max "${OCC_X_MAX:-40}" \
  --occ_y_min "${OCC_Y_MIN:--20}" \
  --occ_y_max "${OCC_Y_MAX:-20}" \
  --occ_cell_size "${OCC_CELL_SIZE:-0.5}" \
  --occ_loss_weight "${OCC_LOSS_WEIGHT:-1.0}" \
  --occ_sigma_scale "${OCC_SIGMA_SCALE:-1.0}" \
  --yaw_loss_weight "${YAW_LOSS_WEIGHT:-0.5}" \
  --wandb_project "${WANDB_PROJECT:-world Model}" \
  --wandb_entity "${WANDB_ENTITY:-scbellad2025-carnegie-mellon-university}" \
  --wandb_name "${WANDB_NAME:-occupancy_worldModel}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --amp
