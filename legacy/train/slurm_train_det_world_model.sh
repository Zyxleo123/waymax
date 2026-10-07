#!/bin/bash
#SBATCH --job-name=world_model
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/world_model
#SBATCH --output=/zfsauton/scratch/sbellad/world_model/world_model_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/world_model/world_model_%j.err

set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/world_model}"
CACHE_DIR="${CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
SCRATCH_WM="${SCRATCH_WM:-/zfsauton/scratch/sbellad/world_model}"
SAVE_DIR="${SAVE_DIR:-${SCRATCH_WM}/ckpts}"
DEVICE="${WORLD_MODEL_TRAIN_DEVICE:-cuda}"

CONDA_ROOT="${CONDA_ROOT:-/zfsauton/scratch/sbellad/miniconda3}"
WAYMAX_ENV="${WAYMAX_ENV:-/zfsauton/scratch/sbellad/conda_envs/waymax_rs}"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

# Create scratch world_model tree only if missing (mkdir -p is a no-op when present).
mkdir -p "${SCRATCH_WM}" "${SAVE_DIR}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-}  python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Full-data pass is ~80k batches/epoch (~1h+ before first val). Cap steps so
# each "epoch" is a SGD chunk + validation (~2k @ ~20 batch/s ≈ 1.5–2 min).
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
