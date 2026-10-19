#!/bin/bash
#SBATCH --job-name=mf_gauss_wm
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs/multiFuture_GaussianWorldModel
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/multifuture_gaussian_world_model_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/multifuture_gaussian_world_model_%j.err

set -euo pipefail

ROOT="${MULTIFUTURE_WM_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs/multiFuture_GaussianWorldModel}"
CACHE_DIR="${CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
SCRATCH_WM="${SCRATCH_WM:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model}"
SAVE_DIR="${SAVE_DIR:-${SCRATCH_WM}/ckpts_multifuture_gaussian}"
DEVICE="${WORLD_MODEL_TRAIN_DEVICE:-cuda}"

CONDA_ROOT="${CONDA_ROOT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3}"
WAYMAX_ENV="${WAYMAX_ENV:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs}"
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
  --num_modes "${NUM_MODES:-6}" \
  --yaw_loss_weight "${YAW_LOSS_WEIGHT:-0.5}" \
  --mode_cls_weight "${MODE_CLS_WEIGHT:-1.0}" \
  --sigma_eps "${SIGMA_EPS:-1e-3}" \
  --planning_mode "${PLANNING_MODE:-top1}" \
  --wandb_project "${WANDB_PROJECT:-world Model}" \
  --wandb_entity "${WANDB_ENTITY:-scbellad2025-carnegie-mellon-university}" \
  --wandb_name "${WANDB_NAME:-multiFuture_GaussianWorldModel}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --amp
