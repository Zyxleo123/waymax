#!/bin/bash
#SBATCH --job-name=mf_gauss_wm
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs/multiFuture_GaussianWorldModel
#SBATCH --output=/zfsauton/scratch/sbellad/world_model/multifuture_gaussian_world_model_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/world_model/multifuture_gaussian_world_model_%j.err

# Train the multi-future Gaussian world model (multiFuture_GaussianWorldModel/).
#   bash slurm/train/mf_wm.sh                       # submits itself
#   NUM_MODES=8 EPOCHS=300 bash slurm/train/mf_wm.sh
# Hyper-parameters are environment variables (inherited by sbatch; no --export).
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p /zfsauton/scratch/sbellad/world_model
  exec sbatch "$(readlink -f "$0")" "$@"
fi

REPO="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
ROOT="${REPO}/multiFuture_GaussianWorldModel"
CACHE_DIR="${CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
SCRATCH_WM="${SCRATCH_WM:-/zfsauton/scratch/sbellad/world_model}"
SAVE_DIR="${SAVE_DIR:-${SCRATCH_WM}/ckpts_multifuture_gaussian}"
DEVICE="${WORLD_MODEL_TRAIN_DEVICE:-cuda}"

set --
set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
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
