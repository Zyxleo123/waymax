#!/bin/bash
#SBATCH --job-name=cv_vs_wm
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/cv_vs_wm_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/cv_vs_wm_%j.err

# Offline CV vs World-Model forecast accuracy vs logged GT (no ego planner).
#
#   sbatch slurm_compare_cv_vs_wm.sh
#   SPLIT=all MAX_BATCHES=0 sbatch slurm_compare_cv_vs_wm.sh
#   SPLIT=val MAX_BATCHES=200 sbatch --job-name=cv_vs_wm_smoke slurm_compare_cv_vs_wm.sh
#
set -euo pipefail
mkdir -p /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs

ROOT="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
WM_ROOT="${WORLD_MODEL_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/world_model}"
cd "${ROOT}"

CONDA_ROOT="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3"
WAYMAX_ENV="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

CACHE_DIR="${CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"
MAX_BATCHES="${MAX_BATCHES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ANCHOR_STEPS="${ANCHOR_STEPS:-20,30,40}"
NEARBY_RADIUS_M="${NEARBY_RADIUS_M:-20}"
OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp}"
TAG="${TAG:-${SPLIT}_full}"

export WORLD_MODEL_ROOT="${WM_ROOT}"
export PYTHONPATH="${WM_ROOT}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS=ignore

echo "host=$(hostname)  device=${DEVICE}  split=${SPLIT}  max_batches=${MAX_BATCHES}"
echo "ckpt=${WORLD_MODEL_CKPT}"
echo "cache=${CACHE_DIR}"
nvidia-smi || true

python "${ROOT}/scripts/compare_cv_vs_wm_forecast.py" \
  --cache_dir "${CACHE_DIR}" \
  --world_model_ckpt "${WORLD_MODEL_CKPT}" \
  --device "${DEVICE}" \
  --split "${SPLIT}" \
  --max_batches "${MAX_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --anchor_steps "${ANCHOR_STEPS}" \
  --nearby_radius_m "${NEARBY_RADIUS_M}" \
  --out_dir "${OUT_DIR}" \
  --tag "${TAG}"
