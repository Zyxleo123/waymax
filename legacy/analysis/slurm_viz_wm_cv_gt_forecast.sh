#!/bin/bash
#SBATCH --job-name=viz_wm_cv_gt
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/viz_wm_cv_gt_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/viz_wm_cv_gt_%j.err

# Overlay GT vs CV vs WM forecasts on scenes where CV and GT succeed
# and WM fails with overlap.
#   sbatch slurm_viz_wm_cv_gt_forecast.sh
#   MAX_SCENES=3 sbatch slurm_viz_wm_cv_gt_forecast.sh
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
cd "${ROOT}"
mkdir -p /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs

CONDA_ROOT="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3"
WAYMAX_ENV="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS=ignore
export MPLBACKEND=Agg

MAX_SCENES="${MAX_SCENES:-0}"
CMD=(
  python "${ROOT}/scripts/viz_wm_cv_gt_forecast.py"
)
if [[ "${MAX_SCENES}" != "0" ]]; then
  CMD+=(--max_scenes "${MAX_SCENES}")
fi

nvidia-smi || true
"${CMD[@]}"
