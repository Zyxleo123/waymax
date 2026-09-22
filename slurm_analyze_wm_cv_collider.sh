#!/bin/bash
#SBATCH --job-name=wm_cv_collider
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/wm_cv_collider_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/wm_cv_collider_%j.err

# Collision-critical CV vs WM errors on GT-ok / WM-overlap / CV-overlap scenes.
#   sbatch slurm_analyze_wm_cv_collider.sh
#   MAX_SCENES=2 sbatch slurm_analyze_wm_cv_collider.sh
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
cd "${ROOT}"
mkdir -p /zfsauton/scratch/sbellad/immediate_exp/logs

CONDA_ROOT="/zfsauton/scratch/sbellad/miniconda3"
WAYMAX_ENV="/zfsauton/scratch/sbellad/conda_envs/waymax_rs"
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
  python "${ROOT}/scripts/analyze_wm_cv_collider_errors.py"
)
if [[ "${MAX_SCENES}" != "0" ]]; then
  CMD+=(--max_scenes "${MAX_SCENES}")
fi

nvidia-smi || true
"${CMD[@]}"
