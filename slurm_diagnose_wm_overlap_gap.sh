#!/bin/bash
#SBATCH --job-name=wm_overlap_gap
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/wm_overlap_gap_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/wm_overlap_gap_%j.err

# Diagnose why WM does not reduce overlap vs GT-only wins.
#   sbatch slurm_diagnose_wm_overlap_gap.sh
#   MAX_SCENES=3 sbatch slurm_diagnose_wm_overlap_gap.sh
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

MAX_SCENES="${MAX_SCENES:-0}"
CMD=(
  python "${ROOT}/scripts/diagnose_wm_overlap_gap.py"
)
if [[ "${MAX_SCENES}" != "0" ]]; then
  CMD+=(--max_scenes "${MAX_SCENES}")
fi

nvidia-smi || true
"${CMD[@]}"
