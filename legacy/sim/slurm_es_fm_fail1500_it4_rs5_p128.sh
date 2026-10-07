#!/bin/bash
#SBATCH --job-name=es_fm_fail1500_it4_rs5_p128
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/Flow_matching_sim/logs/es_fail1500_it4_rs5_p128_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/Flow_matching_sim/logs/es_fail1500_it4_rs5_p128_%j.err

# FM-ES fail1500: match Diffusion resample budget (5 steps), default ES iters=4,
# larger pop/elite (128/16). Env set in job body (no sbatch --export).
#
# Submit:
#   sbatch --job-name=es_fm_gt_fail1500_it4_rs5_p128_e16 slurm_es_fm_fail1500_it4_rs5_p128.sh gt
#   bash slurm_submit_es_fm_fail1500_rs5_p128.sh
#
set -euo pipefail

FORECAST="${1:-}"
FORECAST="$(echo "${FORECAST}" | tr '[:upper:]' '[:lower:]')"
if [[ "${FORECAST}" != "gt" && "${FORECAST}" != "cv" ]]; then
  echo "Usage: sbatch slurm_es_fm_fail1500_it4_rs5_p128.sh {gt|cv}"
  exit 1
fi

export NUM_ES_ITERATIONS=4
export RESAMPLE_NUM_STEPS=5
export RESAMPLE_T_START=0.2
export FM_NUM_STEPS=50
export FM_SOLVER=euler
export POPULATION_SIZE=128
export ELITE_SIZE=16
ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
exec bash "${ROOT}/legacy/sim/slurm_es_failure_subset_1500.sh" es_fm "${FORECAST}"
