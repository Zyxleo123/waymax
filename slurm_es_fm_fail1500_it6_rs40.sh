#!/bin/bash
#SBATCH --job-name=es_fm_fail1500_it6_rs40
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/Flow_matching_sim/logs/es_fail1500_it6_rs40_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/Flow_matching_sim/logs/es_fail1500_it6_rs40_%j.err

# FM-ES fail1500 with Phase-2 best settings from n=250 probe:
#   t_start=0.2, num_es_iterations=6, resample_num_steps=40, fm_num_steps=50, euler
#
# Hparams are set HERE in the job body (not via sbatch --export — that breaks
# on this cluster with "user env retrieval failed").
#
# Submit (from rhea/lov4), NO --export:
#   bash slurm_submit_es_fm_fail1500_it6_rs40.sh
# or manually:
#   sbatch --job-name=es_fm_gt_fail1500_it6_rs40 slurm_es_fm_fail1500_it6_rs40.sh gt
#   sbatch --job-name=es_fm_cv_fail1500_it6_rs40 slurm_es_fm_fail1500_it6_rs40.sh cv
#
set -euo pipefail

FORECAST="${1:-}"
FORECAST="$(echo "${FORECAST}" | tr '[:upper:]' '[:lower:]')"
if [[ "${FORECAST}" != "gt" && "${FORECAST}" != "cv" ]]; then
  echo "Usage: sbatch slurm_es_fm_fail1500_it6_rs40.sh {gt|cv}"
  exit 1
fi

# Phase-2 winner settings (applied inside the compute job).
export NUM_ES_ITERATIONS=6
export RESAMPLE_NUM_STEPS=40
export RESAMPLE_T_START=0.2
export FM_NUM_STEPS=50
export FM_SOLVER=euler
export RUN_SUFFIX=it6_rs40
export POPULATION_SIZE="${POPULATION_SIZE:-64}"
export ELITE_SIZE="${ELITE_SIZE:-8}"

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
exec bash "${ROOT}/slurm_es_failure_subset_1500.sh" es_fm "${FORECAST}"
