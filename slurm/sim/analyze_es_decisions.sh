#!/bin/bash
#SBATCH --job-name=es_decisions
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/es_discriminator/logs/analyze_es_decisions_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/es_discriminator/logs/analyze_es_decisions_%j.err

# Label consistency, controllable failures and long-horizon decision quality for the
# recorded fail1500 Diffusion-ES run (es_discriminator/analyze_es_decisions.py):
#   1. Waymax-exact offroad labels for all 3.84M candidates -> <batch>.labels_v2.npz
#   2. per-step Waymax overlap/offroad of every executed episode -> scenes.npz
#   3. report.md / report.json (+ invalid_at_start.json)
# Submit from rhea or lov4:  sbatch slurm/sim/analyze_es_decisions.sh
# Resumable: finished labels_v2 files and scenes.npz are reused.
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
RUN=${SCRATCH}/es_discriminator/es_candidate_data/fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed0
TFRECORDS=${SCRATCH}/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords/training
OUT=${SCRATCH}/es_discriminator/decision_analysis
set --  # conda's activate script reads "$@"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

set +u  # conda activate.d hooks reference unset variables
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} out=${OUT}"

python -u -m es_discriminator.analyze_es_decisions all \
  --data_dir "${RUN}/es_candidates" --tfrecord_dir "${TFRECORDS}" --out_dir "${OUT}"
echo "[$(date)] done: ${OUT}/report.md"
