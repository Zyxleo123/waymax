#!/bin/bash
#SBATCH --job-name=cache_wm
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/es_discriminator/logs/cache_wm_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/es_discriminator/logs/cache_wm_%j.err

# One frozen multi-future Gaussian WM forward per recorded Diffusion-ES replan (on the executed
# ego state the planner saw), 32 candidate-independent agents (current + predicted occupancy),
# 6 modes x 25 steps of means / std / covariance / probabilities, and 256 road-edge tokens, for
# all 300 fail1500 batch files. Validates every file against the recorded live-planner forecast
# (fails the job on mismatch), plus the logged-state control and agent / edge coverage on 6 files.
# Resumable: batches with both caches are skipped (--overwrite to recompute).
# Submit from rhea or lov4:  sbatch slurm/train/cache_wm_predictions.sh
#
# Output: ${SCRATCH}/es_discriminator/{wm_predictions,road_edges}/<run>/  (+ manifest.json, summary.json)
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
RUN=fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed0
DATA=${SCRATCH}/es_discriminator/es_candidate_data/${RUN}/es_candidates
TFRECORDS=${SCRATCH}/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords/training
WM_DIR=${SCRATCH}/es_discriminator/wm_predictions/${RUN}
EDGE_DIR=${SCRATCH}/es_discriminator/road_edges/${RUN}
EXTRA=("$@")
set --  # conda's activate script reads "$@"
mkdir -p "${WM_DIR}" "${EDGE_DIR}"
exec > >(tee -a "${WM_DIR}/job_${SLURM_JOB_ID:-local}.log") 2>&1

set +u  # conda activate.d hooks reference unset variables
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} wm=${WM_DIR} edges=${EDGE_DIR}"

python -u -m es_discriminator.real_es.wm_cache --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
  --wm_dir "${WM_DIR}" --edge_dir "${EDGE_DIR}" --validate_files 6 \
  --summary_out "${WM_DIR}/summary_${SLURM_JOB_ID:-local}.json" "${EXTRA[@]}"

n_wm=$(find "${WM_DIR}" -name '*.wm.npz' | wc -l)
n_edge=$(find "${EDGE_DIR}" -name '*.edges.npz' | wc -l)
echo "[$(date)] cached ${n_wm}/300 WM files, ${n_edge}/300 edge files"
[[ ${n_wm} -eq 300 && ${n_edge} -eq 300 ]] || { echo "ERROR: incomplete cache" >&2; exit 1; }
echo "[$(date)] done"
