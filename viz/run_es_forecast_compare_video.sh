#!/bin/bash
#SBATCH --job-name=es_gt_cv_wm_video
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/viz/logs/es_gt_cv_wm_video_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/viz/logs/es_gt_cv_wm_video_%j.err

# Side-by-side GT / CV / WM Diffusion-ES videos for the validation failures.
# Output: /zfsauton/scratch/sbellad/viz/es_gt_cv_wm_val_failures_<stamp>/index.html
#
#   mkdir -p /zfsauton/scratch/sbellad/viz/logs && sbatch viz/run_es_forecast_compare_video.sh
#   OUT_DIR=/path EXTRA="--scene_set cv" sbatch viz/run_es_forecast_compare_video.sh
#   python viz/es_forecast_compare_video.py --phase render --out_dir /path     # re-render only (CPU)
#
# Re-submitting with the same OUT_DIR resumes: finished scenes are skipped.
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
OUT_DIR=${OUT_DIR:-${SCRATCH}/viz/es_gt_cv_wm_val_failures_$(date +%Y%m%d_%H%M%S)}
EXTRA=${EXTRA:-}

set +u
# shellcheck source=/dev/null
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
# XLA CUDA command buffers intermittently abort long runs; the torch WM shares the GPU with JAX.
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out_dir=${OUT_DIR} extra=${EXTRA}"
nvidia-smi || true
# shellcheck disable=SC2086
python viz/es_forecast_compare_video.py --phase all --out_dir "${OUT_DIR}" ${EXTRA}
