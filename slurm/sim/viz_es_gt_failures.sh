#!/bin/bash
#SBATCH --job-name=viz_es_gt_fail
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Diffusion_ES_sim/logs/viz_es_gt_failures_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Diffusion_ES_sim/logs/viz_es_gt_failures_%j.err

# Re-simulate the Diffusion-ES (GT forecast) train/val failures with ES recording, then render.
#
#   sbatch slurm/sim/viz_es_gt_failures.sh
#   OUT_DIR=/path EXTRA="--max_batches 1" sbatch slurm/sim/viz_es_gt_failures.sh
#   python scripts/viz_es_gt_failures.py --phase render --out_dir /path    # re-render on CPU
set -euo pipefail

ROOT=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad
OUT_DIR=${OUT_DIR:-${SCRATCH}/Diffusion_ES_sim/viz_es_gt_failures_$(date +%Y%m%d_%H%M%S)}
EXTRA=${EXTRA:-}

set +u
# shellcheck source=/dev/null
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out_dir=${OUT_DIR} extra=${EXTRA}"
nvidia-smi || true
# shellcheck disable=SC2086
python scripts/viz_es_gt_failures.py --phase all --out_dir "${OUT_DIR}" ${EXTRA}
