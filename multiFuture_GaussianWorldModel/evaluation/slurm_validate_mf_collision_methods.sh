#!/bin/bash
#SBATCH --job-name=mf_coll_validate
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation/_validation/validate_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation/_validation/validate_%j.err

# Offline checks for the MF collision ablation (10 scenes x 3 replan steps, no rollout):
#   sbatch multiFuture_GaussianWorldModel/evaluation/slurm_validate_mf_collision_methods.sh
set -euo pipefail

ROOT="/zfsauton2/home/sbellad/baseline/waymax_rs"
OUT_DIR="/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation/_validation"
mkdir -p "${OUT_DIR}"

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}"

python multiFuture_GaussianWorldModel/evaluation/validate_mf_collision_methods.py \
  --device cuda \
  --out_json "${OUT_DIR}/offline_checks_$(date +%Y%m%d_%H%M%S).json"
