#!/bin/bash
#SBATCH --job-name=mf_coll_validate
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation/_validation/validate_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation/_validation/validate_%j.err

# Offline checks for the MF collision ablation (10 scenes x 3 replan steps, no rollout):
#   sbatch multiFuture_GaussianWorldModel/evaluation/slurm_validate_mf_collision_methods.sh
set -euo pipefail

ROOT="/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs"
OUT_DIR="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation/_validation"
mkdir -p "${OUT_DIR}"

set +u
# shellcheck source=/dev/null
source /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3/bin/activate /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}"

python multiFuture_GaussianWorldModel/evaluation/validate_mf_collision_methods.py \
  --device cuda \
  --out_json "${OUT_DIR}/offline_checks_$(date +%Y%m%d_%H%M%S).json"
