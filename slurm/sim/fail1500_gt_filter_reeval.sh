#!/bin/bash
#SBATCH --job-name=gt_filter_reeval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_reeval_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_reeval_%j.err

# Build the GT filter (scenes that fail even with GT forecasts) and re-evaluate
# runs against it. Submits itself:  bash slurm/sim/fail1500_gt_filter_reeval.sh
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs
  exec sbatch "$(readlink -f "$0")" "$@"
fi

set --
set +u
# shellcheck source=/dev/null
source /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3/bin/activate /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore
python scripts/filter_gt_failures_and_reeval.py --batch_size 10 \
  --tf_root /data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example
