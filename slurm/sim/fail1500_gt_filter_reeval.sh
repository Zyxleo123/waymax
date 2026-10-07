#!/bin/bash
#SBATCH --job-name=gt_filter_reeval
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_reeval_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_reeval_%j.err

# Build the GT filter (scenes that fail even with GT forecasts) and re-evaluate
# runs against it. Submits itself:  bash slurm/sim/fail1500_gt_filter_reeval.sh
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p /zfsauton/scratch/sbellad/immediate_exp/logs
  exec sbatch "$(readlink -f "$0")" "$@"
fi

set --
set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="/zfsauton2/home/sbellad/baseline/waymax_rs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore
python scripts/filter_gt_failures_and_reeval.py --batch_size 10 \
  --tf_root /zfsauton/datasets/womd/tf_example
