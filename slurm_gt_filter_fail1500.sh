#!/bin/bash
#SBATCH --job-name=gt_filter_fail1500
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_fail1500_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_fail1500_%j.err

set -euo pipefail
mkdir -p /zfsauton/scratch/sbellad/immediate_exp/logs
set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="/zfsauton2/home/sbellad/baseline/waymax_rs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore

# Reuse the shared fail1500 GT filter by default (same 1500 scenes). Override to rebuild:
#   GT_FILTER_SUMMARY= sbatch slurm_gt_filter_fail1500.sh
GT_FILTER_SUMMARY="${GT_FILTER_SUMMARY:-/zfsauton/scratch/sbellad/immediate_exp/gt_filtered_fail1500_20260911_170730/gt_filter/gt_filter_summary.json}"

CMD=(python scripts/filter_gt_failures_fail1500.py --batch_size 5)
if [[ -n "${GT_FILTER_SUMMARY}" ]]; then
  CMD+=(--gt_filter_summary "${GT_FILTER_SUMMARY}")
fi
"${CMD[@]}"
