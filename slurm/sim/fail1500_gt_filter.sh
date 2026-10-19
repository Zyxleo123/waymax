#!/bin/bash
#SBATCH --job-name=gt_filter_fail1500
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_fail1500_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/gt_filter_fail1500_%j.err

# Re-score fail1500 runs on the GT-filtered subset (drops scenes that fail even
# with GT forecasts; n=1102 of 1500). Submits itself.
#   bash slurm/sim/fail1500_gt_filter.sh                 # reuse the shared GT filter
#   GT_FILTER_SUMMARY= bash slurm/sim/fail1500_gt_filter.sh   # rebuild the filter
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

GT_FILTER_SUMMARY="${GT_FILTER_SUMMARY-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/gt_filtered_fail1500_20260911_170730/gt_filter/gt_filter_summary.json}"
CMD=(python scripts/filter_gt_failures_fail1500.py --batch_size 5
     --tf_root /data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training)
if [[ -n "${GT_FILTER_SUMMARY}" ]]; then
  CMD+=(--gt_filter_summary "${GT_FILTER_SUMMARY}")
fi
"${CMD[@]}"
