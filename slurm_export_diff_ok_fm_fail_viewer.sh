#!/bin/bash
#SBATCH --job-name=diff_fm_viewer50
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/diff_fm_viewer50_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/diff_fm_viewer50_%j.err

# Select 50 Diff-GT OK / FM-GT fail cases, replan both, save full scenario videos + diagnostics.
# Elite endpoint score/mag label is omitted (export script updated).
set -euo pipefail
mkdir -p /zfsauton/scratch/sbellad/immediate_exp/logs
set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="/zfsauton2/home/sbellad/baseline/waymax_rs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore

python scripts/export_diff_ok_fm_fail_viewer.py \
  --mode gt \
  --n 50 \
  --seed 0 \
  --lane_graph_dir /zfsauton/scratch/mineuih/waymax_rs/lane_graphs \
  --out_dir /zfsauton/scratch/sbellad/immediate_exp/diff_ok_fm_fail_viewer_gt
