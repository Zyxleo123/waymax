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

set -euo pipefail
mkdir -p /zfsauton/scratch/sbellad/immediate_exp/logs
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
export PYTHONPATH="/zfsauton2/home/sbellad/baseline/waymax_rs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore
python scripts/filter_gt_failures_and_reeval.py --batch_size 10
