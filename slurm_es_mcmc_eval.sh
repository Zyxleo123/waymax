#!/bin/bash
#SBATCH --job-name=es_mcmc_eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/zfsauton/scratch/ishanj/es_mcmc_eval/logs/%x_%j.log

# Usage: sbatch slurm_es_mcmc_eval.sh <out_tag> [es_mcmc_eval.py args...]
# e.g.   sbatch slurm_es_mcmc_eval.sh v1 --scenarios 00017:173 --variants baseline mcmc_lex
source ~/.bashrc
conda activate waymax_rs
cd /zfsauton/scratch/ishanj/waymax_rs
TAG=$1; shift
python notebooks/es_mcmc_eval.py --out /zfsauton/scratch/ishanj/es_mcmc_eval/${TAG} "$@"
