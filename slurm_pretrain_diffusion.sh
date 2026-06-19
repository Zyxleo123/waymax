#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate waymax_rs

SUBGOAL_CONDITIONED="${SUBGOAL_CONDITIONED:-false}"
TAG="${TAG:-}"
RESUME_PATH="${RESUME_PATH:-}"

if [ "$SUBGOAL_CONDITIONED" = "true" ]; then
    echo "Running subgoal-conditioned pretraining with tag: ${TAG}"
    python -m train_diffusion.pretrain_diffusion_policy --subgoal_conditioned --tag "${TAG}" --resume_path "${RESUME_PATH}"
else
    echo "Running standard pretraining with tag: ${TAG}"
    python -m train_diffusion.pretrain_diffusion_policy --tag "${TAG}" --resume_path "${RESUME_PATH}"
fi