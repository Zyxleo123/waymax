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

GOAL_CONDITIONED="${GOAL_CONDITIONED:-true}"
TAG="${TAG:-}"
RESUME_PATH="${RESUME_PATH:-}"

if [ "$GOAL_CONDITIONED" = "false" ]; then
    echo "Running scene-only diffusion training with tag: ${TAG}"
    python -m train_diffusion.train_diffusion_policy_v2 --no-goal_conditioned --tag "${TAG}" --resume_path "${RESUME_PATH}"
else
    echo "Running goal-conditioned diffusion training with tag: ${TAG}"
    python -m train_diffusion.train_diffusion_policy_v2 --tag "${TAG}" --resume_path "${RESUME_PATH}"
fi