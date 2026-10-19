#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate waymax_rs

PREDICT_SUBGOAL="${PREDICT_SUBGOAL:-false}"
TAG="${TAG:-None}"

if [ "$PREDICT_SUBGOAL" = "true" ]; then
    echo "Running VLA pretraining with subgoal prediction and tag: ${TAG}"
    python -m train_vla.pretrain_vla --predict_subgoal --tag "${TAG}"
else
    echo "Running standard VLA pretraining with tag: ${TAG}"
    python -m train_vla.pretrain_vla --tag "${TAG}"
fi
