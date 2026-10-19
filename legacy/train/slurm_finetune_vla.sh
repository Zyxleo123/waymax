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

SCENE_TOKENIZER_CKPT_PATH="${SCENE_TOKENIZER_CKPT_PATH:-}"
PREDICT_SUBGOAL="${PREDICT_SUBGOAL:-false}"
TAG="${TAG:-}"

if [ "$PREDICT_SUBGOAL" = "true" ]; then
    echo "Running VLA finetuning with subgoal prediction and tag: ${TAG}"
    python -m train_vla.finetuning_vla --add_eos --scene_tokenizer_ckpt "${SCENE_TOKENIZER_CKPT_PATH}" --predict_subgoal --tag "${TAG}"
else
    echo "Running VLA finetuning without subgoal prediction and tag: ${TAG}"
    python -m train_vla.finetuning_vla --add_eos --scene_tokenizer_ckpt "${SCENE_TOKENIZER_CKPT_PATH}" --tag "${TAG}"
fi
