#!/bin/bash
set -euo pipefail

GOAL_CONDITIONED="${1:-true}"
TAG="${2:-None}"
RESUME_PATH="${3:-}"

RUN_NAME="train_diffusion"

mkdir -p logs

echo "Submitting job with:"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  --export=GOAL_CONDITIONED="${GOAL_CONDITIONED}",TAG="${TAG}",RESUME_PATH="${RESUME_PATH}" \
  slurm_pretrain_diffusion.sh