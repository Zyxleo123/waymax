#!/bin/bash
set -euo pipefail


PREDICT_SUBGOAL="${1:-false}"
TAG="${2:-None}"


RUN_NAME="pretrain_vla"

mkdir -p logs

echo "Submitting job with:"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  --export=PREDICT_SUBGOAL="${PREDICT_SUBGOAL}",TAG="${TAG}" \
  slurm_pretrain_vla.sh