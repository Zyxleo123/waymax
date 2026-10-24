#!/bin/bash
# Submits slurm_train_diffusion_v2.sh: branch new_model's diffusion policy (model/diffusion/diffusion_policy_v2.py)
# trained with train_diffusion/train_diffusion_policy_v2.py.
# Usage: ./slurm_train_diffusion_v2_submit.sh [GOAL_CONDITIONED=true] [TAG] [RESUME_PATH]
set -euo pipefail

GOAL_CONDITIONED="${1:-true}"
TAG="${2:-None}"
RESUME_PATH="${3:-}"

RUN_NAME="train_diffusion_v2"

mkdir -p logs

echo "Submitting job with:"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

# Variables go in the environment (sbatch exports it); --export=VAR=... gets jobs held on the general partition.
GOAL_CONDITIONED="${GOAL_CONDITIONED}" TAG="${TAG}" RESUME_PATH="${RESUME_PATH}" sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  slurm_train_diffusion_v2.sh
