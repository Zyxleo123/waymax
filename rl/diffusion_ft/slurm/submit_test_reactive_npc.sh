#!/bin/bash
# Submitter for the reactive-NPC mechanics gates. Only calls sbatch.
#
# Writes a timestamped env file and passes its path as a positional arg (never
# `sbatch --export`, which requeue-holds jobs on this cluster).
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_test_reactive_npc.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft/env"

STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="$ROOT/logs/dft/env/reactunit_${STAMP}.env"
{
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0-7}"
  echo "DFT_REACTIVE=${DFT_REACTIVE:-idm_brake}"
} > "$ENVF"
echo "Wrote env file: $ENVF"
cat "$ENVF"

# Scrub SLURM_* so submitting from inside an allocation does not leak the
# parent job's environment into the child (and never use --export).
env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_SUBMIT_DIR \
    -u SLURM_JOB_NAME -u SLURM_NTASKS -u SLURM_CPUS_PER_TASK \
  sbatch rl/diffusion_ft/slurm/test_reactive_npc.sbatch "$ENVF"
