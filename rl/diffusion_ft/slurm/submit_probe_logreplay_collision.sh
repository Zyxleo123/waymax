#!/bin/bash
# Submitter for the log-replay collision probe. Only calls sbatch.
#
# Writes a timestamped env file and passes its path as a positional arg (never
# `sbatch --export`, which requeue-holds jobs on this cluster).
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_probe_logreplay_collision.sh
#   # finer sweep near the interesting region:
#   DFT_LATERAL="0,0.1,0.2,0.3,0.5,0.75,1.0" \
#     bash rl/diffusion_ft/slurm/submit_probe_logreplay_collision.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft/env"

STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="$ROOT/logs/dft/env/lrprobe_${STAMP}.env"
{
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0-31}"
  echo "DFT_LATERAL=${DFT_LATERAL:-0,0.25,0.5,1.0,1.5,2.0,3.0}"
  echo "DFT_DELAY=${DFT_DELAY:-0,1,2,3,5}"
  echo "DFT_PROBE_OUT=${DFT_PROBE_OUT:-runs/dft_logreplay_probe/collision_vs_deviation_$STAMP.json}"
} > "$ENVF"
echo "Wrote env file: $ENVF"
cat "$ENVF"

# Scrub SLURM_* so submitting from inside an allocation does not leak the
# parent job's environment into the child (and never use --export).
env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_SUBMIT_DIR \
    -u SLURM_JOB_NAME -u SLURM_NTASKS -u SLURM_CPUS_PER_TASK \
  sbatch rl/diffusion_ft/slurm/probe_logreplay_collision.sbatch "$ENVF"
