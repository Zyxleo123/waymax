#!/bin/bash
# Submitter for the one-time WOMD byte-offset index build. Only calls sbatch.
#
#   bash rl/diffusion_ft/slurm/submit_build_womd_offsets.sh
#
# Env knobs:
#   DFT_SHARD_GLOB  which shards to index (default: all 1000 WOMD training shards)
#   DFT_INDEX_DIR   where the .offsets.npy files go (default: our own scratch)
#   DFT_ARRAY       array spec (default 0-19, i.e. 20 parallel tasks x 50 shards)
#
# Resumable: rerun the same command to fill in anything a task did not finish.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/offsets_${STAMP}.env"

{
  echo "DFT_SHARD_GLOB='${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*-of-01000}'"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --array="${DFT_ARRAY:-0-19}" \
  rl/diffusion_ft/slurm/build_womd_offsets.sbatch "$ENV_FILE"
