#!/bin/bash
# Submitter for the scene-sampler gate. Only calls sbatch.
#
#   bash rl/diffusion_ft/slurm/submit_test_scene_sampler.sh
#
# Run it after the offset-index build (submit_build_womd_offsets.sh) and before
# any full-corpus training run.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
ENV_FILE="$ENV_DIR/sampler_gate_$(date +%Y%m%d_%H%M%S).env"

echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}" > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

SB_FLAGS=()
[[ -n "${DFT_DEPEND:-}" ]] && SB_FLAGS+=(--dependency="$DFT_DEPEND")

env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch "${SB_FLAGS[@]}" rl/diffusion_ft/slurm/test_scene_sampler.sbatch "$ENV_FILE"
