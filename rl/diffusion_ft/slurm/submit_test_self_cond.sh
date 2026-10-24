#!/bin/bash
# Submitter for the FlowMol3 self-cond / distortion gates. Only calls sbatch.
#
#   bash rl/diffusion_ft/slurm/submit_test_self_cond.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/selfcond_gates_${STAMP}.env"

{
  echo "JAX_PLATFORMS=cpu"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/test_self_cond.sbatch "$ENV_FILE"
