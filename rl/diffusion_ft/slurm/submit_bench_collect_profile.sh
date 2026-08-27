#!/bin/bash
# Submitter for the update-path GPU benchmark. Only calls sbatch.
#
# Usage (from repo root):
#   bash rl/diffusion_ft/slurm/submit_bench_collect_profile.sh
#
# Override any DFT_* by exporting it before calling, e.g.
#   DFT_INDICES=0,1,2,3 DFT_PPO_EPOCHS=2 bash rl/diffusion_ft/slurm/submit_bench_collect_profile.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/prof_${STAMP}.env"

{
  [[ -n "${DFT_CKPT:-}" ]]        && echo "DFT_CKPT=${DFT_CKPT}"
  [[ -n "${DFT_TFRECORD:-}" ]]    && echo "DFT_TFRECORD=${DFT_TFRECORD}"
  [[ -n "${DFT_INDICES:-}" ]]     && echo "DFT_INDICES=${DFT_INDICES}"
  [[ -n "${DFT_K_TRAINABLE:-}" ]] && echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE}"
  [[ -n "${DFT_PREFIX_LEN:-}" ]]  && echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN}"
  [[ -n "${DFT_PPO_EPOCHS:-}" ]]  && echo "DFT_PPO_EPOCHS=${DFT_PPO_EPOCHS}"
  [[ -n "${DFT_SIGMA_FLOOR:-}" ]] && echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR}"
  [[ -n "${DFT_TARGET_KL:-}" ]]   && echo "DFT_TARGET_KL=${DFT_TARGET_KL}"
  :
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# Scrub SLURM_* so submitting from inside an allocation does not leak job state.
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/bench_collect_profile.sbatch "$ENV_FILE"
