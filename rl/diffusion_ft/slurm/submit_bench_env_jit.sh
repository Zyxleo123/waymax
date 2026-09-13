#!/bin/bash
# Submitter for the eager-vs-jitted env replan A/B. Only calls sbatch.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea):
#   bash rl/diffusion_ft/slurm/submit_bench_env_jit.sh
#   DFT_NODELIST=gpu26 bash rl/diffusion_ft/slurm/submit_bench_env_jit.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/envjit_${STAMP}.env"
OUT_DIR="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bench_envjit/${STAMP}}"

{
  echo "DFT_OUT_DIR=$OUT_DIR"
  echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-32}"
  echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_BC_WEIGHT=${DFT_BC_WEIGHT:-300}"
  echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
  echo "DFT_ROUNDS=${DFT_ROUNDS:-4}"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

JOB_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/bench_env_jit.sbatch "$ENV_FILE")
echo "bench -> job $JOB_ID  ($OUT_DIR)"
echo "table:  grep -A12 'jit_replan A/B' $ROOT/logs/dft/slurm-dft_envjit-${JOB_ID}.out"
