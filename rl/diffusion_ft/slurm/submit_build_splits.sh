#!/bin/bash
# Build the fixed eval splits (failure / success held-out + train) ONCE.
# Only calls sbatch. Chain: GPU headroom-scan array -> CPU split builder, afterok.
#
# The scan rolls the base policy (det + stochastic) and the expert log over a
# seeded WOMD sample; the builder keeps the expert-clean scenes and partitions
# them by whether the base already solves them. Output eval_splits.json is a
# build-once asset: point DFT_SPLITS_JSON at it and re-score any checkpoint
# cheaply with submit_eval_reach.sh.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_build_splits.sh
#   DFT_SCAN_SCENES=8192 DFT_ARRAY=0-15 bash rl/diffusion_ft/slurm/submit_build_splits.sh
#
# Knobs: DFT_SCAN_SCENES (default 4096), DFT_ARRAY (task range, default 0-7),
#   DFT_SCAN_SEED, DFT_HELDOUT_FAILURE/SUCCESS (default 64 each), DFT_SPLIT_SEED,
#   DFT_SHARD_GLOB, and placement DFT_PARTITION DFT_QOS DFT_NODELIST DFT_EXCLUDE.
# GPU class: the scan needs a6000/a5000 (general). If it lands on a v100/2080 Ti
# it dies at cuDNN autotune -- pin with DFT_EXCLUDE if a scan task fails at iter 0.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

DFT_SCAN_SEED="${DFT_SCAN_SEED:-0}"
DFT_SCAN_OUT="${DFT_SCAN_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed${DFT_SCAN_SEED}}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/buildsplits_${STAMP}.env"
{
  echo "DFT_SCAN_OUT=$DFT_SCAN_OUT"
  echo "DFT_SCAN_SEED=$DFT_SCAN_SEED"
  echo "DFT_SHARD_GLOB=$DFT_SHARD_GLOB"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_SCAN_SCENES=${DFT_SCAN_SCENES:-4096}"
  echo "DFT_SCAN_BATCH=${DFT_SCAN_BATCH:-32}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.03}"
  echo "DFT_NUM_PATHS=${DFT_NUM_PATHS:-45}"
  echo "DFT_NUM_POINTS_PER_PATH=${DFT_NUM_POINTS_PER_PATH:-800}"
  echo "DFT_SPLITS_JSON=${DFT_SPLITS_JSON:-${DFT_SCAN_OUT}/eval_splits.json}"
  echo "DFT_HELDOUT_FAILURE=${DFT_HELDOUT_FAILURE:-64}"
  echo "DFT_HELDOUT_SUCCESS=${DFT_HELDOUT_SUCCESS:-64}"
  echo "DFT_SPLIT_SEED=${DFT_SPLIT_SEED:-0}"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

# Scrub SLURM_* so a submission from inside an allocation does not inherit it.
SCAN=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --array="${DFT_ARRAY:-0-7}" "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/scan_womd_headroom.sbatch "$ENV_FILE")
echo "scan array -> job $SCAN  (GPU, ${DFT_ARRAY:-0-7})"

# afterok on an array waits for every task; the builder reads whatever rows exist.
BUILD=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency=afterok:"$SCAN" \
  rl/diffusion_ft/slurm/build_eval_splits.sbatch "$ENV_FILE")
echo "build      -> job $BUILD  (CPU, afterok:$SCAN)"
echo
echo "splits json when done: ${DFT_SPLITS_JSON:-${DFT_SCAN_OUT}/eval_splits.json}"
echo "then score a checkpoint on the splits:"
echo "  DFT_OUT_DIR=<run> DFT_SPLITS_JSON=${DFT_SCAN_OUT}/eval_splits.json \\"
echo "    bash rl/diffusion_ft/slurm/submit_eval_reach.sh"
