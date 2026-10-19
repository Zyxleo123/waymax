#!/bin/bash
# Submitter for the 8-panel post-hoc probe. Only calls sbatch. Chains:
#   GPU  probe_offline (classify at last ckpt, pick K/bucket, re-roll)
#     ->(afterok) CPU  viz_probe_slider (render the 8-column slider)
# so the figure is redrawable without recomputation (rerun the viz alone).
#
# Config travels as positional env FILEs (never `sbatch --export`). List-valued
# and space-bearing values are single-quoted so the SOURCED file preserves them.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_probe_offline.sh
#   DFT_OFFLINE_RUN_DIR=/path/to/run DFT_OFFLINE_K=4 bash rl/diffusion_ft/slurm/submit_probe_offline.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

RUN_DIR="${DFT_OFFLINE_RUN_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/base_anchor_sweep/20260908_225908_492838/beta100_succ}"
SPLITS_JSON="${DFT_SPLITS_JSON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_verified.json}"
K="${DFT_OFFLINE_K:-3}"
ARM_NAME="${DFT_ARM_NAME:-beta100}"
OFFLINE_DIR="${DFT_OFFLINE_DIR:-$RUN_DIR/probe_offline}"
FIG_DIR="${DFT_PROBE_FIG_DIR:-$RUN_DIR/figures_offline}"
TITLE="${DFT_PROBE_TITLE:-$ARM_NAME -- 8 transition buckets (originally fail/success x now fail/success x train/heldout)}"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
GPU_ENV="$ENV_DIR/probeoff_gpu_${STAMP}.env"
VIZ_ENV="$ENV_DIR/probeoff_viz_${STAMP}.env"
{
  echo "DFT_OFFLINE_RUN_DIR='$RUN_DIR'"
  echo "DFT_OFFLINE_DIR='$OFFLINE_DIR'"
  echo "DFT_OFFLINE_K='$K'"
  echo "DFT_SPLITS_JSON='$SPLITS_JSON'"
} > "$GPU_ENV"
{
  echo "DFT_PROBE_ARMS='$ARM_NAME=$OFFLINE_DIR'"
  echo "DFT_PROBE_TITLE='$TITLE'"
  echo "DFT_PROBE_FIG_DIR='$FIG_DIR'"
} > "$VIZ_ENV"

echo "gpu env -> $GPU_ENV"
echo "viz env -> $VIZ_ENV"
echo "run     -> $RUN_DIR"
echo "figs    -> $FIG_DIR"

# Scrub SLURM_* so a submission from inside an allocation does not set
# SLURM_GET_USER_ENV; env files ride through as untouched positional args.
JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable rl/diffusion_ft/slurm/probe_offline.sbatch "$GPU_ENV")
echo "submitted GPU probe_offline job $JID"
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --dependency="afterok:$JID" rl/diffusion_ft/slurm/viz_probe_slider.sbatch "$VIZ_ENV"
echo "submitted CPU viz (afterok:$JID)"
