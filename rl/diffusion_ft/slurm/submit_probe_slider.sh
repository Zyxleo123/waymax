#!/bin/bash
# Submitter for the optimization-step probe slider. Only calls sbatch; writes one
# timestamped env file and passes its path positionally to the batch script (no
# `sbatch --export`, which triggers SLURM_GET_USER_ENV and requeue-holds the job).
#
# CPU-only: it reads each arm's probe/ dumps (static.npz + iter_*.npz), so it
# re-renders in seconds against a still-running run and can be resubmitted as more
# iter_*.npz land. The three base_anchor arms are the default; override with
# DFT_PROBE_ARMS='name=dir name=dir ...'.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea, so a
# compute node fails with "Access/permission denied"):
#   bash rl/diffusion_ft/slurm/submit_probe_slider.sh
#   DFT_PROBE_ARMS='beta3=/a beta10=/b' bash rl/diffusion_ft/slurm/submit_probe_slider.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

SWEEP="${DFT_SWEEP_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/base_anchor_sweep/20260909_094413_534731}"
ARMS="${DFT_PROBE_ARMS:-beta3=$SWEEP/beta3_succ beta10=$SWEEP/beta10_succ beta30=$SWEEP/beta30_succ}"
TITLE="${DFT_PROBE_TITLE:-base_anchor sweep -- recovered / regressed through training}"
FIG_DIR="${DFT_PROBE_FIG_DIR:-$SWEEP/figures}"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/probeslider_${STAMP}.env"
# The batch script SOURCES this file, so every value with spaces (the arm list,
# the title) MUST be single-quoted -- an unquoted `TITLE=a b` runs `b` as a
# command, and an unquoted arm list silently truncates to its first entry.
{
  echo "DFT_PROBE_ARMS='$ARMS'"
  echo "DFT_PROBE_TITLE='$TITLE'"
  echo "DFT_PROBE_FIG_DIR='$FIG_DIR'"
} > "$ENV_FILE"

echo "env  -> $ENV_FILE"
echo "arms -> $ARMS"
echo "figs -> $FIG_DIR"

# Scrub SLURM_* so a submission from inside an allocation does not set
# SLURM_GET_USER_ENV; pass the env file as a positional arg (travels untouched).
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/viz_probe_slider.sbatch "$ENV_FILE"
