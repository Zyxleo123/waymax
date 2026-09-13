#!/bin/bash
# Submit the parameter-distance probe. Only calls sbatch.
#
# Usage (from repo root, ON RHEA):
#   DFT_ARMS='/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_anchor_sweep/20260902_200735_785143/*/' \
#     bash rl/diffusion_ft/slurm/submit_param_distance.sh
#
# DFT_ARMS is a glob of RUN DIRECTORIES (each holding latest/tparams.npz and
# train_log.jsonl). SINGLE-QUOTE it at the call site: it is written verbatim into
# the env file and expanded by python's `glob` inside the job, so an unquoted
# pattern would be expanded by the calling shell into many words and only the
# first would survive the assignment.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/paramdist_${STAMP}.env"

: "${DFT_ARMS:?set DFT_ARMS to a glob of run dirs, e.g. '/path/sweep/*/'}"

{
  echo "DFT_ARMS=${DFT_ARMS}"
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_LR=${DFT_LR:-1e-5}"
  echo "DFT_DIST_OUT=${DFT_DIST_OUT:-$ROOT/logs/dft/param_distance_${STAMP}.json}"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

# Strip inherited SLURM_* so submitting from inside an allocation cannot leak an
# --export-like environment into the new job.
JOB=$(env $(compgen -v | grep '^SLURM_' | sed 's/^/-u /') \
      sbatch --parsable rl/diffusion_ft/slurm/probe_param_distance.sbatch "$ENV_FILE")
echo "submitted param-distance probe: $JOB"
echo "  tail -f $ROOT/logs/dft/slurm-dft_paramdist-$JOB.out"
