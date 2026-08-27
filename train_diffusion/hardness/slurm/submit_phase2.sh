#!/bin/bash
# Phase 2 data prep: descriptors for all 1000 shards, then fit + score the risk
# model. Only calls sbatch. Training is submitted separately by
# submit_alpha_sweep.sh, so you can read risk.json before spending GPU days.
#
# Usage (from the repo root, on rhea):
#     bash train_diffusion/hardness/slurm/submit_phase2.sh
#     HRD_ARRAY_TASKS=32 bash train_diffusion/hardness/slurm/submit_phase2.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

ENV_DIR="$ROOT/logs/hardness/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/hardness"
ENV_FILE="$ENV_DIR/phase2_$(date +%Y%m%d_%H%M%S).env"
ARRAY_TASKS="${HRD_ARRAY_TASKS:-16}"

# Values are emitted quoted: this file is sourced, so an unquoted list value
# parses as an assignment followed by a command.
emit() { printf '%s=%q\n' "$1" "$2"; }
{
  emit HRD_CACHE_DIR "${HRD_CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/}"
  emit HRD_OUT "${HRD_OUT:-$ROOT/runs/hardness}"
  emit HRD_START_T "${HRD_START_T:-10}"
  emit HRD_ANCHORS "${HRD_ANCHORS:-0 10 20 30 40}"
  emit HRD_ANCHOR_SPAN "${HRD_ANCHOR_SPAN:-50}"
  emit HRD_FIT_SLICE "${HRD_FIT_SLICE:-expert_clean}"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

SUBMIT=(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch)

JOB_DESC="$("${SUBMIT[@]}" --parsable --array=0-$((ARRAY_TASKS - 1)) \
  train_diffusion/hardness/slurm/descriptors_all.sbatch "$ENV_FILE")"
echo "stage 1 descriptors (1000 shards) : $JOB_DESC (array of $ARRAY_TASKS)"

JOB_RISK="$("${SUBMIT[@]}" --parsable --dependency=afterok:"$JOB_DESC" \
  train_diffusion/hardness/slurm/risk_model.sbatch "$ENV_FILE")"
echo "stage 2 risk model                : $JOB_RISK"

echo
echo "Watch:  squeue -u \$USER -o '%.10i %.9P %.14j %.2t %.10M %R'"
echo "Result: ${HRD_OUT:-$ROOT/runs/hardness}/risk.json  (read this before the sweep)"
