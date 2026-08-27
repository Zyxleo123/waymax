#!/bin/bash
# Submitter for the hardness-heuristic ranking pipeline. Only calls sbatch.
#
#   stage 1  labels      join the eval JSONs into one label table
#   stage 2  descriptors families B/C/D/E over the labelled cache shards (array)
#   stage 3  rank        AUC / decile lift / grouped-CV, with the go-no-go gate
#
# Stages chain with --dependency=afterok. Configuration is written to one
# timestamped env file whose *path* is passed as a positional argument: an
# explicit `sbatch --export` list (any form, including ALL and NONE) sets
# SLURM_GET_USER_ENV=1, the compute node then fails to rebuild the login
# environment, and the job is requeued and held.
#
# Usage (from the repo root, on rhea -- every partition sets AllocNodes=rhea):
#     bash train_diffusion/hardness/slurm/submit_hardness.sh
#     HRD_ARRAY_TASKS=16 bash train_diffusion/hardness/slurm/submit_hardness.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

ENV_DIR="$ROOT/logs/hardness/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/hardness"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/hardness_${STAMP}.env"

ARRAY_TASKS="${HRD_ARRAY_TASKS:-8}"

# Every value is emitted QUOTED. The batch script `source`s this file, so an
# unquoted list value (HRD_ANCHORS=0 10 20 30 40) parses as an assignment
# followed by a command, and the job dies with "10: command not found".
emit() { printf '%s=%q\n' "$1" "$2"; }

{
  # The 08-16 policy eval: 73,920 scenarios over 154 shards, 3,135 failures.
  emit HRD_POLICY_DIR "${HRD_POLICY_DIR:-/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_diffusion_training_20260816_141401}"
  # Expert log replay over 21 of those shards: the validity gate.
  emit HRD_EXPERT_DIR "${HRD_EXPERT_DIR:-/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_log-replay_training_20260716_025213}"
  emit HRD_CACHE_DIR "${HRD_CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/}"
  emit HRD_OUT "${HRD_OUT:-$ROOT/runs/hardness}"
  # Must match the eval's --start_timestep (simulation/run_simulation.py default 10).
  emit HRD_START_T "${HRD_START_T:-10}"
  emit HRD_ANCHORS "${HRD_ANCHORS:-0 10 20 30 40}"
  emit HRD_ANCHOR_SPAN "${HRD_ANCHOR_SPAN:-50}"
  emit HRD_FOLDS "${HRD_FOLDS:-5}"
  emit HRD_GATE_AUC "${HRD_GATE_AUC:-0.60}"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"
cat "$ENV_FILE"
echo

# Scrub SLURM_* so a submission from inside an allocation does not leak state.
SUBMIT=(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch)

JOB_LABELS="$("${SUBMIT[@]}" --parsable train_diffusion/hardness/slurm/labels.sbatch "$ENV_FILE")"
echo "stage 1 labels      : $JOB_LABELS"

JOB_DESC="$("${SUBMIT[@]}" --parsable \
  --dependency=afterok:"$JOB_LABELS" \
  --array=0-$((ARRAY_TASKS - 1)) \
  train_diffusion/hardness/slurm/descriptors.sbatch "$ENV_FILE")"
echo "stage 2 descriptors : $JOB_DESC (array of $ARRAY_TASKS)"

JOB_RANK="$("${SUBMIT[@]}" --parsable \
  --dependency=afterok:"$JOB_DESC" \
  train_diffusion/hardness/slurm/rank.sbatch "$ENV_FILE")"
echo "stage 3 rank        : $JOB_RANK"

echo
echo "Watch:   squeue -u \$USER -o '%.10i %.9P %.14j %.2t %.10M %R'"
echo "Result:  ${HRD_OUT:-$ROOT/runs/hardness}/ranking.json"
echo "Log:     $ROOT/logs/hardness/slurm-hrd_rank-$JOB_RANK.out"
