#!/bin/bash
# Submitter for the DPPO exploration-noise sweep: how large may sigma_floor be
# before the behaviour policy DPPO collects with can no longer drive?
#
# Only calls sbatch. Chains: GPU sweep (one diagnose_drive per sigma) -> CPU
# aggregate (table + figure), afterok.
#
# Writes a timestamped env file and passes its path as a positional arg to the
# batch scripts (no `sbatch --export`, which requeue-holds on this cluster).
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_sweep_sigma.sh
#   DFT_SIGMA_LIST="0.001 0.003 0.006" bash rl/diffusion_ft/slurm/submit_sweep_sigma.sh
#
# Node: general + gpu28. NOT project/gpu2 -- this account holds no association
# for either project QOS, so sbatch dies on "Invalid qos specification" however
# idle `myfree` claims the h200 is (measured 2026-08-21).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/sigsweep_${STAMP}.env"

DFT_SWEEP_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/sigma_sweep/${STAMP}}"

# A list-valued variable travels fine because the config is a FILE, not an
# `sbatch --export` list -- no comma encoding needed.
{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
  echo "DFT_ARMS=${DFT_ARMS:-base=}"
  echo "DFT_DIAG_SEEDS=${DFT_DIAG_SEEDS:-4}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_REPLAN=${DFT_REPLAN:-10}"
  echo "DFT_SIGMA_LIST=\"${DFT_SIGMA_LIST:-0.003 0.01 0.02 0.03 0.05 0.1}\""
  echo "DFT_SIGMA_TOL=${DFT_SIGMA_TOL:-0.10}"
  # Parity reward, verbatim from job 34866.
  echo "DFT_REWARD_PROGRESS=${DFT_REWARD_PROGRESS:-0.2}"
  echo "DFT_REWARD_OFFROAD=${DFT_REWARD_OFFROAD:--1.0}"
  echo "DFT_REWARD_COLLISION=${DFT_REWARD_COLLISION:--1.0}"
  echo "DFT_REWARD_GOAL_BONUS=${DFT_REWARD_GOAL_BONUS:-5.0}"
  echo "DFT_REWARD_OFF_ROUTE=${DFT_REWARD_OFF_ROUTE:--0.2}"
  echo "DFT_GOAL_SHAPING_COEF=${DFT_GOAL_SHAPING_COEF:-1.0}"
  echo "DFT_NUM_PATHS=${DFT_NUM_PATHS:-10}"
  echo "DFT_NUM_POINTS_PER_PATH=${DFT_NUM_POINTS_PER_PATH:-300}"
  echo "DFT_GAMMA=${DFT_GAMMA:-0.99}"
  echo "DFT_SWEEP_OUT=$DFT_SWEEP_OUT"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# `general` runs qos_general as its default QOS, so no --qos flag is needed.
DFT_PARTITION="${DFT_PARTITION:-general}"
DFT_QOS="${DFT_QOS:-}"
DFT_NODELIST="${DFT_NODELIST:-gpu28}"
SB_FLAGS=(--partition="$DFT_PARTITION")
[[ -n "$DFT_QOS" ]]      && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "$DFT_NODELIST" ]] && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]] && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
echo "sbatch flags: ${SB_FLAGS[*]}"

# Scrub SLURM_* so submitting from inside an allocation cannot leak this job's
# environment into the new one.
JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/sweep_sigma.sbatch "$ENV_FILE")
echo "sigma sweep job:  $JID"

AID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency=afterok:"$JID" \
  rl/diffusion_ft/slurm/aggregate_sigma_sweep.sbatch "$DFT_SWEEP_OUT" "$ENV_FILE")
echo "aggregate job:    $AID  (afterok:$JID)"

echo
echo "read the TAIL of logs/dft/slurm-dft_sigagg-${AID}.out for the threshold."
echo "artifacts -> $DFT_SWEEP_OUT/sigma_sweep.{md,png,jsonl}"
echo "the sweep is resumable: resubmitting skips points that already wrote rollouts.jsonl."
