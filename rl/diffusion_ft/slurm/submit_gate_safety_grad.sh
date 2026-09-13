#!/bin/bash
# Submitter for the analytic-gradient gate. Only calls sbatch.
#
# Run this BEFORE any safety fine-tuning run. It answers, in ~minutes, the four
# questions a long run cannot answer cheaply: does the gradient reach the
# denoiser, does it leave the frozen modules alone, does it descend, and does the
# expert anchor stop the safety term collapsing the trajectory to a standstill.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_gate_safety_grad.sh
#
# Knobs: DFT_GRAD_STEPS, DFT_SAFETY_LR, DFT_SAFETY_WEIGHT, DFT_ANCHOR_WEIGHT,
#        DFT_INDICES, DFT_PARTITION/DFT_QOS/DFT_NODELIST.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"; ENV_FILE="$ENV_DIR/safegrad_${STAMP}.env"
DFT_GATE_OUT="${DFT_GATE_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_grad/${STAMP}}"
{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7}"
  echo "DFT_GRAD_STEPS=${DFT_GRAD_STEPS:-30}"
  echo "DFT_SAFETY_LR=${DFT_SAFETY_LR:-1e-5}"
  echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
  echo "DFT_ANCHOR_WEIGHT=${DFT_ANCHOR_WEIGHT:-1.0}"
  echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
  echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
  echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
  echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.03}"
  echo "DFT_GATE_OUT=$DFT_GATE_OUT"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"
SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/gate_safety_grad.sbatch "$ENV_FILE")
echo "gate job: $JID"
echo "artifacts -> $DFT_GATE_OUT/gate_safety_grad.json"
