#!/bin/bash
# Submitter for the teacher-forced unroll gate. Only calls sbatch.
#
# Run this BEFORE any --unroll_steps training run. In ~minutes it answers what a
# long run cannot answer cheaply: is the unroll a strict generalisation of the
# one-step prediction, is the Jensen gap it exists to close actually present in
# this objective, is it cheaper, is the gradient isolated, and do the cheap steps
# move the free-running sample the deployed policy emits.
#
# Gates 2 and 5 are the ones to read. 2 asks whether the unrolled endpoint is a
# closer stand-in for the driven trajectory than the one-step mean (it was 8x
# closer on job 40471); 5 asks whether the cheap steps move the free-running
# sample at all (they moved it -58.7% against a control's -68.0%). The Jensen
# gap is printed but not gated -- its sign came out opposite to the ring case
# that motivates the mechanic upstream, and that does not invalidate it.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea,lov4):
#   bash rl/diffusion_ft/slurm/submit_gate_bond_unroll.sh
#   DFT_UNROLL_STEPS=5 DFT_UNROLL_ETA=1.0 bash rl/diffusion_ft/slurm/submit_gate_bond_unroll.sh
#
# Knobs: DFT_UNROLL_STEPS DFT_UNROLL_GRAD_STEPS DFT_UNROLL_T_START DFT_UNROLL_T_END
#        DFT_UNROLL_ETA DFT_N_MC DFT_GRAD_STEPS DFT_K_TRAINABLE DFT_INDICES
#        DFT_PARTITION/DFT_QOS/DFT_NODELIST/DFT_EXCLUDE
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
# `$$` guards against two sibling submissions in the same second sharing one env
# file and one out_dir -- see submit_train_safety.sh for the incident.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"; ENV_FILE="$ENV_DIR/bondunroll_${STAMP}.env"
DFT_GATE_OUT="${DFT_GATE_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bond_unroll/${STAMP}}"
{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7}"
  echo "DFT_GRAD_STEPS=${DFT_GRAD_STEPS:-30}"
  echo "DFT_SAFETY_LR=${DFT_SAFETY_LR:-1e-5}"
  echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
  echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
  echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
  echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
  echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-3}"
  echo "DFT_UNROLL_GRAD_STEPS=${DFT_UNROLL_GRAD_STEPS:-1}"
  echo "DFT_UNROLL_T_START=${DFT_UNROLL_T_START:--1}"
  echo "DFT_UNROLL_T_END=${DFT_UNROLL_T_END:-0}"
  echo "DFT_UNROLL_ETA=${DFT_UNROLL_ETA:-0.0}"
  echo "DFT_N_MC=${DFT_N_MC:-16}"
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
  rl/diffusion_ft/slurm/gate_bond_unroll.sbatch "$ENV_FILE")
echo "gate job: $JID"
echo "watch:     tail -f $ROOT/logs/dft/slurm-dft_bondunroll-${JID}.out"
echo "artifacts: $DFT_GATE_OUT/gate_bond_unroll.json"
