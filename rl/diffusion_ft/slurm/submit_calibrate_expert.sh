#!/bin/bash
# Submitter for the expert safety-loss calibration. Only calls sbatch.
#
# Answers one question: does the human demonstration pay a large safety bill?
# If it does, the safety term fights whatever pulls the policy toward the expert,
# and the margin needs lowering (or anchoring per scene) before any training run.
#
# CPU-only and checkpoint-free, so it does not queue behind the GPU probe.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_calibrate_expert.sh
#
# Knobs: DFT_CAL_SCENES (default 64), DFT_TFRECORD, DFT_EDGE_MARGIN, DFT_N_DISCS,
# DFT_AGENT_GEOMETRY (exact|disc). The agent margins on record were measured
# under `disc`, whose cover overhang costs ~1 m per pair of cars; re-run with
# `exact` to see what the human's real box clearance is.
# To calibrate against ordinary driving rather than the curated failure set,
# point DFT_TFRECORD at a WOMD training shard -- the failure set is selected for
# scenes the policy finds hard, which may not represent how humans usually drive.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/expcal_${STAMP}.env"
DFT_CAL_OUT="${DFT_CAL_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/expert_cal/${STAMP}}"

{
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_CAL_SCENES=${DFT_CAL_SCENES:-64}"
  echo "DFT_CAL_START=${DFT_CAL_START:-0}"
  echo "DFT_CAL_CHUNK=${DFT_CAL_CHUNK:-8}"
  echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
  echo "DFT_EDGE_TAU=${DFT_EDGE_TAU:-0.15}"
  echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
  echo "DFT_AGENT_TAU=${DFT_AGENT_TAU:-0.25}"
  echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
  echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
  echo "DFT_CAL_OUT=$DFT_CAL_OUT"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable rl/diffusion_ft/slurm/calibrate_expert.sbatch "$ENV_FILE")
echo "calibration job: $JID  (cpu partition)"
echo
echo "artifacts -> $DFT_CAL_OUT/expert_calibration.{json,jsonl}, expert_clearances.npz"
