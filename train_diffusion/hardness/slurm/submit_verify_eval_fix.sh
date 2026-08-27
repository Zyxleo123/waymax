#!/bin/bash
# Submit the pre-resubmission verification of the two eval fixes.
#
# Run this BEFORE submit_alpha_sweep.sh: the 2026-08-20 sweep burned five GPU
# arms and produced no comparison signal, because the failure-set eval never
# parsed and the WOMD eval was pinned to epoch-5 weights. This job proves both
# are fixed for ~20 minutes of one GPU.
#
# Usage (from the repo root, on rhea):
#     HRD_CKPT=/path/to/pretrained/latest bash train_diffusion/hardness/slurm/submit_verify_eval_fix.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

: "${HRD_CKPT:?set HRD_CKPT to the pretrained diffusion checkpoint}"
if [[ ! -f "$HRD_CKPT/metadata.json" ]]; then
  echo "ERROR: no metadata.json under $HRD_CKPT -- this is not a checkpoint dir." >&2
  exit 1
fi
echo "checkpoint OK: $HRD_CKPT"

ENV_DIR="$ROOT/logs/hardness/env"
mkdir -p "$ENV_DIR"
ENV_FILE="$ENV_DIR/verify_$(date +%Y%m%d_%H%M%S).env"

emit() { printf '%s=%q\n' "$1" "$2"; }
{
  emit HRD_CKPT "$HRD_CKPT"
  emit HRD_EVAL_TFRECORD_DIR "${HRD_EVAL_TFRECORD_DIR:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures}"
  emit HRD_EVAL_LANE_GRAPH_DIR "${HRD_EVAL_LANE_GRAPH_DIR:-/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"
  emit HRD_VERIFY_NUM "${HRD_VERIFY_NUM:-20}"
  emit HRD_VERIFY_OUT "${HRD_VERIFY_OUT:-/zfsauton/scratch/yixiz/waymax_rs/hardness_verify}"
} > "$ENV_FILE"

job="$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch --parsable \
  train_diffusion/hardness/slurm/verify_eval_fix.sbatch "$ENV_FILE")"
echo "submitted hrd_verify job=$job"
echo "Read: logs/hardness/slurm-hrd_verify-$job.out"
