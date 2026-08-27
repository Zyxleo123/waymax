#!/bin/bash
# Submit the hardness diagnostics job (CPU). Reads the risk model + labels +
# descriptors + sim-state cache and renders overview.png plus a per-scenario
# gallery to runs/hardness/viz/. Redrawable: it recomputes nothing, only reads
# saved artefacts, so re-running after new risk.npz just refreshes the figures.
#
# Usage (from the repo root, on rhea):
#     bash train_diffusion/hardness/slurm/submit_viz.sh
#
# Optional overrides via env:  HRD_OUT HRD_DESC_DIR HRD_CACHE_DIR HRD_VIZ_DIR
#                              HRD_N_PER_CELL HRD_SEED
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

HRD_OUT="${HRD_OUT:-$ROOT/runs/hardness}"
if [[ ! -f "$HRD_OUT/risk.npz" || ! -f "$HRD_OUT/labels.npz" ]]; then
  echo "ERROR: $HRD_OUT/{risk,labels}.npz not found. Run submit_phase2.sh first." >&2
  exit 1
fi

ENV_DIR="$ROOT/logs/hardness/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/viz_${STAMP}.env"

emit() { printf '%s=%q\n' "$1" "$2"; }
{
  emit HRD_OUT       "$HRD_OUT"
  emit HRD_DESC_DIR  "${HRD_DESC_DIR:-$HRD_OUT/desc}"
  emit HRD_CACHE_DIR "${HRD_CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/}"
  emit HRD_VIZ_DIR   "${HRD_VIZ_DIR:-$HRD_OUT/viz}"
  emit HRD_N_PER_CELL "${HRD_N_PER_CELL:-6}"
  emit HRD_SEED      "${HRD_SEED:-0}"
} > "$ENV_FILE"

# Scrub SLURM_* so a submit from inside an allocation doesn't trip
# GET_USER_ENV; pass config as a positional argument, never --export.
SUBMIT=(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch)
job="$("${SUBMIT[@]}" --parsable train_diffusion/hardness/slurm/viz.sbatch "$ENV_FILE")"
echo "submitted hrd_viz job=$job  env=$ENV_FILE"
echo "figures will land in ${HRD_VIZ_DIR:-$HRD_OUT/viz}/"
echo "watch: squeue -u \$USER -o '%.10i %.9P %.12j %.2t %.10M %R'"
