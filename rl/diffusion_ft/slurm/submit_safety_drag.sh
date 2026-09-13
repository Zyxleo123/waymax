#!/bin/bash
# Submitter for the draggable safety-loss page. Only calls sbatch.
#
# Chains: CPU dump ->(afterok) CPU render. Both are cheap; the split exists so the
# page can be re-rendered from the saved JSON without touching the corpus again.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_safety_drag.sh
#   DFT_DRAG_INDICES=0,3,7,11 bash rl/diffusion_ft/slurm/submit_safety_drag.sh
#
# Knobs: DFT_DRAG_DIR DFT_DRAG_INDICES DFT_TFRECORD DFT_DRAG_RADIUS
#        DFT_DRAG_PROBES DFT_EDGE_MARGIN DFT_AGENT_MARGIN DFT_DRAG_TITLE
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/drag_${STAMP}.env"
OUT_DIR="${DFT_DRAG_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_drag/${STAMP}}"

{
  echo "DFT_DRAG_DIR=$OUT_DIR"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_DRAG_INDICES=${DFT_DRAG_INDICES:-0,1,2,3,4,5}"
  echo "DFT_DRAG_RADIUS=${DFT_DRAG_RADIUS:-70}"
  echo "DFT_DRAG_PROBES=${DFT_DRAG_PROBES:-48}"
  echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
  echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
  echo "DFT_EDGE_GEOMETRY=${DFT_EDGE_GEOMETRY:-point}"
  [[ -n "${DFT_DRAG_TITLE:-}" ]] && echo "DFT_DRAG_TITLE=${DFT_DRAG_TITLE}"
  [[ -n "${DFT_PYTHON:-}" ]] && echo "DFT_PYTHON=${DFT_PYTHON}"
} > "$ENV_FILE"

# Submitting from inside an allocation leaks SLURM_* into sbatch's view.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done

J1=$(sbatch --parsable rl/diffusion_ft/slurm/dump_safety_scenes.sbatch "$ENV_FILE")
J2=$(sbatch --parsable --dependency=afterok:"$J1" \
        rl/diffusion_ft/slurm/viz_safety_drag.sbatch "$ENV_FILE")
echo "dump   $J1"
echo "render $J2  -> $OUT_DIR/safety_drag.html"
echo "env    $ENV_FILE"
