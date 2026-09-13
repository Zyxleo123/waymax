#!/bin/bash
# Submitter for the edge-geometry gate. Only calls sbatch.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_gate_edge_geometry.sh
#
# Knobs: DFT_TFRECORD DFT_GATE_INDICES DFT_PYTHON
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
ENV_FILE="$ENV_DIR/edgegeo_$(date +%Y%m%d_%H%M%S)_$$.env"
{
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_GATE_INDICES=${DFT_GATE_INDICES:-0,1,2,3,4,5}"
  [[ -n "${DFT_PYTHON:-}" ]] && echo "DFT_PYTHON=${DFT_PYTHON}"
} > "$ENV_FILE"
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done
J=$(sbatch --parsable rl/diffusion_ft/slurm/gate_edge_geometry.sbatch "$ENV_FILE")
echo "gate $J   log: logs/dft/slurm-dft_edgegeo-$J.out"
