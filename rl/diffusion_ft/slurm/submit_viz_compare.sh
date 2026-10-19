#!/bin/bash
# Submitter for the CPU-only comparison plot. Only calls sbatch; passes the
# group dir as a positional arg to the batch script (no `sbatch --export`).
#
# Usage (from repo root):
#   bash rl/diffusion_ft/slurm/submit_viz_compare.sh /path/to/dppo/<group>
# or:
#   DFT_GROUP_DIR=/path/to/dppo/<group> bash rl/diffusion_ft/slurm/submit_viz_compare.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"

GROUP_DIR="${1:-${DFT_GROUP_DIR:-}}"
if [[ -z "$GROUP_DIR" ]]; then
  echo "Usage: $0 <group_dir>   (or set DFT_GROUP_DIR)" >&2
  exit 1
fi

echo "Submitting viz for group_dir=$GROUP_DIR"
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/viz_compare.sbatch "$GROUP_DIR"
