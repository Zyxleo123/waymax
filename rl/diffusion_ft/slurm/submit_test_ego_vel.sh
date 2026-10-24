#!/bin/bash
# The ego-velocity splice gates on their own. Only calls sbatch. (submit_pace.sh
# submits the same job as the first link of its chain.)
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_test_ego_vel.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_GATE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/egovel_gate/${STAMP}}"
mkdir -p "$OUT" "$ROOT/logs/dft"
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
{ kv DFT_GATE_VERDICT "$OUT/verdict.txt"
  kv DFT_CALIB_TOL "${DFT_CALIB_TOL:-0.01}"; } > "$OUT/gate.env"
JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable rl/diffusion_ft/slurm/test_ego_vel.sbatch "$OUT/gate.env")
echo "gate -> job $JID (CPU) -> $OUT/verdict.txt"
