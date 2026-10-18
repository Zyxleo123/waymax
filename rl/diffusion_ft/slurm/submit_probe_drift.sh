#!/bin/bash
# Submitter for the inference-drift probe. Only calls sbatch.
#
#   bash rl/diffusion_ft/slurm/submit_probe_drift.sh
#   DFT_N_SCENES=128 bash rl/diffusion_ft/slurm/submit_probe_drift.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/drift_${STAMP}.env"
OUT_DIR="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/dft/drift_${STAMP}}"

{
  echo "DFT_OUT_DIR='${OUT_DIR}'"
  echo "DFT_N_SCENES='${DFT_N_SCENES:-64}'"
  echo "DFT_CHUNK='${DFT_CHUNK:-8}'"
  echo "DFT_START='${DFT_START:-0}'"
  echo "DFT_SEED='${DFT_SEED:-0}'"
  [[ -n "${DFT_CKPT:-}" ]] && echo "DFT_CKPT='${DFT_CKPT}'"
  [[ -n "${DFT_TFRECORD:-}" ]] && echo "DFT_TFRECORD='${DFT_TFRECORD}'"
} > "$ENV_FILE"

# Verify the round trip before sbatch: a quoting slip here becomes a job that
# runs with the wrong config and reports it as a result.
( set -euo pipefail; set -a; . "$ENV_FILE"; set +a; \
  [[ "$DFT_OUT_DIR" == "$OUT_DIR" ]] || { echo "FATAL: env round trip broken" >&2; exit 1; } )

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/probe_drift.sbatch "$ENV_FILE"
