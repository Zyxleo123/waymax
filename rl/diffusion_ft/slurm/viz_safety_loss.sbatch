#!/bin/bash
#SBATCH --job-name=dft_vizsafe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Draw the safety-loss probe from the artifacts it wrote. Reads only
# safety_probe.{npz,jsonl} + _summary.json -- no model, no GPU -- so the figures
# re-render cheaply when the question changes. Pass the probe dir as $1 (or
# DFT_PROBE_OUT).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

PROBE_ARG=""
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ -d "$arg" ]]; then
    PROBE_ARG="$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done
PROBE_DIR="${PROBE_ARG:-${DFT_PROBE_OUT:-}}"
if [[ -z "$PROBE_DIR" ]]; then
  echo "FATAL: no probe dir given (positional dir arg or DFT_PROBE_OUT)." >&2
  exit 1
fi
# The upstream GPU job may have failed. Say which, and exit 0 either way -- a
# non-zero exit here would only strand anything queued behind this one.
if [[ ! -f "$PROBE_DIR/safety_probe_summary.json" ]]; then
  echo "GATE: $PROBE_DIR/safety_probe_summary.json missing -- the probe did not finish."
  echo "GATE: nothing to draw; exiting 0 so no dependent job is stranded."
  exit 0
fi

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

DFT_VIZ_SCENES="${DFT_VIZ_SCENES:-12}"
echo "host=${SLURMD_NODENAME:-unknown} probe_dir=$PROBE_DIR scenes=$DFT_VIZ_SCENES"
"$PYTHON" -m rl.diffusion_ft.tests.viz_safety_loss "$PROBE_DIR" --scenes "$DFT_VIZ_SCENES"
