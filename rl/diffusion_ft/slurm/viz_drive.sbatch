#!/bin/bash
#SBATCH --job-name=dft_vizdrive
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Draw the driving diagnosis from the artifacts diagnose_drive wrote. Reads only
# rollouts.jsonl + rollouts.npz -- no model, no GPU -- so figures re-render
# cheaply whenever the question changes. Pass the diag dir as $1 (or DFT_DIAG_OUT).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

# Arguments travel through the job record untouched: source any that is a file
# (the submitter's env file), take a directory argument as the target, and export
# any VAR=value. Same convention as the other jobs in this package.
DIAG_ARG=""
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ -d "$arg" ]]; then
    DIAG_ARG="$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done
# A positional directory wins over the env file's DFT_DIAG_OUT, so re-running the
# viz against an older diag dir does not need the env file edited.
DIAG_DIR="${DIAG_ARG:-${DFT_DIAG_OUT:-}}"
if [[ -z "$DIAG_DIR" ]]; then
  echo "FATAL: no diag dir given (positional dir arg or DFT_DIAG_OUT)." >&2
  exit 1
fi
# The upstream GPU job may have failed. Say which, and exit 0 either way: a
# non-zero exit here would only strand anything queued behind this one.
if [[ ! -f "$DIAG_DIR/rollouts.jsonl" ]]; then
  echo "GATE: $DIAG_DIR/rollouts.jsonl missing -- the diagnose job did not finish."
  echo "GATE: nothing to draw; exiting 0 so no dependent job is stranded."
  exit 0
fi

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

DFT_VIZ_SCENES="${DFT_VIZ_SCENES:-16}"
echo "host=${SLURMD_NODENAME:-unknown} diag_dir=$DIAG_DIR scenes=$DFT_VIZ_SCENES"
"$PYTHON" -m rl.diffusion_ft.tests.viz_drive "$DIAG_DIR" --scenes "$DFT_VIZ_SCENES"
