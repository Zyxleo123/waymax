#!/bin/bash
#SBATCH --job-name=dft_winnability
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Per-scene expert-replay winnability ceiling for the DPPO reward. CPU-only
# (no --gres=gpu): the expert path runs no model forward, so this packs onto
# any node with free cores rather than queueing behind GPU work.
#
# Config via positional VAR=value args (never `sbatch --export`):
#   DFT_TFRECORD   default = failures.tfrecord
#   DFT_INDICES    default = 0-31   (ranges 'a-b' and lists 'a,b,c' both ok)
#   DFT_OUT        default = runs/dft_winnability/<indices>.json
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

DFT_TFRECORD="${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_INDICES="${DFT_INDICES:-0-31}"
DFT_OUT="${DFT_OUT:-$ROOT/runs/dft_winnability/win_$(echo "$DFT_INDICES" | tr ',' '_').json}"

echo "host=${SLURMD_NODENAME:-unknown} tfrecord=$DFT_TFRECORD indices=$DFT_INDICES"

"$PYTHON" -m rl.diffusion_ft.winnability \
  --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" --out "$DFT_OUT"
