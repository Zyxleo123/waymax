#!/bin/bash
#SBATCH --job-name=dft_viz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# CPU-only redraw of a DPPO sweep group's comparison figure. Reads only the
# saved train_log.jsonl rows (no model, no GPU), so it re-renders cheaply and
# stays off the GPU partition. Pass the group dir as $1 (or via DFT_GROUP_DIR).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

# Accept the group dir positionally (preferred) or from the environment.
GROUP_DIR="${1:-${DFT_GROUP_DIR:-}}"
if [[ -z "$GROUP_DIR" ]]; then
  echo "FATAL: no group dir given (positional \$1 or DFT_GROUP_DIR)." >&2
  exit 1
fi
if [[ ! -d "$GROUP_DIR" ]]; then
  echo "FATAL: group dir does not exist: $GROUP_DIR" >&2
  exit 1
fi

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
# Keep JAX/XLA from grabbing anything; this job never touches the GPU.
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

echo "host=${SLURMD_NODENAME:-unknown} group_dir=$GROUP_DIR"
"$PYTHON" -m rl.diffusion_ft.tests.viz_compare "$GROUP_DIR" --out "$GROUP_DIR/compare.png"
