#!/bin/bash
#SBATCH --job-name=dft_sampler_gate
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gate the full-corpus scene sampler: byte-offset assembly must equal the
# scan-based loader leaf for leaf, every draw must keep the same object count,
# training draws must never touch the held-out eval scenes, and the raw-WOMD
# 45x800 route dims must actually yield valid routes. CPU-only -- no model, no
# GPU -- so it does not queue behind training.
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
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"

echo "host=${SLURMD_NODENAME:-unknown} index_dir=$DFT_INDEX_DIR"
"$PYTHON" "$ROOT/rl/diffusion_ft/tests/test_scene_sampler.py" "$DFT_INDEX_DIR"
