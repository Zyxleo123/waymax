#!/bin/bash
#SBATCH --job-name=dft_safeunit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Pure-geometry unit gates for rl/diffusion_ft/safety_loss.py: sanity of the
# synthetic fixture, waymax parity of the road-edge distance, soundness of the
# disc cover, barrier shape, and the gradient (finite, concentrated, matches
# finite differences, and descending it increases real clearance).
#
# Tiny and synthetic -- no checkpoint, no scenario file, seconds of compute, and
# the module pins JAX_PLATFORMS=cpu, so this belongs on the `cpu` partition. It
# is a SLURM job only because the login node's ~60 s reaper kills it partway
# through the gradient group. (Job 39352 ran it on a GPU node and reported
# `jax devices: [CpuDevice(id=0)]` -- a held a6000 doing nothing.)
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
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

echo "host=${SLURMD_NODENAME:-unknown}"
"$PYTHON" -m rl.diffusion_ft.tests.test_safety_loss
