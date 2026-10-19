#!/bin/bash
#SBATCH --job-name=dft_paramdist
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Measure ||theta - theta_0|| for each finished run and compare it to Adam's
# aligned vs random-walk predictions. No model forward pass, no GPU: it restores
# the base params, reads saved npz weights and does linear algebra, so it belongs
# on the CPU partition. `checkpoint.py` carries an orbax patch for exactly this
# (arg-less restore fails on CPU topologies without it).
#
# Memory: one float32 copy of the base plus one displacement vector per arm at
# 31.7M params = 127 MB each. 48G covers ~a dozen arms with room to spare.
#
# Config arrives as a positional env FILE (never `sbatch --export`, which
# triggers SLURM_GET_USER_ENV and requeue-holds the job).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_ARMS="${DFT_ARMS:?set DFT_ARMS to a glob of run dirs}"
DFT_DIST_OUT="${DFT_DIST_OUT:-$ROOT/logs/dft/param_distance_${SLURM_JOB_ID:-local}.json}"
DFT_LR="${DFT_LR:-1e-5}"

echo "host=${SLURMD_NODENAME:-unknown} arms=$DFT_ARMS"
echo "ckpt=$DFT_CKPT out=$DFT_DIST_OUT"

"$PY" -m rl.diffusion_ft.tests.probe_param_distance \
  --ckpt "$DFT_CKPT" --arms "$DFT_ARMS" --lr "$DFT_LR" --out "$DFT_DIST_OUT"

echo "[paramdist] done -> $DFT_DIST_OUT"
