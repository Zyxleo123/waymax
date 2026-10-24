#!/bin/bash
#SBATCH --job-name=dft_drift
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Measure the inference-time drift the FlowMol3 port is aimed at, and read the
# distortion scale off it instead of transplanting FlowMol3's molecular sigma.
#
# GPU CLASS IS NOT NEGOTIABLE: a6000/a5000 on `general`/`preempt`. On a v100 this
# env's jaxlib dies at cuDNN autotune with no Python traceback; the 11 GB
# 2080 Tis OOM. Usable: gpu24-31 (general), gpu27/30/31 + gpu1 (preempt).
#
#     bash rl/diffusion_ft/slurm/submit_probe_drift.sh
#
# Config arrives as a positional env FILE (never --export).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_TFRECORD="${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_N_SCENES="${DFT_N_SCENES:-64}"
DFT_CHUNK="${DFT_CHUNK:-8}"
DFT_START="${DFT_START:-0}"
DFT_SEED="${DFT_SEED:-0}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"

echo "host=${SLURMD_NODENAME:-unknown}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "ckpt=$DFT_CKPT"
echo "out=$DFT_OUT_DIR"

"$PY" -m rl.diffusion_ft.tools.probe_sampler_drift \
  --ckpt "$DFT_CKPT" \
  --tfrecord "$DFT_TFRECORD" \
  --n_scenes "$DFT_N_SCENES" \
  --start "$DFT_START" \
  --chunk "$DFT_CHUNK" \
  --seed "$DFT_SEED" \
  --out_dir "$DFT_OUT_DIR"
