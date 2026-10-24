#!/bin/bash
#SBATCH --job-name=dft_safegrad
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
# Gate the analytic-gradient path before any training run: does the safety-loss
# gradient reach the trainable denoiser, leave the frozen modules alone, actually
# descend, and does the expert anchor stop the safety term collapsing the
# trajectory to "stand still"?
#
# Backprops through k_trainable unrolled denoiser steps, so it holds more than the
# probe does -- keep it off the 11 GB 2080 Tis (`general` is a6000/a5000).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_TFRECORD="${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_INDICES="${DFT_INDICES:-0,1,2,3,4,5,6,7}"
DFT_GRAD_STEPS="${DFT_GRAD_STEPS:-30}"
DFT_SAFETY_LR="${DFT_SAFETY_LR:-1e-5}"
DFT_SAFETY_WEIGHT="${DFT_SAFETY_WEIGHT:-1.0}"
DFT_ANCHOR_WEIGHT="${DFT_ANCHOR_WEIGHT:-1.0}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.03}"
DFT_GATE_OUT="${DFT_GATE_OUT:?set DFT_GATE_OUT}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
mkdir -p "$DFT_GATE_OUT"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

"$PYTHON" -m rl.diffusion_ft.tests.gate_safety_grad \
  --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" \
  --steps "$DFT_GRAD_STEPS" --lr "$DFT_SAFETY_LR" \
  --safety_weight "$DFT_SAFETY_WEIGHT" --anchor_weight "$DFT_ANCHOR_WEIGHT" \
  --edge_margin_m "$DFT_EDGE_MARGIN" --agent_margin_m "$DFT_AGENT_MARGIN" \
  --agent_geometry "$DFT_AGENT_GEOMETRY" \
  --n_discs "$DFT_N_DISCS" --k_trainable "$DFT_K_TRAINABLE" \
  --prefix_len "$DFT_PREFIX_LEN" --sigma_floor "$DFT_SIGMA_FLOOR" \
  --out_dir "$DFT_GATE_OUT"
