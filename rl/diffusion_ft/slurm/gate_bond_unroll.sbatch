#!/bin/bash
#SBATCH --job-name=dft_bondunroll
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gate the teacher-forced unrolled proposal (rl.diffusion_ft.unrolled_prediction)
# before any training run uses it: is K=1 the one-step mean, is the Jensen gap
# real and in the claimed direction, is it cheaper, is the gradient finite and
# isolated, and does it transfer to the free-running sample?
#
# GPU CLASS IS NOT NEGOTIABLE HERE -- it is `general`/`preempt` a6000/a5000 or
# nothing. Measured 2026-09-02 (jobs 40492 gpu20, 40494 gpu21): on a v100 this
# env's jaxlib dies at iteration 0 with every cuDNN autotune config failing --
# `Autotuning failed for HLO: %cudnn-conv-bias-activation`, 205x
# `<unknown cudnn status: 5003>` / CUDNN_STATUS_EXECUTION_FAILED -- on a
# f32[16,256,14] conv, i.e. 57 KB. That is a library/architecture mismatch, not
# memory; raising --mem or shrinking the batch does nothing. The 11 GB 2080 Tis
# (`legacy`, `debug` gpu12/13) are out for the separate reason that they OOM, and
# an XLA allocation failure there kills the process with no Python traceback.
# Usable nodes: gpu24-31 (general), gpu27/30/31 + gpu1 (preempt), gpu1 (debug).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_TFRECORD="${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_INDICES="${DFT_INDICES:-0,1,2,3,4,5,6,7}"
DFT_GRAD_STEPS="${DFT_GRAD_STEPS:-30}"
DFT_SAFETY_LR="${DFT_SAFETY_LR:-1e-5}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
# Match the trainer's floor, not the old gate's 0.03: 0.03 normalized is 3 m of
# per-knot jitter (ego_range=100) and would drown the 0.3/0.75 m margins the
# Jensen-gap measurement is trying to read.
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
# ---- unroll ------------------------------------------------------------------
DFT_UNROLL_STEPS="${DFT_UNROLL_STEPS:-3}"
DFT_UNROLL_GRAD_STEPS="${DFT_UNROLL_GRAD_STEPS:-1}"
DFT_UNROLL_T_START="${DFT_UNROLL_T_START:--1}"   # -1 = k_trainable - 1
DFT_UNROLL_T_END="${DFT_UNROLL_T_END:-0}"
DFT_UNROLL_ETA="${DFT_UNROLL_ETA:-0.0}"
DFT_N_MC="${DFT_N_MC:-16}"
DFT_GATE_OUT="${DFT_GATE_OUT:?set DFT_GATE_OUT}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "unroll K=$DFT_UNROLL_STEPS grad_steps=$DFT_UNROLL_GRAD_STEPS eta=$DFT_UNROLL_ETA"
echo "window t_start=$DFT_UNROLL_T_START -> t_end=$DFT_UNROLL_T_END (k_trainable=$DFT_K_TRAINABLE)"
mkdir -p "$DFT_GATE_OUT"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

"$PYTHON" -m rl.diffusion_ft.tests.gate_bond_unroll \
  --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" \
  --steps "$DFT_GRAD_STEPS" --lr "$DFT_SAFETY_LR" \
  --edge_margin_m "$DFT_EDGE_MARGIN" --agent_margin_m "$DFT_AGENT_MARGIN" \
  --agent_geometry "$DFT_AGENT_GEOMETRY" --n_discs "$DFT_N_DISCS" \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --unroll_steps "$DFT_UNROLL_STEPS" --unroll_grad_steps "$DFT_UNROLL_GRAD_STEPS" \
  --unroll_t_start "$DFT_UNROLL_T_START" --unroll_t_end "$DFT_UNROLL_T_END" \
  --unroll_eta "$DFT_UNROLL_ETA" --n_mc "$DFT_N_MC" \
  --out_dir "$DFT_GATE_OUT"
