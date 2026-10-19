#!/bin/bash
#SBATCH --job-name=dft_envjit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# A/B the jitted replan (DiffusionWaymaxEnv jit_replan) against the eager env:
# open-loop + closed-loop equivalence, then per-phase timings of a real training
# iteration (collect / BC / grads) and one 32-scene eval chunk. Nothing trains.
# Exit 1 iff open-loop equivalence FAILS -- read the table before adopting.
#
# `general` advertises only a6000/a5000; this env dies on v100 and OOMs on the
# 2080 Ti with no traceback. No --qos. Config arrives as a positional env FILE
# (never `sbatch --export`).
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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SCENES_PER_ITER="${DFT_SCENES_PER_ITER:-32}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_BC_WEIGHT="${DFT_BC_WEIGHT:-300}"
DFT_EVAL_SEED="${DFT_EVAL_SEED:-12345}"
DFT_DATA_SEED="${DFT_DATA_SEED:-0}"
DFT_ROUNDS="${DFT_ROUNDS:-4}"

mkdir -p "$DFT_OUT_DIR"
echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m rl.diffusion_ft.tests.bench_env_jit \
  --ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --scenes_per_iter "$DFT_SCENES_PER_ITER" --eval_scenes 0 \
  --data_seed "$DFT_DATA_SEED" \
  --max_replans "$DFT_MAX_REPLANS" --replan_interval_steps "$DFT_REPLAN" \
  --k_trainable "$DFT_K_TRAINABLE" --sigma_floor "$DFT_SIGMA_FLOOR" \
  --bc_weight "$DFT_BC_WEIGHT" --eval_seed "$DFT_EVAL_SEED" \
  --iters 0 --eval_every 0 --no-wandb --rounds "$DFT_ROUNDS"
