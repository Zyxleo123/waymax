#!/bin/bash
#SBATCH --job-name=dft_ddimab
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# A/B the DDIM frozen-prefix striding for the safety fine-tune: how much wall
# time --collect_steps buys per training iteration, and what it costs in
# solved/offroad/drift on the base policy. Read the table, then set
# DFT_COLLECT_STEPS in submit_verified_safety.sh. Nothing trains here.
#
# `general` is the partition because it advertises only a6000/a5000 (gpu24-31).
# This env's jaxlib dies at cuDNN autotune on v100 and OOMs on the 11 GB 2080 Ti,
# where an XLA allocation failure kills the process with NO Python traceback --
# do not widen the partition to legacy/debug. No --qos: every partition mandates
# its own and `general` takes the default.
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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
# Match the real run: the timing is only useful at the batch size and replan
# budget the training arm actually uses.
DFT_SCENES_PER_ITER="${DFT_SCENES_PER_ITER:-32}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_EVAL_SEED="${DFT_EVAL_SEED:-12345}"
DFT_DATA_SEED="${DFT_DATA_SEED:-0}"
# S must lie in [k_trainable, timesteps]; 0 = the full chain and is the
# reference row, so keep it first.
DFT_AB_STEPS="${DFT_AB_STEPS:-0,50,30,20,15}"
DFT_AB_ROUNDS="${DFT_AB_ROUNDS:-5}"

mkdir -p "$DFT_OUT_DIR"
echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ab_steps=$DFT_AB_STEPS rounds=$DFT_AB_ROUNDS"
echo "scenes/iter=$DFT_SCENES_PER_ITER max_replans=$DFT_MAX_REPLANS" \
     "k_trainable=$DFT_K_TRAINABLE"

srun "$PY" -m rl.diffusion_ft.tests.bench_safety_ddim \
  --ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --scenes_per_iter "$DFT_SCENES_PER_ITER" --eval_scenes 0 \
  --data_seed "$DFT_DATA_SEED" \
  --max_replans "$DFT_MAX_REPLANS" --replan_interval_steps "$DFT_REPLAN" \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" --eval_seed "$DFT_EVAL_SEED" \
  --iters 0 --eval_every 0 --no-wandb \
  --ab_steps "$DFT_AB_STEPS" --ab_rounds "$DFT_AB_ROUNDS"
