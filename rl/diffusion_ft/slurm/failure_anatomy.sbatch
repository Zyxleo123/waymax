#!/bin/bash
#SBATCH --job-name=dft_anat
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#
# Anatomy of the residual: per-STEP geometry for every episode a select run
# dumped, recomputed on CPU from the executed world paths
# (rl/diffusion_ft/tools/failure_anatomy.py). No GPU, no checkpoint, no
# re-rolling -- it reads `<arm>/r<rep>/traj` out of feat/*.npz (so the select run
# must have been submitted with DFT_DUMP_TRAJ=1) and reloads each scene's
# roadgraph/agents through the same byte-offset index the trainer uses.
#
# Array mode: $SLURM_ARRAY_TASK_ID is the chunk index over the split's scenes,
# DFT_ANAT_CHUNKS the number of chunks. Set DFT_ANAT_MERGE=1 for the aggregation
# task, which reads only what the array wrote.
#
# Config arrives as a positional env FILE. Never `sbatch --export`.
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
SEL_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR (a select run with --dump_traj)}"
OUT_DIR="${DFT_ANAT_OUT:?set DFT_ANAT_OUT}"
SPLIT="${DFT_ANAT_SPLIT:-failure_heldout}"
ARMS="${DFT_ANAT_ARMS:-base,ft,c_r10}"
REPEATS="${DFT_ANAT_REPEATS:-4}"
CHUNKS="${DFT_ANAT_CHUNKS:-6}"
CAP="${DFT_ANAT_CAP:--1}"
LOAD_BATCH="${DFT_ANAT_LOAD_BATCH:-8}"
SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
SWEEP_ARM="${DFT_ANAT_SWEEP_ARM:-ft}"
SWEEP_MAX="${DFT_ANAT_SWEEP_MAX:-3}"
SWEEP_LAT="${DFT_ANAT_SWEEP_LAT:-4.0}"
SWEEP_STEP="${DFT_ANAT_SWEEP_STEP:-0.1}"
MERGE="${DFT_ANAT_MERGE:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

echo "host=${SLURMD_NODENAME:-unknown}  select_dir=$SEL_DIR  split=$SPLIT  arms=$ARMS"
echo "out=$OUT_DIR  task=$TASK/$CHUNKS  merge=$MERGE  sweep_arm=${SWEEP_ARM:-<none>}"
mkdir -p "$OUT_DIR"

# A quota-full scratch fails the run hours in; fail it in seconds instead.
: > "$OUT_DIR/.writeprobe" && rm -f "$OUT_DIR/.writeprobe"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2

if [[ "$MERGE" == "1" ]]; then
  "$PY" -m rl.diffusion_ft.tools.failure_anatomy --merge \
    --out_dir "$OUT_DIR" --split "$SPLIT"
else
  "$PY" -m rl.diffusion_ft.tools.failure_anatomy \
    --select_dir "$SEL_DIR" --out_dir "$OUT_DIR" --split "$SPLIT" \
    --arms "$ARMS" --repeats "$REPEATS" \
    --chunk_index "$TASK" --num_chunks "$CHUNKS" --cap "$CAP" \
    --load_batch "$LOAD_BATCH" --splits_json "$SPLITS_JSON" \
    --shard_glob "$SHARD_GLOB" --index_dir "$INDEX_DIR" \
    --sweep_arm "$SWEEP_ARM" --sweep_max "$SWEEP_MAX" \
    --sweep_lat "$SWEEP_LAT" --sweep_step "$SWEEP_STEP"
fi
echo "done."
