#!/bin/bash
#SBATCH --job-name=dft_offsets
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#
# One-time: build the byte-offset sidecar index (`<shard>.offsets.npy`) for every
# raw-WOMD training shard, so the DPPO scene sampler can draw a fresh batch from
# anywhere in the 1000-shard corpus in O(1) instead of scanning a 1.2 GB shard.
#
# CPU-only and IO-light: the builder reads 8-byte record headers and *seeks* past
# each payload, so it never transfers the 246 GB of record data. Array-parallel
# over shards, and resumable -- an already-built index is skipped, so a task that
# hits the time limit is fixed by resubmitting.
#
# The indexes are written to our own scratch (DFT_INDEX_DIR), not next to the
# shards: that directory belongs to another user and is not ours to litter.
#
# Submit via rl/diffusion_ft/slurm/submit_build_womd_offsets.sh (never
# `sbatch --export`: it requeue-holds on this cluster).
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

DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*-of-01000}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host=${SLURMD_NODENAME:-unknown} task=${TASK_ID}/${NTASKS}"
echo "glob=$DFT_SHARD_GLOB"
echo "index_dir=$DFT_INDEX_DIR"
mkdir -p "$DFT_INDEX_DIR"

"$PYTHON" "$ROOT/rl/diffusion_ft/tools/build_offsets.py" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --task_id "$TASK_ID" --n_tasks "$NTASKS"
