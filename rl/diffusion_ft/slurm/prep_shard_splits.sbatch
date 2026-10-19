#!/bin/bash
#SBATCH --job-name=dft_shsplit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Byte-offset indexes for a shard glob (skips any already built), then a one-split
# splits JSON over every scene in it (tools/build_shard_splits.py). CPU only.
#
# Config: a positional env file (DFT_SHARD_GLOB, DFT_INDEX_DIR, DFT_SPLIT_NAME,
# DFT_SPLITS_JSON). Never `sbatch --export`.
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
GLOB="${DFT_SHARD_GLOB:?set DFT_SHARD_GLOB}"
INDEX_DIR="${DFT_INDEX_DIR:?set DFT_INDEX_DIR}"
SPLIT="${DFT_SPLIT_NAME:?set DFT_SPLIT_NAME}"
OUT="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"

echo "host=${SLURMD_NODENAME:-unknown} ($(hostname)) job=${SLURM_JOB_ID:-?}"
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M (CPU job, no GPU)"
echo "glob=$GLOB"
echo "index_dir=$INDEX_DIR split=$SPLIT -> $OUT"
mkdir -p "$INDEX_DIR"
"$PY" "$ROOT/rl/diffusion_ft/tools/build_offsets.py" --shard_glob "$GLOB" --index_dir "$INDEX_DIR"
"$PY" -m rl.diffusion_ft.tools.build_shard_splits --shard_glob "$GLOB" \
  --index_dir "$INDEX_DIR" --split "$SPLIT" --out "$OUT"
