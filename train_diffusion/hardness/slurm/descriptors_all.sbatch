#!/bin/bash
#SBATCH --job-name=hrd_desc_all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%A_%a.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%A_%a.out
#
# Phase 2, stage 1: descriptors for ALL 1000 cache shards, not just the 154 that
# were evaluated. This is the point of using a-priori descriptors -- 3,135
# failure labels cannot reweight 2.3M examples, but a model fit on them can score
# every one.
#
# Writes to desc_all/ rather than reusing desc/: the 154 evaluated shards took
# their goal from the label table and the rest fall back to recomputing it, and
# although the rule is identical (ego's last valid log position) a single
# consistent source removes the question entirely. ~7 s/shard measured.
#
# CPU-only. Resumable: existing .desc.npz files are skipped.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/hardness"

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

HRD_CACHE_DIR="${HRD_CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/}"
HRD_OUT="${HRD_OUT:-$ROOT/runs/hardness}"
HRD_START_T="${HRD_START_T:-10}"
HRD_ANCHORS="${HRD_ANCHORS:-0 10 20 30 40}"
HRD_ANCHOR_SPAN="${HRD_ANCHOR_SPAN:-50}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host=${SLURMD_NODENAME:-unknown} task=$TASK_ID/$NUM_TASKS cache=$HRD_CACHE_DIR"

# shellcheck disable=SC2086
"$PYTHON" -m train_diffusion.hardness.descriptors \
  --cache_dir "$HRD_CACHE_DIR" \
  --shard_source cache \
  --out_dir "$HRD_OUT/desc_all" \
  --task_id "$TASK_ID" \
  --num_tasks "$NUM_TASKS" \
  --start_timestep "$HRD_START_T" \
  --anchor_span "$HRD_ANCHOR_SPAN" \
  --anchors $HRD_ANCHORS
