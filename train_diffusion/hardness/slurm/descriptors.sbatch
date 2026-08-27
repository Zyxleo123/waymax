#!/bin/bash
#SBATCH --job-name=hrd_desc
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%A_%a.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%A_%a.out
#
# Stage 2/3: descriptor extraction over the labelled cache shards (array job).
# CPU-only. ~3.4 ms/scenario measured, so a 455-scenario shard costs a few
# seconds plus ~1s to load; the array size is set by the submitter.
#
# Resumable: a shard whose .desc.npz already exists is skipped, and each write
# goes to a .tmp file that is renamed atomically -- a job killed at the time
# limit can be resubmitted without corrupting anything.
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

if [[ ! -f "$HRD_OUT/labels.npz" ]]; then
  # Gate, not a crash: exiting non-zero here would strand the ranking job in
  # DependencyNeverSatisfied with no explanation of what went wrong.
  echo "GATE: $HRD_OUT/labels.npz is missing -- stage 1 did not produce labels. Nothing to do."
  exit 0
fi

echo "host=${SLURMD_NODENAME:-unknown} task=$TASK_ID/$NUM_TASKS cache=$HRD_CACHE_DIR"

# shellcheck disable=SC2086
"$PYTHON" -m train_diffusion.hardness.descriptors \
  --cache_dir "$HRD_CACHE_DIR" \
  --labels "$HRD_OUT/labels.npz" \
  --out_dir "$HRD_OUT/desc" \
  --task_id "$TASK_ID" \
  --num_tasks "$NUM_TASKS" \
  --start_timestep "$HRD_START_T" \
  --anchor_span "$HRD_ANCHOR_SPAN" \
  --anchors $HRD_ANCHORS
