#!/bin/bash
#SBATCH --job-name=hrd_train
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
# 8, not 16: the loader runs num_workers=0, so 4 arms pack onto gpu28's 48 CPUs
# (4 GPUs) instead of only 3 at 16 each.
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Phase 2, stage 3: one arm of the alpha sweep.
#
# a5000 (24G): moved off a6000 because the general a6000s (gpu24-31) sit full and
# the sweep waited forever on (Priority); gpu28 is a free a5000 in the same
# partition. Still >11G, so it clears the rtx_2080_ti death the a6000 pin guarded
# against. Do NOT drop to a 2080_ti: a CUDA/XLA allocation failure there kills the
# process with NO Python traceback, which reads as a mysterious silent FAILED.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
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

export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
# 24G a5000 headroom: grow the GPU allocation on demand instead of grabbing 75%
# up front, so training + the closed-loop eval's waymax rollout coexist.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Belt and braces with data/scenario_loader.py hiding the GPU from TF: if some
# other import gets TF onto the device first, at least it grows instead of
# claiming the card and starving the waymax rollout.
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

HRD_OUT="${HRD_OUT:-$ROOT/runs/hardness}"
HRD_ALPHA="${HRD_ALPHA:-0.0}"
HRD_EXCLUDE_DIRTY="${HRD_EXCLUDE_DIRTY:-1}"
HRD_WEIGHT_CLIP="${HRD_WEIGHT_CLIP:-10.0}"
HRD_CKPT="${HRD_CKPT:?set HRD_CKPT to the pretrained diffusion checkpoint}"
HRD_CACHE_DIR="${HRD_CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/}"
HRD_SAVE_DIR="${HRD_SAVE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/hardness_sweep}"
HRD_EPOCHS="${HRD_EPOCHS:-20}"
HRD_STEPS="${HRD_STEPS:-1000}"
HRD_SEED="${HRD_SEED:-0}"
HRD_SHUFFLE_SEED="${HRD_SHUFFLE_SEED:-0}"

# Closed-loop safe-goal-reach eval (same criterion that produced the failure set:
# success = goal_reached & ~offroad & ~overlap & ~tl_violation). Two wandb series:
# the 218-scene failure set (headroom) and a held-out WOMD slice (general drift).
# Set HRD_EVAL_EVERY=0 to disable.
HRD_EVAL_EVERY="${HRD_EVAL_EVERY:-5}"
HRD_EVAL_TFRECORD_DIR="${HRD_EVAL_TFRECORD_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures}"
HRD_EVAL_NUM="${HRD_EVAL_NUM:-218}"
HRD_EVAL2_TFRECORD_DIR="${HRD_EVAL2_TFRECORD_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training}"
HRD_EVAL2_NUM="${HRD_EVAL2_NUM:-200}"
HRD_EVAL_LANE_GRAPH_DIR="${HRD_EVAL_LANE_GRAPH_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"

RISK_ARGS=()
if [[ "$HRD_ALPHA" != "0.0" || "$HRD_EXCLUDE_DIRTY" == "1" ]]; then
  if [[ ! -f "$HRD_OUT/risk.npz" ]]; then
    echo "GATE: $HRD_OUT/risk.npz missing -- the risk model has not been fit. Nothing to do."
    exit 0
  fi
  RISK_ARGS+=(--risk_table "$HRD_OUT/risk.npz")
fi
if [[ "$HRD_EXCLUDE_DIRTY" == "1" ]]; then
  RISK_ARGS+=(--exclude_dirty)
else
  RISK_ARGS+=(--no_exclude_dirty)
fi

EVAL_ARGS=()
if [[ "$HRD_EVAL_EVERY" != "0" ]]; then
  EVAL_ARGS+=(
    --eval_every "$HRD_EVAL_EVERY"
    --eval_lane_graph_dir "$HRD_EVAL_LANE_GRAPH_DIR"
    --eval_tag failures --eval_tfrecord_dir "$HRD_EVAL_TFRECORD_DIR" --eval_num_scenarios "$HRD_EVAL_NUM"
    --eval2_tag womd --eval2_lane_graph_dir "$HRD_EVAL_LANE_GRAPH_DIR"
    --eval2_tfrecord_dir "$HRD_EVAL2_TFRECORD_DIR" --eval2_num_scenarios "$HRD_EVAL2_NUM"
  )
fi

echo "host=${SLURMD_NODENAME:-unknown} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "alpha=$HRD_ALPHA exclude_dirty=$HRD_EXCLUDE_DIRTY clip=$HRD_WEIGHT_CLIP seed=$HRD_SEED"
echo "eval_every=$HRD_EVAL_EVERY eval=$HRD_EVAL_TFRECORD_DIR eval2=$HRD_EVAL2_TFRECORD_DIR"

"$PYTHON" -m train_diffusion.train_diffusion_policy \
  --pretrained_ckpt "$HRD_CKPT" \
  --cache_dir "$HRD_CACHE_DIR" \
  --save_dir "$HRD_SAVE_DIR" \
  --epochs "$HRD_EPOCHS" \
  --steps_per_epoch "$HRD_STEPS" \
  --seed "$HRD_SEED" \
  --shuffle_seed "$HRD_SHUFFLE_SEED" \
  --hardness_alpha "$HRD_ALPHA" \
  --weight_clip "$HRD_WEIGHT_CLIP" \
  --wandb_name "hardness_alpha${HRD_ALPHA}_dirty${HRD_EXCLUDE_DIRTY}" \
  "${RISK_ARGS[@]}" \
  "${EVAL_ARGS[@]:+"${EVAL_ARGS[@]}"}"
