#!/bin/bash
# Smoke test for the three code paths submit_verified_safety.sh depends on, at a
# size that finishes in minutes. Only calls sbatch. Run this BEFORE the real
# chain: a typo in --verify_only would otherwise surface after an hour of queue
# plus an hour of rollouts.
#
# What it exercises, and what to read in the .out:
#   1. verify   train_safety --verify_only over 8 scenes/split at 1 repeat ->
#               verify_rows.jsonl with one row per scene. Then the CPU rebuild
#               runs on those rows (it will keep very few scenes -- that is
#               fine, the point is that it parses and writes a splits file).
#   2. chunked  a 2-iteration train with --split_cap 24 --eval_chunk 8, i.e.
#      eval     three chunks per split. Read `[safety] eval split <name>: 24
#               scenes (cap 24) in 3 chunk(s)` and check the base reference line
#               prints one number per split.
#   3. balanced a train pool of failure_train+success_train, uncapped, with
#      batches  --train_split_ratio 1,1. Read `[safety] BALANCED batches:` and
#               `balanced 8of564/8of23254` in the sampler's describe() line.
#
# The two jobs are independent, so they queue as siblings.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_smoke_verified.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

SPLITS_JSON="${DFT_SPLITS_JSON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/eval_splits.json}"
BASE="${DFT_SMOKE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/smoke_verified/${STAMP}}"
CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
mkdir -p "$BASE/verify" "$BASE/train"

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }

# ---- 1. verify + rebuild, tiny ----------------------------------------------
ENV_V="$ENV_DIR/smokeverify_${STAMP}.env"
{
  echo "DFT_OUT_DIR=$BASE/verify"
  echo "DFT_CKPT=$CKPT"
  echo "DFT_SPLITS_JSON=$SPLITS_JSON"
  echo "DFT_SHARD_GLOB=$GLOB"
  echo "DFT_INDEX_DIR=$IDX"
  echo "DFT_VERIFY_OUT=$BASE/verify/verify_rows.jsonl"
  echo "DFT_VERIFIED_JSON=$BASE/verify/eval_splits_verified.json"
  echo "DFT_VERIFY_SPLITS=failure_heldout,failure_train,success_heldout,success_train"
  # Leading "-" on purpose: the real chain passes -1 for the uncapped splits,
  # and a value starting with "-" is exactly what argparse mis-read as an option
  # in job 43942. failure_heldout uncapped is 64 scenes at 1 repeat -- ~1 min.
  echo "DFT_VERIFY_CAP=-1,8,8,8"
  echo "DFT_VERIFY_REPEATS=1"
  echo "DFT_VERIFY_SEED=777"
  echo "DFT_EVAL_CHUNK=8"
  echo "DFT_HELDOUT_FAILURE=-1"
  echo "DFT_HELDOUT_SUCCESS=4"
} > "$ENV_V"
VID=$(sb --parsable --time=00:40:00 "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/verify_splits.sbatch "$ENV_V")
echo "verify smoke  -> job $VID"
RID=$(sb --parsable --dependency="afterok:${VID}" \
  rl/diffusion_ft/slurm/rebuild_splits.sbatch "$ENV_V")
echo "rebuild smoke -> job $RID  (CPU, afterok:$VID)"

# ---- 2. chunked eval + balanced batches, 2 iterations ------------------------
ENV_T="$ENV_DIR/smoketrain_${STAMP}.env"
{
  echo "DFT_OUT_DIR=$BASE/train"
  echo "DFT_CKPT=$CKPT"
  echo "DFT_SHARD_GLOB=$GLOB"
  echo "DFT_INDEX_DIR=$IDX"
  echo "DFT_SCENES_PER_ITER=16"
  echo "DFT_EVAL_SCENES=8"
  echo "DFT_SPLITS_JSON=$SPLITS_JSON"
  echo "DFT_SPLITS=failure_heldout,success_heldout,failure_train,success_train"
  echo "DFT_SPLIT_CAP=24"
  echo "DFT_EVAL_CHUNK=8"
  echo "DFT_TRAIN_SPLITS=failure_train,success_train"
  echo "DFT_TRAIN_SPLIT_CAP=-1"
  echo "DFT_TRAIN_SPLIT_RATIO=1,1"
  echo "DFT_PROBE_PER_SPLIT=0"
  echo "DFT_ITERS=2"
  echo "DFT_EVAL_EVERY=1"
  echo "DFT_SAVE_EVERY=2"
  echo "DFT_EVAL_REPEATS=1"
  echo "DFT_ANCHOR_WEIGHT=0.0"
  echo "DFT_SAFETY_WEIGHT=1.0"
  echo "DFT_WANDB=0"
} > "$ENV_T"
TID=$(sb --parsable --time=01:00:00 "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_T")
echo "train smoke   -> job $TID"
echo
echo "read:"
echo "  tail -f $ROOT/logs/dft/slurm-dft_verify-${VID}.out"
echo "  tail -f $ROOT/logs/dft/slurm-dft_safetrain-${TID}.out"
echo "  head -2 $BASE/verify/verify_rows.jsonl"
