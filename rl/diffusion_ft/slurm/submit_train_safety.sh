#!/bin/bash
# Submitter for the analytic safety fine-tune. Only calls sbatch.
#
# Chains: GPU train  ->(afterok)  CPU figures. The CPU job only reads
# train_log.jsonl, so figures are redrawable without recomputation -- rerun
# viz_safety_train.sbatch alone against any finished out_dir.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea,lov4,
# so submitting from a compute node fails with "Access/permission denied"):
#
#   bash rl/diffusion_ft/slurm/submit_train_safety.sh
#   DFT_K_TRAINABLE=3 DFT_ITERS=200 bash rl/diffusion_ft/slurm/submit_train_safety.sh
#   DFT_UNROLL_STEPS=3 DFT_ANCHOR_STATES=log bash rl/diffusion_ft/slurm/submit_train_safety.sh
#
# Knobs (all optional, defaults live in train_safety.sbatch):
#   objective  DFT_ANCHOR_WEIGHT DFT_BC_WEIGHT DFT_SAFETY_WEIGHT
#              DFT_EDGE_MARGIN DFT_AGENT_MARGIN
#   gradient   DFT_K_TRAINABLE DFT_PREFIX_LEN DFT_LR DFT_SIGMA_FLOOR DFT_GRAD_ACCUM
#   eval       DFT_EVAL_SEED DFT_EVAL_REPEATS DFT_EVAL_BASE
#   proposal   DFT_UNROLL_STEPS DFT_UNROLL_GRAD_STEPS DFT_UNROLL_T_START
#              DFT_UNROLL_T_END DFT_UNROLL_ETA DFT_ANCHOR_STATES
#   data       DFT_SHARD_GLOB DFT_SCENES_PER_ITER DFT_EVAL_SCENES
#              (or DFT_TFRECORD + DFT_INDICES for a fixed batch)
#   splits     DFT_SPLITS_JSON DFT_SPLITS DFT_SPLIT_CAP DFT_EVAL_CHUNK
#   train pool DFT_TRAIN_SPLITS DFT_TRAIN_SPLIT_CAP DFT_TRAIN_SPLIT_RATIO
#              (empty DFT_TRAIN_SPLITS = uniform corpus)
#   probe      DFT_PROBE_PER_SPLIT DFT_PROBE_SPLITS DFT_PROBE_EVERY DFT_PROBE_SEED
#   schedule   DFT_ITERS DFT_MAX_REPLANS DFT_EVAL_EVERY DFT_SAVE_EVERY
#   simulator  DFT_SIM_AGENTS (none|idm) DFT_NEIGHBOR_SOURCE (log|sim)
#   logging    DFT_WANDB (0/1) DFT_WANDB_PROJECT DFT_WANDB_GROUP DFT_WANDB_NAME
#   placement  DFT_PARTITION DFT_QOS DFT_NODELIST DFT_EXCLUDE
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
# Two sibling submissions in the SAME SECOND collide: `date +%S` resolution gives
# both the same env filename and the same default out_dir, so the second write
# silently overwrites the first's config and both jobs then run one arm into one
# directory, interleaving train_log.jsonl and racing on latest/. That is exactly
# what "independent stages stay siblings so they queue concurrently" invites, and
# it ate an arm of the 09-02 unroll pair (jobs 40478/40480, both submitted at
# 02:38:02). $$ differs per invocation, so it separates them.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/safetrain_${STAMP}.env"
OUT_DIR="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_train/${STAMP}}"

{
  echo "DFT_OUT_DIR=$OUT_DIR"
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  # data
  [[ -n "${DFT_TFRECORD:-}" ]] && echo "DFT_TFRECORD=${DFT_TFRECORD}"
  [[ -n "${DFT_INDICES:-}" ]]  && echo "DFT_INDICES=${DFT_INDICES}"
  echo "DFT_SHARD_GLOB=${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-32}"
  echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-32}"
  echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
  echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
  # named splits: eval every DFT_EVAL_EVERY, and (DFT_TRAIN_SPLITS) the pool the
  # gradient is actually taken on
  [[ -n "${DFT_SPLITS_JSON:-}" ]] && echo "DFT_SPLITS_JSON=${DFT_SPLITS_JSON}"
  [[ -n "${DFT_SPLITS:-}" ]] && echo "DFT_SPLITS=${DFT_SPLITS}"
  [[ -n "${DFT_SPLIT_CAP:-}" ]] && echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP}"
  [[ -n "${DFT_EVAL_CHUNK:-}" ]] && echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK}"
  [[ -n "${DFT_TRAIN_SPLITS:-}" ]] && echo "DFT_TRAIN_SPLITS=${DFT_TRAIN_SPLITS}"
  [[ -n "${DFT_TRAIN_SPLIT_CAP:-}" ]] && echo "DFT_TRAIN_SPLIT_CAP=${DFT_TRAIN_SPLIT_CAP}"
  [[ -n "${DFT_TRAIN_SPLIT_RATIO:-}" ]] && echo "DFT_TRAIN_SPLIT_RATIO=${DFT_TRAIN_SPLIT_RATIO}"
  echo "DFT_PROBE_PER_SPLIT=${DFT_PROBE_PER_SPLIT:-4}"
  echo "DFT_PROBE_SPLITS=${DFT_PROBE_SPLITS:-failure_train,success_train,failure_heldout,success_heldout}"
  echo "DFT_PROBE_EVERY=${DFT_PROBE_EVERY:-0}"
  echo "DFT_PROBE_SEED=${DFT_PROBE_SEED:-0}"
  # objective
  echo "DFT_ANCHOR_WEIGHT=${DFT_ANCHOR_WEIGHT:-0.0}"
  echo "DFT_BC_WEIGHT=${DFT_BC_WEIGHT:-0.0}"
  echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
  echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
  echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
  echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
  echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
  echo "DFT_BARRIER_SHAPE=${DFT_BARRIER_SHAPE:-softplus}"
  # gradient entry point
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
  echo "DFT_COLLECT_STEPS=${DFT_COLLECT_STEPS:-0}"
  echo "DFT_EVAL_STEPS=${DFT_EVAL_STEPS:-0}"
  # proposal: 0 = free-running chain, K >= 1 = teacher-forced K-step unroll
  echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-0}"
  echo "DFT_UNROLL_GRAD_STEPS=${DFT_UNROLL_GRAD_STEPS:-1}"
  echo "DFT_UNROLL_T_START=${DFT_UNROLL_T_START:--1}"
  echo "DFT_UNROLL_T_END=${DFT_UNROLL_T_END:-0}"
  echo "DFT_UNROLL_ETA=${DFT_UNROLL_ETA:-0.0}"
  echo "DFT_ANCHOR_STATES=${DFT_ANCHOR_STATES:-policy}"
  # schedule
  echo "DFT_ITERS=${DFT_ITERS:-500}"
  echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
  echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
  echo "DFT_REPLAN=${DFT_REPLAN:-10}"
  # the simulator the curve is measured in, and where the loss reads neighbours
  echo "DFT_SIM_AGENTS=${DFT_SIM_AGENTS:-none}"
  echo "DFT_NEIGHBOR_SOURCE=${DFT_NEIGHBOR_SOURCE:-log}"
  # The clock (2026-09-24): `median` unless told otherwise; the batch script's
  # own fallback is `first`, so leaving this line out would silently run the bug.
  echo "DFT_WORLD_DT_METHOD=${DFT_WORLD_DT_METHOD:-median}"
  echo "DFT_LR=${DFT_LR:-1e-5}"
  echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-100}"
  echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
  echo "DFT_GRAD_ACCUM=${DFT_GRAD_ACCUM:-0}"
  echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
  echo "DFT_EVAL_REPEATS=${DFT_EVAL_REPEATS:-1}"
  echo "DFT_EVAL_BASE=${DFT_EVAL_BASE:-1}"
  echo "DFT_RESUME=${DFT_RESUME:-1}"
  echo "DFT_SEED=${DFT_SEED:-0}"
  echo "DFT_WANDB=${DFT_WANDB:-1}"
  echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
  [[ -n "${DFT_WANDB_GROUP:-}" ]] && echo "DFT_WANDB_GROUP=${DFT_WANDB_GROUP}"
  [[ -n "${DFT_WANDB_NAME:-}" ]]  && echo "DFT_WANDB_NAME=${DFT_WANDB_NAME}"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

# Scrub SLURM_* so a submission from inside an allocation does not inherit it.
TRAIN_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_FILE")
echo "train  -> job $TRAIN_ID  ($OUT_DIR)"

VIZ_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency="afterok:${TRAIN_ID}" \
  rl/diffusion_ft/slurm/viz_safety_train.sbatch "$ENV_FILE")
echo "figures-> job $VIZ_ID  (CPU, afterok:${TRAIN_ID})"
echo
echo "watch:  tail -f $ROOT/logs/dft/slurm-dft_safetrain-${TRAIN_ID}.out"
echo "curve:  python3 -c \"import json;[print(r['iter'], r['safety_last'], r.get('eval_offroad_rate'), r.get('eval_drift_m')) for r in map(json.loads, open('$OUT_DIR/train_log.jsonl'))]\""
