#!/bin/bash
# Sweep the safety:anchor weighting as full training runs, to attribute the
# observed post-safety-loss degradation. Only calls sbatch -- one train_safety
# chain per arm. By default the arms run SERIALLY (one GPU at a time; see SERIAL
# below), so the sweep never occupies more than one slot. DFT_SERIAL=0 for the
# old concurrent-siblings behavior.
#
# WHY A RATIO, NOT A GRID. The objective is a single combined loss
#   total = safety_weight * safety + anchor_weight * anchor   (train_safety.py)
# optimized by Adam under clip_by_global_norm (rl_actor.py). Adam is invariant to
# a global scaling of the gradient, and the global-norm clip removes what little
# scale sensitivity is left, so (sw, aw) and (c*sw, c*aw) produce the SAME update
# up to Adam's eps. Only the ratio aw/sw matters. A 2D grid would be duplicate
# cells burning scarce a6000 time; this sweeps the ratio and pins the two
# endpoints that are the actual attribution test:
#
#   1:0   safety only  -- the current default that degraded
#   0:1   anchor only  -- pure BC continuation, the control that should NOT degrade
#
# and intermediate ratios locate where the safety term starts hurting eval. Read
# the arms by their eval_offroad_rate / eval_overlap_rate / reach / drift curves
# (in wandb group, or train_log.jsonl on disk): the lowest anchor weight at which
# eval stops regressing is the usable safety:anchor balance.
#
# Node: `general` advertises only a6000/a5000 (gpu24-31); this env's jaxlib dies
# at cuDNN autotune on v100 and OOMs on the 2080 Ti, so DO NOT widen the
# partition. Chains per arm: GPU train -> CPU figures (afterok), redrawable.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/sweep_safety_anchor.sh
#   DFT_ITERS=500 bash rl/diffusion_ft/slurm/sweep_safety_anchor.sh
#   DFT_WEIGHT_ARMS="1:0 1:100 0:1" bash rl/diffusion_ft/slurm/sweep_safety_anchor.sh
#   DFT_TFRECORD=/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord \
#     bash rl/diffusion_ft/slurm/sweep_safety_anchor.sh
#
# Each arm is "safety_weight:anchor_weight". All other train_safety knobs pass
# through unchanged (see submit_train_safety.sh).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_anchor_sweep/${STAMP}}"
# Arms are safety_weight:anchor_weight. Anchor is pinned at its NATIVE scale
# (aw=1) and safety is scaled DOWN, rather than scaling the tiny anchor up: under
# Adam+global-norm clip the two are numerically identical (only the ratio moves
# the policy), but pinning the anchor keeps the logged anchor value/gradient
# comparable to the BC/pretraining run, so it is easy to eyeball. 1:0 = the
# degrading safety-only default; 0:1 = the pure-BC control. At sw:aw=1:1 safety
# measured ~150x the anchor (39.87 vs 0.2653, failure set), so 0.1:1 is still
# safety-dominated (confirms the safety-only mode is flat), and 0.0003:1 reaches
# the gradient-share scale where the anchor could dominate.
ARMS="${DFT_WEIGHT_ARMS:-1:0 0.1:1 0.01:1 0.001:1 0.0003:1 0:1}"
# Shorter than a committed run: 200 iters at eval_every=10 gives 20 eval points,
# enough to see a degradation trend. Override for a fuller curve.
DFT_ITERS="${DFT_ITERS:-200}"
GROUP="${DFT_WANDB_GROUP:-safety_anchor_${STAMP}}"
# SERIAL (default): run ONE arm at a time. Each arm's train job is submitted with
# `--dependency=afterany:<previous arm's train job>`, so only a single arm is ever
# eligible to run -- the rest sit in the queue in `Dependency` state and consume
# NO GPU, leaving slots free for interactive work. `afterany` (not afterok) so a
# failed/timed-out arm still releases the next rather than stranding the chain.
# This is set up from rhea at submit time on purpose: a compute node cannot sbatch
# (every partition sets AllocNodes=rhea), so a job cannot launch its own successor.
# DFT_SERIAL=0 restores concurrent siblings (all arms queue at once).
SERIAL="${DFT_SERIAL:-1}"
PREV_TRAIN=""

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

echo "sweep    -> $BASE_OUT"
echo "wandb grp-> $GROUP"
echo "arms     -> $ARMS  (iters=$DFT_ITERS each)"
echo

for ARM in $ARMS; do
  SW="${ARM%%:*}"; AW="${ARM##*:}"
  if [[ "$SW" == "$ARM" || -z "$SW" || -z "$AW" ]]; then
    echo "SKIP malformed arm '$ARM' (want safety_weight:anchor_weight)" >&2; continue
  fi
  TAG="sw${SW}_aw${AW}"
  OUT_DIR="$BASE_OUT/$TAG"
  ENV_FILE="$ENV_DIR/safetyanchor_${STAMP}_${TAG}.env"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    # data (defaults to full WOMD corpus; set DFT_TFRECORD for a fixed batch)
    [[ -n "${DFT_TFRECORD:-}" ]] && echo "DFT_TFRECORD=${DFT_TFRECORD}"
    [[ -n "${DFT_INDICES:-}" ]]  && echo "DFT_INDICES=${DFT_INDICES}"
    echo "DFT_SHARD_GLOB=${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
    echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-16}"
    echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-32}"
    echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
    echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
    # named eval splits: per-iter solve/offroad on each, logged as `<split>/*`
    echo "DFT_SPLITS_JSON=${DFT_SPLITS_JSON:-}"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-48}"
    # objective -- the swept axis
    echo "DFT_SAFETY_WEIGHT=$SW"
    echo "DFT_ANCHOR_WEIGHT=$AW"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    # gradient entry point
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    # proposal
    echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-0}"
    echo "DFT_UNROLL_GRAD_STEPS=${DFT_UNROLL_GRAD_STEPS:-1}"
    echo "DFT_UNROLL_T_START=${DFT_UNROLL_T_START:--1}"
    echo "DFT_UNROLL_T_END=${DFT_UNROLL_T_END:-0}"
    echo "DFT_UNROLL_ETA=${DFT_UNROLL_ETA:-0.0}"
    echo "DFT_ANCHOR_STATES=${DFT_ANCHOR_STATES:-policy}"
    # schedule
    echo "DFT_ITERS=$DFT_ITERS"
    echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-10}"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_RESUME=${DFT_RESUME:-1}"
    echo "DFT_SEED=${DFT_SEED:-0}"
    # logging: one shared group, one name per arm, so the curves overlay
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
  } > "$ENV_FILE"

  DEP=()
  if [[ "$SERIAL" == "1" && -n "$PREV_TRAIN" ]]; then
    DEP=(--dependency="afterany:${PREV_TRAIN}")
  fi
  TRAIN_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable "${SB_FLAGS[@]}" "${DEP[@]}" --job-name="dft_${TAG}" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_FILE")
  VIZ_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable --dependency="afterok:${TRAIN_ID}" \
    rl/diffusion_ft/slurm/viz_safety_train.sbatch "$ENV_FILE")
  AFTER=""
  [[ -n "$PREV_TRAIN" && "$SERIAL" == "1" ]] && AFTER="  (waits for $PREV_TRAIN)"
  echo "  $TAG  train->$TRAIN_ID  figures->$VIZ_ID$AFTER  ($OUT_DIR)"
  PREV_TRAIN="$TRAIN_ID"
done

echo
echo "compare arms (offroad / overlap / drift per iter):"
echo "  for d in $BASE_OUT/sw*_aw*/train_log.jsonl; do echo \"== \$d ==\";"
echo "    python3 -c \"import json;rows=[json.loads(l) for l in open('\$d')];"
echo "    [print(r['iter'], r.get('eval_offroad_rate'), r.get('eval_overlap_rate'), r.get('eval_drift_m')) for r in rows if 'eval_offroad_rate' in r]\"; done"
