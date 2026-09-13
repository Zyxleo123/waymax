#!/bin/bash
# THE NULL EXPERIMENT: does the degradation scale with the learning rate?
#
# WHY. Every arm of the 09-02 safety:anchor sweep degraded -- including the two
# that were controls. `sw=0` (pure anchor, a plain BC continuation that optimizes
# nothing new) came out WORST of the six: failure solved 0.191 vs 0.354 for
# safety-only, success solved 0.792 vs 0.806-0.846, and its own training safety
# value rose 17x (0.037 -> 0.641) on states nothing was pushing it toward. A
# degradation that appears at sw=0 AND at aw=0 is not about the objective.
#
# The mechanism it points at is the optimizer. An Adam update is
# `lr * m/(sqrt(v)+eps)`: O(lr) per parameter WHATEVER the gradient norm is. So
# gradient magnitude picks the direction and the step COUNT picks the distance,
# and nothing in train_safety bounded either (`safety_update` skips the DPPO
# trust region -- correctly, there is no importance ratio to protect, but that
# region was also the only leash on how far the weights moved; and
# `clip_by_global_norm` sits BEFORE Adam, where a uniform pre-scale cancels
# against Adam's own RMS normalization).
#
# At 31.7M params, lr=1e-5, 200 iters x 8 contexts = 1600 steps:
#   aligned displacement  lr*N*sqrt(P)   = 90
#   random-walk           lr*sqrt(N*P)   = 2.25
# and the six arms' final weights differed pairwise by 1.35-5.02 -- the walk
# scale. In metres that is ~0.45 m of drift (the anchor-only arm went 0.80 ->
# 1.25 m while optimizing nothing), against an edge_margin of 0.3 m and a median
# failure-set off-road excursion of 0.77 m. The noise moves the car as far as the
# violation being fixed, which is why the threshold metrics read as a lottery.
#
# THE TEST. Run the null objective (safety_weight=0, anchor_weight=1) down an lr
# ladder. If degradation is the walk, it scales with lr and vanishes at 1e-7. If
# it is the anchor objective genuinely doing damage, it persists in shape while
# only the rate changes. Those two look nothing alike on a curve.
#
# GRAD_ACCUM STAYS AT 1 FOR THE LADDER, on purpose: 1 is the pre-09-03 schedule
# (one Adam step per replan context, 1600 steps) that produced the degradation
# being explained. Running the ladder under the new default would confound "lr
# does not matter" with "accumulation already fixed it". The fourth arm is the
# controlled version of exactly that: lr=1e-5 with accum=0 takes 200 steps
# instead of 1600, so its predicted walk shrinks by sqrt(8)=2.8x and it should
# land between the 1e-5 and 1e-6 arms. That is a falsifiable placement, not a
# vibe -- if it lands on top of 1e-5 instead, the walk story is wrong.
#
# All arms share ONE eval seed (DFT_EVAL_SEED) and score the frozen base once on
# it, so every curve is a paired comparison against a constant and arm-vs-arm
# differences are not sampler noise. That is the other half of the 09-02 problem:
# evals drew from the training rng, so an eval depended on how much training had
# preceded it, and arms with identical weights disagreed at iter 0 by 0.3.
#
# Node: `general` advertises a6000/a5000 (gpu24-31); this env's jaxlib dies at
# cuDNN autotune on v100 and OOMs on the 2080 Ti, so DO NOT widen the partition.
# Arms run SERIALLY by default (one GPU at a time); DFT_SERIAL=0 for siblings.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/sweep_null_lr.sh
#   DFT_ITERS=200 DFT_LR_ARMS="1e-5:1 1e-7:1" bash rl/diffusion_ft/slurm/sweep_null_lr.sh
#
# Each arm is "lr:grad_accum".
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/null_lr_sweep/${STAMP}}"

ARMS="${DFT_LR_ARMS:-1e-5:1 1e-6:1 1e-7:1 1e-5:0}"
# 200 to match the sweep being explained, at eval_every=10 => 20 eval points.
DFT_ITERS="${DFT_ITERS:-200}"
GROUP="${DFT_WANDB_GROUP:-null_lr_${STAMP}}"
SERIAL="${DFT_SERIAL:-1}"
PREV_TRAIN=""
TRAIN_IDS=()

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

echo "sweep    -> $BASE_OUT"
echo "wandb grp-> $GROUP"
echo "arms     -> $ARMS  (iters=$DFT_ITERS each, objective = anchor only)"
echo

for ARM in $ARMS; do
  LR="${ARM%%:*}"; GA="${ARM##*:}"
  if [[ "$LR" == "$ARM" || -z "$LR" || -z "$GA" ]]; then
    echo "SKIP malformed arm '$ARM' (want lr:grad_accum)" >&2; continue
  fi
  TAG="lr${LR}_ga${GA}"
  OUT_DIR="$BASE_OUT/$TAG"
  ENV_FILE="$ENV_DIR/nulllr_${STAMP}_${TAG}.env"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    # data -- same corpus and splits as the sweep being explained
    [[ -n "${DFT_TFRECORD:-}" ]] && echo "DFT_TFRECORD=${DFT_TFRECORD}"
    [[ -n "${DFT_INDICES:-}" ]]  && echo "DFT_INDICES=${DFT_INDICES}"
    echo "DFT_SHARD_GLOB=${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
    echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-16}"
    echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-32}"
    echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
    echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
    # PIN the named splits. Empty here means only the random 32-scene
    # held-out batch is scored, and failure_heldout/success_heldout --
    # the splits every claim in this diagnosis is stated on -- are never
    # measured. This is the same file the 09-02 sweep used; a different
    # one would make the arms incomparable to it.
    echo "DFT_SPLITS_JSON=${DFT_SPLITS_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/eval_splits.json}"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-48}"
    # objective: THE NULL. Safety off, anchor at its native scale.
    echo "DFT_SAFETY_WEIGHT=0"
    echo "DFT_ANCHOR_WEIGHT=1"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    # gradient entry point (unchanged from the sweep being explained)
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-0}"
    echo "DFT_UNROLL_GRAD_STEPS=${DFT_UNROLL_GRAD_STEPS:-1}"
    echo "DFT_UNROLL_T_START=${DFT_UNROLL_T_START:--1}"
    echo "DFT_UNROLL_T_END=${DFT_UNROLL_T_END:-0}"
    echo "DFT_UNROLL_ETA=${DFT_UNROLL_ETA:-0.0}"
    echo "DFT_ANCHOR_STATES=${DFT_ANCHOR_STATES:-policy}"
    # schedule -- the swept axis
    echo "DFT_LR=$LR"
    echo "DFT_GRAD_ACCUM=$GA"
    echo "DFT_ITERS=$DFT_ITERS"
    echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-10}"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_RESUME=${DFT_RESUME:-1}"
    echo "DFT_SEED=${DFT_SEED:-0}"
    # eval: ONE seed across every arm, plus the frozen-base reference row
    echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
    echo "DFT_EVAL_REPEATS=${DFT_EVAL_REPEATS:-4}"
    echo "DFT_EVAL_BASE=${DFT_EVAL_BASE:-1}"
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
    sbatch --parsable "${SB_FLAGS[@]}" "${DEP[@]}" --job-name="dft_null_${TAG}" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_FILE")
  VIZ_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable --dependency="afterok:${TRAIN_ID}" \
    rl/diffusion_ft/slurm/viz_safety_train.sbatch "$ENV_FILE")
  AFTER=""
  [[ -n "$PREV_TRAIN" && "$SERIAL" == "1" ]] && AFTER="  (waits for $PREV_TRAIN)"
  echo "  $TAG  train->$TRAIN_ID  figures->$VIZ_ID$AFTER  ($OUT_DIR)"
  TRAIN_IDS+=("$TRAIN_ID")
  PREV_TRAIN="$TRAIN_ID"
done

# The measurement the ladder exists to feed: ||theta - theta_0|| per arm against
# Adam's aligned and random-walk predictions, on the CPU partition so it does not
# queue behind GPU work. `afterany` -- a timed-out arm still saved weights, and a
# partial ladder is still worth measuring.
if ((${#TRAIN_IDS[@]})); then
  DIST_ENV="$ENV_DIR/nulllr_${STAMP}_paramdist.env"
  {
    echo "DFT_ARMS=$BASE_OUT/*/"
    echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    echo "DFT_DIST_OUT=$BASE_OUT/param_distance.json"
  } > "$DIST_ENV"
  DEPS=$(IFS=:; echo "${TRAIN_IDS[*]}")
  DIST_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable --dependency="afterany:${DEPS}" \
    rl/diffusion_ft/slurm/probe_param_distance.sbatch "$DIST_ENV")
  echo
  echo "  param-distance -> $DIST_ID  ($BASE_OUT/param_distance.json)"
fi

echo
echo "read the ladder (solved rate vs its own fixed-seed base reference):"
echo "  for d in $BASE_OUT/lr*/train_log.jsonl; do echo \"== \$d\";"
echo "    python3 -c \"import json;rows=[json.loads(l) for l in open('\$d')];"
echo "    [print(r['iter'], round(r['success_heldout/solved_rate'],3),"
echo "     round(r.get('base_success_heldout/solved_rate',float('nan')),3),"
echo "     round(r['success_heldout/drift_m'],3)) for r in rows if 'success_heldout/solved_rate' in r]\"; done"
