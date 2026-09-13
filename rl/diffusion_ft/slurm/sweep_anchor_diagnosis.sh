#!/bin/bash
# Two questions the 09-02 sweep could not answer, because every arm shared a seed
# and every arm trained on policy-visited states.
#
# CONTEXT. The param-distance probe (job 41725) killed the "it is all optimizer
# noise" reading: displacement cosines between arms run 0.006-0.95, where two
# random walks in 31.7M dimensions would be |cos| ~ 1e-4. The arms sit on a clean
# 2-D structure -- an anchor direction and a safety direction that are ORTHOGONAL
# to each other (sw0_aw1 vs sw1_aw0: cos = +0.0063), with the ratio arms
# interpolating monotonically between them. Both terms move the policy
# coherently. What the probe found instead is that displacement rank-orders
# damage perfectly and inversely (Spearman -1.0 over all six arms):
#
#   ||dtheta||  4.352  4.211  4.176  3.886  3.250  2.527
#   F.solved    0.191  0.300  0.311  0.321  0.348  0.354
#
# and the anchor is what does the moving: anchor-only travels furthest and does
# worst, and adding safety weight monotonically SHRINKS the distance (two
# near-orthogonal terms disagreeing per-coordinate shrink Adam's m/sqrt(v), so
# safety brakes the anchor's march). Anchor-only also moves the weights most
# (4.35) and the car least (drift +0.45 m) while safety-only moves the weights
# least (2.53) and the car most (+1.76 m) -- a large weight change buying almost
# no behaviour, which is what a term spread over all 25 output columns should
# look like when only ~5 are ever executed.
#
# ARM 1 -- IS THE ANCHOR DIRECTION SIGNAL OR SEED? Every arm of the 09-02 sweep
# ran at `--seed 0` with the same data order, so their gradient noise is
# identical and would ALSO produce high cosines between similar objectives. The
# orthogonality of anchor-vs-safety rules that out as the whole story, but not
# for two arms with the SAME objective. `policy:1` is anchor-only at seed 1,
# otherwise identical to the null ladder's `lr1e-5_ga1` arm. Cosine ~0.95 against
# it => the anchor direction is objective signal. Cosine ~0.05 => what looked
# like a reproducible direction was one seed's noise, and the ladder's reading
# changes with it.
#
# ARM 2 -- IS THE ANCHOR HURTING BECAUSE OF WHERE IT IS TRAINED? The default
# `--anchor_states policy` is DAgger-shaped: contexts come from the policy's own
# closed-loop rollout, but the target is the human's future re-expressed from
# whatever drifted pose the policy reached. From a drifted state that target
# demands an instantaneous lateral correction, and a model trained on it learns
# to emit exactly that. `log:0` trains the same objective on the human's own
# states, where the target is feasible by construction. If the anchor's damage is
# the relabeling, `log` is harmless and `policy` is not.
#
# THE SEED-0 REFERENCE IS NOT RE-RUN. It is the null ladder's `lr1e-5_ga1` arm
# (sw=0, aw=1, lr=1e-5, grad_accum=1, seed 0, anchor_states=policy) -- identical
# by construction to `policy:0`. Point DFT_EXTRA_ARMS at that sweep so the
# param-distance probe spans both and computes the cross-sweep cosine. Launch
# this with NO other DFT_* overrides, or the two stop being comparable.
#
# COST. Only the two heldout splits are evaluated (not the *_train pools): at
# --eval_repeats 4 each eval point costs 4 rollouts per split, and the ladder's
# 4-split default roughly doubles an arm's wall clock. ~2.5 h of training per arm
# plus eval; arms run SERIALLY (DFT_SERIAL=0 for siblings).
#
# Node: `general` a6000/a5000 only -- this env's jaxlib dies at cuDNN autotune on
# v100 and OOMs on the 2080 Ti. Do not widen the partition.
#
# Usage (from repo root, ON RHEA):
#   DFT_EXTRA_ARMS='/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/null_lr_sweep/<STAMP>/*/' \
#     bash rl/diffusion_ft/slurm/sweep_anchor_diagnosis.sh
#
# Each arm is "anchor_states:seed".
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/anchor_diagnosis/${STAMP}}"

ARMS="${DFT_DIAG_ARMS:-policy:1 log:0}"
DFT_ITERS="${DFT_ITERS:-200}"
GROUP="${DFT_WANDB_GROUP:-anchor_diag_${STAMP}}"
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
echo "seed-0 policy reference comes from DFT_EXTRA_ARMS=${DFT_EXTRA_ARMS:-<unset>}"
echo

for ARM in $ARMS; do
  AS="${ARM%%:*}"; SD="${ARM##*:}"
  if [[ "$AS" == "$ARM" || -z "$AS" || -z "$SD" ]]; then
    echo "SKIP malformed arm '$ARM' (want anchor_states:seed)" >&2; continue
  fi
  if [[ "$AS" != "policy" && "$AS" != "log" ]]; then
    echo "SKIP arm '$ARM': anchor_states must be policy|log" >&2; continue
  fi
  TAG="as-${AS}_seed${SD}"
  OUT_DIR="$BASE_OUT/$TAG"
  ENV_FILE="$ENV_DIR/anchordiag_${STAMP}_${TAG}.env"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    # data -- must match the null ladder exactly for the cross-sweep comparison
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
    # Heldout only: the *_train pools double the eval cost and answer nothing here.
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-48}"
    # objective: anchor only, same null as the ladder
    echo "DFT_SAFETY_WEIGHT=0"
    echo "DFT_ANCHOR_WEIGHT=1"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-0}"
    echo "DFT_UNROLL_GRAD_STEPS=${DFT_UNROLL_GRAD_STEPS:-1}"
    echo "DFT_UNROLL_T_START=${DFT_UNROLL_T_START:--1}"
    echo "DFT_UNROLL_T_END=${DFT_UNROLL_T_END:-0}"
    echo "DFT_UNROLL_ETA=${DFT_UNROLL_ETA:-0.0}"
    # the swept axes
    echo "DFT_ANCHOR_STATES=$AS"
    echo "DFT_SEED=$SD"
    # schedule -- grad_accum 1 = the schedule that produced the degradation, so
    # this stays comparable to the ladder's lr1e-5_ga1 reference arm.
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_GRAD_ACCUM=${DFT_GRAD_ACCUM:-1}"
    echo "DFT_ITERS=$DFT_ITERS"
    echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-10}"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_RESUME=${DFT_RESUME:-1}"
    # eval seed is NOT the training seed: the arms differ in --seed on purpose,
    # and they must still be scored on identical sampler noise to be compared.
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
    sbatch --parsable "${SB_FLAGS[@]}" "${DEP[@]}" --job-name="dft_diag_${TAG}" \
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

# Param distance across BOTH sweeps: the cross-sweep cosine (as-policy_seed1 vs
# the ladder's lr1e-5_ga1) is the whole point of arm 1.
if ((${#TRAIN_IDS[@]})); then
  DIST_ENV="$ENV_DIR/anchordiag_${STAMP}_paramdist.env"
  ARMS_GLOB="$BASE_OUT/*/"
  [[ -n "${DFT_EXTRA_ARMS:-}" ]] && ARMS_GLOB="$ARMS_GLOB,${DFT_EXTRA_ARMS}"
  {
    echo "DFT_ARMS=$ARMS_GLOB"
    echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    echo "DFT_DIST_OUT=$BASE_OUT/param_distance.json"
  } > "$DIST_ENV"
  DEPS=$(IFS=:; echo "${TRAIN_IDS[*]}")
  DIST_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable --dependency="afterany:${DEPS}" \
    rl/diffusion_ft/slurm/probe_param_distance.sbatch "$DIST_ENV")
  echo
  echo "  param-distance -> $DIST_ID  ($BASE_OUT/param_distance.json)"
  echo "     arms: $ARMS_GLOB"
fi

echo
echo "read it: cosine(as-policy_seed1, lr1e-5_ga1) in param_distance.json"
echo "  ~0.95 => the anchor direction is signal; ~0.05 => it was one seed's noise"
echo "and compare as-log_seed0 vs as-policy_seed1 solved/drift curves:"
echo "  for d in $BASE_OUT/as-*/train_log.jsonl; do echo \"== \$d\";"
echo "    python3 -c \"import json;rows=[json.loads(l) for l in open('\$d')];"
echo "    [print(r['iter'], round(r['failure_heldout/solved_rate'],3),"
echo "     round(r.get('base_failure_heldout/solved_rate',float('nan')),3),"
echo "     round(r['success_heldout/drift_m'],3)) for r in rows if 'failure_heldout/solved_rate' in r]\"; done"
