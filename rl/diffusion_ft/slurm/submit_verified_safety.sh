#!/bin/bash
# The 1000-scene rerun of the arm that worked. Only calls sbatch.
#
# WHAT THIS RERUNS. `probe_anchor/20260904_002609_25528/noanchor` is the only
# configuration measured so far that improved the failure splits without paying
# for it on the successes:
#
#     split             base    iter 299
#     failure_heldout   0.328   0.578   (peak 0.625)
#     failure_train     0.219   0.562
#     success_heldout   0.984   0.922
#     success_train     0.953   1.000
#
# i.e. safety_weight=1, anchor_weight=0, gradient taken on a mixed
# failure/success pool. The anchor arm was worse on failures, and the
# failure-ONLY pool (sweep ..._25576) took success_heldout to 0.688.
#
# WHAT CHANGES HERE, and only this:
#   1. Eval splits go from 48 scenes to DFT_SPLIT_CAP=1000. At 48 the binomial
#      floor is ~0.07 on a rate near 0.8 -- the same size as the effect. Splits
#      longer than DFT_EVAL_CHUNK are rolled in chunks and aggregated
#      scene-weighted (a single 1000-scene batched state does not fit on the GPU).
#   2. The failure splits are REBUILT from scenes the base verifiably fails under
#      the eval protocol. The old labels came from the scan's deterministic
#      rollout, under which the base "fails" scenes it solves 0.328 of the time
#      when scored the way the curve scores it.
#   3. The gradient sees ALL the data (`--train_split_cap -1`) with a per-batch
#      quota (`DFT_TRAIN_SPLIT_RATIO=1,1`) instead of a 256/256 capped pool, so
#      no scene is discarded to balance the mix.
#
# CHAIN:  verify (GPU) -> rebuild (CPU) -> train (GPU) -> figures (CPU)
# Each stage is afterok on the previous. The verify and rebuild stages are
# build-once assets: once DFT_VERIFIED_JSON exists, set DFT_SKIP_VERIFY=1 and
# this submits the training arm alone against it.
#
# COST. Eval is linear in cap x repeats x |splits| and is charged at every
# DFT_EVAL_EVERY -- and it, not the safety loss, is what bounds this arm.
# Measured 2026-09-09: training is 11 s/iter (`iter_s` is stamped BEFORE the eval
# block, so eval is charged on top), so 300 iters is 55 min of training against
# 12 eval points x ~25 min = 5 h of eval. Eval was ~85% of the wall clock, which
# is why the longest arm to date reached only ~700 iterations (~11k scenes) and
# why speeding up the loss would have bought ~3%.
#
# Both eval knobs are now set from that measurement:
#   * repeats 2 -> 1. The repeats were calibrated at cap=48, where one pass has a
#     binomial floor of ~0.07. At cap=1000 that floor is ~0.013 in a SINGLE pass;
#     the extra repeats re-rolled the same fixed 1000 scenes for precision the
#     cap already bought. Restore them only if you drop the cap back down.
#   * every 25 -> 100. Still ~30 curve points over a 3000-iteration run.
# With DFT_SCENES_PER_ITER=32 that is ~12x the scenes per GPU-hour.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea, so a
# compute node fails with "Access/permission denied"):
#
#   bash rl/diffusion_ft/slurm/submit_verified_safety.sh
#   DFT_SKIP_VERIFY=1 DFT_VERIFIED_JSON=<path> \
#     bash rl/diffusion_ft/slurm/submit_verified_safety.sh
#
# Node: `general` advertises only a6000/a5000 (gpu24-31). This env's jaxlib dies
# at cuDNN autotune on v100 and OOMs on the 2080 Ti -- do NOT widen the partition.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
# $$ as well as the second: two submissions in the same second would otherwise
# share an env filename and an out_dir and interleave one log.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

# ---- where things live -------------------------------------------------------
SPLITS_DIR="${DFT_SPLITS_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0}"
# The splits being VERIFIED. A different file is silently wrong: scene ids are
# positions in the scan's shard_glob order, so arms scored against another file
# are not comparable to anything already measured, and nothing errors.
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-${SPLITS_DIR}/eval_splits.json}"
VERIFY_DIR="${DFT_VERIFY_DIR:-${SPLITS_DIR}/verified_seed777}"
DFT_VERIFY_OUT="${DFT_VERIFY_OUT:-${VERIFY_DIR}/verify_rows.jsonl}"
DFT_VERIFIED_JSON="${DFT_VERIFIED_JSON:-${VERIFY_DIR}/eval_splits_verified.json}"
OUT_DIR="${DFT_OUT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_train/verified_${STAMP}}"
SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

sb() {  # scrub SLURM_* so a submission from inside an allocation does not inherit it
  env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"
}

# ---- stage 1+2: verify the labels, rebuild the splits -------------------------
DEP=()
if [[ "${DFT_SKIP_VERIFY:-0}" != "1" ]]; then
  mkdir -p "$VERIFY_DIR"
  ENV_V="$ENV_DIR/verify_${STAMP}.env"
  {
    echo "DFT_OUT_DIR=$VERIFY_DIR"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SPLITS_JSON=$DFT_SPLITS_JSON"
    echo "DFT_SHARD_GLOB=$SHARD_GLOB"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
    echo "DFT_VERIFY_OUT=$DFT_VERIFY_OUT"
    echo "DFT_VERIFIED_JSON=$DFT_VERIFIED_JSON"
    echo "DFT_VERIFY_SPLITS=${DFT_VERIFY_SPLITS:-failure_heldout,failure_train,success_heldout,success_train}"
    # All 628 failures; success_train capped -- only enough verified successes to
    # fill the 1000-scene held-out set are needed, the other 21k are pool.
    echo "DFT_VERIFY_CAP=${DFT_VERIFY_CAP:--1,-1,-1,2500}"
    echo "DFT_VERIFY_REPEATS=${DFT_VERIFY_REPEATS:-4}"
    echo "DFT_VERIFY_SEED=${DFT_VERIFY_SEED:-777}"
    echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-32}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    # No DDIM striding here on purpose: verify is what DEFINES the split labels,
    # so it has to run the same full chain the curve is later scored on.
    echo "DFT_HELDOUT_FAILURE=${DFT_HELDOUT_FAILURE:--1}"
    echo "DFT_HELDOUT_SUCCESS=${DFT_HELDOUT_SUCCESS:-1000}"
    echo "DFT_SPLIT_SEED=${DFT_SPLIT_SEED:-0}"
  } > "$ENV_V"
  echo "Wrote env file: $ENV_V"; cat "$ENV_V"; echo

  VID=$(sb --parsable "${SB_FLAGS[@]}" \
    rl/diffusion_ft/slurm/verify_splits.sbatch "$ENV_V")
  echo "verify  -> job $VID  (GPU a6000/a5000, ~1 h)"
  RID=$(sb --parsable --dependency="afterok:${VID}" \
    rl/diffusion_ft/slurm/rebuild_splits.sbatch "$ENV_V")
  echo "rebuild -> job $RID  (CPU, afterok:$VID)"
  DEP=(--dependency="afterok:${RID}")
else
  [[ -s "$DFT_VERIFIED_JSON" ]] || {
    echo "DFT_SKIP_VERIFY=1 but $DFT_VERIFIED_JSON does not exist" >&2; exit 1; }
  echo "skipping verify/rebuild; using $DFT_VERIFIED_JSON"
fi

# ---- stage 3: the training arm ----------------------------------------------
ENV_T="$ENV_DIR/verifiedsafety_${STAMP}.env"
{
  echo "DFT_OUT_DIR=$OUT_DIR"
  echo "DFT_CKPT=$CKPT"
  echo "DFT_SHARD_GLOB=$SHARD_GLOB"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-32}"
  echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-16}"
  echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
  echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
  # the rebuilt splits, not the scan's
  echo "DFT_SPLITS_JSON=$DFT_VERIFIED_JSON"
  echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
  echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-1000}"
  echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-64}"
  # ALL the data, balanced per batch -- not a capped sample of it.
  echo "DFT_TRAIN_SPLITS=${DFT_TRAIN_SPLITS:-failure_train,success_train}"
  echo "DFT_TRAIN_SPLIT_CAP=${DFT_TRAIN_SPLIT_CAP:--1}"
  echo "DFT_TRAIN_SPLIT_RATIO=${DFT_TRAIN_SPLIT_RATIO:-1,1}"
  echo "DFT_PROBE_PER_SPLIT=${DFT_PROBE_PER_SPLIT:-6}"
  echo "DFT_PROBE_SPLITS=${DFT_PROBE_SPLITS:-failure_train,success_train,failure_heldout,success_heldout}"
  echo "DFT_PROBE_EVERY=${DFT_PROBE_EVERY:-25}"
  echo "DFT_PROBE_SEED=${DFT_PROBE_SEED:-0}"
  # the objective that won: safety only, no denoising anchor
  echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
  echo "DFT_ANCHOR_WEIGHT=${DFT_ANCHOR_WEIGHT:-0.0}"
  echo "DFT_BC_WEIGHT=${DFT_BC_WEIGHT:-0.0}"
  echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
  echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
  echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
  echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
  # DDIM striding of the frozen prefix: cuts BOTH the collection rollout and the
  # gradient pass from 100 denoiser evals to S. Eval stays on the full chain so
  # the curve still measures the deployable policy. 0/0 = unchanged.
  echo "DFT_COLLECT_STEPS=${DFT_COLLECT_STEPS:-0}"
  echo "DFT_EVAL_STEPS=${DFT_EVAL_STEPS:-0}"
  echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-0}"
  echo "DFT_ANCHOR_STATES=${DFT_ANCHOR_STATES:-policy}"
  echo "DFT_ITERS=${DFT_ITERS:-300}"
  echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
  echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
  echo "DFT_REPLAN=${DFT_REPLAN:-10}"
  echo "DFT_LR=${DFT_LR:-1e-5}"
  # 25, not 10: a 1000-scene eval point costs ~25 min (see COST above).
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
  echo "DFT_WANDB_GROUP=${DFT_WANDB_GROUP:-verified_splits_${STAMP}}"
  echo "DFT_WANDB_NAME=${DFT_WANDB_NAME:-noanchor_cap1000}"
} > "$ENV_T"
echo "Wrote env file: $ENV_T"; cat "$ENV_T"; echo

TID=$(sb --parsable "${DEP[@]}" "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_T")
echo "train   -> job $TID  ($OUT_DIR)"

VZ=$(sb --parsable --dependency="afterok:${TID}" \
  rl/diffusion_ft/slurm/viz_safety_train.sbatch "$ENV_T")
echo "figures -> job $VZ  (CPU, afterok:${TID})"
echo
echo "watch:    tail -f $ROOT/logs/dft/slurm-dft_safetrain-${TID}.out"
echo "splits:   $DFT_VERIFIED_JSON"
echo "curve:    python3 -c \"import json;[print(r['iter'], r.get('failure_heldout/solved_rate'), r.get('success_heldout/solved_rate')) for r in map(json.loads, open('$OUT_DIR/train_log.jsonl')) if 'failure_heldout/solved_rate' in r]\""
