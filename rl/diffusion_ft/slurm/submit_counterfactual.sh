#!/bin/bash
# Repair search -> correction training, and selector labels -> learned + look-ahead
# selection. Only calls sbatch.
#
# CHAIN (independent branches stay siblings so they queue concurrently)
#   0. build (CPU, only if missing): the three-stratum splits file.
#   A1. repair (GPU array, shards): failure_train base failures -> rewind, search
#       timing / lateral / re-noise, keep repairs whose whole continuation solves.
#   A2. train (GPU array, afterok A1): stopping-sweep protocol, 3 seeds each of
#         repair     safety term OFF, repair_weight W, base anchor on
#         repair_pres  + preserve_weight 3, safety_rows unsolved
#       (arms exit 0 with a message if the search produced no repair data).
#   B1. labels (GPU array, shards): one-switch labels vs the fine-tune at
#       stopping base_s0@110, on failure_train + a success_train cap.
#   B2. fit (CPU, afterok B1): selector.json (+ repair_summary.json if A1 is done).
#   B3. select (GPU array, afterok B2): base, bf_cons, bf_look, bf_learned on the
#       held-out strata, same seed/shards/chunk as submit_select_eval.sh, so its
#       base/bf_cons rows pair with an Experiment-1 run as well.
#   B4. analyze (CPU, afterok B3).
#
# COST (a6000, guesses from ~0.33 s/scene eval): A1 ~30-60 min; B1 ~1-1.5 h/shard;
# B3 base_s0 only ~2-3 h/shard (look-ahead cost scales with how often a candidate
# is within 0.3 m of a hazard); A2 ~5 h/arm, 6 arms at 3-wide.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_counterfactual.sh
#   DFT_CF_SMOKE=1 bash ...            # 8 scenes/split, 1 shard, ITERS=2 training
#   DFT_CF_PARTS="repair" bash ...     # or "labels", default "repair labels"
#   # grow the repair set: also search the success pool's own base failures
#   DFT_CF_PARTS=repair DFT_NUM_SHARDS=4 DFT_REPAIR_SPLITS=failure_train,success_train \
#     DFT_REPAIR_CAPS=-1,4000 bash ...
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CF_DIR="${DFT_CF_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/counterfactual/${STAMP}}"
VDIR=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777
SPLITS3="${DFT_SPLITS3_JSON:-$VDIR/eval_splits_3strata.json}"
RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
FT_CKPT="${DFT_FT_CKPT:-$RUN/base_s0/iter_000110}"
PARTS="${DFT_CF_PARTS:-repair labels}"
SHARDS="${DFT_NUM_SHARDS:-2}"
CHUNK="${DFT_EVAL_CHUNK:-32}"
SUCC_CAP="${DFT_LABEL_SUCCESS_CAP:-2000}"
REPAIR_WEIGHT="${DFT_REPAIR_WEIGHT:-100}"
# Which scenes the repair search runs on. `failure_train` is only 173 scenes, and
# the search keeps only what it can verifiably fix, so the cheap way to grow the
# target set is to point it at the SUCCESS pool too: the base fails ~2% of those,
# and those failures are just as repairable and just as much training data. Only
# training pools are legal here (the module refuses a held-out split).
REPAIR_SPLITS="${DFT_REPAIR_SPLITS:-failure_train}"
REPAIR_CAPS="${DFT_REPAIR_CAPS:--1}"
CANDS="${DFT_CANDS:-}"
REWINDS="${DFT_REWINDS:-3}"
ITERS="${DFT_ITERS:-150}"
SEL_CAP=-1; REPEATS="${DFT_SELECT_REPEATS:-4}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"
EXCLUDE="${DFT_EXCLUDE-gpu28}"
RISK="$ROOT/runs/hardness/risk.npz"
if [[ -n "${DFT_CF_SMOKE:-}" ]]; then
  SHARDS=1; CHUNK=8; SUCC_CAP=8; ITERS=2; SEL_CAP=4; REPEATS=1
  echo "SMOKE: 8 scenes/split, 1 shard, 2 training iters, 4 selection scenes/split"
fi
[[ -s "$FT_CKPT/tparams.npz" ]] || { echo "FATAL: missing $FT_CKPT/tparams.npz" >&2; exit 1; }
mkdir -p "$CF_DIR" "$ROOT/logs/dft"

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
SB_FLAGS=(); [[ -n "$EXCLUDE" ]] && SB_FLAGS+=(--exclude="$EXCLUDE")

DEP=()
if [[ ! -s "$SPLITS3" ]]; then
  BID=$(sb --parsable rl/diffusion_ft/slurm/build_intermittent.sbatch "DFT_SPLITS3_JSON=$SPLITS3")
  DEP=(--dependency="afterok:${BID}")
  echo "build    -> job $BID (CPU)"
fi

cf_env() {  # dir mode shard splits caps extra...
  local d="$1" mode="$2" s="$3" splits="$4" caps="$5"; shift 5
  mkdir -p "$d"
  {
    echo "DFT_OUT_DIR=$d"; echo "DFT_CF_MODE=$mode"; echo "DFT_SPLITS_JSON=$SPLITS3"
    echo "DFT_CF_SPLITS=$splits"; echo "DFT_CF_CAPS=$caps"
    echo "DFT_SHARD_INDEX=$s"; echo "DFT_NUM_SHARDS=$SHARDS"; echo "DFT_EVAL_CHUNK=$CHUNK"
    echo "DFT_K_TRAINABLE=10"; echo "DFT_PREFIX_LEN=5"; echo "DFT_SIGMA_FLOOR=1e-4"
    echo "DFT_REPLAN=10"; echo "DFT_MAX_REPLANS=8"
    for kv in "$@"; do echo "$kv"; done
  }
}

REPAIR_AID=""
if [[ " $PARTS " == *" repair "* ]]; then
  E="$CF_DIR/env_repair"; mkdir -p "$E"
  rsplits="$REPAIR_SPLITS"; rcaps="$REPAIR_CAPS"
  if [[ -n "${DFT_CF_SMOKE:-}" ]]; then rsplits=failure_train; rcaps=8; fi
  for ((s = 0; s < SHARDS; s++)); do
    extra=(); [[ -n "$CANDS" ]] && extra+=("DFT_CANDS=$CANDS")
    extra+=("DFT_REWINDS=$REWINDS")
    cf_env "$CF_DIR/repair/shard$s" repair "$s" "$rsplits" "$rcaps" \
      ${extra[@]+"${extra[@]}"} > "$E/$s.env"
  done
  echo "repair pool -> $rsplits (caps $rcaps), rewinds $REWINDS"
  REPAIR_AID=$(sb --parsable ${DEP[@]+"${DEP[@]}"} ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} \
    --array="0-$((SHARDS - 1))%${MAX_CONCURRENT}" --job-name=dft_cf_repair \
    rl/diffusion_ft/slurm/counterfactual.sbatch "$E")
  echo "repair   -> array $REPAIR_AID"

  # A2: repair-trained fine-tunes, stopping-sweep protocol.
  T="$CF_DIR/train"; TE="$T/env"; mkdir -p "$TE"; i=0
  for arm in repair repair_pres; do
    for s in 0 1 2; do
      [[ -n "${DFT_CF_SMOKE:-}" && "$s" != 0 ]] && continue
      tag="${arm}_s$s"; mkdir -p "$T/$tag"
      {
        echo "DFT_OUT_DIR=$T/$tag"
        echo "DFT_SCENES_PER_ITER=32"; echo "DFT_EVAL_SCENES=16"; echo "DFT_PREFETCH=1"
        echo "DFT_SPLITS_JSON=$SPLITS3"
        echo "DFT_SPLITS=failure_heldout,success_heldout,intermittent_heldout"
        echo "DFT_SPLIT_CAP=$([[ -n "${DFT_CF_SMOKE:-}" ]] && echo 8 || echo 1000)"
        echo "DFT_EVAL_CHUNK=32"
        echo "DFT_TRAIN_SPLITS=failure_train,success_train"; echo "DFT_TRAIN_SPLIT_CAP=-1"
        echo "DFT_TRAIN_SPLIT_RATIO=1,1"; echo "DFT_PROBE_PER_SPLIT=0"; echo "DFT_PROBE_EVERY=0"
        echo "DFT_ANCHOR_WEIGHT=0.0"; echo "DFT_BC_WEIGHT=0.0"
        echo "DFT_BASE_ANCHOR_WEIGHT=100"; echo "DFT_BASE_ANCHOR_SUCCESS_ONLY=1"
        echo "DFT_K_TRAINABLE=10"; echo "DFT_PREFIX_LEN=5"; echo "DFT_SIGMA_FLOOR=1e-4"
        echo "DFT_ITERS=$ITERS"; echo "DFT_EPOCHS=1"; echo "DFT_MAX_REPLANS=8"; echo "DFT_REPLAN=10"
        echo "DFT_LR=1e-5"; echo "DFT_GRAD_ACCUM=0"
        echo "DFT_EVAL_EVERY=$([[ -n "${DFT_CF_SMOKE:-}" ]] && echo 1 || echo 10)"
        echo "DFT_SAVE_EVERY=25"; echo "DFT_KEEP_EVERY=10"
        echo "DFT_EVAL_SEED=12345"; echo "DFT_EVAL_REPEATS=1"; echo "DFT_EVAL_BASE=1"; echo "DFT_RESUME=1"
        echo "DFT_REPAIR_DATA=$CF_DIR/repair/shard*/repair_data/*.npz"
        echo "DFT_REPAIR_WEIGHT=$REPAIR_WEIGHT"
        echo "DFT_SEED=$s"; echo "DFT_DATA_SEED=$s"
        echo "DFT_WANDB=${DFT_WANDB:-1}"; echo "DFT_WANDB_GROUP=cf_${STAMP}"; echo "DFT_WANDB_NAME=$tag"
        if [[ "$arm" == repair ]]; then
          echo "DFT_SAFETY_WEIGHT=0.0"
        else
          echo "DFT_SAFETY_WEIGHT=1.0"; echo "DFT_SAFETY_ROWS=unsolved"
          echo "DFT_PRESERVE_WEIGHT=3"; echo "DFT_PRESERVE_RISK=$RISK"
        fi
      } > "$TE/$i.env"
      printf '   train %2d %s\n' "$i" "$tag"
      i=$((i + 1))
    done
  done
  TID=$(sb --parsable --dependency="afterok:${REPAIR_AID}" ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} \
    --array="0-$((i - 1))%${MAX_CONCURRENT}" --job-name=dft_cf_train \
    rl/diffusion_ft/slurm/train_safety.sbatch "$TE")
  echo "train    -> array $TID ($i arms, afterok:$REPAIR_AID)"
fi

if [[ " $PARTS " == *" labels "* ]]; then
  E="$CF_DIR/env_labels"; mkdir -p "$E"
  fcap=-1; [[ -n "${DFT_CF_SMOKE:-}" ]] && fcap=8
  for ((s = 0; s < SHARDS; s++)); do
    cf_env "$CF_DIR/labels/shard$s" labels "$s" failure_train,success_train "$fcap,$SUCC_CAP" \
      "DFT_FT_CKPT=$FT_CKPT" "DFT_LABEL_ALT=ft" > "$E/$s.env"
  done
  LID=$(sb --parsable ${DEP[@]+"${DEP[@]}"} ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} \
    --array="0-$((SHARDS - 1))%${MAX_CONCURRENT}" --job-name=dft_cf_labels \
    rl/diffusion_ft/slurm/counterfactual.sbatch "$E")
  echo "labels   -> array $LID"
  FENV="$CF_DIR/fit.env"
  { echo "DFT_CF_DIR=$CF_DIR"; echo "DFT_SPLITS_JSON=$SPLITS3"; } > "$FENV"
  FID=$(sb --parsable --dependency="afterok:${LID}" rl/diffusion_ft/slurm/fit_selector.sbatch "$FENV")
  echo "fit      -> job $FID (CPU, afterok:$LID)"

  SEL="$CF_DIR/select"; SE="$SEL/env"; mkdir -p "$SE" "$SEL/tasks"
  tag="$(basename "$(dirname "$FT_CKPT")")_$(basename "$FT_CKPT")"
  SSH=$SHARDS; [[ -n "${DFT_CF_SMOKE:-}" ]] || SSH="${DFT_SELECT_SHARDS:-2}"
  for ((s = 0; s < SSH; s++)); do
    d="$SEL/tasks/${tag}_shard$s"; mkdir -p "$d"
    {
      echo "DFT_OUT_DIR=$d"; echo "DFT_SPLITS_JSON=$SPLITS3"
      echo "DFT_SELECT_SPLITS=failure_heldout,success_heldout,intermittent_heldout"
      echo "DFT_FT_CKPT=$FT_CKPT"
      echo "DFT_ARMS=base,bf_cons=ft:cons:20,bf_look=ft:look:20,bf_learned=ft:learned:20"
      echo "DFT_SELECTOR_JSON=$CF_DIR/selector.json"
      echo "DFT_LOOK_REPLANS=${DFT_LOOK_REPLANS:-2}"
      echo "DFT_SELECT_SEED=4242"; echo "DFT_SELECT_REPEATS=$REPEATS"
      echo "DFT_SHARD_INDEX=$s"; echo "DFT_NUM_SHARDS=$SSH"
      echo "DFT_EVAL_CHUNK=$([[ -n "${DFT_CF_SMOKE:-}" ]] && echo 4 || echo 32)"
      echo "DFT_SELECT_CAP=$SEL_CAP"
      echo "DFT_K_TRAINABLE=10"; echo "DFT_PREFIX_LEN=5"; echo "DFT_SIGMA_FLOOR=1e-4"
      echo "DFT_REPLAN=10"; echo "DFT_MAX_REPLANS=8"
    } > "$SE/$s.env"
  done
  SID=$(sb --parsable --dependency="afterok:${FID}" ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} \
    --array="0-$((SSH - 1))%${MAX_CONCURRENT}" --job-name=dft_cf_select \
    rl/diffusion_ft/slurm/select_eval.sbatch "$SE")
  AENV="$SEL/analyze.env"
  { echo "DFT_SELECT_DIR=$SEL"; echo "DFT_SPLITS_JSON=$SPLITS3"; echo "DFT_SELECT_REPEATS=$REPEATS"; } > "$AENV"
  AID=$(sb --parsable --dependency="afterok:${SID}" rl/diffusion_ft/slurm/select_analysis.sbatch "$AENV")
  echo "select   -> array $SID (afterok:$FID); analyze -> job $AID"
fi
echo "out      -> $CF_DIR"
