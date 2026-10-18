#!/bin/bash
# Distil the ship bank into the weights: behaviour-clone b_l24_r20 on the fixed
# clock as a fine-tune of the ORIGINAL policy. Only calls sbatch.
#
# WHY. On the corrected clock the ship bank (l24, ramp 20, cubic, cons w20) scores
# a corpus marginal3 of 0.9921 (base 0.9715, m3 +0.0206 [+0.0184,+0.0233];
# retrain_worlddt/20260924_235925_2645687/eval_old). The best WEIGHT-space arm,
# the safety fine-tune retrained on the same clock, is m3 -0.0023 (eval_new). The
# bank is a deploy-time filter that reads the other cars' recorded future; the
# question is how much of it survives being learned. Using the oracle version is
# fine here: it only LABELS the training data, the student never sees the future.
#
# CHAIN
#   1. collect (GPU array, select_eval.sbatch --dump_bc): the EXACT eval_old arm
#      -- base proposes, b_l24_r20 = bank@l24r20:cons:20 (neighbours=log), median
#      clock, rails -- driven on the TRAINING pools, never the held-out ones:
#        failure_train  all 173 scenes x 4 repeats
#        success_train  every 22nd scene (shards 0,1 of 22, ~2024 scenes) x 1 --
#                       the file is sorted by scene id, so a stride spreads the
#                       sample over the whole corpus where a cap would take one
#                       contiguous slice of it.
#      Per alive replan: (condition, the plan the arm EXECUTED) -- the base's own
#      sample where the bank kept it, the edit where it switched -- plus the
#      episode outcome. Sampler seed 2718 (not 777/12345/4242).
#   2. train (GPU array, train_safety.sbatch via sweep_safety_stopping.sh, 3
#      seeds, afterok:collect): the 09-11 recipe's shape and eval, with the
#      safety term and the base anchor OFF. The only objective is the late-step
#      denoising loss (t < k_trainable, the steps the trainable copy drives) on
#      the rows of SOLVED episodes (filtered BC), 1:1 failure_train:success_train
#      per batch of 32, 20 Adam steps per iteration, 151 iterations = 3000 steps,
#      lr 1e-5. The success rows are the base's own samples: they are the tether,
#      so there is no separate anchor. Eval every 10 on failure/success heldout
#      (seed 12345) for the curve; checkpoints every 25.
#   3. eval (GPU array, afterok:train): base | b_l24_r20 | BC ft @150 x 3 seeds,
#      seed 4242, 2 shards, chunk 32 -- the same draws as eval_old and eval_new,
#      so all three runs' `base` arms are identical episodes.
#   4. compare (CPU, afterok:eval): paired per (scene, repeat)
#        compare_vs_bank.json       BC ft minus the bank it imitates (same run)
#        compare_vs_safety_ft.json  BC ft minus the retrained safety ft (eval_new)
#
# THE SCORED POINT IS PRE-REGISTERED: iter 150 (3000 Adam steps). If the curve
# peaks elsewhere, re-score that checkpoint in a separate select run; do not pick
# it off the held-out curve and quote the same held-out number.
#
# COST. collect ~3 tasks x ~15 min. train ~2.5 h/seed (16 eval points dominate;
# the 20 BC steps are a few seconds per iteration). eval 5 arm-units x 1450 x 4
# over 6 tasks. Disk: bc_data ~0.7 GB (cond1 is 32 KB/row), checkpoints 7 x 121
# MB x 3 + latest/opt_state ~0.6 GB x 3 = ~4.5 GB.
#
# RE-TRAINING ON EXISTING ROWS (2026-09-29). The first run (bank_bc/20260928_121223_
# 39589) recovered ~6% of the bank's gain; its probe found the edit rows under-fit
# (loss 0.68x base, still falling) at ~7% of a batch, and the frozen prefix NOT the
# limit. DFT_BCD_DATA_FROM=<run dir> skips collection and trains on that run's rows;
# DFT_BCD_SWITCH_SHARE puts a fixed share of every batch on the edited rows. The
# eval then runs `base` only and the bank is read from the source run's eval (same
# draws); a third compare pairs the new BC arm with the source run's.
#   5. probe (GPU, afterok:train): tools/probe_bc_transfer.py on the new checkpoints.
#
# Usage (from repo root, on rhea or lov4):
#   DFT_BCD_SMOKE=1 bash rl/diffusion_ft/slurm/submit_bank_bc.sh   # ~8 scenes, 3 iters
#   bash rl/diffusion_ft/slurm/submit_bank_bc.sh
#   DFT_BCD_DATA_FROM=$ZFS/.../bank_bc/20260928_121223_39589 DFT_BCD_SWITCH_SHARE=0.5 \
#     DFT_BCD_TAG=bcsw50 bash rl/diffusion_ft/slurm/submit_bank_bc.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_BCD_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bank_bc/${STAMP}}"
REF=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/retrain_worlddt/20260924_235925_2645687
SPLITS3=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
ARM="b_l24_r20"
L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
SMOKE="${DFT_BCD_SMOKE:-}"
DATA_FROM="${DFT_BCD_DATA_FROM:-}"
SHARE="${DFT_BCD_SWITCH_SHARE:-0.0}"
TAG="${DFT_BCD_TAG:-bcd}"
[[ -s "$SPLITS3" ]] || { echo "FATAL: no splits file $SPLITS3" >&2; exit 1; }
mkdir -p "$OUT/collect/env" "$OUT/collect/tasks" "$ROOT/logs/dft"

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

# ---- 1. collect (GPU) --------------------------------------------------------- #
# split | repeats | shard | num_shards | cap
if [[ -n "$SMOKE" ]]; then
  TASKS=("failure_train|1|0|1|8" "success_train|1|0|2000|-1")
  CHUNK=8; ITERS=3; EVAL_EVERY=2; KEEP_EVERY=2; STEPS=2; SPLIT_CAP=8; ITER=000002
  echo "SMOKE: 8 failure + ~12 success scenes, 1 repeat; 3 iters x 2 steps; eval cap 8"
else
  TASKS=("failure_train|4|0|1|-1" "success_train|1|0|22|-1" "success_train|1|1|22|-1")
  CHUNK=32; ITERS=151; EVAL_EVERY=10; KEEP_EVERY=25; STEPS=20; SPLIT_CAP=1000; ITER=000150
fi
if [[ -n "$DATA_FROM" ]]; then
  DATA="$DATA_FROM/collect/tasks/*/bc_data/$ARM/*.npz"
  compgen -G "$DATA" > /dev/null || { echo "FATAL: no rows match $DATA" >&2; exit 1; }
  [[ -s "$DATA_FROM/eval/select_summary.json" ]] || {
    echo "FATAL: $DATA_FROM/eval has no select_summary.json (its bank arm is the reference)" >&2; exit 1; }
  TASKS=(); CID=""
  echo "collect -> skipped: reusing $(compgen -G "$DATA" | wc -l) files from $DATA_FROM"
fi
i=0
for t in ${TASKS[@]+"${TASKS[@]}"}; do
  IFS='|' read -r split reps shard nsh cap <<< "$t"
  d="$OUT/collect/tasks/${split}_shard${shard}"; mkdir -p "$d"
  {
    kv DFT_OUT_DIR "$d"
    kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_SELECT_SPLITS "$split"
    kv DFT_FT_CKPT ""
    kv DFT_ARMS "${ARM}=bank@l24r20:cons:20"
    kv DFT_SELECT_SEED 2718
    kv DFT_SELECT_REPEATS "$reps"
    kv DFT_SHARD_INDEX "$shard"
    kv DFT_NUM_SHARDS "$nsh"
    kv DFT_EVAL_CHUNK "$CHUNK"
    kv DFT_SELECT_CAP "$cap"
    kv DFT_SELECT_BANKS "l24r20=${L24}"
    kv DFT_SELECT_BANK_RAMPS "l24r20=20"
    kv DFT_SELECT_BANK_BLENDS "l24r20=cubic"
    kv DFT_SELECT_BANK_NEIGHBORS "l24r20=log"
    kv DFT_SELECT_TOL_M 1.0; kv DFT_SELECT_MARGIN_M 0.3; kv DFT_SELECT_DELTA_M 0.2
    kv DFT_DUMP_BC 1
    # rollout shape = eval_old exactly
    kv DFT_K_TRAINABLE 10; kv DFT_PREFIX_LEN 5; kv DFT_SIGMA_FLOOR 1e-4
    kv DFT_REPLAN 10; kv DFT_MAX_REPLANS 8
    kv DFT_SIM_AGENTS none; kv DFT_EGO_VEL_SOURCE commanded; kv DFT_WORLD_DT_METHOD median
  } > "$OUT/collect/env/$i.env"
  # Same invariant as submit_select_eval.sh: every key round-trips through `source`.
  while IFS= read -r k; do
    want="$(sed -n "s/^${k}='\(.*\)'$/\1/p" "$OUT/collect/env/$i.env")"
    got="$(env -i bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2}"' _ "$OUT/collect/env/$i.env" "$k")"
    [[ "$want" == "$got" ]] || { echo "FATAL: $k does not round-trip in $i.env" >&2; exit 1; }
  done < <(sed -n 's/^\(DFT_[A-Z_]*\)=.*/\1/p' "$OUT/collect/env/$i.env")
  printf '%2d  %-14s shard %s/%s  repeats %s  cap %s\n' "$i" "$split" "$shard" "$nsh" "$reps" "$cap"
  i=$((i + 1))
done
if [[ -z "$DATA_FROM" ]]; then
  CID=$(sb --parsable --exclude=gpu28 --array="0-$((i - 1))%3" --job-name=dft_bcd_collect \
    rl/diffusion_ft/slurm/select_eval.sbatch "$OUT/collect/env")
  echo "collect -> array $CID ($i tasks, GPU) -> $OUT/collect/tasks/*/bc_data/$ARM/"
  DATA="$OUT/collect/tasks/*/bc_data/$ARM/*.npz"
fi
echo

# ---- 2. train (GPU) ----------------------------------------------------------- #
XENV="$OUT/bc.env"
{
  echo "# BC of $ARM (submit_bank_bc.sh): the repair term is the whole objective"
  kv DFT_REPAIR_DATA "$DATA"
  kv DFT_REPAIR_WEIGHT 1.0
  kv DFT_REPAIR_BATCH 32
  kv DFT_REPAIR_STEPS "$STEPS"
  kv DFT_REPAIR_ROWS solved
  kv DFT_REPAIR_SPLIT_RATIO "failure_train=1,success_train=1"
  kv DFT_REPAIR_SWITCH_SHARE "$SHARE"
} > "$XENV"
DFT_SWEEP_OUT="$OUT/train" DFT_EXTRA_ENV_FILE="$XENV" \
DFT_ARM_SPECS="${TAG}_s0|DFT_SAFETY_WINDOW=0,DFT_SEED=0,DFT_DATA_SEED=0 ${TAG}_s1|DFT_SAFETY_WINDOW=0,DFT_SEED=1,DFT_DATA_SEED=1 ${TAG}_s2|DFT_SAFETY_WINDOW=0,DFT_SEED=2,DFT_DATA_SEED=2" \
DFT_ARMS="$([[ -n "$SMOKE" ]] && echo "${TAG}_s0" || echo "${TAG}_s0 ${TAG}_s1 ${TAG}_s2")" \
DFT_DEPENDENCY="${CID:+afterok:$CID}" DFT_SPLITS_JSON="$SPLITS3" DFT_SPLIT_CAP="$SPLIT_CAP" \
DFT_SAFETY_WEIGHT=0.0 DFT_BASE_ANCHOR_WEIGHT=0.0 DFT_WORLD_DT_METHOD=median \
DFT_ITERS="$ITERS" DFT_EVAL_EVERY="$EVAL_EVERY" DFT_KEEP_EVERY="$KEEP_EVERY" \
DFT_WANDB_GROUP="bank_bc_${STAMP}" \
  bash rl/diffusion_ft/slurm/sweep_safety_stopping.sh | sed -n '/^ *[0-9]/p;/^submitted/p'
TID="$(cat "$OUT/train/jobid.txt")"
grep -q "^DFT_REPAIR_STEPS='$STEPS'$" "$OUT/train/env/0.env" \
  && grep -q "^DFT_SAFETY_WEIGHT=0.0$" "$OUT/train/env/0.env" || {
  echo "FATAL: train env files do not carry the BC config -- scancel $TID $CID" >&2; exit 1; }
echo "train   -> array $TID (GPU${CID:+, afterok:$CID}) -> $OUT/train/${TAG}_s*/  (switch share $SHARE)"
echo

# ---- 3. eval: the eval_old / eval_new draws (GPU) ------------------------------ #
CKPTS=""
for s in $([[ -n "$SMOKE" ]] && echo 0 || echo 0 1 2); do CKPTS+="$OUT/train/${TAG}_s$s/iter_$ITER "; done
# With reused rows the bank is already scored on these draws in the source run.
BASE_ARMS="base,${ARM}=bank@l24r20:cons:20"; BANK_RUN="$OUT/eval"
[[ -n "$DATA_FROM" ]] && { BASE_ARMS="base"; BANK_RUN="$DATA_FROM/eval"; }
DFT_SELECT_DIR="$OUT/eval" DFT_SELECT_DEP="afterok:$TID" DFT_SELECT_CKPTS="${CKPTS% }" \
DFT_SPLITS3_JSON="$SPLITS3" DFT_BASE_ARMS="$BASE_ARMS" DFT_FT_ARMS=ft \
DFT_SELECT_BANKS="l24r20=${L24}" DFT_SELECT_BANK_RAMPS="l24r20=20" \
DFT_SELECT_BANK_BLENDS="l24r20=cubic" DFT_SELECT_BANK_NEIGHBORS="l24r20=log" \
DFT_SELECT_SEED=4242 DFT_SELECT_REPEATS=4 DFT_NUM_SHARDS=2 DFT_EVAL_CHUNK=32 \
DFT_SIM_AGENTS=none DFT_EGO_VEL_SOURCE=commanded DFT_WORLD_DT_METHOD=median \
DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" DFT_SELECT_SMOKE="$SMOKE" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh
SID="$(cat "$OUT/eval/jobid.txt")"
echo

# ---- 4. compare (CPU) --------------------------------------------------------- #
REPS=4; [[ -n "$SMOKE" ]] && REPS=1
{ kv DFT_CMP_A "$BANK_RUN"; kv DFT_CMP_B "$OUT/eval"
  kv DFT_CMP_ARM_A "$ARM"; kv DFT_CMP_ARM_B ft
  kv DFT_CMP_LABEL_A bank; kv DFT_CMP_LABEL_B bc
  kv DFT_SPLITS_JSON "$SPLITS3"; kv DFT_SELECT_REPEATS "$REPS"
  kv DFT_CMP_OUT "$OUT/eval/compare_vs_bank.json"; } > "$OUT/compare_bank.env"
X1=$(sb --parsable --dependency="afterok:$SID" rl/diffusion_ft/slurm/compare_select.sbatch "$OUT/compare_bank.env")
echo "compare -> job $X1 (CPU, afterok:$SID) -> $OUT/eval/compare_vs_bank.json"
if [[ -z "$SMOKE" ]]; then
  { kv DFT_CMP_A "$REF/eval_new"; kv DFT_CMP_B "$OUT/eval"
    kv DFT_CMP_LABEL_A safety_ft; kv DFT_CMP_LABEL_B bc
    kv DFT_SPLITS_JSON "$SPLITS3"; kv DFT_SELECT_REPEATS 4
    kv DFT_CMP_OUT "$OUT/eval/compare_vs_safety_ft.json"; } > "$OUT/compare_ft.env"
  X2=$(sb --parsable --dependency="afterok:$SID" rl/diffusion_ft/slurm/compare_select.sbatch "$OUT/compare_ft.env")
  echo "compare -> job $X2 (CPU, afterok:$SID) -> $OUT/eval/compare_vs_safety_ft.json"
fi
if [[ -n "$DATA_FROM" && -z "$SMOKE" ]]; then
  { kv DFT_CMP_A "$DATA_FROM/eval"; kv DFT_CMP_B "$OUT/eval"
    kv DFT_CMP_LABEL_A bc_src; kv DFT_CMP_LABEL_B "$TAG"
    kv DFT_SPLITS_JSON "$SPLITS3"; kv DFT_SELECT_REPEATS 4
    kv DFT_CMP_OUT "$OUT/eval/compare_vs_source_bc.json"; } > "$OUT/compare_src.env"
  X3=$(sb --parsable --dependency="afterok:$SID" rl/diffusion_ft/slurm/compare_select.sbatch "$OUT/compare_src.env")
  echo "compare -> job $X3 (CPU, afterok:$SID) -> $OUT/eval/compare_vs_source_bc.json"
fi

# ---- 5. probe: did the edits fit, and do deploy samples move toward them (GPU) --- #
PCK="$OUT/train/${TAG}_s0/iter_$(printf '%06d' "$KEEP_EVERY")"
for s in $([[ -n "$SMOKE" ]] && echo 0 || echo 0 1 2); do PCK+=" $OUT/train/${TAG}_s$s/iter_$ITER"; done
{ kv DFT_BC_DATA "$DATA"; kv DFT_FT_CKPTS "$PCK"
  kv DFT_OUT "$OUT/probe/bc_transfer.json"
  kv DFT_N_ROWS "$([[ -n "$SMOKE" ]] && echo 32 || echo 512)"; } > "$OUT/probe.env"
mkdir -p "$OUT/probe"
PID=$(sb --parsable --dependency="afterok:$TID" rl/diffusion_ft/slurm/probe_bc_transfer.sbatch "$OUT/probe.env")
echo "probe   -> job $PID (GPU, afterok:$TID) -> $OUT/probe/bc_transfer.json"
echo
echo "all under $OUT"
echo "  collect/tasks/*/rows.jsonl        the bank on the TRAINING pools (sanity: ~0.77 failure)"
echo "  train/${TAG}_s*/train_log.jsonl   curves: repair_loss, marginal/delta_vs_base"
echo "  probe/bc_transfer.json            edit-row loss vs base; closed_free on edit vs kept rows"
echo "  eval/select_summary.json          base | bank | BC ft @$ITER, marginal3"
echo "  eval/compare_vs_*.json            paired deltas"
