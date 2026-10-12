#!/bin/bash
# Diffusion-DPO baseline on the bank's (condition, rejected base draw, accepted
# edit) pairs, as a fine-tune of the ORIGINAL policy. Only calls sbatch.
#
# WHY. Repaired-trajectory BC of the same bank (bank_bc 09-28/30) either moved
# nothing (plain, m3 +0.0012) or fit the edits by shifting the whole conditional
# and broke successes (half-edit batches, m3 -0.020). The working explanation
# (docs/bank_bc_distillation.md section 5): the bank edits THIS draw when it is
# bad, and a denoising loss on the targets alone never shows the student its own
# bad draw. A preference pair does -- "the edit over this draw", relative to the
# frozen base -- which is idea 2 of that doc. No RL, critic or reward model.
#
# CHAIN
#   1. collect (GPU array, select_eval.sbatch --dump_bc): the bank-BC collection
#      EXACTLY (arm b_l24_r20 = bank@l24r20:cons:20, oracle neighbours, median
#      clock, rails, sampler seed 2718; failure_train 173 x 4, success_train
#      stride 22) -- so the pairs should be BC's 611 edit rows -- now also storing
#      `rejected`, the base draw each edit replaced (proposal_select._bc_rows).
#   2. pair check (CPU, afterok:collect, informational): tools/pref_pairs_check.py
#      -- pair stats, and whether the rows are bank-BC run 1's row for row.
#   3. train (GPU array via sweep_safety_stopping.sh, afterok:collect): safety
#      term and rollout anchor OFF; the objective is
#          L = L_DPO(pairs) + 100 * base_anchor_KL(success_train kept rows)
#      both on every Adam step, 20 steps/iter x 151 iters = 3020 steps, lr 1e-5,
#      16 pairs + 16 anchor rows per step (~79 passes over the pairs, = bcsw50).
#      k_trainable 10 as in BC. Arms (3 seeds each; vanilla first in the array):
#        b1e4   beta 1e4, t ~ U[0, 10)  (the steps the trainable copy drives)
#        b1e3   beta 1e3, same          (10x heavier anchor, relatively)
#        lowt   beta 1e4, t ~ U[0, 3)   (low-noise arm A)
#        edit   beta 1e4, steps where |good - bad| > 0.1 m weigh 5 (arm B)
#      WHY NOT beta 1 / 0.1 (the plan's values, still available as b1 / b01): the
#      smoke (job 70446) measured z ~ 1e-4 and, after one step, a DPO gradient
#      of ~0.1 against an anchor gradient of 70-390 (x100 weight). Under the 1.0
#      global-norm clip the step is all anchor, so those arms would re-measure the
#      base. In the unsaturated regime the DPO gradient scales with beta, and the
#      safety recipe this anchor weight comes from ran anchor:task ~20:1 (grad
#      norm 1526 vs 78); beta 1e4 lands near that, 1e3 is 10x more anchored. Both
#      bracket Diffusion-DPO's own 2500-5000.
#      Eval every 10 on failure/success held-out (seed 12345, 1 repeat) for the
#      curve; checkpoints every 25.
#   4. eval (GPU array per arm, afterok on THAT arm's three tasks): base | ft @
#      iters 50 and 150 x 3 seeds, seed 4242, 4 repeats, 2 shards, chunk 32 --
#      the bank-BC eval draws, so base | bank | bcd | bcsw50 | DPO are all paired.
#   5. report (CPU per arm, afterok:eval): compare_select vs base / bank / bcd /
#      bcsw50 per scored iteration, then tools/pref_summary.py -> summary.md.
#   6. probe (GPU, afterok:train): probe_bc_transfer on every arm's s0 @150 next
#      to bcd_s0 / bcsw50_s0 @150 -- edit-row loss, closed_free, and d_rej_m (did
#      deploy samples move OFF the rejected draw).
#
# THE SCORED ITERATIONS ARE FIXED NOW: 50 and 150. 150 is BC's (3000 steps); 50
# because bcsw50's failure gain was in by iteration 10 and its success damage
# grew with steps. summary.md also reports the plan's checkpoint rule (most
# failure gain with success within 0.005) applied to the held-out training
# curve, labelled optimistic: it is selected on the scenes it is scored on.
#
# COST. collect 3 tasks x ~15 min. train ~2.5 h/task x 12 at 4-wide. eval per arm
# 7 arm-units (base + 6 ckpts) x 1450 x 4 over 12 tasks. Disk: 12 x (7 x 121 MB
# + ~0.6 GB latest) ~ 17 GB.
#
# JOB BUDGET. qos_general caps a user at MaxSubmitJobsPU=50 queued jobs, array
# tasks counted one by one (`scontrol show assoc_mgr flags=qos qos=qos_general`).
# The whole chain is 3 + 12 + 48 + 1 = 64 GPU jobs, so it goes in two STAGES, each
# refusing up front (DFT_PREF_JOB_LIMIT, default 48) rather than half-submitting:
#   DFT_PREF_STAGE=train   steps 1-3 + 6 (16 GPU jobs)
#   DFT_PREF_STAGE=eval    steps 4-5 for DFT_PREF_ARMS of an existing DFT_PREF_DIR
#                          (12 GPU jobs per arm). Chains on that arm's training
#                          tasks while they are queued; checks checkpoints after.
#   DFT_PREF_STAGE=all     everything at once (the default; fits only the smoke)
#
# Usage (from repo root; lov4 submits directly):
#   DFT_PREF_SMOKE=1 bash rl/diffusion_ft/slurm/submit_pref_dpo.sh   # b1e4 s0, 8 scenes, 3 iters
#   DFT_PREF_STAGE=train bash rl/diffusion_ft/slurm/submit_pref_dpo.sh
#   DFT_PREF_STAGE=eval DFT_PREF_DIR=$ZFS/.../pref_dpo/<run> DFT_PREF_ARMS="b1e4" bash ...
#   DFT_PREF_DATA_FROM=$ZFS/.../pref_dpo/<run> DFT_PREF_ARMS="b1" bash ...   # reuse pairs
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_PREF_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/pref_dpo/${STAMP}}"
BCD=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bank_bc/20260928_121223_39589
BCSW=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bank_bc/20260930_000014_1275746
SPLITS3=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
ARM="b_l24_r20"
L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
SMOKE="${DFT_PREF_SMOKE:-}"
DATA_FROM="${DFT_PREF_DATA_FROM:-}"
WANT_ARMS="${DFT_PREF_ARMS:-b1e4 b1e3 lowt edit}"
SEEDS="${DFT_PREF_SEEDS:-0 1 2}"
SCORE_ITERS="${DFT_PREF_SCORE_ITERS:-50 150}"
STAGE="${DFT_PREF_STAGE:-all}"
LIMIT="${DFT_PREF_JOB_LIMIT:-48}"
case "$STAGE" in all|train|eval) ;; *) echo "FATAL: DFT_PREF_STAGE=$STAGE (all|train|eval)" >&2; exit 1;; esac
if [[ "$STAGE" == eval ]]; then
  [[ -n "${DFT_PREF_DIR:-}" && -s "$OUT/train/jobid.txt" && -s "$OUT/train/arms.txt" ]] || {
    echo "FATAL: the eval stage needs DFT_PREF_DIR = a run whose train stage was submitted" >&2; exit 1; }
fi
[[ -s "$SPLITS3" ]] || { echo "FATAL: no splits file $SPLITS3" >&2; exit 1; }
for d in "$BCD/eval" "$BCSW/eval"; do
  [[ -s "$d/select_summary.json" ]] || { echo "FATAL: reference eval $d missing" >&2; exit 1; }
done
mkdir -p "$OUT/collect/env" "$OUT/collect/tasks" "$ROOT/logs/dft"

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
roundtrip() {   # every DFT_ key of an env file comes back verbatim through `source`
  while IFS= read -r k; do
    want="$(sed -n "s/^${k}='\(.*\)'$/\1/p" "$1")"
    got="$(env -i bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2}"' _ "$1" "$k")"
    [[ "$want" == "$got" ]] || { echo "FATAL: $k does not round-trip in $1" >&2; exit 1; }
  done < <(sed -n "s/^\(DFT_[A-Z0-9_]*\)='.*/\1/p" "$1")
}
budget() {      # n_jobs label: refuse BEFORE anything is submitted
  local used; used=$(squeue -u "$USER" -h -r -o "%q" | grep -c "^qos_general$" || true)
  if (( used + $1 > LIMIT )); then
    echo "REFUSING: $2 needs $1 qos_general jobs; $used already queued; limit $LIMIT." >&2
    echo "  submit fewer arms (DFT_PREF_ARMS) or wait for queued jobs to finish" >&2; exit 1
  fi
  echo "budget  -> $2: $1 qos_general jobs + $used queued <= $LIMIT"
}

# arm tag -> overrides (no commas inside values: sweep_safety_stopping splits on them)
declare -A SPEC=(
  [b1e4]="DFT_PREF_BETA=10000"
  [b1e3]="DFT_PREF_BETA=1000"
  [lowt]="DFT_PREF_BETA=10000,DFT_PREF_T_MAX=3"
  [edit]="DFT_PREF_BETA=10000,DFT_PREF_EDIT_WEIGHT=4.0,DFT_PREF_EDIT_DELTA_M=0.1"
  [b1]="DFT_PREF_BETA=1.0"
  [b01]="DFT_PREF_BETA=0.1"
)
for a in $WANT_ARMS; do [[ -n "${SPEC[$a]:-}" ]] || { echo "FATAL: unknown arm '$a'" >&2; exit 1; }; done

# ---- 1. collect (GPU) --------------------------------------------------------- #
# split | repeats | shard | num_shards | cap
if [[ -n "$SMOKE" ]]; then
  TASKS=("failure_train|1|0|1|8" "success_train|1|0|2000|-1")
  CHUNK=8; ITERS=3; EVAL_EVERY=2; KEEP_EVERY=2; STEPS=2; SPLIT_CAP=8
  WANT_ARMS="b1e4"; SEEDS="0"; SCORE_ITERS="2"; NROWS=32
  echo "SMOKE: 8 failure + ~12 success scenes, 1 repeat; arm b1e4 s0, 3 iters x 2 steps; eval cap 8"
else
  TASKS=("failure_train|4|0|1|-1" "success_train|1|0|22|-1" "success_train|1|1|22|-1")
  CHUNK=32; ITERS=151; EVAL_EVERY=10; KEEP_EVERY=25; STEPS=20; SPLIT_CAP=1000; NROWS=512
fi
for it in $SCORE_ITERS; do
  (( it % KEEP_EVERY == 0 && it < ITERS )) || {
    echo "FATAL: scored iteration $it has no iter_* snapshot (keep every $KEEP_EVERY, $ITERS iters)" >&2; exit 1; }
done
NA=$(wc -w <<< "$WANT_ARMS"); NS=$(wc -w <<< "$SEEDS"); NI=$(wc -w <<< "$SCORE_ITERS")
NSH=2; [[ -n "$SMOKE" ]] && NSH=1 && NI=1 && NS=1
NC=${#TASKS[@]}; [[ -n "$DATA_FROM" ]] && NC=0
N_TRAIN=$(( NC + NA * NS + 1 ))   # collect + training tasks + probe
N_EVAL=$(( NA * NI * NS * NSH ))
case "$STAGE" in
  train) budget "$N_TRAIN" "train stage" ;;
  eval)  budget "$N_EVAL" "eval stage ($WANT_ARMS)" ;;
  all)   budget "$((N_TRAIN + N_EVAL))" "whole chain" ;;
esac
echo
if [[ "$STAGE" != eval ]]; then
CID=""
if [[ -n "$DATA_FROM" ]]; then
  DATA="$DATA_FROM/collect/tasks/*/bc_data/$ARM/*.npz"
  compgen -G "$DATA" > /dev/null || { echo "FATAL: no rows match $DATA" >&2; exit 1; }
  TASKS=()
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
    kv DFT_K_TRAINABLE 10; kv DFT_PREFIX_LEN 5; kv DFT_SIGMA_FLOOR 1e-4
    kv DFT_REPLAN 10; kv DFT_MAX_REPLANS 8
    kv DFT_SIM_AGENTS none; kv DFT_EGO_VEL_SOURCE commanded; kv DFT_WORLD_DT_METHOD median
  } > "$OUT/collect/env/$i.env"
  roundtrip "$OUT/collect/env/$i.env"
  printf '%2d  %-14s shard %s/%s  repeats %s  cap %s\n' "$i" "$split" "$shard" "$nsh" "$reps" "$cap"
  i=$((i + 1))
done
if [[ -z "$DATA_FROM" ]]; then
  CID=$(sb --parsable --exclude=gpu28 --array="0-$((i - 1))%3" --job-name=dft_pref_collect \
    rl/diffusion_ft/slurm/select_eval.sbatch "$OUT/collect/env")
  echo "collect -> array $CID ($i tasks, GPU) -> $OUT/collect/tasks/*/bc_data/$ARM/"
  DATA="$OUT/collect/tasks/*/bc_data/$ARM/*.npz"
fi
echo

# ---- 2. pair check (CPU, informational) ----------------------------------------- #
{ kv DFT_PREF_DATA "$DATA"; kv DFT_PREF_CHECK_OUT "$OUT/pairs_check.json"
  if [[ -z "$SMOKE" ]]; then kv DFT_PREF_REF_DATA "$BCD/collect/tasks/*/bc_data/$ARM/*.npz"; fi
} > "$OUT/pairs_check.env"
roundtrip "$OUT/pairs_check.env"
XID=$(sb --parsable ${CID:+--dependency=afterok:$CID} rl/diffusion_ft/slurm/pref_pairs_check.sbatch "$OUT/pairs_check.env")
echo "pairs   -> job $XID (CPU${CID:+, afterok:$CID}) -> $OUT/pairs_check.json"
echo

# ---- 3. train (GPU) ----------------------------------------------------------- #
XENV="$OUT/pref.env"
{
  echo "# Diffusion-DPO (submit_pref_dpo.sh): pairs + success anchor are the whole objective"
  kv DFT_PREF_DATA "$DATA"
  kv DFT_PREF_WEIGHT 1.0
  kv DFT_PREF_BATCH 16
  kv DFT_PREF_ANCHOR_BATCH 16
  kv DFT_PREF_ANCHOR_WEIGHT 100
  kv DFT_PREF_STEPS "$STEPS"
  kv DFT_PREF_ROWS solved
} > "$XENV"
roundtrip "$XENV"
SPECS=""; NAMES=""; declare -A TASK_IX; n=0
for a in $WANT_ARMS; do
  for s in $SEEDS; do
    SPECS+="dpo_${a}_s${s}|${SPEC[$a]},DFT_SAFETY_WINDOW=0,DFT_SEED=$s,DFT_DATA_SEED=$s "
    NAMES+="dpo_${a}_s${s} "
    TASK_IX[$a]+="${TASK_IX[$a]:+ }$n"; n=$((n + 1))
  done
done
DFT_SWEEP_OUT="$OUT/train" DFT_EXTRA_ENV_FILE="$XENV" \
DFT_ARM_SPECS="${SPECS% }" DFT_ARMS="${NAMES% }" \
DFT_DEPENDENCY="${CID:+afterok:$CID}" DFT_SPLITS_JSON="$SPLITS3" DFT_SPLIT_CAP="$SPLIT_CAP" \
DFT_SAFETY_WEIGHT=0.0 DFT_BASE_ANCHOR_WEIGHT=0.0 DFT_WORLD_DT_METHOD=median \
DFT_ITERS="$ITERS" DFT_EVAL_EVERY="$EVAL_EVERY" DFT_KEEP_EVERY="$KEEP_EVERY" \
DFT_MAX_CONCURRENT="${DFT_PREF_TRAIN_CONCURRENT:-4}" DFT_WANDB_GROUP="pref_dpo_${STAMP}" \
  bash rl/diffusion_ft/slurm/sweep_safety_stopping.sh | sed -n '/^ *[0-9]/p;/^submitted/p'
TID="$(cat "$OUT/train/jobid.txt")"
grep -q "^DFT_PREF_STEPS='$STEPS'$" "$OUT/train/env/0.env" \
  && grep -q "^DFT_SAFETY_WEIGHT=0.0$" "$OUT/train/env/0.env" \
  && grep -q "^DFT_BASE_ANCHOR_WEIGHT=0.0$" "$OUT/train/env/0.env" || {
  echo "FATAL: train env files do not carry the DPO config -- scancel $TID $CID" >&2; exit 1; }
echo "train   -> array $TID (GPU${CID:+, afterok:$CID}) -> $OUT/train/dpo_<arm>_s*/"
echo
fi   # train side

# ---- 4-5. per arm: eval on the bank-BC draws (GPU), then report (CPU) ---------- #
if [[ "$STAGE" != train ]]; then
if [[ "$STAGE" == eval ]]; then
  TID="$(cat "$OUT/train/jobid.txt")"; declare -A TASK_IX
  for a in $WANT_ARMS; do
    TASK_IX[$a]="$(awk -v p="^dpo_${a}_s[0-9]+$" '$2 ~ p {printf "%s ", $1}' "$OUT/train/arms.txt")"
    [[ -n "${TASK_IX[$a]// }" ]] || { echo "FATAL: arm $a was not trained in $OUT" >&2; exit 1; }
  done
fi
for a in $WANT_ARMS; do
  [[ ! -s "$OUT/eval_$a/jobid.txt" ]] || {
    echo "FATAL: $OUT/eval_$a was already submitted (job $(cat "$OUT/eval_$a/jobid.txt"))" >&2; exit 1; }
done
for a in $WANT_ARMS; do
  CKPTS=""; DEP="afterok"
  for it in $SCORE_ITERS; do
    for s in $SEEDS; do CKPTS+="$OUT/train/dpo_${a}_s${s}/iter_$(printf '%06d' "$it") "; done
  done
  # Chain on this arm's training tasks that are still queued; a task that has
  # already left the queue must have left its checkpoints behind. A task rerun
  # outside the array (e.g. after a fix) is not found that way: DFT_PREF_EVAL_DEP
  # gives the dependency verbatim instead.
  for x in ${TASK_IX[$a]}; do
    if squeue -h -j "${TID}_${x}" 2>/dev/null | grep -q .; then DEP+=":${TID}_${x}"; fi
  done
  if [[ -n "${DFT_PREF_EVAL_DEP:-}" ]]; then
    DEP="$DFT_PREF_EVAL_DEP"
  elif [[ "$DEP" == afterok ]]; then
    DEP=""
    for c in $CKPTS; do
      [[ -s "$c/tparams.npz" ]] || { echo "FATAL: arm $a: training is done but $c is missing -- read its .out" >&2; exit 1; }
    done
  fi
  DFT_SELECT_DIR="$OUT/eval_$a" DFT_SELECT_DEP="$DEP" DFT_SELECT_CKPTS="${CKPTS% }" \
  DFT_SPLITS3_JSON="$SPLITS3" DFT_BASE_ARMS="base" DFT_FT_ARMS=ft \
  DFT_SELECT_SEED=4242 DFT_SELECT_REPEATS=4 DFT_NUM_SHARDS=2 DFT_EVAL_CHUNK=32 \
  DFT_SIM_AGENTS=none DFT_EGO_VEL_SOURCE=commanded DFT_WORLD_DT_METHOD=median \
  DFT_MAX_CONCURRENT="${DFT_PREF_EVAL_CONCURRENT:-3}" DFT_SELECT_SMOKE="$SMOKE" \
    bash rl/diffusion_ft/slurm/submit_select_eval.sh | sed -n '/^score/p;/^analyze/p'
  SID="$(cat "$OUT/eval_$a/jobid.txt")"
  { kv DFT_PREF_RUN "$OUT"; kv DFT_PREF_EVAL "$OUT/eval_$a"; kv DFT_PREF_ITERS "$SCORE_ITERS"
    kv DFT_BCD_RUN "$BCD"; kv DFT_BCSW_RUN "$BCSW"; kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_SELECT_REPEATS "$([[ -n "$SMOKE" ]] && echo 1 || echo 4)"; } > "$OUT/report_$a.env"
  roundtrip "$OUT/report_$a.env"
  RID=$(sb --parsable --dependency="afterok:$SID" rl/diffusion_ft/slurm/pref_report.sbatch "$OUT/report_$a.env")
  echo "arm $a  -> eval array $SID (${DEP:-checkpoints on disk})  report job $RID (CPU) -> $OUT/eval_$a/compare_*.json"
done
echo
fi   # eval side

# ---- 6. probe (GPU) ----------------------------------------------------------- #
if [[ "$STAGE" != eval ]]; then
LAST="$(echo "$SCORE_ITERS" | awk '{print $NF}')"
PCK=""
for a in $WANT_ARMS; do PCK+="$OUT/train/dpo_${a}_s${SEEDS%% *}/iter_$(printf '%06d' "$LAST") "; done
if [[ -z "$SMOKE" ]]; then PCK+="$BCD/train/bcd_s0/iter_000150 $BCSW/train/bcsw50_s0/iter_000150"; fi
{ kv DFT_BC_DATA "$DATA"; kv DFT_FT_CKPTS "${PCK% }"
  kv DFT_OUT "$OUT/probe/pref_transfer.json"; kv DFT_N_ROWS "$NROWS"; } > "$OUT/probe.env"
roundtrip "$OUT/probe.env"
mkdir -p "$OUT/probe"
PID=$(sb --parsable --exclude=gpu28 --time=3:00:00 --dependency="afterok:$TID" rl/diffusion_ft/slurm/probe_bc_transfer.sbatch "$OUT/probe.env")
echo "probe   -> job $PID (GPU, afterok:$TID) -> $OUT/probe/pref_transfer.json"
echo
fi
echo "all under $OUT"
echo "  pairs_check.json                  pairs, edit sizes; same rows as bank-BC run 1?"
echo "  train/dpo_*/train_log.jsonl       pref/{loss,logit_z,accuracy,*_loss,grad_norm}, curves"
echo "  eval_<arm>/compare_it*_vs_*.json  paired deltas vs base / bank / bcd / bcsw50"
echo "  summary.md                        everything in one table (rewritten by each report)"
echo "  probe/pref_transfer.json          edit-row loss, closed_free, d_rej_m"
