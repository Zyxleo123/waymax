#!/bin/bash
# Gated tail adapter pilot: distill the repair optimizer (o_lpA3) into the diffusion
# policy's last K sampling steps. Only calls sbatch.
#
# CHAIN (rl/diffusion_ft/train_adapter.py, docs/adapter_distill.md)
#   1. collect (GPU array, select_eval.sbatch --dump_pairs): base + o_lpA3 on the
#      TRAINING pools, every alive replan stored replayable (frozen prefix output +
#      per-row late noise) next to the base plan and the executed plan.
#        failure_train  all 173 scenes x 4 repeats (2 tasks)
#        success_train  every 56th scene (~400) x 2 repeats (1 task)
#      seed 2718 (not 4242, the eval's; not 777/12345, the selection/train-eval seeds)
#   2. train (GPU): preflight (identity + gradient; a FAIL exits 0 without a
#      checkpoint), stages A 500 / B 200 / relabel / C 500, strength-0 re-check,
#      offline strength sweep, latency bench.
#   3. eval (GPU array, 4 shards x chunk 32, seed 4242, 4 repeats -- the reference
#      repair run's exact draws): base | adp_open (gate forced 1) | adp_gate (learned).
#      The teacher (base + optimizer) is read from that reference run, paired via
#      its base rows (the report checks they match).
#   4. sweep (GPU, 1 task): strengths 0 / 0.25 / 0.5 / 0.75 / 1 closed loop on the
#      first 48 scenes of each held-out split, 1 repeat, its own base arm.
#   5. CPU: select_analysis, paired compares (adapter arms vs base, vs each other,
#      adp_gate vs o_lpA3 across runs; term + legacy), ride quality, the report.
#
# Usage (repo root; on lov4 sbatch is local):
#   DFT_ADP_SMOKE=1 bash rl/diffusion_ft/slurm/submit_adapter_pilot.sh   # ~8 scenes, few steps
#   bash rl/diffusion_ft/slurm/submit_adapter_pilot.sh
#   DFT_ADP_PAIRS_FROM=<earlier run dir> bash ...     # reuse its collection, retrain + eval
#   DFT_ADP_EXTRA='--adapter_K 10' DFT_ADP_TAG=k10 DFT_ADP_PAIRS_FROM=... bash ...  # escalation
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_ADP_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/adapter/${STAMP}${DFT_ADP_TAG:+_$DFT_ADP_TAG}}"
# The verified three-stratum splits with `shard_glob` moved to /zfsauton/datasets/womd
# (eshau vanished ~10-02; same shards, offsets verified 10-03). train_safety refuses a
# splits file whose shard_glob resolves to a different file list than --shard_glob.
SPLITS3="${DFT_ADP_SPLITS:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata_womd.json}"
SHARD_GLOB=/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*
REF="${DFT_ADP_REF:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_proj_20260930_171718_1665097/select}"
TEACHER=o_lpA3
SMOKE="${DFT_ADP_SMOKE:-}"
PAIRS_FROM="${DFT_ADP_PAIRS_FROM:-}"
[[ -s "$SPLITS3" ]] || { echo "FATAL: no splits file $SPLITS3" >&2; exit 1; }
mkdir -p "$OUT" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
# Single-quote every value (a ';' in a bank spec would otherwise end the assignment).
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
# Source what was written in a clean shell; every key must come back byte-identical.
roundtrip() {
  local f="$1" k want got
  while IFS= read -r k; do
    want="$(sed -n "s/^${k}='\(.*\)'$/\1/p" "$f")"
    got="$(env -i bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2}"' _ "$f" "$k")"
    [[ "$want" == "$got" ]] || { echo "FATAL: $f: $k wrote [$want] read back [$got]" >&2; exit 1; }
  done < <(sed -n 's/^\(DFT_[A-Z_0-9]*\)=.*/\1/p' "$f")
}
# The reference run's rollout shape and teacher configuration, verbatim
# (lateral_opt_proj_20260930_171718_1665097/select/env/0.env).
select_common() {
  kv DFT_SPLITS_JSON "$SPLITS3"; kv DFT_SHARD_GLOB "$SHARD_GLOB"
  kv DFT_FT_CKPT ""; kv DFT_SELECTOR_JSON ""; kv DFT_LOOK_REPLANS 2
  kv DFT_SELECT_BANK "tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5"
  kv DFT_SELECT_RAMP 10; kv DFT_SELECT_BANKS "olp="; kv DFT_SELECT_BANK_RAMPS "olp=20"
  kv DFT_SELECT_BANK_BLENDS "olp=cubic"; kv DFT_SELECT_BANK_NEIGHBORS "olp=cv"
  kv DFT_SELECT_BANK_OPTS "olp=0,lon0,prev,lonprev"; kv DFT_SELECT_OPT_CFG ""
  kv DFT_SELECT_BANK_OPT_CFGS ""; kv DFT_SELECT_BANK_PREFER ""; kv DFT_SELECT_ES ""
  kv DFT_SELECT_TOL_M 1.0; kv DFT_SELECT_MARGIN_M 0.3; kv DFT_SELECT_DELTA_M 0.2
  kv DFT_K_TRAINABLE 10; kv DFT_PREFIX_LEN 5; kv DFT_SIGMA_FLOOR 1e-4
  kv DFT_REPLAN 10; kv DFT_MAX_REPLANS 8; kv DFT_SIM_AGENTS none
  kv DFT_EGO_VEL_SOURCE commanded; kv DFT_WORLD_DT_METHOD median
  kv DFT_EDGE_MARGIN 0.3; kv DFT_AGENT_MARGIN 0.75; kv DFT_REQUIRE_GATE ""
}

# ---- 1. collect ---------------------------------------------------------------- #
DEP_T=()
if [[ -n "$PAIRS_FROM" ]]; then
  PAIRS="$PAIRS_FROM/collect/tasks/*/pair_data/*/*.npz"
  n=$(compgen -G "$PAIRS" | wc -l || true)
  [[ "$n" -gt 0 ]] || { echo "FATAL: no pair files under $PAIRS_FROM/collect" >&2; exit 1; }
  echo "collect -> skipped: reusing $n files from $PAIRS_FROM"
else
  mkdir -p "$OUT/collect/env" "$OUT/collect/tasks"
  if [[ -n "$SMOKE" ]]; then
    TASKS=("failure_train|0|1|1|8" "success_train|0|1|1|8"); CHUNK=8
  else
    # split | shard | num_shards | repeats | cap
    TASKS=("failure_train|0|2|4|-1" "failure_train|1|2|4|-1" "success_train|0|56|2|-1"); CHUNK=16
  fi
  i=0
  for t in "${TASKS[@]}"; do
    IFS='|' read -r split shard nsh reps cap <<< "$t"
    d="$OUT/collect/tasks/${split}_shard${shard}"; mkdir -p "$d"
    f="$OUT/collect/env/$i.env"
    { kv DFT_OUT_DIR "$d"; select_common
      kv DFT_SELECT_SPLITS "$split"; kv DFT_ARMS "base,${TEACHER}=bank@olp:proj:20:0.3"
      kv DFT_SELECT_SEED 2718; kv DFT_SELECT_REPEATS "$reps"
      kv DFT_SHARD_INDEX "$shard"; kv DFT_NUM_SHARDS "$nsh"; kv DFT_EVAL_CHUNK "$CHUNK"
      kv DFT_SELECT_CAP "$cap"; kv DFT_DUMP_TRAJ 0; kv DFT_DUMP_PAIRS 1; kv DFT_ADAPTER_CKPT ""
    } > "$f"
    roundtrip "$f"
    printf '  collect %d: %s shard %s/%s x %s repeats (cap %s)\n' "$i" "$split" "$shard" "$nsh" "$reps" "$cap"
    i=$((i + 1))
  done
  echo "+ sbatch --parsable --array=0-$((i - 1))%3 --job-name=dft_adp_collect rl/diffusion_ft/slurm/select_eval.sbatch $OUT/collect/env"
  CID=$(sb --parsable --array="0-$((i - 1))%3" --job-name=dft_adp_collect \
    rl/diffusion_ft/slurm/select_eval.sbatch "$OUT/collect/env")
  echo "collect -> array $CID ($i tasks, GPU) -> $OUT/collect/tasks/*/pair_data/"
  PAIRS="$OUT/collect/tasks/*/pair_data/*/*.npz"
  DEP_T=(--dependency="afterok:$CID")
fi

# ---- 2. train ---------------------------------------------------------------- #
TRAIN="$OUT/train"; mkdir -p "$TRAIN"
f="$OUT/train.env"
{ kv DFT_ADAPTER_OUT "$TRAIN"; kv DFT_ADAPTER_PAIRS "$PAIRS"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_SHARD_GLOB "$SHARD_GLOB"; kv DFT_TEACHER_ARM "$TEACHER"
  kv DFT_ADAPTER_K "${DFT_ADP_K:-5}"
  kv DFT_ADAPTER_EXTRA "${DFT_ADP_EXTRA:-}"
  if [[ -n "$SMOKE" ]]; then
    kv DFT_STAGE_A 20; kv DFT_STAGE_B 10; kv DFT_STAGE_C 20; kv DFT_VAL_EVERY 10
  else
    kv DFT_STAGE_A "${DFT_ADP_STAGE_A:-500}"; kv DFT_STAGE_B "${DFT_ADP_STAGE_B:-200}"
    kv DFT_STAGE_C "${DFT_ADP_STAGE_C:-500}"; kv DFT_VAL_EVERY 100
  fi
} > "$f"
roundtrip "$f"
echo "+ sbatch --parsable ${DEP_T[*]:-} rl/diffusion_ft/slurm/train_adapter.sbatch $f"
TID=$(sb --parsable ${DEP_T[@]+"${DEP_T[@]}"} rl/diffusion_ft/slurm/train_adapter.sbatch "$f")
echo "train   -> job $TID (GPU) -> $TRAIN"
CKPT="$TRAIN/adapter_final"

# ---- 3. eval: the reference run's draws ------------------------------------------- #
EVAL="$OUT/eval"; mkdir -p "$EVAL/env" "$EVAL/tasks"
# a6000 only for the real eval (gpu28 is the a5000): the teacher's rows came from
# a6000s, and the pairing rests on this eval's base arm reproducing them bit for bit.
if [[ -n "$SMOKE" ]]; then E_SH=1; E_REP=1; E_CHUNK=4; E_CAP=4; EXCL=()
else E_SH=4; E_REP=4; E_CHUNK=32; E_CAP=-1; EXCL=(--exclude=gpu28); fi
for ((s = 0; s < E_SH; s++)); do
  d="$EVAL/tasks/adapter_shard${s}"; mkdir -p "$d"; f="$EVAL/env/$s.env"
  { kv DFT_OUT_DIR "$d"; select_common
    kv DFT_SELECT_SPLITS "failure_heldout,success_heldout,intermittent_heldout"
    kv DFT_ARMS "base,adp_open=adapter:open:1,adp_gate=adapter:learned:1"
    kv DFT_ADAPTER_CKPT "$CKPT"; kv DFT_SELECT_SEED 4242; kv DFT_SELECT_REPEATS "$E_REP"
    kv DFT_SHARD_INDEX "$s"; kv DFT_NUM_SHARDS "$E_SH"; kv DFT_EVAL_CHUNK "$E_CHUNK"
    kv DFT_SELECT_CAP "$E_CAP"; kv DFT_DUMP_TRAJ 1; kv DFT_DUMP_PAIRS 0
  } > "$f"
  roundtrip "$f"
done
echo "+ sbatch --parsable --dependency=afterok:$TID ${EXCL[*]:-} --array=0-$((E_SH - 1))%4 --job-name=dft_adp_eval rl/diffusion_ft/slurm/select_eval.sbatch $EVAL/env"
EID=$(sb --parsable --dependency="afterok:$TID" ${EXCL[@]+"${EXCL[@]}"} --array="0-$((E_SH - 1))%4" \
  --job-name=dft_adp_eval rl/diffusion_ft/slurm/select_eval.sbatch "$EVAL/env")
echo "eval    -> array $EID ($E_SH tasks, GPU, afterok:$TID) -> $EVAL"

# ---- 4. strength sweep (closed loop, capped) ------------------------------------- #
SWEEP="$OUT/sweep"; mkdir -p "$SWEEP/tasks/sweep_shard0"
f="$SWEEP/sweep.env"
{ kv DFT_OUT_DIR "$SWEEP/tasks/sweep_shard0"; select_common
  kv DFT_SELECT_SPLITS "failure_heldout,success_heldout,intermittent_heldout"
  kv DFT_ARMS "base,adp_s0=adapter:learned:0,adp_s25=adapter:learned:0.25,adp_s50=adapter:learned:0.5,adp_s75=adapter:learned:0.75,adp_s100=adapter:learned:1"
  kv DFT_ADAPTER_CKPT "$CKPT"; kv DFT_SELECT_SEED 4242; kv DFT_SELECT_REPEATS 1
  kv DFT_SHARD_INDEX 0; kv DFT_NUM_SHARDS 1; kv DFT_EVAL_CHUNK 16
  kv DFT_SELECT_CAP "$([[ -n "$SMOKE" ]] && echo 4 || echo 48)"; kv DFT_DUMP_TRAJ 0; kv DFT_DUMP_PAIRS 0
} > "$f"
roundtrip "$f"
echo "+ sbatch --parsable --dependency=afterok:$TID --job-name=dft_adp_sweep rl/diffusion_ft/slurm/select_eval.sbatch $f"
SID=$(sb --parsable --dependency="afterok:$TID" --job-name=dft_adp_sweep \
  rl/diffusion_ft/slurm/select_eval.sbatch "$f")
echo "sweep   -> job $SID (GPU, afterok:$TID) -> $SWEEP"

# ---- 5. CPU: analysis, compares, ride, report ------------------------------------ #
REPS="$E_REP"
{ kv DFT_SELECT_DIR "$EVAL"; kv DFT_SPLITS_JSON "$SPLITS3"; kv DFT_SELECT_REPEATS "$REPS"; } > "$EVAL/analyze.env"
AID=$(sb --parsable --dependency="afterok:$EID" rl/diffusion_ft/slurm/select_analysis.sbatch "$EVAL/analyze.env")
CIDS=()
for spec in "adp_gate:base:$EVAL" "adp_open:base:$EVAL" "adp_gate:adp_open:$EVAL" "adp_gate:${TEACHER}:$REF"; do
  IFS=':' read -r arm ref refdir <<< "$spec"
  [[ -n "$SMOKE" && "$refdir" == "$REF" ]] && continue      # smoke draws are not the reference's
  for term in "" 1; do
    f="$EVAL/cmp_${arm}_vs_${ref}${term:+_term}.env"
    { kv DFT_CMP_A "$refdir"; kv DFT_CMP_B "$EVAL"; kv DFT_SPLITS_JSON "$SPLITS3"
      kv DFT_CMP_ARM_A "$ref"; kv DFT_CMP_ARM_B "$arm"; kv DFT_CMP_LABEL_A "$ref"; kv DFT_CMP_LABEL_B "$arm"
      kv DFT_CMP_TERM "$term"; kv DFT_SELECT_REPEATS "$REPS"
      kv DFT_CMP_OUT "$EVAL/compare_${arm}_vs_${ref}${term:+_term}.json"; } > "$f"
    CIDS+=("$(sb --parsable --dependency="afterok:$EID" rl/diffusion_ft/slurm/compare_select.sbatch "$f")")
  done
done
{ kv DFT_SELECT_DIR "$EVAL"; kv DFT_RIDE_SPLITS "failure_heldout,intermittent_heldout,success_heldout"; } > "$EVAL/ride.env"
RID=$(sb --parsable --dependency="afterok:$EID" rl/diffusion_ft/slurm/viz_repair.sbatch "$EVAL/ride.env")
f="$OUT/report.env"
{ kv DFT_RPT_EVAL "$EVAL"; kv DFT_RPT_REF "$([[ -n "$SMOKE" ]] || echo "$REF")"
  kv DFT_RPT_TRAIN "$TRAIN"; kv DFT_RPT_SWEEP "$SWEEP"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_RPT_OUT "$OUT/report"; } > "$f"
roundtrip "$f"
# afterany on the ride job too: the report reads its ride_*.json, and without the
# dependency it raced the ride job and printed the teacher alone (k10 run, 10-05).
PID=$(sb --parsable --dependency="afterok:$EID,afterany:$SID,afterany:$RID" rl/diffusion_ft/slurm/adapter_report.sbatch "$f")
echo "cpu     -> analysis $AID, compares ${CIDS[*]:-none}, ride $RID, report $PID"
echo "$OUT" > "$ROOT/logs/dft/adapter_pilot_latest.txt"
echo "all under $OUT; read report.md (and train/preflight.json first if it stops early)"
