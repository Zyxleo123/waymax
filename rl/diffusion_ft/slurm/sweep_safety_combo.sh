#!/bin/bash
# Do the four independent wins from the 2026-09-09 one-at-a-time ablation ADD UP?
# Only calls sbatch.
#
# WHAT THIS FOLLOWS FROM. `sweep_safety_ablation.sh` (array 46733, 12 arms x 300
# iters, ablation/20260909_171022_586834) varied six axes one at a time against a
# beta=100 success-only centre. Base reference: failure_heldout 0.017, success
# 0.997. Result, sorted by marginal delta:
#
#     arm          fail   succ    margD      arm          fail   succ    margD
#     tau_lo       0.335  0.979  -0.0130     k25          0.329  0.961  -0.0308
#     swin_full    0.318  0.975  -0.0172     k3           0.347  0.960  -0.0315
#     awin_exec    0.358  0.972  -0.0195     tau_hi       0.364  0.961  -0.0303
#     swin_25      0.324  0.972  -0.0200     BASE         0.335  0.954  -0.0376
#     margin_lo    0.347  0.970  -0.0217     margin_hi    0.376  0.949  -0.0419
#                                            anchor_bc    0.399  0.947  -0.0435
#                                            anchor_none  0.422  0.756  -0.2313
#
# READ THE SUCCESS COLUMN, NOT THE FAILURE COLUMN. `failure_heldout` is 173
# scenes (it cannot reach the 1000 cap -- only ~628 failure candidates exist in
# the scan at all), so at p~0.35 its se is 0.036 and EVERY arm except
# `anchor_none` is statistically tied on failure. `success_heldout` is a real
# 1000 (se ~0.006), and the marginal weights success at 0.985 -- so the marginal
# ranking IS the success ranking, and that part is sound. The ablation did not
# find arms that fix more failures; it found arms that fix the SAME failures for
# less collateral damage.
#
# THE HYPOTHESIS. Four separate one-knob changes -- narrower tau, smaller margin,
# anchor charged only on the executed window, safety scored past the replan --
# each land in the -0.013..-0.022 band against the centre's -0.0376, and all four
# are the same KIND of change: stop applying force where there is nothing to fix.
# If their effects are independent they should compose. Break-even needs success
# >= 0.992 (logged as `marginal/breakeven_success_rate`); the best single arm
# reached 0.979, so the combination is the first configuration with a real chance
# of clearing it. If it does, the fine-tune stops being a corpus-level net loss.
#
# THE DESIGN IS LEAVE-ONE-OUT, not one-at-a-time. The centre here is all four
# wins ON; each arm turns exactly ONE of them back OFF. So an arm that is much
# worse than `combo` identifies a component that is carrying real weight, and an
# arm indistinguishable from `combo` identifies one that is redundant once the
# others are present. `ref_k3` reverts all four at once and is the reference
# point -- it is the previous sweep's centre, so it says how much the whole
# combination bought.
#
# k3 IS IN THE CENTRE, and that is a new assumption. k_trainable measured FLAT in
# the ablation (k3 -0.0315, k10 -0.0376, k25 -0.0308, all inside the success
# noise), so k3 is adopted for cost, not for effect. `combo_k10` exists because
# "flat on its own" does not prove "flat in combination" -- if k3 interacts with
# the narrowed geometry, that arm is what catches it.
#
# SCHEDULE. 200 iterations, not 300: in the ablation the centre lost success
# 0.98 -> 0.95 over the last third, and both `swin` arms gave their failure gain
# back (swin_full peaked 0.405 at iter 200 and fell to 0.318 by 299). eval_every
# and keep_every are 50 here rather than 100, so the peak is actually resolvable
# and there is a checkpoint at it. ~1.75 h/arm, 7 arms at 3-wide = ~4.5 h.
#
# CONCURRENCY. One job array with `%3`, same as the ablation: at most 3 GPUs at
# any moment. Each task sources `<env_dir>/<task_id>.env` (the directory branch
# at the top of train_safety.sbatch). Arms are fixed-length and DFT_RESUME=1, so
# re-running the same command finishes anything the wall clock cut short.
#
# Node: `general` = a6000/a5000 (gpu24-31) only. This env's jaxlib dies at cuDNN
# autotune on v100 and OOMs on the 2080 Ti -- do NOT widen the partition.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea):
#   bash rl/diffusion_ft/slurm/sweep_safety_combo.sh
#   DFT_ARMS="combo ref_k3" bash rl/diffusion_ft/slurm/sweep_safety_combo.sh
#   # resume/extend: re-run with the SAME out dir
#   DFT_SWEEP_OUT=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/combo/<STAMP> \
#     bash rl/diffusion_ft/slurm/sweep_safety_combo.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/combo/${STAMP}}"
# Must outlive the whole array (later tasks source it hours after submission), so
# it lives under the sweep's out dir, not the shared logs/dft/env scratch.
ENV_DIR="$BASE_OUT/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

SPLITS_JSON="${DFT_SPLITS_JSON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_verified.json}"
CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"

# ---- the centre: ALL FOUR WINS ON, plus k3 -----------------------------------
# Anchor strength is unchanged from the ablation (beta=100 success-only): the
# ablation confirmed the frozen-ckpt anchor beats both the log-target anchor
# (-0.0435) and no anchor at all (-0.2313, success collapsed to 0.756, drift
# 3.12 m against the anchored arms' 0.72 m), so it is not in question here.
B_BASE_ANCHOR="${DFT_BASE_ANCHOR_WEIGHT:-100}"
B_BC="${DFT_BC_WEIGHT:-0.0}"
B_EDGE_TAU="${DFT_EDGE_TAU:-0.075}"      # win 1: half the knee width
B_AGENT_TAU="${DFT_AGENT_TAU:-0.125}"
B_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.15}" # win 2: half the margin
B_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.375}"
B_SWIN="${DFT_SAFETY_WINDOW:-25}"        # win 3: score 2.5 s, not the driven 1 s
B_AWIN="${DFT_ANCHOR_WINDOW:-0}"         # win 4: anchor the executed window only
B_K="${DFT_K_TRAINABLE:-3}"              # adopted for cost; measured flat

ITERS="${DFT_ITERS:-200}"
EVAL_EVERY="${DFT_EVAL_EVERY:-50}"
KEEP_EVERY="${DFT_KEEP_EVERY:-50}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"

[[ -s "$SPLITS_JSON" ]] || {
  echo "FATAL: splits file not found: $SPLITS_JSON" >&2
  echo "  Every arm is read off the four-panel failure/success frontier." >&2
  exit 1; }

# ---- the arms: leave-one-out against the centre ------------------------------
# name|VAR=value[,...]  -- each arm turns exactly ONE win back OFF.
ALL_ARMS=(
  "combo|"
  "combo_noTau|DFT_EDGE_TAU=0.15,DFT_AGENT_TAU=0.25"
  "combo_noMargin|DFT_EDGE_MARGIN=0.3,DFT_AGENT_MARGIN=0.75"
  "combo_noSwin|DFT_SAFETY_WINDOW=0"
  "combo_noAwin|DFT_ANCHOR_WINDOW=-1"
  # Not a win being removed -- a check that k3 does not INTERACT with the
  # narrowed geometry. It measured flat alone; that is not the same claim.
  "combo_k10|DFT_K_TRAINABLE=10"
  # All four reverted at once = the ablation's centre, on k3. The reference the
  # combination has to beat; deliberately not a one-knob arm.
  "ref_k3|DFT_EDGE_TAU=0.15,DFT_AGENT_TAU=0.25,DFT_EDGE_MARGIN=0.3,DFT_AGENT_MARGIN=0.75,DFT_SAFETY_WINDOW=0,DFT_ANCHOR_WINDOW=-1"
)

WANT="${DFT_ARMS:-}"
ARMS=()
if [[ -z "$WANT" ]]; then
  ARMS=("${ALL_ARMS[@]}")
else
  for w in $WANT; do
    hit=""
    for a in "${ALL_ARMS[@]}"; do
      [[ "${a%%|*}" == "$w" ]] && { ARMS+=("$a"); hit=1; break; }
    done
    [[ -n "$hit" ]] || { echo "FATAL: unknown arm '$w'" >&2; exit 1; }
  done
fi
N=${#ARMS[@]}
GROUP="${DFT_WANDB_GROUP:-combo_${STAMP}}"

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
[[ -n "${DFT_TIME:-}" ]]      && SB_FLAGS+=(--time="$DFT_TIME")

echo "sweep     -> $BASE_OUT"
echo "wandb grp -> $GROUP"
echo "splits    -> $SPLITS_JSON"
echo "arms      -> $N, at most $MAX_CONCURRENT running at once"
echo "centre    -> tau=$B_EDGE_TAU/$B_AGENT_TAU margin=$B_EDGE_MARGIN/$B_AGENT_MARGIN" \
     "swin=$B_SWIN awin=$B_AWIN k=$B_K beta=$B_BASE_ANCHOR"
echo "schedule  -> $ITERS iters, eval/keep every $EVAL_EVERY"
echo

MANIFEST="$BASE_OUT/arms.txt"
: > "$MANIFEST"
IDX_N=0
for spec in "${ARMS[@]}"; do
  TAG="${spec%%|*}"; OVR="${spec#*|}"
  [[ "$OVR" == "$spec" ]] && OVR=""
  OUT_DIR="$BASE_OUT/$TAG"
  mkdir -p "$OUT_DIR"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SHARD_GLOB=$GLOB"
    echo "DFT_INDEX_DIR=$IDX"
    echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-32}"
    echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-16}"
    echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
    echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
    echo "DFT_SPLITS_JSON=$SPLITS_JSON"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-1000}"
    echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-32}"
    echo "DFT_TRAIN_SPLITS=${DFT_TRAIN_SPLITS:-failure_train,success_train}"
    echo "DFT_TRAIN_SPLIT_CAP=${DFT_TRAIN_SPLIT_CAP:--1}"
    echo "DFT_TRAIN_SPLIT_RATIO=${DFT_TRAIN_SPLIT_RATIO:-1,1}"
    echo "DFT_PROBE_PER_SPLIT=0"
    echo "DFT_PROBE_EVERY=0"
    # objective
    echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
    echo "DFT_ANCHOR_WEIGHT=${DFT_ANCHOR_WEIGHT:-0.0}"
    echo "DFT_BC_WEIGHT=$B_BC"
    echo "DFT_BASE_ANCHOR_WEIGHT=$B_BASE_ANCHOR"
    echo "DFT_BASE_ANCHOR_SUCCESS_ONLY=${DFT_BASE_ANCHOR_SUCCESS_ONLY:-1}"
    echo "DFT_EDGE_MARGIN=$B_EDGE_MARGIN"
    echo "DFT_AGENT_MARGIN=$B_AGENT_MARGIN"
    echo "DFT_EDGE_TAU=$B_EDGE_TAU"
    echo "DFT_AGENT_TAU=$B_AGENT_TAU"
    echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    echo "DFT_SAFETY_WINDOW=$B_SWIN"
    echo "DFT_ANCHOR_WINDOW=$B_AWIN"
    echo "DFT_K_TRAINABLE=$B_K"
    echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    echo "DFT_COLLECT_STEPS=${DFT_COLLECT_STEPS:-0}"
    echo "DFT_EVAL_STEPS=${DFT_EVAL_STEPS:-0}"
    echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-0}"
    echo "DFT_ANCHOR_STATES=${DFT_ANCHOR_STATES:-policy}"
    # schedule
    echo "DFT_ITERS=$ITERS"
    echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_GRAD_ACCUM=${DFT_GRAD_ACCUM:-0}"
    echo "DFT_EVAL_EVERY=$EVAL_EVERY"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_KEEP_EVERY=$KEEP_EVERY"
    echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
    echo "DFT_EVAL_REPEATS=${DFT_EVAL_REPEATS:-1}"
    echo "DFT_EVAL_BASE=${DFT_EVAL_BASE:-1}"
    echo "DFT_RESUME=1"
    echo "DFT_SEED=${DFT_SEED:-0}"
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
    # The override goes LAST so it wins: the file is sourced top to bottom.
    if [[ -n "$OVR" ]]; then
      echo "# --- arm override: $TAG ---"
      IFS=',' read -ra KV <<< "$OVR"
      for kv in "${KV[@]}"; do echo "$kv"; done
    fi
  } > "$ENV_DIR/${IDX_N}.env"
  printf '%2d  %-15s %s\n' "$IDX_N" "$TAG" "${OVR:-<centre: all four wins>}" | tee -a "$MANIFEST"
  IDX_N=$((IDX_N + 1))
done
echo
echo "arm manifest -> $MANIFEST"
echo

# No --export of any kind: an explicit export list sets SLURM_GET_USER_ENV=1, the
# compute node fails to rebuild the login environment, and the job is requeued
# and HELD. The env dir travels as a positional argument.
ARRAY_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" \
    --job-name="dft_combo" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_DIR")

echo "submitted array $ARRAY_ID  (${N} tasks, %${MAX_CONCURRENT})"
echo
cat <<EOF
watch:
  squeue -j $ARRAY_ID -r -o '%.14i %.12j %.9T %.10M %R'

read it (does the combination clear break-even?):
  python3 - <<'PY'
import json,glob
for p in sorted(glob.glob('$BASE_OUT/*/train_log.jsonl')):
    arm=p.split('/')[-2]
    ev=[json.loads(l) for l in open(p) if l.strip()]
    ev=[r for r in ev if 'failure_heldout/solved_rate' in r]
    if not ev: continue
    r=ev[-1]
    be=r.get('marginal/breakeven_success_rate',0); sc=r.get('success_heldout/solved_rate',0)
    print('%-15s fail %.3f  succ %.3f  margD %+.4f  breakeven %.4f  %s' % (
        arm, r.get('failure_heldout/solved_rate',0), sc,
        r.get('marginal/delta_vs_base',0), be, 'CLEARS' if sc>=be else ''))
PY
EOF
