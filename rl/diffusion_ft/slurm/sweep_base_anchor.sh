#!/bin/bash
# Sweep the FROZEN-BASE-POLICY anchor weight (beta) as FREE-RUNNING training
# runs, to find the largest failure-set gain that costs no success regression.
# Only calls sbatch -- one train_safety chain per arm. Arms run CONCURRENTLY
# (siblings); within each arm a chain of afterany SEGMENTS lets the run continue
# PAST the 12 h wall clock indefinitely (a compute node cannot sbatch its own
# successor -- every partition sets AllocNodes=rhea -- so the whole chain is laid
# down here, at submit time, from rhea).
#
# WHAT THE KNOB IS. train_safety optimizes
#   total = safety_weight*safety + anchor_weight*anchor + bc_weight*bc
#           + base_anchor_weight*base_anchor
# The first three anchors (when on) pull toward the HUMAN LOG; --base_anchor_weight
# pulls toward the PRETRAINED POLICY itself (Gaussian KL between the trainable and
# the frozen base denoiser's one-step posterior means, on the policy's own visited
# states; see DiffusionRLActor.base_anchor_loss). Its minimizer IS the base, so it
# is zero where the policy still matches base and applies force only against drift
# -- the do-no-harm leash on the successes, which the log-target anchors are not
# (the base already departs from the log while succeeding, so pulling toward the
# log perturbs a working policy). beta=0 is the safety-only default that regressed
# success-heldout; a large beta pins the policy to base and the safety term barely
# moves it. Read the arms by the paired failure-vs-success frontier:
#   success_heldout/solved_rate  vs  base_*  (retention)   -- must not drop
#   failure_heldout/solved_rate  vs  base_*  (the gain)     -- want it up
# The usable beta is the LARGEST failure gain at success retention >= base - eps.
#
# SIZING beta. The base_anchor KL is not on the same scale as the bc/anchor
# denoising losses, so their weight->share numbers do NOT transfer. Read the FIRST
# job's iter-0 `base_anchor_share` in train_log.jsonl (and then the MEDIAN over the
# run -- the KL collapses within a few steps, so iter 0 overstates it), and move
# the ladder so the arms straddle share ~0.2-0.6. The default ladder is geometric
# and wide on purpose; re-center it once the share is known.
#
# Node: `general` = a6000/a5000 (gpu24-31) only; this env's jaxlib dies at cuDNN
# autotune on v100 and OOMs on the 2080 Ti (see train_safety.sbatch). Do NOT widen
# the partition. The chain is GPU train jobs ONLY; redraw figures on demand with
# `sbatch rl/diffusion_ft/slurm/viz_safety_train.sbatch <env file>` (CPU, cheap).
#
# Usage (from repo root, ON RHEA):
#   DFT_SPLITS_JSON=/path/to/splits.json \
#     bash rl/diffusion_ft/slurm/sweep_base_anchor.sh
#   DFT_BASE_ANCHOR_ARMS="0 100 300 1000" DFT_SEGMENTS=24 \
#     DFT_SPLITS_JSON=/path/to/splits.json \
#     bash rl/diffusion_ft/slurm/sweep_base_anchor.sh
#   # extend an existing sweep past its chain: re-run with the SAME out dir
#   DFT_SWEEP_OUT=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/base_anchor_sweep/<STAMP> \
#     bash rl/diffusion_ft/slurm/sweep_base_anchor.sh
#
# Each arm is one beta (base_anchor_weight). safety_weight is pinned at 1.0. All
# other train_safety knobs pass through (same env-var names as train_safety.sbatch).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/base_anchor_sweep/${STAMP}}"

# The swept axis: beta = base_anchor_weight. 0 = the safety-only baseline that
# regressed; the rest is a wide geometric ladder to be re-centered from the logged
# base_anchor_share (see SIZING above).
ARMS="${DFT_BASE_ANCHOR_ARMS:-0 30 100 300 1000}"

# FREE RUN. Each arm is a chain of SEGMENTS afterany train jobs. A segment ends at
# the 12 h wall clock (not at an iter count), and the next resumes from the log --
# so the arm runs ~SEGMENTS*12 h. DFT_ITERS is set absurdly high on purpose: the
# loop is `range(start_iter, iters)` and resume continues from the log, so the
# cap must sit above anything a chain could reach, letting wall clock be the only
# thing that ends a segment. To run LONGER than the chain, re-run this script with
# the same DFT_SWEEP_OUT; resume picks up every arm where it stopped.
# SEGMENTS is capped by the QOS, not by taste: the cluster enforces
# QOSMaxSubmitJobPerUserLimit over ALL of a user's queued jobs, and a chain
# counts against it in full the moment it is laid down. The first version of this
# script asked for 24 segments x 2 jobs x 5 arms = 240 submissions and died
# partway through (job 45581 sweep, 2026-09-08), leaving two arms chained and
# three never submitted. Keep arms*SEGMENTS under the budget below, and buy wall
# clock with DFT_TIME (a longer per-segment limit) rather than more segments.
SEGMENTS="${DFT_SEGMENTS:-6}"
DFT_ITERS="${DFT_ITERS:-1000000}"
GROUP="${DFT_WANDB_GROUP:-base_anchor_${STAMP}}"
# Per-segment wall clock. The sbatch asks for 12 h; raise this (if the partition
# MaxTime allows) to cover more hours per submitted job -- the cheap axis.
DFT_TIME="${DFT_TIME:-}"
# Refuse to submit a chain bigger than this rather than half-submitting it. Set
# it from `sacctmgr show qos <your qos> format=MaxSubmitJobsPU` minus whatever is
# already queued.
MAX_JOBS="${DFT_MAX_JOBS:-40}"

# afterany (not afterok) between segments so a timed-out or preempted segment
# still releases its successor rather than stranding the chain. A segment that
# dies for a REAL reason will also release the next, which then resumes from the
# same log and likely dies the same way -- watch the first .out of each arm.
SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
[[ -n "${DFT_TIME:-}" ]]      && SB_FLAGS+=(--time="$DFT_TIME")

if [[ -z "${DFT_SPLITS_JSON:-}" ]]; then
  echo "WARNING: DFT_SPLITS_JSON is empty -- without it the four-panel eval" >&2
  echo "  (failure/success held-out + train) does not run, so there is nothing" >&2
  echo "  to read the failure-vs-success frontier off. Set it unless you mean to." >&2
fi

echo "sweep     -> $BASE_OUT"
echo "wandb grp -> $GROUP"
echo "arms      -> beta in { $ARMS }"
echo "free run  -> $SEGMENTS segments/arm x 12h  (iters cap=$DFT_ITERS)"
echo "splits    -> ${DFT_SPLITS_JSON:-<none>}  train_pool=${DFT_TRAIN_SPLITS:-uniform-corpus}"

# Count the arms the loop will actually accept, and refuse the whole submission if
# the chain exceeds the QOS budget. Half a sweep is worse than none: the arms that
# did land keep running and burn GPU while the comparison they exist for is
# missing, and the out_dirs look legitimate.
N_ARMS=0
for _a in $ARMS; do
  _b="${_a%%:*}"
  case "$_b" in ''|*[!0-9.eE+-]*) continue;; esac
  N_ARMS=$((N_ARMS + 1))
done
TOTAL_JOBS=$((N_ARMS * SEGMENTS))
echo "budget    -> $N_ARMS arms x $SEGMENTS segments = $TOTAL_JOBS jobs (limit $MAX_JOBS)"
if (( TOTAL_JOBS > MAX_JOBS )); then
  echo >&2
  echo "REFUSING: $TOTAL_JOBS jobs exceeds DFT_MAX_JOBS=$MAX_JOBS." >&2
  echo "  The QOS counts every queued job in a chain, so this would fail partway" >&2
  echo "  and leave an incomplete sweep running. Options:" >&2
  echo "    - fewer segments:  DFT_SEGMENTS=$((MAX_JOBS / (N_ARMS > 0 ? N_ARMS : 1)))" >&2
  echo "    - longer segments: DFT_TIME=1-00:00:00 (if partition MaxTime allows)" >&2
  echo "    - fewer arms:      DFT_BASE_ANCHOR_ARMS=\"0 100 1000\"" >&2
  echo "    - raise the cap:   DFT_MAX_JOBS=N (check MaxSubmitJobsPU first)" >&2
  exit 1
fi
echo

for ARM in $ARMS; do
  # Arm spec is BETA[:all|:succ]. The masking is part of the ARM, not a
  # sweep-wide env var, so `100` and `100:all` can run side by side -- otherwise
  # the two masking variants of one beta collide on the same tag and out_dir.
  BETA="${ARM%%:*}"; MODE="${ARM#*:}"
  [[ "$MODE" == "$ARM" ]] && MODE=""
  case "$BETA" in
    ''|*[!0-9.eE+-]*) echo "SKIP malformed beta '$ARM'" >&2; continue;;
  esac
  case "$MODE" in
    "")     SO="${DFT_BASE_ANCHOR_SUCCESS_ONLY:-1}"; TAG="beta${BETA}";;
    succ)   SO=1; TAG="beta${BETA}_succ";;
    all)    SO=0; TAG="beta${BETA}_all";;
    *) echo "SKIP arm '$ARM': mode must be 'all' or 'succ'" >&2; continue;;
  esac
  OUT_DIR="$BASE_OUT/$TAG"
  ENV_FILE="$ENV_DIR/baseanchor_${STAMP}_${TAG}.env"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    # data (defaults to full WOMD corpus; set DFT_TFRECORD for a fixed batch)
    [[ -n "${DFT_TFRECORD:-}" ]] && echo "DFT_TFRECORD=${DFT_TFRECORD}"
    [[ -n "${DFT_INDICES:-}" ]]  && echo "DFT_INDICES=${DFT_INDICES}"
    echo "DFT_SHARD_GLOB=${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
    echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-16}"
    echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-32}"
    echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
    echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
    # named eval splits -- the paired failure/success frontier the sweep is read on
    echo "DFT_SPLITS_JSON=${DFT_SPLITS_JSON:-}"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-1000}"
    echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-32}"
    # balanced training pool (the 50/50 failure/success batch), passed through
    echo "DFT_TRAIN_SPLITS=${DFT_TRAIN_SPLITS:-}"
    echo "DFT_TRAIN_SPLIT_CAP=${DFT_TRAIN_SPLIT_CAP:--1}"
    echo "DFT_TRAIN_SPLIT_RATIO=${DFT_TRAIN_SPLIT_RATIO:-}"
    # per-iter probe pictures under the curve
    echo "DFT_PROBE_PER_SPLIT=${DFT_PROBE_PER_SPLIT:-4}"
    echo "DFT_PROBE_SPLITS=${DFT_PROBE_SPLITS:-failure_train,success_train,failure_heldout,success_heldout}"
    echo "DFT_PROBE_EVERY=${DFT_PROBE_EVERY:-0}"
    echo "DFT_PROBE_SEED=${DFT_PROBE_SEED:-0}"
    # objective -- the swept axis is DFT_BASE_ANCHOR_WEIGHT; the log-target anchors
    # default OFF so the base anchor is isolated.
    echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
    echo "DFT_BASE_ANCHOR_WEIGHT=$BETA"
    echo "DFT_BASE_ANCHOR_SUCCESS_ONLY=$SO"
    echo "DFT_ANCHOR_WEIGHT=${DFT_ANCHOR_WEIGHT:-0.0}"
    echo "DFT_BC_WEIGHT=${DFT_BC_WEIGHT:-0.0}"
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
    # schedule -- free run: huge iter cap, wall clock ends the segment
    echo "DFT_ITERS=$DFT_ITERS"
    echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_GRAD_ACCUM=${DFT_GRAD_ACCUM:-0}"
    # Eval cadence is sized for a FREE RUN, not a 200-iter sweep. A full eval
    # pass is 2346 scenes (173+1000+173+1000); at repeats=4/every=10 that is 938
    # scene-rollouts charged per training iteration against a ~17 s iteration --
    # 5x the load of the bc_sweep that these splits were validated on
    # (repeats=2/every=25, 188/iter). Eval time does NOT show up in `iter_s`, so
    # this overspend is invisible in the log; it only shows as a run that never
    # gets anywhere. repeats=4 was chosen back when --split_cap was 48 and
    # binomial noise (~0.07) dominated; at cap 1000 the floor is ~0.015 already,
    # so averaging 4 seeds is paying twice for the same noise reduction.
    echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-100}"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    # a free run overwrites `latest` forever, so keep periodic weight-only
    # snapshots or "the success split was best 3000 iters ago" is unanswerable.
    echo "DFT_KEEP_EVERY=${DFT_KEEP_EVERY:-200}"
    echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
    echo "DFT_EVAL_REPEATS=${DFT_EVAL_REPEATS:-2}"
    echo "DFT_EVAL_BASE=${DFT_EVAL_BASE:-1}"
    echo "DFT_RESUME=1"
    echo "DFT_SEED=${DFT_SEED:-0}"
    # logging: one shared group, one name per arm, so the curves overlay
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
  } > "$ENV_FILE"

  # Lay down the segment chain for this arm: segment k+1 waits on segment k
  # (afterany), each resuming from the shared out_dir's log.
  #
  # NO per-segment figures job. It doubled the submission count against the QOS
  # limit to redraw a plot that `viz_safety_train.sbatch <env file>` regenerates
  # from train_log.jsonl at any time, on CPU, for free -- the figures were made
  # redrawable precisely so they need not ride the training chain.
  PREV_SEG=""
  for ((s = 1; s <= SEGMENTS; s++)); do
    DEP=(); [[ -n "$PREV_SEG" ]] && DEP=(--dependency="afterany:${PREV_SEG}")
    if ! SEG_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
      sbatch --parsable "${SB_FLAGS[@]}" "${DEP[@]}" \
      --job-name="dft_${TAG}_s${s}" \
      rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_FILE"); then
      echo "  $TAG  sbatch FAILED at segment $s -- arm is short but the" >&2
      echo "        segments already queued will still run. Fix the limit and" >&2
      echo "        re-run with DFT_SWEEP_OUT=$BASE_OUT to extend." >&2
      break
    fi
    if [[ "$s" == "1" ]]; then echo "  $TAG  seg1->$SEG_ID  ($OUT_DIR)"; fi
    PREV_SEG="$SEG_ID"
  done
  echo "        chained to $PREV_SEG"
done

echo
echo "read the frontier (failure gain vs success retention, per iter):"
cat <<EOF
  for d in $BASE_OUT/beta*/train_log.jsonl; do echo "== \$d =="
    python3 -c "import json
rows=[json.loads(l) for l in open('\$d') if l.strip()]
for r in rows:
    if 'failure_heldout/solved_rate' in r:
        print(r['iter'],
              'fail', round(r.get('failure_heldout/solved_rate',0),3),
              'fail_base', round(r.get('base_failure_heldout/solved_rate',0),3),
              'succ', round(r.get('success_heldout/solved_rate',0),3),
              'succ_base', round(r.get('base_success_heldout/solved_rate',0),3),
              'ba_share', round(r.get('base_anchor_share',0),3))"
  done
EOF
