#!/bin/bash
# One-change-at-a-time ablation of the safety fine-tune's objective geometry.
# Only calls sbatch.
#
# SIX AXES, ONE KNOB PER ARM. Every arm below differs from `base` in exactly one
# setting, so a difference in the curve is attributable. The axes:
#
#   1. tau        the WIDTH OF THE KNEE of `tau*softplus((margin-c)/tau)`.
#                 Distinct from the margin: the margin says WHERE cost starts,
#                 tau says HOW SHARPLY, and it also sets the ~2.7*tau band past
#                 the margin over which gradient still exists. Small tau is a
#                 near-hinge that only pushes frames already inside the margin;
#                 large tau is a soft field that also pulls frames that are clear.
#   2. margin     where the barrier turns on. 0.3 m edge / 0.75 m agent are
#                 calibrated against measured human clearance, so this asks
#                 whether the calibration or a deliberate over/under-shoot wins.
#   3. k_trainable  how many of the 100 reverse steps carry a gradient. On the
#                 cosine schedule the residual xy noise is 2.5 m at t=0, 5.8 m at
#                 t=2, 16.7 m at t=9 -- so k=3 trains only the precise-geometry
#                 end and k=25 reaches steps that still choose the manoeuvre.
#   4. anchor source   frozen CHECKPOINT (--base_anchor_weight: Gaussian KL to
#                 the pretrained denoiser, minimizer IS the base) vs EXPERT
#                 behaviour (--bc_weight: the pretraining denoising loss on the
#                 human's own states). They are not the same leash -- the base
#                 already departs from the log while succeeding, so pulling
#                 toward the log perturbs a working policy. `none` is the control.
#   5. safety window   how much of the candidate the safety term is charged on.
#                 The candidate is ~5 s but the env drives 1 s before replanning.
#                 Scoring only the executed 1 s prices only trajectory that
#                 happens (the `safety_objective` docstring's argument) -- but it
#                 also cannot see a violation the candidate commits at 2.5 s, so
#                 the policy is free to aim at one and replan out of it forever.
#   6. anchor window   the same question for the anchor: is the tether charged on
#                 the whole 5 s horizon or only the executed part. A tether on
#                 unexecuted trajectory constrains shape that is never driven.
#
# CONCURRENCY. One job ARRAY with `%3`, so at most 3 GPUs are held at any moment
# no matter how many arms there are -- independent sibling jobs (what every other
# sweep in this directory submits) have no such throttle and would take the whole
# partition. Each array task sources `<env_dir>/<task_id>.env`; see the directory
# branch at the top of train_safety.sbatch.
#
# RESUMABILITY. Arms are fixed-length (DFT_ITERS, default 300), not free-running,
# because a segment chain does not compose with an array throttle. Every arm has
# DFT_RESUME=1 against its own out_dir, so an arm cut off by the 12 h wall clock
# is finished by re-submitting the SAME array (the script prints the command);
# arms that already reached DFT_ITERS exit immediately.
#
# COST. ~11-15 s/iter training plus ~13 min per eval point (2346 scenes at
# cap 1000, repeats 1). At 300 iters / eval_every 100 that is ~2.5 h per arm, so
# 12 arms at 3-wide is ~10-11 h wall. The `swin_*` arms score 2.5-5x the steps in
# the safety term, but the 100-step sampler dominates the gradient pass, so
# expect ~+20-50% on those, not 5x.
#
# Node: `general` = a6000/a5000 (gpu24-31) only. This env's jaxlib dies at cuDNN
# autotune on v100 and OOMs on the 2080 Ti -- do NOT widen the partition.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea):
#   bash rl/diffusion_ft/slurm/sweep_safety_ablation.sh
#   DFT_ARMS="base tau_lo tau_hi" bash rl/diffusion_ft/slurm/sweep_safety_ablation.sh
#   DFT_MAX_CONCURRENT=2 DFT_ITERS=500 bash rl/diffusion_ft/slurm/sweep_safety_ablation.sh
#   # resume/extend: re-run with the SAME out dir
#   DFT_SWEEP_OUT=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/ablation/<STAMP> \
#     bash rl/diffusion_ft/slurm/sweep_safety_ablation.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/ablation/${STAMP}}"
# Array mode needs ONE directory holding <task_id>.env, and it must survive the
# whole array (tasks 9..11 source it hours after submission), so it lives under
# the sweep's own out dir rather than in the shared logs/dft/env scratch.
ENV_DIR="$BASE_OUT/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

# ---- the centre of the star --------------------------------------------------
# Everything here is the currently-best-measured configuration, and every arm is
# this with ONE line replaced. Anchor centre = the frozen-base KL at beta=100
# success-only, which measured -0.004..-0.012 marginal against beta=0's
# -0.02..-0.15 and the log-target bc300's -0.031 (2026-09-08).
SPLITS_JSON="${DFT_SPLITS_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_verified.json}"
CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"

B_BASE_ANCHOR="${DFT_BASE_ANCHOR_WEIGHT:-100}"
B_BC="${DFT_BC_WEIGHT:-0.0}"
B_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
B_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
B_EDGE_TAU="${DFT_EDGE_TAU:-0.15}"
B_AGENT_TAU="${DFT_AGENT_TAU:-0.25}"
B_K="${DFT_K_TRAINABLE:-10}"
B_SWIN="${DFT_SAFETY_WINDOW:-0}"     # 0 = the executed window (1 s)
B_AWIN="${DFT_ANCHOR_WINDOW:--1}"    # -1 = the whole 5 s horizon

ITERS="${DFT_ITERS:-300}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"

[[ -s "$SPLITS_JSON" ]] || {
  echo "FATAL: splits file not found: $SPLITS_JSON" >&2
  echo "  Every arm is read off the four-panel failure/success frontier, so" >&2
  echo "  without it there is nothing to compare. Build it with" >&2
  echo "  submit_verified_safety.sh, or point DFT_SPLITS_JSON at an existing one." >&2
  exit 1; }

# ---- the arms ----------------------------------------------------------------
# name|VAR=value[,VAR=value...]   -- the overrides against the centre above.
# A comma-separated list is still ONE axis (edge and agent are the two halves of
# the same knob); no arm changes two different questions at once.
ALL_ARMS=(
  "base|"
  # 1. tau: the knee width, halved and doubled on both barriers.
  "tau_lo|DFT_EDGE_TAU=0.075,DFT_AGENT_TAU=0.125"
  "tau_hi|DFT_EDGE_TAU=0.30,DFT_AGENT_TAU=0.50"
  # 2. margin: where the barrier turns on, halved and doubled on both.
  "margin_lo|DFT_EDGE_MARGIN=0.15,DFT_AGENT_MARGIN=0.375"
  "margin_hi|DFT_EDGE_MARGIN=0.60,DFT_AGENT_MARGIN=1.50"
  # 3. how many reverse steps the gradient reaches.
  "k3|DFT_K_TRAINABLE=3"
  "k25|DFT_K_TRAINABLE=25"
  # 4. what the policy is anchored TO. CAVEAT: weight 100 on both does NOT make
  #    these equal-strength -- the two terms are on different scales (bc 100
  #    measured ~0.30 of the objective; base_anchor's `share` is ~3e-5 and
  #    useless, since its VALUE is ~0 by construction while its GRADIENT is
  #    large, and gradients are clipped to global norm 1 so only direction
  #    survives). So read this axis by where each arm lands on the
  #    failure-gain vs success-retention frontier, NOT as a paired comparison at
  #    matched strength. To match strength instead, size bc from the `grad_norm`
  #    ratio against `anchor_none` (measured 78 unanchored -> 467 at beta=100).
  "anchor_bc|DFT_BASE_ANCHOR_WEIGHT=0.0,DFT_BC_WEIGHT=100"
  "anchor_none|DFT_BASE_ANCHOR_WEIGHT=0.0"
  # 5. safety loss on the unexecuted part of the candidate (2.5 s, then all 5 s).
  "swin_25|DFT_SAFETY_WINDOW=25"
  "swin_full|DFT_SAFETY_WINDOW=-1"
  # 6. anchor on the unexecuted part: charge it on the executed 1 s only.
  "awin_exec|DFT_ANCHOR_WINDOW=0"
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
GROUP="${DFT_WANDB_GROUP:-ablation_${STAMP}}"

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
echo "centre    -> beta=$B_BASE_ANCHOR bc=$B_BC margin=$B_EDGE_MARGIN/$B_AGENT_MARGIN" \
     "tau=$B_EDGE_TAU/$B_AGENT_TAU k=$B_K swin=$B_SWIN awin=$B_AWIN iters=$ITERS"
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
    # the four-panel frontier every arm is read on
    echo "DFT_SPLITS_JSON=$SPLITS_JSON"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-1000}"
    echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-32}"
    # ALL the data, balanced 50/50 per batch -- the mixed pool is what does the
    # do-no-harm work (the failure-only pool took success_heldout to 0.688).
    echo "DFT_TRAIN_SPLITS=${DFT_TRAIN_SPLITS:-failure_train,success_train}"
    echo "DFT_TRAIN_SPLIT_CAP=${DFT_TRAIN_SPLIT_CAP:--1}"
    echo "DFT_TRAIN_SPLIT_RATIO=${DFT_TRAIN_SPLIT_RATIO:-1,1}"
    # probe pictures OFF: each probe point is an extra eval pass, and with 12
    # arms the eval budget is the binding constraint. Re-render from
    # train_log.jsonl on CPU instead (viz_safety_train.sbatch).
    echo "DFT_PROBE_PER_SPLIT=0"
    echo "DFT_PROBE_EVERY=0"
    # ---- the objective: the centre, then this arm's one override -------------
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
    # ---- schedule -----------------------------------------------------------
    echo "DFT_ITERS=$ITERS"
    echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_GRAD_ACCUM=${DFT_GRAD_ACCUM:-0}"
    echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-100}"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_KEEP_EVERY=${DFT_KEEP_EVERY:-100}"
    echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
    echo "DFT_EVAL_REPEATS=${DFT_EVAL_REPEATS:-1}"
    echo "DFT_EVAL_BASE=${DFT_EVAL_BASE:-1}"
    echo "DFT_RESUME=1"
    echo "DFT_SEED=${DFT_SEED:-0}"
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
    # The arm's override goes LAST so it wins over the centre above -- the whole
    # file is sourced top to bottom, so a later assignment simply replaces the
    # earlier one. This is the only line that differs between arms.
    if [[ -n "$OVR" ]]; then
      echo "# --- arm override: $TAG ---"
      IFS=',' read -ra KV <<< "$OVR"
      for kv in "${KV[@]}"; do echo "$kv"; done
    fi
  } > "$ENV_DIR/${IDX_N}.env"
  printf '%2d  %-12s %s\n' "$IDX_N" "$TAG" "${OVR:-<centre>}" | tee -a "$MANIFEST"
  IDX_N=$((IDX_N + 1))
done
echo
echo "arm manifest -> $MANIFEST"
echo "env files    -> $ENV_DIR/{0..$((N - 1))}.env"
echo

# Scrub SLURM_* so a submission from inside an allocation does not inherit it.
# NOTE: no `--export` of any kind -- an explicit export list sets
# SLURM_GET_USER_ENV=1, the compute node fails to rebuild the login environment,
# and the job is requeued and HELD. The env file travels as a positional arg.
ARRAY_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" \
    --job-name="dft_abl" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_DIR")

echo "submitted array $ARRAY_ID  (${N} tasks, %${MAX_CONCURRENT})"
echo
cat <<EOF
watch:
  squeue -j $ARRAY_ID -o '%.18i %.12j %.8T %.10M %R'
  tail -f $ROOT/logs/dft/slurm-dft_abl-*.out

resume anything the 12 h wall clock cut short (same env dir, arms that already
reached $ITERS iterations exit immediately):
  DFT_SWEEP_OUT=$BASE_OUT bash rl/diffusion_ft/slurm/sweep_safety_ablation.sh

read the frontier (failure gain vs success retention, per arm, at the last eval):
  for d in $BASE_OUT/*/train_log.jsonl; do
    python3 -c "import json,sys
rows=[json.loads(l) for l in open('\$d') if l.strip()]
ev=[r for r in rows if 'failure_heldout/solved_rate' in r]
if ev:
    r=ev[-1]
    print('%-12s it%-5d fail %.3f (base %.3f)  succ %.3f (base %.3f)  marg %s' % (
        '\$d'.split('/')[-2], r['iter'],
        r.get('failure_heldout/solved_rate',0), r.get('base_failure_heldout/solved_rate',0),
        r.get('success_heldout/solved_rate',0), r.get('base_success_heldout/solved_rate',0),
        round(r.get('marginal/delta_vs_base',0),4)))"
  done
EOF
