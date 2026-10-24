#!/bin/bash
# Experiment 2: protect successful SAMPLED trajectories directly. Only calls sbatch.
#
# QUESTION. The only do-no-harm lever that has worked is the frozen-base anchor,
# and it measures the wrong thing (one-step posterior means on noised HUMAN
# trajectories, at t over the whole schedule, on the fine-tune's drifted states;
# its logged share of the objective was ~1e-5). Does a term that protects what the
# base actually DRIVES when it succeeds -- same prefix, same noise, compare the
# executed continuations -- plus no barrier pressure on solved rows, stop the
# success-split damage that sinks every arm so far?
#
# ARMS (each x 3 seeds, the stopping-sweep protocol exactly: 150 iters, eval and
# keep every 10, eval seed 12345, base anchor beta=100 success-only, k=10):
#   srow      --safety_rows unsolved only                (attributes the mask)
#   pres3     unsolved + preserve_weight 3               (preserve ~ safety scale
#   pres30    unsolved + preserve_weight 30               at 0.5-1 m drift)
#   ctrl      the stopping-sweep `base` config, rerun    (NOT in the default set:
#             the 09-11 base_s{0,1,2} curves are the control; add it with
#             DFT_ARMS if the trainer change needs re-proving bit-for-bit)
# Preservation draws 16 success-pool scenes/iter: up to 8 replayed recently-broken
# successes, half the rest from the top risk quintile (runs/hardness/risk.npz).
#
# Eval adds intermittent_heldout (logged per split; the trainer's marginal stays
# the two-stratum one, so curves compare directly with the stopping sweep).
#
# READ: marginal/delta_vs_base per seed at matched iterations vs the stopping
# sweep's base_s*, then score the chosen snapshot per scene with
# submit_select_eval.sh (DFT_SELECT_CKPTS=...) for the paired recovered/broken
# split and the three-stratum marginal. Single-run differences under ~0.03 are noise.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/sweep_preserve.sh
#   DFT_ARMS="pres3_s0" DFT_ITERS=2 DFT_EVAL_EVERY=1 DFT_SPLIT_CAP=8 bash ...   # smoke
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/preserve/${STAMP}}"
ENV_DIR="$BASE_OUT/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

VDIR=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777
SPLITS_JSON="${DFT_SPLITS3_JSON:-$VDIR/eval_splits_3strata.json}"
CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
RISK="${DFT_PRESERVE_RISK:-$ROOT/runs/hardness/risk.npz}"
[[ -s "$RISK" ]] || { echo "FATAL: no risk table $RISK (set DFT_PRESERVE_RISK= to sample uniformly)" >&2; exit 1; }

ITERS="${DFT_ITERS:-150}"
EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
KEEP_EVERY="${DFT_KEEP_EVERY:-10}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"

ALL_ARMS=()
for s in 0 1 2; do
  ALL_ARMS+=("srow_s$s|DFT_SAFETY_ROWS=unsolved,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("pres3_s$s|DFT_SAFETY_ROWS=unsolved,DFT_PRESERVE_WEIGHT=3,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("pres30_s$s|DFT_SAFETY_ROWS=unsolved,DFT_PRESERVE_WEIGHT=30,DFT_SEED=$s,DFT_DATA_SEED=$s")
done
EXTRA_ARMS=("ctrl_s0|DFT_SEED=0,DFT_DATA_SEED=0" "ctrl_s1|DFT_SEED=1,DFT_DATA_SEED=1"
            "ctrl_s2|DFT_SEED=2,DFT_DATA_SEED=2")

WANT="${DFT_ARMS:-}"
ARMS=()
if [[ -z "$WANT" ]]; then ARMS=("${ALL_ARMS[@]}"); else
  for w in $WANT; do
    hit=""
    for a in "${ALL_ARMS[@]}" "${EXTRA_ARMS[@]}"; do
      [[ "${a%%|*}" == "$w" ]] && { ARMS+=("$a"); hit=1; break; }
    done
    [[ -n "$hit" ]] || { echo "FATAL: unknown arm '$w'" >&2; exit 1; }
  done
fi
N=${#ARMS[@]}
GROUP="${DFT_WANDB_GROUP:-preserve_${STAMP}}"

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
SB_FLAGS=()
[[ -n "${DFT_EXCLUDE-}" ]] && SB_FLAGS+=(--exclude="${DFT_EXCLUDE-}")
[[ -n "${DFT_TIME:-}" ]] && SB_FLAGS+=(--time="$DFT_TIME")

DEP=()
if [[ ! -s "$SPLITS_JSON" ]]; then
  BID=$(sb --parsable rl/diffusion_ft/slurm/build_intermittent.sbatch "DFT_SPLITS3_JSON=$SPLITS_JSON")
  DEP=(--dependency="afterok:${BID}")
  echo "build     -> job $BID (CPU) -> $SPLITS_JSON"
fi

echo "sweep     -> $BASE_OUT"
echo "arms      -> $N, at most $MAX_CONCURRENT at once; $ITERS iters, eval+keep every $EVAL_EVERY"
echo

MANIFEST="$BASE_OUT/arms.txt"; : > "$MANIFEST"
IDX_N=0
for spec in "${ARMS[@]}"; do
  TAG="${spec%%|*}"; OVR="${spec#*|}"; [[ "$OVR" == "$spec" ]] && OVR=""
  OUT_DIR="$BASE_OUT/$TAG"; mkdir -p "$OUT_DIR"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SHARD_GLOB=$GLOB"
    echo "DFT_INDEX_DIR=$IDX"
    echo "DFT_SCENES_PER_ITER=32"
    echo "DFT_EVAL_SCENES=16"
    echo "DFT_PREFETCH=1"
    echo "DFT_SPLITS_JSON=$SPLITS_JSON"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,intermittent_heldout}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-1000}"
    echo "DFT_EVAL_CHUNK=32"
    echo "DFT_TRAIN_SPLITS=failure_train,success_train"
    echo "DFT_TRAIN_SPLIT_CAP=-1"
    echo "DFT_TRAIN_SPLIT_RATIO=1,1"
    echo "DFT_PROBE_PER_SPLIT=0"
    echo "DFT_PROBE_EVERY=0"
    echo "DFT_SAFETY_WEIGHT=1.0"
    echo "DFT_ANCHOR_WEIGHT=0.0"
    echo "DFT_BC_WEIGHT=0.0"
    echo "DFT_BASE_ANCHOR_WEIGHT=100"
    echo "DFT_BASE_ANCHOR_SUCCESS_ONLY=1"
    echo "DFT_EDGE_MARGIN=0.3"
    echo "DFT_AGENT_MARGIN=0.75"
    echo "DFT_EDGE_TAU=0.15"
    echo "DFT_AGENT_TAU=0.25"
    echo "DFT_AGENT_GEOMETRY=exact"
    echo "DFT_N_DISCS=5"
    echo "DFT_ANCHOR_WINDOW=-1"
    echo "DFT_SAFETY_WINDOW=0"
    echo "DFT_K_TRAINABLE=10"
    echo "DFT_PREFIX_LEN=5"
    echo "DFT_SIGMA_FLOOR=1e-4"
    echo "DFT_COLLECT_STEPS=0"
    echo "DFT_EVAL_STEPS=0"
    echo "DFT_UNROLL_STEPS=0"
    echo "DFT_ANCHOR_STATES=policy"
    echo "DFT_ITERS=$ITERS"
    echo "DFT_EPOCHS=1"
    echo "DFT_MAX_REPLANS=8"
    echo "DFT_REPLAN=10"
    echo "DFT_LR=1e-5"
    echo "DFT_GRAD_ACCUM=0"
    echo "DFT_EVAL_EVERY=$EVAL_EVERY"
    echo "DFT_SAVE_EVERY=25"
    echo "DFT_KEEP_EVERY=$KEEP_EVERY"
    echo "DFT_EVAL_SEED=12345"
    echo "DFT_EVAL_REPEATS=1"
    echo "DFT_EVAL_BASE=1"
    echo "DFT_RESUME=1"
    echo "DFT_PRESERVE_RISK=$RISK"
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
    echo "# --- arm: $TAG ---"
    IFS=',' read -ra KV <<< "$OVR"; for kv in "${KV[@]}"; do echo "$kv"; done
  } > "$ENV_DIR/${IDX_N}.env"
  printf '%2d  %-11s %s\n' "$IDX_N" "$TAG" "$OVR" | tee -a "$MANIFEST"
  IDX_N=$((IDX_N + 1))
done
echo

ARRAY_ID=$(sb --parsable ${DEP[@]+"${DEP[@]}"} "${SB_FLAGS[@]}" \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" --job-name="dft_pres" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_DIR")
echo "submitted array $ARRAY_ID  (${N} tasks, %${MAX_CONCURRENT})"
echo "first check: iter-0 .out prints '[safety] PRESERVE ...' and rows carry preserve_rms_m > 0 after iter 1"
