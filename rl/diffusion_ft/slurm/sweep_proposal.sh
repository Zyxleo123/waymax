#!/bin/bash
# Where the safety loss is evaluated: five proposals, one protocol. Only calls sbatch.
#
# QUESTION. Every safety run so far scores the endpoint of a FREE-RUNNING 100-step
# sample taken at the policy's own visited state. That is faithful and expensive,
# and only 10 of the 100 evaluations carry a gradient. Two cheaper objects are
# available, and they differ in WHICH state the denoiser is corrected at:
#
#   teacher-forced (--unroll_steps K)   state = q_sample(x0_expert, t): a noised
#       HUMAN trajectory. Cheap, and mismatched -- the sampler's own x_t at that
#       timestep is not drawn from this distribution (`unrolled_prediction`'s
#       "diffusion-state mismatch").
#   replayed transition (--tb_weight)   state = an x_t the sampler ACTUALLY passed
#       through, harvested from the collection rollout's trace (which already
#       computes it and was throwing it away) and replayed from a buffer. Same
#       cost per gradient, no mismatch, at the price of staleness.
#
# Crossed with what is scored at that state: the one-step clean prediction
# x0_hat (K=1, a posterior MEAN -- the Jensen gap `unrolled_prediction` argues
# about) or the clean prediction after a short unroll toward a sample (K=4).
#
# ARMS (each x 3 seeds; 150 iters, eval+keep every 10, eval seed 12345, k=10,
# base anchor beta=100 success-only, 50/50 failure/success batches -- the
# preserve/stopping-sweep protocol exactly, so curves compare across sweeps):
#   ctrl   free-running 100-step sample, the current default        (the control)
#   tf1    teacher-forced, K=1: score x0_hat(q_sample(x0_expert,t))
#   tf4    teacher-forced, K=4: unroll, then score the clean prediction
#   tb1    replayed transition, K=1: score x0_hat(x_t_stored, t)
#   tb4    replayed transition, K=4: unroll from x_t_stored, then score
# The tb arms run with --safety_weight 0, so their geometric gradient comes from
# the buffer ALONE and the comparison is about the proposal, not about a blend.
# The rollout still runs in those arms -- it is what fills the buffer, and it is
# what the roll_* metrics report.
#
# READ, in this order:
#   1. iter-0 .out prints '[safety] TRANSITION REPLAY ...' (tb arms) and
#      '[safety] proposal=teacher-forced unroll' (tf arms). If it prints
#      'proposal=free-running' on a tf arm the flag did not take.
#   2. tb arms: `tb_size` must climb to capacity and `tb_age` settle. tb_age
#      rising without bound means the buffer is not being refilled and the
#      gradient is being taken at states an old policy visited.
#   3. `tb_share` / `safety_last` for the term sizes, then marginal/delta_vs_base
#      per seed at matched iterations. Single-run differences under ~0.03 are
#      noise (measured seed spread), so read the 3 seeds together.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/sweep_proposal.sh
#   DFT_ARMS="tb1_s0" DFT_ITERS=3 DFT_EVAL_EVERY=1 DFT_SPLIT_CAP=8 bash rl/diffusion_ft/slurm/sweep_proposal.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/proposal/${STAMP}}"
ENV_DIR="$BASE_OUT/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

VDIR=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777
SPLITS_JSON="${DFT_SPLITS3_JSON:-$VDIR/eval_splits_3strata.json}"
CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"

ITERS="${DFT_ITERS:-150}"
EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
KEEP_EVERY="${DFT_KEEP_EVERY:-10}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"

# K for the unrolled arms. 4 sits inside the trainable window (t <= k-1 = 9) with
# room for t_start ~ U[1, 9]; the trainer rejects a t_start above it.
KU="${DFT_SWEEP_K:-4}"

ALL_ARMS=()
for s in 0 1 2; do
  ALL_ARMS+=("ctrl_s$s|DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("tf1_s$s|DFT_UNROLL_STEPS=1,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("tf${KU}_s$s|DFT_UNROLL_STEPS=$KU,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("tb1_s$s|DFT_TB_WEIGHT=1.0,DFT_SAFETY_WEIGHT=0,DFT_TB_UNROLL_STEPS=1,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("tb${KU}_s$s|DFT_TB_WEIGHT=1.0,DFT_SAFETY_WEIGHT=0,DFT_TB_UNROLL_STEPS=$KU,DFT_SEED=$s,DFT_DATA_SEED=$s")
done
# Not in the default set. `tbmix` keeps the on-policy term AND the buffer (the
# hybrid); `tbfresh` is the on-policy control for the buffer -- capacity == batch
# makes every drawn transition one this iteration produced, so a difference
# between it and tb1 is staleness and nothing else.
EXTRA_ARMS=()
for s in 0 1 2; do
  EXTRA_ARMS+=("tbmix_s$s|DFT_TB_WEIGHT=1.0,DFT_TB_UNROLL_STEPS=1,DFT_SEED=$s,DFT_DATA_SEED=$s")
  EXTRA_ARMS+=("tbfresh_s$s|DFT_TB_WEIGHT=1.0,DFT_SAFETY_WEIGHT=0,DFT_TB_UNROLL_STEPS=1,DFT_TB_CAPACITY=32,DFT_SEED=$s,DFT_DATA_SEED=$s")
done

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
GROUP="${DFT_WANDB_GROUP:-proposal_${STAMP}}"

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
SB_FLAGS=()
[[ -n "${DFT_EXCLUDE-gpu28}" ]] && SB_FLAGS+=(--exclude="${DFT_EXCLUDE-gpu28}")
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
    echo "DFT_UNROLL_GRAD_STEPS=1"
    echo "DFT_UNROLL_T_START=-1"
    echo "DFT_UNROLL_T_END=0"
    echo "DFT_UNROLL_ETA=0.0"
    echo "DFT_ANCHOR_STATES=policy"
    echo "DFT_TB_WEIGHT=0.0"
    echo "DFT_TB_CAPACITY=8192"
    echo "DFT_TB_BATCH=32"
    echo "DFT_TB_STORE_PER_REPLAN=1"
    echo "DFT_TB_MAX_USES=0"
    echo "DFT_TB_GRAD_STEPS=1"
    echo "DFT_TB_UNROLL_T_END=0"
    echo "DFT_TB_UNROLL_ETA=0.0"
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
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
    echo "# --- arm: $TAG ---"
    IFS=',' read -ra KV <<< "$OVR"; for kv in "${KV[@]}"; do echo "$kv"; done
  } > "$ENV_DIR/${IDX_N}.env"
  printf '%2d  %-12s %s\n' "$IDX_N" "$TAG" "$OVR" | tee -a "$MANIFEST"
  IDX_N=$((IDX_N + 1))
done
echo

ARRAY_ID=$(sb --parsable ${DEP[@]+"${DEP[@]}"} "${SB_FLAGS[@]}" \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" --job-name="dft_prop" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_DIR")
echo "submitted array $ARRAY_ID  (${N} tasks, %${MAX_CONCURRENT})"
echo "first check: each arm's iter-0 .out names its proposal; tb arms must show tb_size climbing"
