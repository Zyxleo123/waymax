#!/bin/bash
# Barrier shape A/B on the arms that worked. Only calls sbatch.
#
# QUESTION. The safety barrier `tau*softplus((margin-c)/tau)` has INFINITE
# support, and it is summed over 4 ego corners x every valid agent x every
# scored step. A corner or an agent comfortably clear of its margin still
# contributes value and gradient, so part of every scene's safety loss is
# charged for the scene's traffic density and length rather than for anything
# the policy did. `hinge2` -- the quadratic-smoothed hinge -- keeps the knee
# width, the unbounded ramp and the C^1 smoothness and drops exactly that tail:
# zero value AND zero gradient at or beyond the margin.
#
# Everything else is the preserve/stopping/proposal protocol, unchanged, so
# these curves compare against every sweep since 09-09.
#
# ARMS (2 shapes x 2 proposals x 3 seeds = 12 tasks, 150 iters):
#   tf1_sp_s{0,1,2}    teacher-forced K=1, softplus   <- control (best proposal
#                                                        on record: fail 0.239,
#                                                        marginal -0.0026)
#   tf1_h2_s{0,1,2}    teacher-forced K=1, hinge2
#   ctrl_sp_s{0,1,2}   free-running, softplus         <- control (the arm every
#                                                        geometry/stopping sweep
#                                                        used)
#   ctrl_h2_s{0,1,2}   free-running, hinge2
# Both proposals are run because they put the safety term at very different
# states: `tf1` scores a noised HUMAN trajectory, which is mostly clear of both
# margins, so hinge2 can zero far more of its loss there than on the policy's
# own visited states. A result on one proposal alone could not tell a barrier
# effect from a proposal interaction.
#
# READ, in this order:
#   1. `tail_share` on the softplus arms at iter 0. It is the new per-iteration
#      metric: the fraction of the safety loss charged to geometry already clear
#      of the margin. THIS IS THE DECIDING NUMBER and it is available within one
#      iteration. If it is ~0.02, the two shapes are the same objective and a
#      null A/B is the expected result, not a failed experiment. If it is 0.2+,
#      a fifth of the term is density, and the arms should separate.
#      On the hinge2 arms it must read exactly 0.0 -- that is the flag taking.
#   2. `safety_last` / `edge_last` / `agent_last`: hinge2 removes the tail, so
#      its loss is strictly lower. Do NOT read that as an improvement; it is a
#      different constant. What matters is the split rates.
#   3. `failure_heldout` and `success_heldout` vs their `base_*` rows, per seed,
#      at MATCHED iterations -- and read the whole curve, not the endpoint. On a
#      non-monotonic curve a single late point ranks decay rates, not
#      configurations (the 09-09 ablation's whole lesson). Single-run
#      differences under ~0.03 are inside the measured seed spread.
#
# EXPECTATION, on record before the run: every geometry knob tried so far (tau,
# margin, both windows) moved the marginal by <=0.01, inside the noise, while
# the stopping point moved it by 0.03. `barrier_shape` is a knob of that same
# family and the honest prior is that it lands there too. What makes it worth a
# run anyway is that it is not a re-parameterisation of the same objective: it
# removes a term that exists in every scene whether or not anything is wrong,
# and `tail_share` says exactly how large that term is -- so this run returns a
# measured number even if the rates do not separate.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/sweep_barrier_shape.sh
#   DFT_ARMS="tf1_sp_s0 tf1_h2_s0" DFT_ITERS=3 DFT_EVAL_EVERY=1 DFT_SPLIT_CAP=8 \
#     bash rl/diffusion_ft/slurm/sweep_barrier_shape.sh      # smoke, ~15 min
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/barrier/${STAMP}}"
ENV_DIR="$BASE_OUT/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

VDIR=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777
SPLITS_JSON="${DFT_SPLITS3_JSON:-$VDIR/eval_splits_3strata.json}"
CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"

ITERS="${DFT_ITERS:-150}"
EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
KEEP_EVERY="${DFT_KEEP_EVERY:-10}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"

ALL_ARMS=()
for s in 0 1 2; do
  ALL_ARMS+=("tf1_sp_s$s|DFT_UNROLL_STEPS=1,DFT_BARRIER_SHAPE=softplus,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("tf1_h2_s$s|DFT_UNROLL_STEPS=1,DFT_BARRIER_SHAPE=hinge2,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("ctrl_sp_s$s|DFT_BARRIER_SHAPE=softplus,DFT_SEED=$s,DFT_DATA_SEED=$s")
  ALL_ARMS+=("ctrl_h2_s$s|DFT_BARRIER_SHAPE=hinge2,DFT_SEED=$s,DFT_DATA_SEED=$s")
done
# Not in the default set. The compact barrier reaches a distance shorter than
# the softplus's ~2.7*tau of effective reach, so `_wide` asks whether the two
# shapes agree once hinge2 is given back that reach through its margin -- i.e.
# whether what matters is the tail or the reach. Only worth submitting if the
# default pairs separate.
EXTRA_ARMS=()
for s in 0 1 2; do
  EXTRA_ARMS+=("tf1_h2wide_s$s|DFT_UNROLL_STEPS=1,DFT_BARRIER_SHAPE=hinge2,DFT_EDGE_MARGIN=0.7,DFT_AGENT_MARGIN=1.4,DFT_SEED=$s,DFT_DATA_SEED=$s")
  EXTRA_ARMS+=("tf1_hinge_s$s|DFT_UNROLL_STEPS=1,DFT_BARRIER_SHAPE=hinge,DFT_SEED=$s,DFT_DATA_SEED=$s")
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
GROUP="${DFT_WANDB_GROUP:-barrier_${STAMP}}"

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
    echo "DFT_UNROLL_GRAD_STEPS=1"
    echo "DFT_UNROLL_T_START=-1"
    echo "DFT_UNROLL_T_END=0"
    echo "DFT_UNROLL_ETA=0.0"
    echo "DFT_ANCHOR_STATES=policy"
    echo "DFT_TB_WEIGHT=0.0"
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
    # Overrides go LAST so an arm's DFT_BARRIER_SHAPE wins over the block above.
    IFS=',' read -ra KV <<< "$OVR"; for kv in "${KV[@]}"; do echo "$kv"; done
  } > "$ENV_DIR/${IDX_N}.env"
  printf '%2d  %-14s %s\n' "$IDX_N" "$TAG" "$OVR" | tee -a "$MANIFEST"
  IDX_N=$((IDX_N + 1))
done
echo

ARRAY_ID=$(sb --parsable ${DEP[@]+"${DEP[@]}"} "${SB_FLAGS[@]}" \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" --job-name="dft_barr" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_DIR")
echo "submitted array $ARRAY_ID  (${N} tasks, %${MAX_CONCURRENT})"
echo
echo "FIRST CHECK, at iter 0 of any arm:"
echo "  grep -h 'shape=' logs/dft/slurm-dft_barr-*.out | head    # the flag took"
echo "  grep -ho 'tail [0-9.]*' logs/dft/slurm-dft_barr-*.out | head"
echo "    softplus arms: how much of the safety loss is charged to geometry"
echo "                   already clear of the margin (the deciding number)"
echo "    hinge2 arms:   must be exactly 0.000"
