#!/bin/bash
# The BC ladder, run for DAYS. Same three settings as sweep_bc_weight.sh
# (bc_weight 0 / 100 / 300, safety-only otherwise); what changes is the schedule.
# Only calls sbatch.
#
# HOW A MULTI-DAY RUN IS SHAPED HERE. `general` caps a job at 2 days
# (MaxTime=2-00:00:00), so N days is a CHAIN of 2-day jobs, each resuming from
# `<out_dir>/latest`. The chain is a job array with `%1`: the tasks are
# guaranteed to run one at a time, in order, and each one resumes where the last
# stopped. That is one submission per arm rather than N, and a task that dies
# early just means the next one picks up from the last checkpoint.
#
# THE RESUME HAD TO BE FIXED FIRST (2026-09-08), and this is why it matters more
# here than anywhere else in this package:
#   * `save` now writes `opt_state.npz` too. Weights alone made every restart a
#     NEW run: Adam's moments and step counter reset, and bias correction makes
#     the first steps after a restart full-size whatever the gradient is (the
#     null objective moves drift 0.412 -> 0.717 m in ONE iteration, then only
#     -> 0.810 over the next 80). Over a 5-day chain that is 2-3 uncontrolled
#     kicks, in a project where displacement rank-orders damage.
#   * the sampler is seeded from the resume point, so a restarted run continues
#     the batch stream instead of re-drawing the batches it already trained on.
#   * the BC pass itself is cheap -- 9% of an iteration, measured.
#   * `--keep_every` writes un-overwritten snapshots. `latest` cannot answer
#     "the success split was best 2000 iterations ago, give me that model",
#     which over days is the question you will actually have.
#
# EVAL BUDGET IS THE REAL SCHEDULING DECISION. An eval point is
# cap x repeats x |splits| = 2346 scenes x repeats, ~13 min at repeats=1 and
# ~27 min at 2. At DFT_EVAL_EVERY=25 (the 300-iteration default) a multi-day run
# would spend HALF its wall clock evaluating. Defaults here: repeats 1,
# every 200 iterations -> ~18% overhead and ~36 eval points over 5 days.
# repeats=1 is not a compromise at this size: the 4-repeat default was set when
# a split was 48 scenes and its binomial floor was ~0.07; at 1000 scenes the
# split average already integrates sampler noise across scenes.
#
# Usage (from repo root, ON RHEA):
#   DFT_SKIP_VERIFY=1 bash rl/diffusion_ft/slurm/sweep_bc_long.sh
#   DFT_DAYS=10 DFT_ITERS=40000 bash rl/diffusion_ft/slurm/sweep_bc_long.sh
#   DFT_BC_LADDER=0,100 bash rl/diffusion_ft/slurm/sweep_bc_long.sh
#
# To STOP early: the arms checkpoint every DFT_SAVE_EVERY iterations, so
# `scancel` at any point leaves a resumable `latest` and a complete curve.
#
# Node: `general` a6000/a5000 only -- this env's jaxlib dies at cuDNN autotune on
# v100 and OOMs on the 2080 Ti. Do not widen the partition. The 24 GB a5000
# (gpu28) is the tight one: at DFT_EVAL_CHUNK=64 both arms that landed there died
# with a 1.84 GiB OOM at their first eval point while the a6000 arm ran fine
# (jobs 44862/44864 vs 44866). Fixed by chunk 32 + XLA_PYTHON_CLIENT_MEM_FRACTION
# 0.95 in the sbatch; if an OOM ever recurs, `DFT_EXCLUDE=gpu28` is the fallback.
# A crash is survivable here anyway -- the next array task resumes from `latest`.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

DFT_BC_LADDER="${DFT_BC_LADDER:-0,100,300}"
# Days of wall clock per arm, as a chain of 2-day array tasks (general's max).
DFT_DAYS="${DFT_DAYS:-6}"
CHUNK_H="${DFT_CHUNK_HOURS:-48}"
NTASKS=$(( (DFT_DAYS * 24 + CHUNK_H - 1) / CHUNK_H ))
# `--iters` is the ceiling the chain trains toward, not a promise: the last task
# simply stops when the wall clock runs out, and the curve is whatever it
# reached. ~16-20 s/iter plus eval => ~4000 iters/day.
DFT_ITERS="${DFT_ITERS:-24000}"

SPLITS_DIR="${DFT_SPLITS_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0}"
DFT_VERIFIED_JSON="${DFT_VERIFIED_JSON:-${SPLITS_DIR}/verified_seed777/eval_splits_verified.json}"
BASE_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bc_long/${STAMP}}"
GROUP="${DFT_WANDB_GROUP:-bc_long_${STAMP}}"
SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"

[[ -s "$DFT_VERIFIED_JSON" ]] || {
  echo "no verified splits at $DFT_VERIFIED_JSON -- run submit_verified_safety.sh first" >&2
  exit 1; }

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }

echo "sweep  -> $BASE_OUT"
echo "ladder -> bc_weight in {$DFT_BC_LADDER}"
echo "budget -> ${DFT_DAYS}d/arm = ${NTASKS} x ${CHUNK_H}h chained (array %1), iters ceiling $DFT_ITERS"
echo "splits -> $DFT_VERIFIED_JSON"
echo

IFS=',' read -ra RUNGS <<< "$DFT_BC_LADDER"
for BW in "${RUNGS[@]}"; do
  TAG="bc${BW}"
  OUT_DIR="$BASE_OUT/$TAG"
  ENV_T="$ENV_DIR/bclong_${STAMP}_${TAG}.env"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SHARD_GLOB=$SHARD_GLOB"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
    echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-16}"
    echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-16}"
    echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
    echo "DFT_SPLITS_JSON=$DFT_VERIFIED_JSON"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-1000}"
    echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-32}"
    echo "DFT_TRAIN_SPLITS=${DFT_TRAIN_SPLITS:-failure_train,success_train}"
    echo "DFT_TRAIN_SPLIT_CAP=-1"
    echo "DFT_TRAIN_SPLIT_RATIO=${DFT_TRAIN_SPLIT_RATIO:-1,1}"
    echo "DFT_PROBE_PER_SPLIT=${DFT_PROBE_PER_SPLIT:-6}"
    echo "DFT_PROBE_EVERY=${DFT_PROBE_EVERY:-200}"
    echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
    echo "DFT_ANCHOR_WEIGHT=0.0"
    echo "DFT_BC_WEIGHT=$BW"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    echo "DFT_ITERS=$DFT_ITERS"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    # see EVAL BUDGET above
    echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-200}"
    echo "DFT_EVAL_REPEATS=${DFT_EVAL_REPEATS:-1}"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-50}"
    echo "DFT_KEEP_EVERY=${DFT_KEEP_EVERY:-1000}"
    echo "DFT_GRAD_ACCUM=0"
    echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
    echo "DFT_EVAL_BASE=1"
    # Every task after the first MUST resume, or the chain silently restarts
    # training from the base checkpoint on each 2-day boundary.
    echo "DFT_RESUME=1"
    echo "DFT_SEED=${DFT_SEED:-0}"
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
  } > "$ENV_T"

  # %1: at most one task of this array runs at a time, so task k always starts
  # from task k-1's checkpoint. Without it they would run concurrently, three
  # processes writing one out_dir.
  TID=$(sb --parsable --array="0-$((NTASKS - 1))%1" --time="${CHUNK_H}:00:00" \
    "${SB_FLAGS[@]}" rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_T")
  echo "arm $TAG -> array $TID  ($NTASKS x ${CHUNK_H}h, %1)  $OUT_DIR"
  # afterany, not afterok: the last task is EXPECTED to be killed by the wall
  # clock, and afterok would then never draw the figures.
  sb --parsable --dependency="afterany:${TID}" \
    rl/diffusion_ft/slurm/viz_safety_train.sbatch "$ENV_T" >/dev/null
done
echo
echo "watch the ladder (any time -- the log is appended, not rewritten):"
echo "  python3 -c \"import json,glob"
echo "for d in sorted(glob.glob('$BASE_OUT/*/train_log.jsonl')):"
echo "    rows=[json.loads(l) for l in open(d) if 'failure_heldout/solved_rate' in l]"
echo "    print(d.split('/')[-2], [(r['iter'], round(r['failure_heldout/solved_rate'],3), round(r['success_heldout/solved_rate'],3)) for r in rows[::4]])\""
