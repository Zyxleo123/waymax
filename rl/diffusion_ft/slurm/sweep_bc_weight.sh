#!/bin/bash
# Does mixing the ORIGINAL pretraining loss into the safety objective stop the
# regression on the success splits? One arm per --bc_weight. Only calls sbatch.
#
# THE TERM. `--bc_weight` adds `diffusion.loss_per_example` -- the exact
# denoising loss the checkpoint was pretrained with -- computed on the HUMAN's
# states, from a second sampling-free pass over the same batch. It is NOT
# `--anchor_weight`, which calls the same function on POLICY-visited states with
# the human future relabeled from whatever drifted pose the policy reached. That
# relabeling is DAgger-shaped: from a drifted state the target demands an
# instantaneous lateral correction, and a model trained on it learns to emit
# exactly that. Measured, the anchor was a large weight change buying almost no
# behaviour (moves the weights most, the car least) and it was the worse arm on
# failures in the 09-04 probe sweep. The BC term trains where the target is
# feasible by construction, which is the only version of "stay on the pretraining
# manifold" that is actually the pretraining objective.
#
# WHAT THE ARMS ARE. The weight has to be read against the size of the term. At
# `anchor_weight=1` the anchor measured **0.66% of the objective**, i.e. the
# denoising loss is ~0.007 against a safety term of ~1. The gate's share ladder
# read 30 -> 17%, 100 -> 40%, 300 -> 67%. The BC loss is the same function on a
# different state distribution, so the same scale applies to within a factor.
# `bc_weight=0` is the control and is the arm submit_verified_safety.sh runs.
#
# Read the MEDIAN `bc_share` over the run, NOT its value at iter 0. Measured
# 2026-09-08 on the bc300 arm: `bc_last` is 0.058 at iter 0 and 0.0044 on the
# median, because the term is minimized within a few iterations -- so iter 0
# reads it at ~13x its steady size and said 0.95 where the truth is 0.57. On the
# median, weight 100 -> 0.30 of the objective and 300 -> 0.57, i.e. this ladder
# is correctly spread. ~0 would mean the arm is the control with extra compute;
# ~1 would mean the safety term is switched off.
#
# COST. The BC pass is 8 env steps against the policy pass's 8 x 100 denoiser
# evals: measured at 9% of an iteration (`bc_collect_s` 3.9 s of 43 s), not 2x.
#
# CHAIN: verify (GPU) -> rebuild (CPU) -> N training arms as SIBLINGS -> figures.
# The verify/rebuild stages are build-once; with DFT_VERIFIED_JSON pointing at a
# finished one, DFT_SKIP_VERIFY=1 fans the arms out immediately.
#
# Usage (from repo root, ON RHEA):
#   DFT_SKIP_VERIFY=1 bash rl/diffusion_ft/slurm/sweep_bc_weight.sh
#   DFT_BC_LADDER="0,100" bash rl/diffusion_ft/slurm/sweep_bc_weight.sh
#
# Node: `general` a6000/a5000 only -- this env's jaxlib dies at cuDNN autotune on
# v100 and OOMs on the 2080 Ti. Do not widen the partition. The 24 GB a5000
# (gpu28) is the tight one: at DFT_EVAL_CHUNK=64 both arms that landed there died
# with a 1.84 GiB OOM at their first eval point, while the a6000 arm ran to
# completion (jobs 44862/44864 vs 44866). Chunk 32 + XLA_PYTHON_CLIENT_MEM_FRACTION
# 0.95 (set in the sbatch) is the fix; DFT_EXCLUDE=gpu28 is the fallback.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

DFT_BC_LADDER="${DFT_BC_LADDER:-0,100,300}"
SPLITS_DIR="${DFT_SPLITS_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-${SPLITS_DIR}/eval_splits.json}"
VERIFY_DIR="${DFT_VERIFY_DIR:-${SPLITS_DIR}/verified_seed777}"
DFT_VERIFY_OUT="${DFT_VERIFY_OUT:-${VERIFY_DIR}/verify_rows.jsonl}"
DFT_VERIFIED_JSON="${DFT_VERIFIED_JSON:-${VERIFY_DIR}/eval_splits_verified.json}"
BASE_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bc_sweep/${STAMP}}"
GROUP="${DFT_WANDB_GROUP:-bc_weight_${STAMP}}"
SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }

echo "sweep  -> $BASE_OUT"
echo "ladder -> bc_weight in {$DFT_BC_LADDER}"
echo "splits -> $DFT_VERIFIED_JSON"
echo

# ---- verify + rebuild, once for every arm -----------------------------------
DEP=()
if [[ "${DFT_SKIP_VERIFY:-0}" != "1" ]]; then
  mkdir -p "$VERIFY_DIR"
  ENV_V="$ENV_DIR/bcsweep_${STAMP}_verify.env"
  {
    echo "DFT_OUT_DIR=$VERIFY_DIR"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SPLITS_JSON=$DFT_SPLITS_JSON"
    echo "DFT_SHARD_GLOB=$SHARD_GLOB"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
    echo "DFT_VERIFY_OUT=$DFT_VERIFY_OUT"
    echo "DFT_VERIFIED_JSON=$DFT_VERIFIED_JSON"
    echo "DFT_VERIFY_CAP=${DFT_VERIFY_CAP:--1,-1,-1,2500}"
    echo "DFT_VERIFY_REPEATS=${DFT_VERIFY_REPEATS:-4}"
    echo "DFT_VERIFY_SEED=${DFT_VERIFY_SEED:-777}"
    echo "DFT_EVAL_CHUNK=32"
    echo "DFT_HELDOUT_FAILURE=${DFT_HELDOUT_FAILURE:--1}"
    echo "DFT_HELDOUT_SUCCESS=${DFT_HELDOUT_SUCCESS:-1000}"
    echo "DFT_SPLIT_SEED=${DFT_SPLIT_SEED:-0}"
  } > "$ENV_V"
  VID=$(sb --parsable "${SB_FLAGS[@]}" \
    rl/diffusion_ft/slurm/verify_splits.sbatch "$ENV_V")
  echo "verify  -> job $VID  (GPU)"
  RID=$(sb --parsable --dependency="afterok:${VID}" \
    rl/diffusion_ft/slurm/rebuild_splits.sbatch "$ENV_V")
  echo "rebuild -> job $RID  (CPU, afterok:$VID)"
  DEP=(--dependency="afterok:${RID}")
else
  [[ -s "$DFT_VERIFIED_JSON" ]] || {
    echo "DFT_SKIP_VERIFY=1 but $DFT_VERIFIED_JSON does not exist" >&2; exit 1; }
  echo "skipping verify/rebuild; using $DFT_VERIFIED_JSON"
fi
echo

# ---- one training arm per rung, all siblings on the rebuild -----------------
IFS=',' read -ra RUNGS <<< "$DFT_BC_LADDER"
for BW in "${RUNGS[@]}"; do
  TAG="bc${BW}"
  OUT_DIR="$BASE_OUT/$TAG"
  ENV_T="$ENV_DIR/bcsweep_${STAMP}_${TAG}.env"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SHARD_GLOB=$SHARD_GLOB"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
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
    echo "DFT_PROBE_EVERY=${DFT_PROBE_EVERY:-25}"
    # the objective: safety + the pretraining loss, and NEVER the anchor -- two
    # BC-shaped terms at once would make neither attributable.
    echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
    echo "DFT_ANCHOR_WEIGHT=0.0"
    echo "DFT_BC_WEIGHT=$BW"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    echo "DFT_ITERS=${DFT_ITERS:-300}"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-25}"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_GRAD_ACCUM=0"
    echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
    echo "DFT_EVAL_REPEATS=${DFT_EVAL_REPEATS:-2}"
    echo "DFT_EVAL_BASE=1"
    echo "DFT_SEED=${DFT_SEED:-0}"
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
  } > "$ENV_T"
  TID=$(sb --parsable "${DEP[@]}" "${SB_FLAGS[@]}" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_T")
  echo "arm $TAG -> job $TID  ($OUT_DIR)"
  sb --parsable --dependency="afterok:${TID}" \
    rl/diffusion_ft/slurm/viz_safety_train.sbatch "$ENV_T" >/dev/null
done
echo
echo "read FIRST (iter 0 of any arm), before waiting hours for curves:"
echo "  grep -h 'bc_share\\|BALANCED' $BASE_OUT/*/train_log.jsonl | head"
echo "  python3 -c \"import json,glob;[print(d.split('/')[-2], [ (r['iter'], round(r.get('bc_share',0),3), round(r.get('bc_last',0),5)) for r in map(json.loads, open(d))][:3]) for d in sorted(glob.glob('$BASE_OUT/*/train_log.jsonl'))]\""
