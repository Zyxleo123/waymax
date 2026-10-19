#!/bin/bash
# Submitter for the post-hoc reach/solved eval of a finished safety_train run.
# Only calls sbatch. Chains: GPU eval (scores base+ft, dumps driving artifacts)
# ->(afterok) CPU viz (reach_summary.png + reach_paths.png).
#
# The eval re-runs the run's OWN held-out batch: same shard_glob + eval_scenes +
# data_seed => the identical 32 scenes, so base-vs-ft is apples to apples and the
# ft numbers match what train_log.jsonl would have logged had reach been tracked.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea, so a
# compute node fails with "Access/permission denied"):
#
#   DFT_OUT_DIR=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_train/<STAMP> \
#     bash rl/diffusion_ft/slurm/submit_eval_reach.sh
#
# Knobs (optional): DFT_LOAD_CKPT (default <OUT_DIR>/latest), DFT_EVAL_SCENES
#   (held-out, default 32), DFT_EVAL_TRAIN (0/1, default 1 = also score a
#   train-pool batch), DFT_EVAL_TRAIN_SCENES (default 100), DFT_DATA_SEED,
#   DFT_SHARD_GLOB, DFT_VIZ_SCENES, and placement DFT_PARTITION DFT_QOS
#   DFT_NODELIST DFT_EXCLUDE. Model config (k_trainable, margins) is read from
#   <OUT_DIR>/latest/meta.json when present, else falls back to defaults.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR to a finished safety_train run dir}"
[[ -d "$DFT_OUT_DIR" ]] || { echo "no such dir: $DFT_OUT_DIR" >&2; exit 1; }
LOAD_CKPT="${DFT_LOAD_CKPT:-${DFT_OUT_DIR}/latest}"
META="${LOAD_CKPT}/meta.json"

# Pull model config from the checkpoint's meta.json so a non-default run
# (different k_trainable / margins) is scored with the actor it was trained with,
# not the defaults -- a leaf-count mismatch on load would otherwise abort.
read_meta() {  # key default
  local v=""
  if [[ -f "$META" ]]; then
    v=$(python3 -c "import json,sys; print(json.load(open('$META')).get('$1',''))" 2>/dev/null || true)
  fi
  echo "${v:-$2}"
}
K_TRAINABLE="${DFT_K_TRAINABLE:-$(read_meta k_trainable 10)}"
EDGE_MARGIN="${DFT_EDGE_MARGIN:-$(read_meta edge_margin_m 0.3)}"
AGENT_MARGIN="${DFT_AGENT_MARGIN:-$(read_meta agent_margin_m 0.75)}"
BASE_CKPT="${DFT_CKPT:-$(read_meta base_ckpt /data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest)}"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/evalreach_${STAMP}.env"
{
  echo "DFT_OUT_DIR=$DFT_OUT_DIR"
  echo "DFT_LOAD_CKPT=$LOAD_CKPT"
  echo "DFT_DUMP_DIR=${DFT_DUMP_DIR:-${DFT_OUT_DIR}/eval}"
  echo "DFT_CKPT=$BASE_CKPT"
  echo "DFT_SHARD_GLOB=${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-32}"
  echo "DFT_EVAL_TRAIN=${DFT_EVAL_TRAIN:-1}"
  echo "DFT_EVAL_TRAIN_SCENES=${DFT_EVAL_TRAIN_SCENES:-100}"
  [[ -n "${DFT_SPLITS_JSON:-}" ]] && echo "DFT_SPLITS_JSON=${DFT_SPLITS_JSON}"
  echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout}"
  echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-64}"
  echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
  echo "DFT_K_TRAINABLE=$K_TRAINABLE"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
  echo "DFT_EDGE_MARGIN=$EDGE_MARGIN"
  echo "DFT_AGENT_MARGIN=$AGENT_MARGIN"
  echo "DFT_REPLAN=${DFT_REPLAN:-10}"
  echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
  echo "DFT_SEED=${DFT_SEED:-0}"
  echo "DFT_VIZ_SCENES=${DFT_VIZ_SCENES:-16}"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

# Scrub SLURM_* so a submission from inside an allocation does not inherit it.
EVAL_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/eval_reach.sbatch "$ENV_FILE")
echo "eval   -> job $EVAL_ID  (GPU a6000/a5000)"

VIZ_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency="afterok:${EVAL_ID}" \
  rl/diffusion_ft/slurm/viz_reach.sbatch "$ENV_FILE")
echo "figures-> job $VIZ_ID  (CPU, afterok:${EVAL_ID})"
echo
echo "watch:  tail -f $ROOT/logs/dft/slurm-dft_evalreach-${EVAL_ID}.out"
echo "figs :  ${DFT_DUMP_DIR:-${DFT_OUT_DIR}/eval}/reach_summary_heldout.png , reach_paths_heldout.png"
echo "        (+ reach_summary_train.png , reach_paths_train.png when DFT_EVAL_TRAIN=1)"
