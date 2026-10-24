#!/bin/bash
# Submitter for the V-Max SAC hardness-balanced fine-tune pipeline. Only calls
# sbatch -- no work of its own. Chains four stages with afterok:
#
#   [offsets] -> rescore (GPU) -> refit (CPU) -> train (GPU)
#
# The offsets stage (byte-offset sidecar index over the WOMD shards) is a one-time
# prerequisite and is submitted ONLY when the index files are missing.
#
# Config is by environment variable; defaults live in _preamble.sh. Override by
# exporting before calling, e.g.:
#   VMH_ALPHA=0.5 VMH_TOTAL_TIMESTEPS=8000000 bash rl/vmax_hardness/slurm/submit.sh
#
# Never uses `sbatch --export` (requeue-holds on this cluster): one timestamped
# env file is written and passed as a positional argument to each stage.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
SLURM_DIR="$ROOT/rl/vmax_hardness/slurm"
ENV_DIR="$ROOT/logs/vmax_hardness/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/vmax_hardness"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/vmh_${STAMP}.env"

# Persist any overrides the caller exported, so all stages read identical config.
: > "$ENV_FILE"
for v in PYTHON VMH_RUN_DIR VMH_BEST_CKPT VMH_SHARD_GLOB VMH_INDEX_DIR VMH_OUT \
         VMH_RESCORE_DIR VMH_RISK_OUT VMH_EXPERT_DIR VMH_DESC_DIR \
         VMH_N_SHARDS VMH_PER_SHARD VMH_MAX_NUM_OBJECTS VMH_NUM_PATHS VMH_NUM_POINTS_PER_PATH \
         VMH_FIT_SLICE VMH_ALPHA VMH_WEIGHT_CLIP VMH_EXCLUDE_DIRTY VMH_TOTAL_TIMESTEPS \
         VMH_SAVE_ROOT VMH_NAME_RUN VMH_WANDB VMH_WANDB_PROJECT; do
  if [[ -n "${!v:-}" ]]; then printf '%s=%q\n' "$v" "${!v}" >> "$ENV_FILE"; fi
done
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

INDEX_DIR="${VMH_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
SHARD_GLOB="${VMH_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*-of-01000}"

# Clean environment so no SLURM_* from an enclosing allocation leaks into sbatch.
submit() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }

DEP=""
if ! compgen -G "$INDEX_DIR/*.offsets.npy" > /dev/null; then
  echo "WOMD byte-offset indexes missing under $INDEX_DIR -> submitting one-time build."
  OFF=$(submit --parsable --array=0-19 \
        "$ROOT/rl/diffusion_ft/slurm/build_womd_offsets.sbatch" \
        "DFT_INDEX_DIR=$INDEX_DIR" "DFT_SHARD_GLOB=$SHARD_GLOB")
  echo "  offsets (array) job: $OFF"
  DEP="afterok:$OFF"
fi

RESCORE=$(submit --parsable ${DEP:+--dependency=$DEP} "$SLURM_DIR/rescore.sbatch" "$ENV_FILE")
echo "rescore  (GPU) job: $RESCORE"
REFIT=$(submit --parsable --dependency="afterok:$RESCORE" "$SLURM_DIR/refit.sbatch" "$ENV_FILE")
echo "refit    (CPU) job: $REFIT"
TRAIN=$(submit --parsable --dependency="afterok:$REFIT" "$SLURM_DIR/train.sbatch" "$ENV_FILE")
echo "train    (GPU) job: $TRAIN"
echo
echo "Chain: ${DEP:+offsets -> }rescore($RESCORE) -> refit($REFIT) -> train($TRAIN)"
