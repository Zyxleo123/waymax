#!/bin/bash
# Re-evaluate the alpha-sweep arms offline with the FIXED eval.
#
# The 2026-08-20 eval_logs are invalid: the planner was pinned to the weights
# traced at the first eval (identical rollouts every epoch) and the 218-scene
# failure set parsed under the wrong route dims (loaded zero scenarios). Both are
# fixed (verify job 35289). This re-scores the saved epoch_0010 checkpoints --
# the fair equal-compute point available for all five arms -- WITHOUT retraining.
#
# Five sibling GPU eval jobs (each scores failures + womd), then one CPU
# aggregation job afterok that prints the comparison table.
#
# Usage (from the repo root, on rhea):
#     bash train_diffusion/hardness/slurm/submit_reeval.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

SWEEP="${SWEEP:-/zfsauton/scratch/yixiz/waymax_rs/hardness_sweep}"
RE_OUT="${RE_OUT:-/zfsauton/scratch/yixiz/waymax_rs/hardness_reeval}"

# arm -> the run whose epoch_0010 is the best (full-length 08-19 original).
declare -A RUN=(
  [control]="control/hardness_alpha0.0_dirty0_20260819_133502"
  [cleanonly]="cleanonly/hardness_alpha0.0_dirty1_20260819_133502"
  [alpha03]="alpha03/hardness_alpha0.3_dirty1_20260819_140423"
  [alpha05]="alpha05/hardness_alpha0.5_dirty1_20260819_142722"
  [alpha10]="alpha10/hardness_alpha1.0_dirty1_20260819_161305"
)
ARMS=(control cleanonly alpha03 alpha05 alpha10)

# Pre-flight every checkpoint HERE, not on the compute node: five siblings from
# one bad path is five queued jobs that each fail seconds in.
for arm in "${ARMS[@]}"; do
  ck="$SWEEP/${RUN[$arm]}/epoch_0010"
  if [[ ! -f "$ck/metadata.json" ]]; then
    echo "ERROR: $arm: no metadata.json under $ck" >&2
    exit 1
  fi
done
echo "all five epoch_0010 checkpoints OK"

ENV_DIR="$ROOT/logs/hardness/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

emit() { printf '%s=%q\n' "$1" "$2"; }
# Scrub SLURM_* so an in-allocation submit does not trip the user-env-retrieval
# requeue-held trap; pass the env file as a positional arg, never --export.
SUBMIT=(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch)

eval_jobs=()
for arm in "${ARMS[@]}"; do
  ck="$SWEEP/${RUN[$arm]}/epoch_0010"
  env_file="$ENV_DIR/reeval_${STAMP}_${arm}.env"
  {
    emit RE_ARM "$arm"
    emit RE_CKPT "$ck"
    emit RE_OUT "$RE_OUT"
    emit RE_FAIL_TFRECORD_DIR "${RE_FAIL_TFRECORD_DIR:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures}"
    emit RE_FAIL_NUM "${RE_FAIL_NUM:-218}"
    emit RE_WOMD_TFRECORD_DIR "${RE_WOMD_TFRECORD_DIR:-/zfsauton/scratch/eshau/womd/tf_example/training}"
    emit RE_WOMD_NUM "${RE_WOMD_NUM:-200}"
    emit RE_LANE_GRAPH_DIR "${RE_LANE_GRAPH_DIR:-/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"
    emit RE_NUM_WORLDS "${RE_NUM_WORLDS:-10}"
  } > "$env_file"
  job="$("${SUBMIT[@]}" --parsable --job-name="hrd_reeval_$arm" \
    train_diffusion/hardness/slurm/reeval_arm.sbatch "$env_file")"
  echo "submitted eval $arm: job $job  (env $env_file)"
  eval_jobs+=("$job")
done

# afterok on ALL five: the table should reflect a complete sweep, and any arm
# failing is worth noticing rather than silently averaging over four.
dep="afterok:$(IFS=:; echo "${eval_jobs[*]}")"
agg_env="$ENV_DIR/reeval_${STAMP}_agg.env"
{ emit RE_OUT "$RE_OUT"; } > "$agg_env"
agg="$("${SUBMIT[@]}" --parsable --dependency="$dep" --job-name=hrd_reeval_agg \
  train_diffusion/hardness/slurm/aggregate_reeval.sbatch "$agg_env")"
echo "submitted aggregate: job $agg  (afterok ${eval_jobs[*]})"
echo
echo "table will be at: $RE_OUT/comparison.json"
echo "and printed in:   logs/hardness/slurm-hrd_reeval_agg-$agg.out"
