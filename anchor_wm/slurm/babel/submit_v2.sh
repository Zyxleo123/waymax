#!/bin/bash
# AnchorWM v2 on babel, from scratch (no encoder warm start), as one dependency chain:
#   1. cache    array job (CPU, preempt partition): WOMD training tfrecords -> 1000 npz shards
#   2. prep     anchors (k-means) + profiling of both scorer losses -> step count for TARGET_HOURS
#   3. train    two joint runs, --scorer_loss current and simple, started once prep has succeeded
# Run on the login node:  bash anchor_wm/slurm/babel/submit_v2.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
mkdir -p "${LOG_DIR}"

export NUM_TASKS="${NUM_TASKS:-50}" EXPECTED_SHARDS="${EXPECTED_SHARDS:-1000}"
export TARGET_HOURS="${TARGET_HOURS:-14}" NUM_WORKERS="${NUM_WORKERS:-12}"
# Same flags as train.sh RECIPE=v2, plus the from-scratch encoder learning rate and a later joint ramp
# (the scratch decoder needs longer before its forecasts mean anything to the scorer).
export SCRATCH_ARGS="--encoder_lr 3e-4 --joint_warmup_steps 10000 --joint_ramp_steps 10000"
export V2_ARGS="--goal_features --route_features --rule_features --input_heading --aux_heads \
--scorer_agents 32 --scorer_modes 6 --scorer_near_agents 8 --scorer_near_modes 12 --time_remaining \
--anchor_steps 20 30 40 50 60 70 80 --interaction_weight kernel --sigma_eps 0.002 --no_decay_norm_bias ${SCRATCH_ARGS}"

# BUILD_CACHE=0 when CACHE_DIR is already complete (prep.sh still checks the shard count).
prep_dep=()
if [[ "${BUILD_CACHE:-1}" == "1" ]]; then
  cache_job=$(sbatch --parsable -p preempt --qos=preempt_cpu_qos --requeue --array=0-$((NUM_TASKS - 1)) \
    --cpus-per-task=4 --mem=24G --time=02:00:00 -J awm_cache -o "${LOG_DIR}/cache_%A_%a.out" "${HERE}/cache.sh")
  echo "cache  ${cache_job}"
  prep_dep=(--dependency=afterok:"${cache_job}")
fi

prep_job=$(sbatch --parsable -p general --qos=normal --gres="${GPU_GRES}" --cpus-per-task=16 --mem=96G \
  --time=03:00:00 -J awm_prep -o "${LOG_DIR}/prep_%j.out" "${prep_dep[@]}" "${HERE}/prep.sh")
echo "prep   ${prep_job}"

# wandb: online if credentials exist when the job starts (WANDB_API_KEY or `wandb login`), else offline
# (sync later with `wandb sync <run dir>/wandb/offline-run-*`).
for loss in current simple; do
  train_job=$(SCORER_LOSS="${loss}" RECIPE=v2 ARM=joint WARM_START=0 RUN_NAME="v2_scratch_joint_${loss}" \
    MAX_HOURS="$(python3 -c "print(${TARGET_HOURS} + 1.5)")" EXTRA_ARGS="${SCRATCH_ARGS}" \
    sbatch --parsable -p general --qos=normal --gres="${GPU_GRES}" --cpus-per-task=16 --mem=96G \
    --time=17:30:00 -J "awm_v2_${loss}" -o "${LOG_DIR}/train_${loss}_%j.out" -e "${LOG_DIR}/train_${loss}_%j.err" \
    --dependency=afterok:"${prep_job}" --wrap 'if [[ -n "${WANDB_API_KEY:-}" ]] || grep -qs api.wandb.ai ~/.netrc; then export WANDB_MODE=online; else export WANDB_MODE=offline; fi; echo "wandb ${WANDB_MODE}"; exec bash "${ROOT}/anchor_wm/slurm/train.sh"')
  echo "train  ${train_job}  (${loss})"
done
