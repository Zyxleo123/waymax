#!/bin/bash
# After the cache: check it is complete, fit the k-means anchors, then profile one v2 joint step per scorer
# loss (from scratch) and write the step count that fits TARGET_HOURS of training to MAX_STEPS_FILE.
# Submitted by submit_v2.sh.
set -euo pipefail
source "${ROOT}/anchor_wm/slurm/common.sh"

n=$(ls "${CACHE_DIR}"/*.sim_state_cache.npz | wc -l)
echo "cache shards: ${n} (expected ${EXPECTED_SHARDS})"
if (( n < EXPECTED_SHARDS )); then echo "ERROR: cache incomplete" >&2; exit 1; fi

mkdir -p "$(dirname "${ANCHORS}")"
if [[ -f "${ANCHORS}" ]]; then
  echo "anchors exist: ${ANCHORS}"
else
  python -m anchor_wm.anchors --cache_dir "${CACHE_DIR}" --out "${ANCHORS%.npz}" --num_anchors 64
fi
cat "${ANCHORS%.npz}.json"

best=0
for loss in current simple; do
  # shellcheck disable=SC2086
  out=$(python -m anchor_wm.train --cache_dir "${CACHE_DIR}" --anchors "${ANCHORS}" --save_dir "${OUT_ROOT}/runs" \
        --run_name "profile_${loss}" --profile_steps 60 --wandb_mode disabled --num_workers "${NUM_WORKERS}" \
        --joint_grad_scale 1 --joint_warmup_steps 0 --joint_ramp_steps 1 ${V2_ARGS} --scorer_loss "${loss}" \
        | grep '^PROFILE ')
  echo "${loss}: ${out}"
  s=$(python -c "import json,sys; print(json.loads(sys.argv[1][8:])['total_s_per_step'])" "${out}")
  best=$(python -c "print(max(${best}, ${s}))")
done
# 10% margin for validation passes and checkpoints; round down to a multiple of 1000.
steps=$(python -c "print(int(${TARGET_HOURS} * 3600 / (${best} * 1.10)) // 1000 * 1000)")
echo "slowest s/step ${best} -> ${steps} steps for ${TARGET_HOURS} h"
echo "${steps}" > "${MAX_STEPS_FILE}"
