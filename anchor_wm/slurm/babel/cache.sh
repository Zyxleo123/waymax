#!/bin/bash
# Array task: WOMD training tfrecords -> sim_state cache npz shards (anchor_wm.build_cache, single pass).
# Each task takes every NUM_TASKS-th file and skips shards that already exist, so a requeued or
# re-submitted task only fills what is missing. Submitted by submit_v2.sh.
set -euo pipefail
source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}"
python -m anchor_wm.build_cache --tfrecord_dir "${TF_ROOT}/training" --output_dir "${CACHE_DIR}" \
  --task_id "${SLURM_ARRAY_TASK_ID}" --num_tasks "${NUM_TASKS}"
