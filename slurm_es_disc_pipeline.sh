#!/bin/bash
# Submits the discriminator pipeline on the cluster (run on rhea, not with sbatch):
# slurm_es_disc_collect.sh (array, one task per chunk of fail1500), then
# slurm_es_disc_train.sh once every collection task finished successfully (afterok).
# WORLD_MODEL_CKPT / OFFROAD_MODE / NUM_CHUNKS / RUN_NAME are passed on to both jobs.
#
#   bash slurm_es_disc_pipeline.sh            # full fail1500 run
#   bash slurm_es_disc_pipeline.sh --smoke    # 20 scenes of chunk 0, 1 training epoch
set -euo pipefail

SMOKE_ARGS=()
COLLECT_ARRAY=()
if [[ "${1:-}" == --smoke ]]; then
  SMOKE_ARGS=(--smoke)
  COLLECT_ARRAY=(--array=0)
elif [[ $# -gt 0 ]]; then
  echo "ERROR: unknown argument '$1'" >&2; exit 1
fi

cd /zfsauton/scratch/ishanj/waymax_rs
mkdir -p /zfsauton/scratch/ishanj/es_discriminator/logs
collect=$(sbatch --parsable ${COLLECT_ARRAY[@]+"${COLLECT_ARRAY[@]}"} slurm_es_disc_collect.sh ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"})
train=$(sbatch --parsable --dependency=afterok:"${collect}" slurm_es_disc_train.sh ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"})
echo "collect job ${collect} -> train job ${train} (afterok)"
