#!/bin/bash
#SBATCH --job-name=awm_cache
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-21
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/cache_bank_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/cache_bank_%A_%a.err
#
# AnchorWM inputs + per-candidate per-step geometry for the v3 discriminator's ES bank
# (anchor_wm/cache_bank.py), CPU only. One array task per shard of the sorted batch files;
# resubmitting skips finished files.
#
#   SMOKE=1 sbatch --array=0 --time=00:30:00 anchor_wm/slurm/cache_bank.sh     # 2 files + label check
#   sbatch anchor_wm/slurm/cache_bank.sh                                       # all 1,260 files, 22 shards
set -euo pipefail
source /zfsauton/scratch/ishanj/waymax_rs/anchor_wm/slurm/common.sh

SET="${SET:-fail1500_plus_womd_train_s02_11}"
BANK_DIR="${BANK_DIR_V3:-/zfsauton/scratch/sbellad/es_discriminator/training_sets/${SET}/es_candidates}"
TF_DIR="${TF_DIR:-/zfsauton/datasets/womd/tf_example/training}"
CACHE_OUT="${CACHE_OUT:-${OUT_ROOT}/bank_cache/${SET}}"
NUM_SHARDS="${NUM_SHARDS:-22}"
EXTRA=()
# BY_TFRECORD=1: remaining files grouped by source tfrecord, one tfrecord pass per unit (much less I/O).
if [[ "${BY_TFRECORD:-0}" == "1" ]]; then EXTRA+=(--by_tfrecord --files_per_unit "${FILES_PER_UNIT:-24}"); fi
if [[ -n "${ONLY_FILES:-}" ]]; then EXTRA+=(--only_files "${ONLY_FILES}"); fi
# BASE_SHARDS="12,13,...": only the files of these shards of an earlier BASE_NUM_SHARDS split.
if [[ -n "${BASE_SHARDS:-}" ]]; then EXTRA+=(--base_shards "${BASE_SHARDS}" --base_num_shards "${BASE_NUM_SHARDS:-22}"); fi
SHARD_OFFSET="${SHARD_OFFSET:-0}"     # lets two arrays (e.g. debug + cpu) split one NUM_SHARDS plan
if [[ "${SMOKE:-0}" == "1" ]]; then
  CACHE_OUT="${CACHE_OUT}_smoke"; NUM_SHARDS=1; EXTRA=(--max_files 2 --check)
fi
mkdir -p "${CACHE_OUT}"
python -m anchor_wm.cache_bank --bank_dir "${BANK_DIR}" --tfrecord_dir "${TF_DIR}" --out_dir "${CACHE_OUT}" \
  --shard "$(( ${SLURM_ARRAY_TASK_ID:-0} + SHARD_OFFSET ))" --num_shards "${NUM_SHARDS}" \
  --threads "${SLURM_CPUS_PER_TASK:-16}" "${EXTRA[@]}"
