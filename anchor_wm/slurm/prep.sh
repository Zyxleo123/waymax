#!/bin/bash
#SBATCH --job-name=awm_prep
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/prep_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/prep_%j.err
#
# Step 1 of the 1-day test: unit tests, k-means anchors (+ oracle coverage report) and the
# online-label vs Waymax-label agreement check on a few ES-bank files.
#
#   sbatch anchor_wm/slurm/prep.sh
#   NUM_ANCHORS=32 LABEL_FILES=5 sbatch anchor_wm/slurm/prep.sh
set -euo pipefail
source "$(dirname "$0")/common.sh" 2>/dev/null || source /zfsauton/scratch/ishanj/waymax_rs/anchor_wm/slurm/common.sh

NUM_ANCHORS="${NUM_ANCHORS:-64}"
LABEL_FILES="${LABEL_FILES:-3}"
mkdir -p "${OUT_ROOT}/anchors" "${OUT_ROOT}/label_check"

python -m unittest discover -s anchor_wm/tests -t . -v

if [[ -f "${OUT_ROOT}/anchors/anchors_k${NUM_ANCHORS}.npz" && "${REFIT:-0}" != "1" ]]; then
  echo "anchors exist (REFIT=1 to refit):"; cat "${OUT_ROOT}/anchors/anchors_k${NUM_ANCHORS}.json"
else
  python -m anchor_wm.anchors --cache_dir "${CACHE_DIR}" --out "${OUT_ROOT}/anchors/anchors_k${NUM_ANCHORS}" \
    --num_anchors "${NUM_ANCHORS}"
fi

python -m anchor_wm.label_check --bank_dir "${BANK_DIR}" --tfrecord_dir "${TFRECORDS}" \
  --max_files "${LABEL_FILES}" --out "${OUT_ROOT}/label_check/label_check_${SLURM_JOB_ID:-local}.json"
