#!/bin/bash
# Submit FM-ES fail1500 GT+CV with it6_rs40 (Phase-2 best on n=250).
# Does NOT touch the in-progress default it4_rs20 GT job.
#
# Usage (from rhea or lov4):
#   bash slurm_submit_es_fm_fail1500_it6_rs40.sh
#
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
SCRIPT="${ROOT}/slurm_es_fm_fail1500_it6_rs40.sh"
LOG_DIR="/zfsauton/scratch/sbellad/Flow_matching_sim/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT}"
  exit 1
fi

echo "Submitting FM-ES fail1500 it6_rs40 (t=0.2, iters=6, resample_steps=40, n=50)"
echo "  Outputs: ${LOG_DIR%/logs}/es_fm_{gt,cv}_fail1500_it6_rs40/"
echo "  (leaves existing es_fm_gt_fail1500 / es_fm_cv_fail1500 alone)"
echo

jid_gt=$(sbatch --parsable \
  --job-name=es_fm_gt_fail1500_it6_rs40 \
  --output="${LOG_DIR}/es_fail1500_it6_rs40_%j.out" \
  --error="${LOG_DIR}/es_fail1500_it6_rs40_%j.err" \
  "${SCRIPT}" gt)
echo "  GT -> ${jid_gt}"

jid_cv=$(sbatch --parsable \
  --job-name=es_fm_cv_fail1500_it6_rs40 \
  --output="${LOG_DIR}/es_fail1500_it6_rs40_%j.out" \
  --error="${LOG_DIR}/es_fail1500_it6_rs40_%j.err" \
  "${SCRIPT}" cv)
echo "  CV -> ${jid_cv}"

echo
squeue -u "${USER}" -o '%.18i %.12P %.28j %.2t %.30R'
echo
echo "Expect PD (Priority) or R — NOT 'user env retrieval failed'."
