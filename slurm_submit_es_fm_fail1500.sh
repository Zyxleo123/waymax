#!/bin/bash
# Submit FM-ES fail1500 GT + CV the same way Phase-2 probe jobs succeed:
# absolute script path, explicit --output/--error, NO --export flags.
#
# Usage (from rhea or lov4):
#   bash slurm_submit_es_fm_fail1500.sh
#
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
SCRIPT="${ROOT}/slurm_es_failure_subset_1500.sh"
LOG_DIR="/zfsauton/scratch/sbellad/Flow_matching_sim/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT}"
  exit 1
fi

echo "Cancelling any held es_fm_*_fail1500 jobs (if present)..."
# Only cancel held fail1500 FM jobs; leave probe jobs alone.
squeue -u "${USER}" -h -o '%i %T %r %j' | while read -r jid state reason name; do
  case "${name}" in
    es_fm_gt_fail1500|es_fm_cv_fail1500)
      if [[ "${reason}" == *user_env_retrieval* ]] || [[ "${state}" == "PENDING" && "${reason}" == *held* ]]; then
        echo "  scancel ${jid} (${name}: ${reason})"
        scancel "${jid}" || true
      fi
      ;;
  esac
done

echo
echo "Submitting (no --export)..."
jid_gt=$(sbatch --parsable \
  --job-name=es_fm_gt_fail1500 \
  --output="${LOG_DIR}/es_fail1500_%j.out" \
  --error="${LOG_DIR}/es_fail1500_%j.err" \
  "${SCRIPT}" es_fm gt)
echo "  GT -> ${jid_gt}"

jid_cv=$(sbatch --parsable \
  --job-name=es_fm_cv_fail1500 \
  --output="${LOG_DIR}/es_fail1500_%j.out" \
  --error="${LOG_DIR}/es_fail1500_%j.err" \
  "${SCRIPT}" es_fm cv)
echo "  CV -> ${jid_cv}"

echo
squeue -u "${USER}" -o '%.18i %.12P %.22j %.2t %.30R'
echo
echo "Expect PD (Priority) or R — NOT 'user env retrieval failed'."
