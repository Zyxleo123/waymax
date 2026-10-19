#!/bin/bash
# Submit FM-ES fail1500 GT with rs=5 (Diffusion-matched), pop=128, elite=16.
# Two variants: it4 and it6. Does not touch existing fail1500 runs.
#
# Usage:
#   bash slurm_submit_es_fm_fail1500_rs5_p128.sh
#
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
SCRIPT_IT4="${ROOT}/legacy/sim/slurm_es_fm_fail1500_it4_rs5_p128.sh"
SCRIPT_IT6="${ROOT}/legacy/sim/slurm_es_fm_fail1500_it6_rs5_p128.sh"
LOG_DIR="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/logs"
mkdir -p "${LOG_DIR}"

for s in "${SCRIPT_IT4}" "${SCRIPT_IT6}"; do
  if [[ ! -f "${s}" ]]; then
    echo "ERROR: missing ${s}"
    exit 1
  fi
done

echo "Submitting FM-ES fail1500 GT:"
echo "  it4_rs5_p128_e16: iters=4 resample=5 pop=128 elite=16 t=0.2"
echo "  it6_rs5_p128_e16: iters=6 resample=5 pop=128 elite=16 t=0.2"
echo "  Outputs:"
echo "    /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/es_fm_gt_fail1500_it4_rs5_p128_e16/"
echo "    /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/es_fm_gt_fail1500_it6_rs5_p128_e16/"
echo

jid_it4=$(sbatch --parsable \
  --job-name=es_fm_gt_fail1500_it4_rs5_p128_e16 \
  --output="${LOG_DIR}/es_fail1500_it4_rs5_p128_%j.out" \
  --error="${LOG_DIR}/es_fail1500_it4_rs5_p128_%j.err" \
  "${SCRIPT_IT4}" gt)
echo "  it4 GT -> ${jid_it4}"

jid_it6=$(sbatch --parsable \
  --job-name=es_fm_gt_fail1500_it6_rs5_p128_e16 \
  --output="${LOG_DIR}/es_fail1500_it6_rs5_p128_%j.out" \
  --error="${LOG_DIR}/es_fail1500_it6_rs5_p128_%j.err" \
  "${SCRIPT_IT6}" gt)
echo "  it6 GT -> ${jid_it6}"

echo
squeue -u "${USER}" -o '%.18i %.12P %.36j %.2t %.30R'
echo
echo "Expect PD (Priority) or R — NOT 'user env retrieval failed'."
