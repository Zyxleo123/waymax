#!/bin/bash
# Submit FM-ES fail1500 with deterministic WM and multifuture WM at it6 / rs=40.
# FM setting: t_start=0.2, iters=6, resample_num_steps=40, fm_num_steps=50, pop=64, elite=8
# WM ckpts = slurm_es_failure_subset_1500.sh defaults (same as the Diffusion-ES WM runs).
#
# CV at this setting already exists (1500/1500) in
#   Flow_matching_sim/es_fm_cv_fail1500_it6_rs40_p64_e8/
# so it is skipped unless INCLUDE_CV=1.
#
# Usage (from rhea or lov4, NO --export; settings are passed via the environment):
#   bash slurm_submit_es_fm_fail1500_cv_wm_it6_rs40.sh
#   INCLUDE_CV=1 bash slurm_submit_es_fm_fail1500_cv_wm_it6_rs40.sh
#
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
SCRIPT="${ROOT}/legacy/sim/slurm_es_failure_subset_1500.sh"
LOG_DIR="/zfsauton/scratch/sbellad/Flow_matching_sim/logs"
mkdir -p "${LOG_DIR}"

export NUM_ES_ITERATIONS=6
export RESAMPLE_NUM_STEPS=40
export POPULATION_SIZE=64
export ELITE_SIZE=8
unset RUN_SUFFIX

echo "Submitting FM-ES fail1500 (it6, rs=40, pop 64 / elite 8, t=0.2):"

if [[ "${INCLUDE_CV:-0}" == "1" ]]; then
  jid_cv=$(sbatch --parsable --job-name=es_fm_cv_it6_rs40 \
    --output="${LOG_DIR}/es_fm_cv_fail1500_it6_rs40_%j.out" --error="${LOG_DIR}/es_fm_cv_fail1500_it6_rs40_%j.err" \
    "${SCRIPT}" es_fm cv)
  echo "  CV     -> ${jid_cv}  (Flow_matching_sim/es_fm_cv_fail1500_it6_rs40_p64_e8/)"
fi

jid_det=$(sbatch --parsable --job-name=es_fm_wm_det_it6_rs40 \
  --output="${LOG_DIR}/es_fm_wm_det_fail1500_it6_rs40_%j.out" --error="${LOG_DIR}/es_fm_wm_det_fail1500_it6_rs40_%j.err" \
  "${SCRIPT}" es_fm wm_det)
echo "  WM det -> ${jid_det}  (Flow_matching_sim/es_fm_wm_det_epoch0500_fail1500_it6_rs40_p64_e8/)"

jid_mf=$(sbatch --parsable --job-name=es_fm_wm_mf_it6_rs40 \
  --output="${LOG_DIR}/es_fm_wm_mf_fail1500_it6_rs40_%j.out" --error="${LOG_DIR}/es_fm_wm_mf_fail1500_it6_rs40_%j.err" \
  "${SCRIPT}" es_fm wm_mf expected_independent)
echo "  WM mf  -> ${jid_mf}  (Flow_matching_sim/es_fm_wm_mf_epoch0450_expected_independent_fail1500_it6_rs40_p64_e8/)"

echo
squeue -u "${USER}" -o '%.18i %.12P %.36j %.2t %.30R'
