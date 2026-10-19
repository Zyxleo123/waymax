#!/bin/bash
# Submit FM-ES fail1500 with CV, deterministic WM and multifuture WM.
# FM setting = script defaults (same ES budget as Diffusion-ES):
#   t_start=0.2, iters=4, resample_num_steps=20, fm_num_steps=50, pop=64, elite=8
# WM ckpts = slurm_es_failure_subset_1500.sh defaults (same as the Diffusion-ES WM runs):
#   wm_det: world_model_20260912_170741/epoch_0500.pt
#   wm_mf : multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt, expected_independent
#
# Usage (from rhea or lov4, NO --export):
#   bash slurm_submit_es_fm_fail1500_cv_wm.sh
#
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
SCRIPT="${ROOT}/legacy/sim/slurm_es_failure_subset_1500.sh"
LOG_DIR="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/logs"
mkdir -p "${LOG_DIR}"

echo "Submitting FM-ES fail1500 (it4, rs=20, pop 64 / elite 8, t=0.2):"
echo "  CV     -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/es_fm_cv_fail1500_it4_rs20_p64_e8/"
echo "  WM det -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/es_fm_wm_det_epoch0500_fail1500_it4_rs20_p64_e8/"
echo "  WM mf  -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/es_fm_wm_mf_epoch0450_expected_independent_fail1500_it4_rs20_p64_e8/"
echo

jid_cv=$(sbatch --parsable --job-name=es_fm_cv_fail1500 \
  --output="${LOG_DIR}/es_fm_cv_fail1500_%j.out" --error="${LOG_DIR}/es_fm_cv_fail1500_%j.err" \
  "${SCRIPT}" es_fm cv)
echo "  CV     -> ${jid_cv}"

jid_det=$(sbatch --parsable --job-name=es_fm_wm_det_fail1500 \
  --output="${LOG_DIR}/es_fm_wm_det_fail1500_%j.out" --error="${LOG_DIR}/es_fm_wm_det_fail1500_%j.err" \
  "${SCRIPT}" es_fm wm_det)
echo "  WM det -> ${jid_det}"

jid_mf=$(sbatch --parsable --job-name=es_fm_wm_mf_fail1500 \
  --output="${LOG_DIR}/es_fm_wm_mf_fail1500_%j.out" --error="${LOG_DIR}/es_fm_wm_mf_fail1500_%j.err" \
  "${SCRIPT}" es_fm wm_mf expected_independent)
echo "  WM mf  -> ${jid_mf}"

echo
squeue -u "${USER}" -o '%.18i %.12P %.36j %.2t %.30R'
