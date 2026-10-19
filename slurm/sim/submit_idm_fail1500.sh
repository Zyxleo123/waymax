#!/bin/bash
# fail1500 runs with reactive IDM other agents (Waymax default IDM settings).
# Submit from rhea or lov4 (sbatch is denied on lov1):
#
#   bash slurm/sim/submit_idm_fail1500.sh              # all 7 jobs
#   bash slurm/sim/submit_idm_fail1500.sh --dry_run    # print only
#
#   diffusion only               -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/diffusion_sim/diffusion_only_IDM_fail1500_seed0
#   diffusion-ES  gt/cv/wm_mf    -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_diffusion_IDM_*
#   flow-matching-ES gt/cv/wm_mf -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/es_fm_IDM_*  (it6/rs40;
#                                   override with FM_IT=4 FM_RS=20 bash ...)
# Extra args are forwarded to both ES submissions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DRY_RUN=0
ES_ARGS=()
for a in "$@"; do
  [[ "$a" == --dry_run ]] && DRY_RUN=1
  ES_ARGS+=("$a")
done

DIFF_LOG_DIR=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/diffusion_sim/logs_IDM
DIFF_CMD=(sbatch --job-name=diffusion_only_IDM_fail1500
  --output=${DIFF_LOG_DIR}/diffusion_only_IDM_fail1500_%j.out
  --error=${DIFF_LOG_DIR}/diffusion_only_IDM_fail1500_%j.err
  slurm/sim/run_diffusion_fail1500.sh --other_agents idm)
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[dry-run] ${DIFF_CMD[*]}"
else
  mkdir -p "${DIFF_LOG_DIR}"
  "${DIFF_CMD[@]}"
fi

bash slurm/sim/fail1500.sh --planner diffusion --forecast gt,cv,wm_mf --other_agents idm "${ES_ARGS[@]}"
bash slurm/sim/fail1500.sh --planner fm --forecast gt,cv,wm_mf --other_agents idm \
  --it "${FM_IT:-6}" --rs "${FM_RS:-40}" "${ES_ARGS[@]}"
