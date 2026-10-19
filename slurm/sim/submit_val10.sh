#!/bin/bash
# All planners on WOMD validation shards 00000-00009 (2916 scenes; 2895 run, since
# each shard's leftover partial batch of 5 is skipped). 8 jobs per --other_agents value:
#
#   diffusion only                -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/diffusion_sim/diffusion_only_val10_seed0
#   flow matching only            -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/fm_only_val10_seed0
#   diffusion-ES  gt/cv/wm_mf     -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_diffusion_*_val10
#   flow-matching-ES gt/cv/wm_mf  -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/es_fm_*_val10_it6_rs40_p64_e8
#
# wm_mf = multi-future world model, expected-independence scoring. Submit from rhea or lov4:
#
#   bash slurm/sim/submit_val10.sh                         # log replay (8 jobs)
#   bash slurm/sim/submit_val10.sh --other_agents idm      # reactive IDM (8 jobs, names carry IDM)
#   bash slurm/sim/submit_val10.sh --other_agents log,idm  # both (16 jobs)
#   bash slurm/sim/submit_val10.sh --dry_run               # print only
#   FM_IT=4 FM_RS=20 bash slurm/sim/submit_val10.sh        # other FM-ES budget
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DRY_RUN=0
OTHER_AGENTS_LIST=log
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN=1; shift ;;
    --other_agents) OTHER_AGENTS_LIST="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
DRY_FLAG=()
(( DRY_RUN )) && DRY_FLAG=(--dry_run)

IFS=',' read -r -a OTHER_AGENTS_VALUES <<< "${OTHER_AGENTS_LIST}"
for other in "${OTHER_AGENTS_VALUES[@]}"; do
  for planner in diffusion fm; do
    args=(--subset val10 --planner "${planner}" --other_agents "${other}")
    info="$(bash slurm/sim/run_diffusion_fail1500.sh "${args[@]}" --print_paths)"
    run_name="$(sed -n 's/^RUN_NAME=//p' <<< "${info}")"
    log_dir="$(sed -n 's/^LOG_DIR=//p' <<< "${info}")"
    cmd=(sbatch --job-name="${run_name}"
         --output="${log_dir}/${run_name}_%j.out" --error="${log_dir}/${run_name}_%j.err"
         slurm/sim/run_diffusion_fail1500.sh "${args[@]}")
    if (( DRY_RUN )); then
      echo "[dry-run] ${run_name}"
      echo "    $(sed -n 's/^EXP_DIR=//p' <<< "${info}")"
      echo "    ${cmd[*]}"
    else
      mkdir -p "${log_dir}"
      "${cmd[@]}"
    fi
  done

  bash slurm/sim/val10.sh --planner diffusion --forecast gt,cv,wm_mf --other_agents "${other}" "${DRY_FLAG[@]}"
  bash slurm/sim/val10.sh --planner fm --forecast gt,cv,wm_mf --other_agents "${other}" \
    --it "${FM_IT:-6}" --rs "${FM_RS:-40}" "${DRY_FLAG[@]}"
done
