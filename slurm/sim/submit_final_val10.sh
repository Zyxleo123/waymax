#!/bin/bash
# Final paper runs on WOMD validation shards 00000-00009 -> /zfsauton/scratch/sbellad/final_runs
#
# Without IDM (log replay):
#   Diffusion only
#   Diffusion-ES + CV
#   Diffusion-ES + WM (expected_independent)
#   Diffusion-ES + WM + discriminator
#
# With IDM:
#   Diffusion only
#   Diffusion-ES + CV
#   Diffusion-ES + WM (expected_independent)
#
#   bash slurm/sim/submit_final_val10.sh --dry_run
#   bash slurm/sim/submit_final_val10.sh            # submit from rhea / lov4
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SCRATCH=/zfsauton/scratch/sbellad
EXP_ROOT=${SCRATCH}/final_runs
DISC_CKPT=${SCRATCH}/es_discriminator/safety_progress/safety_progress_full_gaussian_wm_20261013_090539_81674/best.pt
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN=1; shift ;;
    --exp_root) EXP_ROOT="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done

submit_diff() {
  local other="$1"
  local args=(--subset val10 --planner diffusion --other_agents "${other}" --exp_root "${EXP_ROOT}")
  local info run_name log_dir exp_dir
  info="$(bash slurm/sim/run_diffusion_fail1500.sh "${args[@]}" --print_paths)"
  run_name="$(sed -n 's/^RUN_NAME=//p' <<< "${info}")"
  log_dir="$(sed -n 's/^LOG_DIR=//p' <<< "${info}")"
  exp_dir="$(sed -n 's/^EXP_DIR=//p' <<< "${info}")"
  if (( DRY_RUN )); then
    echo "[dry-run] ${run_name}"
    echo "    ${exp_dir}"
    return
  fi
  mkdir -p "${log_dir}"
  jid="$(sbatch --parsable --job-name="${run_name}" \
         --output="${log_dir}/${run_name}_%j.out" --error="${log_dir}/${run_name}_%j.err" \
         slurm/sim/run_diffusion_fail1500.sh "${args[@]}")"
  echo "  ${run_name} -> job ${jid}"
  echo "      ${exp_dir}"
}

submit_es() {
  local args=("$@")
  local info run_name log_dir exp_dir jid
  info="$(bash slurm/sim/run_es.sh "${args[@]}" --print_paths)"
  run_name="$(sed -n 's/^RUN_NAME=//p' <<< "${info}")"
  log_dir="$(sed -n 's/^LOG_DIR=//p' <<< "${info}")"
  exp_dir="$(sed -n 's/^EXP_DIR=//p' <<< "${info}")"
  if (( DRY_RUN )); then
    echo "[dry-run] ${run_name}"
    echo "    ${exp_dir}"
    return
  fi
  mkdir -p "${log_dir}"
  jid="$(sbatch --parsable --job-name="${run_name}" \
         --output="${log_dir}/${run_name}_%j.out" --error="${log_dir}/${run_name}_%j.err" \
         slurm/sim/run_es.sh "${args[@]}")"
  echo "  ${run_name} -> job ${jid}"
  echo "      ${exp_dir}"
}

[[ -e "${DISC_CKPT}" ]] || { echo "ERROR: missing disc ckpt ${DISC_CKPT}" >&2; exit 1; }
mkdir -p "${EXP_ROOT}"

echo "=== final_runs root: ${EXP_ROOT} ==="
echo "--- log replay ---"
submit_diff log
submit_es --subset val10 --planner diffusion --forecast cv --other_agents log --exp_root "${EXP_ROOT}"
submit_es --subset val10 --planner diffusion --forecast wm_mf --mf_mode expected_independent \
  --other_agents log --exp_root "${EXP_ROOT}"
submit_es --subset val10 --planner diffusion --forecast wm_mf --mf_mode expected_independent \
  --other_agents log --exp_root "${EXP_ROOT}" \
  --disc_ckpt "${DISC_CKPT}"

echo "--- IDM ---"
submit_diff idm
submit_es --subset val10 --planner diffusion --forecast cv --other_agents idm --exp_root "${EXP_ROOT}"
submit_es --subset val10 --planner diffusion --forecast wm_mf --mf_mode expected_independent \
  --other_agents idm --exp_root "${EXP_ROOT}"

if (( DRY_RUN )); then
  echo "7 job(s) would be submitted."
else
  echo "7 job(s) submitted."
  squeue -u "${USER}" -o '%.10i %.45j %.2t %.10M %.20R' || true
fi
