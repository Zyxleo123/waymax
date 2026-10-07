#!/bin/bash
# Shared submitter behind trainval.sh / fail1500.sh / probe250.sh.
# Every run_es.sh flag is accepted; comma-separated values expand into one job
# per combination (duplicates with the same run name are submitted once).
#
#   --dry_run        print each job's run name, dirs and command; submit nothing
#   --exclude NODES  sbatch --exclude (e.g. gpu26,gpu31)
#
# Flags are passed to sbatch as script arguments (never via --export, which
# fails on this cluster). Submit from rhea or lov4; sbatch is denied on lov1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ES="${HERE}/run_es.sh"
LIST_FLAGS=" --planner --forecast --mf_mode --mf_collision --M --it --pop --elite --rs --t_start --fm_steps --solver --resample_timesteps --wm_refresh --seed --tag_suffix "
BOOL_FLAGS=" --print_cmd --print_paths "

DRY_RUN=0
EXCLUDE=""
combos=("")
while [[ $# -gt 0 ]]; do
  flag="$1"
  case "${flag}" in
    --dry_run) DRY_RUN=1; shift; continue ;;
    --exclude) EXCLUDE="${2:?--exclude needs a value}"; shift 2; continue ;;
    -h|--help) bash "${RUN_ES}" --help; exit 0 ;;
    --*) ;;
    *) echo "ERROR: unexpected argument '${flag}'" >&2; exit 1 ;;
  esac
  if [[ "${BOOL_FLAGS}" == *" ${flag} "* ]]; then
    echo "ERROR: ${flag} is for run_es.sh directly; use --dry_run here" >&2
    exit 1
  fi
  [[ $# -ge 2 ]] || { echo "ERROR: ${flag} needs a value" >&2; exit 1; }
  value="$2"
  shift 2
  if [[ "${LIST_FLAGS}" == *" ${flag} "* && "${value}" == *,* ]]; then
    IFS=',' read -r -a values <<< "${value}"
  else
    values=("${value}")
  fi
  expanded=()
  for c in "${combos[@]}"; do
    for v in "${values[@]}"; do
      expanded+=("${c} ${flag} ${v}")
    done
  done
  combos=("${expanded[@]}")
done

declare -A seen=()
n_jobs=0
for c in "${combos[@]}"; do
  read -r -a args <<< "${c}"
  info="$(bash "${RUN_ES}" "${args[@]}" --print_paths)"
  run_name="$(sed -n 's/^RUN_NAME=//p' <<< "${info}")"
  exp_dir="$(sed -n 's/^EXP_DIR=//p' <<< "${info}")"
  log_dir="$(sed -n 's/^LOG_DIR=//p' <<< "${info}")"
  [[ -n "${seen[${run_name}]:-}" ]] && continue
  seen[${run_name}]=1
  n_jobs=$((n_jobs + 1))
  if (( DRY_RUN )); then
    echo "[dry-run] ${run_name}"
    bash "${RUN_ES}" "${args[@]}" --print_cmd | sed -n 's/^/    /p' | grep -v 'RUN_NAME='
    continue
  fi
  mkdir -p "${log_dir}"
  sbatch_args=(--parsable --job-name="${run_name}"
               --output="${log_dir}/${run_name}_%j.out" --error="${log_dir}/${run_name}_%j.err")
  [[ -n "${EXCLUDE}" ]] && sbatch_args+=(--exclude="${EXCLUDE}")
  jid="$(sbatch "${sbatch_args[@]}" "${RUN_ES}" "${args[@]}")"
  echo "  ${run_name} -> job ${jid}"
  echo "      ${exp_dir}"
done

if (( DRY_RUN )); then
  echo "${n_jobs} job(s) would be submitted."
else
  echo "${n_jobs} job(s) submitted."
  squeue -u "${USER}" -o '%.10i %.40j %.2t %.10M %.20R' || true
fi
