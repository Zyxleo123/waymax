#!/bin/bash
# Every simulation submission goes through here. Submit from rhea or lov4 (sbatch is denied on lov1).
#
# Ad-hoc job grids: any run_es.sh / run_policy.sh flag; comma-separated values of the
# list flags below expand into one job per combination (same run name -> submitted once).
#
#   bash slurm/sim/submit.sh es --subset fail1500 --planner fm --forecast cv,wm_mf --it 6 --rs 40
#   bash slurm/sim/submit.sh es --subset trainval --planner diffusion --forecast wm_mf --mf_mode top1,worst_case_independent
#   bash slurm/sim/submit.sh es --subset fail1500 --planner diffusion --forecast wm_mf --mf_collision gaussian_uniform --M 4,8,16
#   bash slurm/sim/submit.sh policy --subset val10 --planner diffusion,fm --other_agents log,idm
#
# Named experiments (the exact job sets behind the reported results):
#
#   val10_all [--other_agents log,idm]  plain diffusion / FM + Diffusion-ES and FM-ES (gt, cv, wm_mf)
#                                       on val10; FM-ES budget it6/rs40 (override: FM_IT=4 FM_RS=20)
#   idm_fail1500                        the same planners on fail1500 with IDM other agents
#   final_val10 [--exp_root R]          paper runs on val10 -> final_runs: diffusion only, Diffusion-ES
#                                       CV / WM, and WM + discriminator trained on fail1500 (log replay);
#                                       diffusion only, CV, WM (IDM)
#   disc_v3                             discriminator v3 (A0): val10 log + IDM -> final_runs, trainval
#   dtfix                               fail1500 after the world-dt fix -> final_runs_dtfix:
#                                       Diffusion-ES + WM, and + discriminator v3
#   disc_v4                             discriminator v4 (A4) on fail1500 -> final_runs_dtfix
#
# Global options (anywhere): --dry_run (print run names, dirs and commands; submit nothing),
# --exclude NODES (sbatch --exclude, e.g. gpu26,gpu31).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SCRATCH=/zfsauton/scratch/sbellad
SP=${SCRATCH}/es_discriminator/safety_progress
DISC_FAIL1500=${SP}/safety_progress_full_gaussian_wm_20261013_090539_81674/best.pt   # trained on fail1500 only
DISC_V3=${SP}/safety_progress_v3_20261016_155456_86103/best.pt                         # A0, fail1500 + WOMD train shards 02-11
DISC_V4=${SP}/safety_progress_v3_A4_20261017_212234_87831/best.pt                      # A4, same data
LIST_FLAGS=" --subset --planner --forecast --other_agents --mf_mode --mf_collision --M --it --pop --elite --rs --t_start --fm_steps --solver --resample_timesteps --seed "
WM=(--planner diffusion --forecast wm_mf --mf_mode expected_independent)

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; }

DRY_RUN=0 EXCLUDE="" ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN=1; shift ;;
    --exclude) EXCLUDE="${2:?--exclude needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]}"
[[ $# -gt 0 ]] || { usage >&2; exit 1; }
WHAT="$1"; shift

N_JOBS=0
declare -A SEEN=()

# submit JOB ARGS...: one sbatch job of slurm/sim/JOB, named after the job's run name.
submit() {
  local job="slurm/sim/$1"; shift
  local info run_name exp_dir log_dir jid
  info="$(bash "${job}" "$@" --print_paths)"
  run_name="$(sed -n 's/^RUN_NAME=//p' <<< "${info}")"
  exp_dir="$(sed -n 's/^EXP_DIR=//p' <<< "${info}")"
  log_dir="$(sed -n 's/^LOG_DIR=//p' <<< "${info}")"
  [[ -n "${SEEN[${run_name}]:-}" ]] && return 0
  SEEN[${run_name}]=1
  N_JOBS=$((N_JOBS + 1))
  if (( DRY_RUN )); then
    echo "[dry-run] ${run_name}"
    echo "    ${exp_dir}"
    echo "    sbatch ${job} $*"
    [[ "${job}" == */run_es.sh ]] && bash "${job}" "$@" --print_cmd | grep '^CMD' | sed 's/^/    /'
    return 0
  fi
  mkdir -p "${log_dir}"
  local sb=(--parsable --job-name="${run_name}"
            --output="${log_dir}/${run_name}_%j.out" --error="${log_dir}/${run_name}_%j.err")
  [[ -n "${EXCLUDE}" ]] && sb+=(--exclude="${EXCLUDE}")
  jid="$(sbatch "${sb[@]}" "${job}" "$@")"
  echo "  ${run_name} -> job ${jid}"
  echo "      ${exp_dir}"
}

# grid JOB --flag value ...: submit() for every combination of comma-separated LIST_FLAGS values.
grid() {
  local job="$1"; shift
  local combos=("") expanded values c v flag value
  while [[ $# -gt 0 ]]; do
    flag="$1"
    [[ "${flag}" == --* && $# -ge 2 ]] || { echo "ERROR: expected '--flag value', got '${flag}'" >&2; exit 1; }
    value="$2"; shift 2
    if [[ "${LIST_FLAGS}" == *" ${flag} "* && "${value}" == *,* ]]; then
      IFS=',' read -r -a values <<< "${value}"
    else
      values=("${value}")
    fi
    expanded=()
    for c in "${combos[@]}"; do
      for v in "${values[@]}"; do expanded+=("${c} ${flag} ${v}"); done
    done
    combos=("${expanded[@]}")
  done
  local args
  for c in "${combos[@]}"; do
    read -r -a args <<< "${c}"
    submit "${job}" "${args[@]}"
  done
}

need_ckpt() { [[ -e "$1" ]] || { echo "ERROR: missing discriminator checkpoint $1" >&2; exit 1; }; }
exp_root_arg() {  # --exp_root R (optional) for the named experiments that write to one root
  if [[ $# -eq 0 ]]; then return; fi
  [[ "$1" == --exp_root && $# -eq 2 ]] || { echo "ERROR: ${WHAT} only takes --exp_root R" >&2; exit 1; }
  echo "$2"
}
no_args() { [[ $# -eq 0 ]] || { echo "ERROR: ${WHAT} takes no arguments (got '$*')" >&2; exit 1; }; }

case "${WHAT}" in
  es) grid run_es.sh "$@" ;;
  policy) grid run_policy.sh "$@" ;;
  val10_all)
    others=log
    if [[ $# -gt 0 ]]; then
      [[ "$1" == --other_agents && $# -eq 2 ]] || { echo "ERROR: val10_all only takes --other_agents L" >&2; exit 1; }
      others="$2"
    fi
    grid run_policy.sh --subset val10 --planner diffusion,fm --other_agents "${others}"
    grid run_es.sh --subset val10 --planner diffusion --forecast gt,cv,wm_mf --other_agents "${others}"
    grid run_es.sh --subset val10 --planner fm --forecast gt,cv,wm_mf --other_agents "${others}" \
      --it "${FM_IT:-6}" --rs "${FM_RS:-40}"
    ;;
  idm_fail1500)
    no_args "$@"
    submit run_policy.sh --other_agents idm
    grid run_es.sh --subset fail1500 --planner diffusion --forecast gt,cv,wm_mf --other_agents idm
    grid run_es.sh --subset fail1500 --planner fm --forecast gt,cv,wm_mf --other_agents idm \
      --it "${FM_IT:-6}" --rs "${FM_RS:-40}"
    ;;
  final_val10)
    root="$(exp_root_arg "$@")"; root="${root:-${SCRATCH}/final_runs}"
    need_ckpt "${DISC_FAIL1500}"
    for other in log idm; do
      R=(--subset val10 --other_agents "${other}" --exp_root "${root}")
      submit run_policy.sh --planner diffusion "${R[@]}"
      submit run_es.sh --planner diffusion --forecast cv "${R[@]}"
      submit run_es.sh "${WM[@]}" "${R[@]}"
    done
    submit run_es.sh "${WM[@]}" --subset val10 --other_agents log --exp_root "${root}" --disc_ckpt "${DISC_FAIL1500}"
    ;;
  disc_v3)
    no_args "$@"
    need_ckpt "${DISC_V3}"
    for other in log idm; do
      submit run_es.sh --subset val10 "${WM[@]}" --disc_ckpt "${DISC_V3}" --run_suffix v3 \
        --other_agents "${other}" --exp_root "${SCRATCH}/final_runs"
    done
    submit run_es.sh --subset trainval "${WM[@]}" --disc_ckpt "${DISC_V3}" --run_suffix v3 --other_agents log
    ;;
  dtfix)
    root="$(exp_root_arg "$@")"; root="${root:-${SCRATCH}/final_runs_dtfix}"
    need_ckpt "${DISC_V3}"
    submit run_es.sh --subset fail1500 "${WM[@]}" --other_agents log --exp_root "${root}"
    submit run_es.sh --subset fail1500 "${WM[@]}" --other_agents log --exp_root "${root}" \
      --disc_ckpt "${DISC_V3}" --run_suffix v3
    ;;
  disc_v4)
    root="$(exp_root_arg "$@")"; root="${root:-${SCRATCH}/final_runs_dtfix}"
    need_ckpt "${DISC_V4}"
    submit run_es.sh --subset fail1500 "${WM[@]}" --other_agents log --exp_root "${root}" \
      --disc_ckpt "${DISC_V4}" --run_suffix v4
    ;;
  *) echo "ERROR: unknown '${WHAT}'" >&2; usage >&2; exit 1 ;;
esac

if (( DRY_RUN )); then
  echo "${N_JOBS} job(s) would be submitted."
else
  echo "${N_JOBS} job(s) submitted."
  squeue -u "${USER}" -o '%.10i %.45j %.2t %.10M %.20R' || true
fi
