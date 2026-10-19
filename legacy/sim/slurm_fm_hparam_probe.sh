#!/bin/bash
#SBATCH --job-name=fm_hparam_probe
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/logs/fm_hparam_probe_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/logs/fm_hparam_probe_%j.err

# FM-ES hyperparam probe on a fixed GT subset (default n=250).
# ALL outputs go under /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim
#
# Usage:
#   # 0) once: create log dir (script also mkdir -p)
#   mkdir -p /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/logs
#
#   # 1) build the shared n=250 subset (CPU ok; no GPU needed)
#   sbatch --job-name=fm_probe_build --gres=gpu:0 slurm_fm_hparam_probe.sh build
#   # or interactively:
#   bash slurm_fm_hparam_probe.sh build
#
#   # 2a) launch full Phase-1 sweep (Diff GT baseline + FM t_start grid)
#   bash slurm_fm_hparam_probe.sh sweep
#
#   # 2b) or run one job manually:
#   sbatch --job-name=fm_probe_diff  slurm_fm_hparam_probe.sh diff
#   sbatch --job-name=fm_probe_t0.2  slurm_fm_hparam_probe.sh fm t0.2
#   sbatch --job-name=fm_probe_t0.5  slurm_fm_hparam_probe.sh fm 0.5   # bare float also ok
#
#   # 3) summarize
#   python scripts/summarize_fm_hparam_probe.py
#
# Fixed for this checkpoint (do not change without reason):
#   model_time scale  = t * 99   (default_num_steps=100)
#   fm_num_steps      = 50       (full ODE)
#   resample_num_steps= 20
#   solver            = euler
#   other_agent       = gt
#
# Sweep knob (Phase 1): resample_t_start — current default grid below.
# Override via env, e.g.:
#   T_STARTS="0.2 0.3" POPULATION_SIZE=128 bash slurm_fm_hparam_probe.sh sweep_fm
#
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
cd "${ROOT}"

FM_ROOT="${FM_SIM_ROOT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim}"
LOG_DIR="${FM_ROOT}/logs"
PROBE_N="${PROBE_N:-250}"
PROBE_SEED="${PROBE_SEED:-0}"
PROBE_SOURCE="${PROBE_SOURCE:-fail1500}"   # fail1500 | mixed
NUM_WORLDS="${NUM_WORLDS:-1}"

SUBSET_DIR="${SUBSET_DIR:-${FM_ROOT}/fm_hparam_probe_n${PROBE_N}_seed${PROBE_SEED}}"
RUNS_DIR="${RUNS_DIR:-${FM_ROOT}/fm_hparam_probe_runs}"
TARGET_JSON="${SUBSET_DIR}/target_scenarios.json"
TFRECORD_DIR="${SUBSET_DIR}/tfrecords"

LANE_GRAPH_DIR="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
FM_CKPT="/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500"

POPULATION_SIZE="${POPULATION_SIZE:-64}"
ELITE_SIZE="${ELITE_SIZE:-8}"
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
RESAMPLE_TIMESTEPS="${RESAMPLE_TIMESTEPS:-5}"          # Diff-ES only
RESAMPLE_NUM_STEPS="${RESAMPLE_NUM_STEPS:-20}"         # FM-ES resample ODE steps
FM_NUM_STEPS="${FM_NUM_STEPS:-50}"                     # FM full-sample ODE steps
FM_SOLVER="${FM_SOLVER:-euler}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"

# Default Phase-1 t_start grid (already completed on n=250)
T_STARTS="${T_STARTS:-0.5 0.6 0.8}"

# Phase-2 (beat Diff): freeze t_start=0.20 from Phase-1; vary budget / ODE quality only.
# Skip: iters 0/1/2 (weaker than current), rs 5/10 (coarser ODE unlikely to help),
#       full steps 20/30 (we already match Diff at 50), annealing (not implemented yet).
PHASE2_T_START="${PHASE2_T_START:-0.20}"
PHASE2_CONFIGS="${PHASE2_CONFIGS:-it6_rs20 it4_rs40 it6_rs40}"

MODE="${1:-}"
ARG2="${2:-}"
set --

mkdir -p "${LOG_DIR}" "${RUNS_DIR}" "${FM_ROOT}"

CONDA_ROOT="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3"
WAYMAX_ENV="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs"
activate_env() {
  if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
    set -u
  fi
}

tag_from_tstart() {
  local t="$1"
  # Normalize 0.50 -> 0.5 for tags, keep leading form used in CLI.
  echo "fm_gt_t${t}_pop${POPULATION_SIZE}_it${NUM_ES_ITERATIONS}_rs${RESAMPLE_NUM_STEPS}_n${FM_NUM_STEPS}_${FM_SOLVER}"
}

parse_tstart() {
  # Accept "t0.2" or "0.2"
  local raw="$1"
  if [[ "${raw}" =~ ^t([0-9]*\.?[0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "${raw}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "${raw}"
  else
    echo "ERROR: bad t_start '${raw}' (want t0.2 or 0.2)" >&2
    exit 1
  fi
}

cmd_build() {
  activate_env
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  echo "Building probe subset n=${PROBE_N} source=${PROBE_SOURCE} -> ${SUBSET_DIR}"
  python "${ROOT}/scripts/build_fm_hparam_probe_subset.py" \
    --source "${PROBE_SOURCE}" \
    --n "${PROBE_N}" \
    --seed "${PROBE_SEED}" \
    --num_worlds "${NUM_WORLDS}" \
    --out_dir "${SUBSET_DIR}"
  echo "DONE build: ${SUBSET_DIR}"
  ls -la "${SUBSET_DIR}"
}

require_subset() {
  if [[ ! -f "${TARGET_JSON}" ]]; then
    echo "ERROR: missing ${TARGET_JSON}"
    echo "Run first:  bash slurm_fm_hparam_probe.sh build"
    exit 1
  fi
  if [[ ! -d "${TFRECORD_DIR}/training" ]]; then
    echo "ERROR: missing ${TFRECORD_DIR}/training"
    exit 1
  fi
}

run_one() {
  local planner="$1"   # es | es_fm
  local tag="$2"
  local t_start="${3:-}"

  activate_env
  require_subset
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  local exp_dir="${RUNS_DIR}/${tag}"
  local run_stamp
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${tag}_${run_stamp}.log"
  mkdir -p "${exp_dir}"

  {
    echo "============================================================"
    echo " FM hparam probe"
    echo " host=$(/usr/bin/hostname 2>/dev/null || echo unknown)"
    echo " date=$(date -Is)"
    echo " planner=${planner}"
    echo " tag=${tag}"
    echo " subset_dir=${SUBSET_DIR}"
    echo " exp_dir=${exp_dir}"
    echo " log_file=${log_file}"
    echo " population_size=${POPULATION_SIZE} elite_size=${ELITE_SIZE} num_es_iterations=${NUM_ES_ITERATIONS}"
    echo " num_worlds=${NUM_WORLDS}"
    if [[ "${planner}" == "es_fm" ]]; then
      echo " resample_t_start=${t_start}"
      echo " resample_num_steps=${RESAMPLE_NUM_STEPS}"
      echo " fm_num_steps=${FM_NUM_STEPS}"
      echo " fm_solver=${FM_SOLVER}"
      echo " fm_ckpt=${FM_CKPT}"
    else
      echo " resample_timesteps=${RESAMPLE_TIMESTEPS}"
      echo " diffusion_ckpt=${DIFFUSION_CKPT}"
    fi
    echo "============================================================"
    nvidia-smi || true
    echo
    python -c "import json; from pathlib import Path; m=json.loads(Path('${SUBSET_DIR}/subset_manifest.json').read_text()); print('n_final', m['n_final'], 'source', m.get('source'), 'buckets', m.get('bucket_counts'))"
    echo
  } | tee -a "${log_file}"

  local cmd=(
    python "${ROOT}/simulation/run_simulation.py"
    --planner "${planner}"
    --tfrecord_dir "${TFRECORD_DIR}"
    --lane_graph_dir "${LANE_GRAPH_DIR}"
    --exp_dir "${exp_dir}"
    --split training
    --num_worlds "${NUM_WORLDS}"
    --num_scenarios 100000
    --population_size "${POPULATION_SIZE}"
    --elite_size "${ELITE_SIZE}"
    --num_es_iterations "${NUM_ES_ITERATIONS}"
    --replan_interval_steps "${REPLAN_INTERVAL_STEPS}"
    --other_agent_forecast gt
    --target_scenario_path "${TARGET_JSON}"
    --visualize_mode none
    --tag "${tag}"
  )

  if [[ "${planner}" == "es" ]]; then
    cmd+=(--diffusion_ckpt_path "${DIFFUSION_CKPT}" --resample_timesteps "${RESAMPLE_TIMESTEPS}")
  else
    cmd+=(
      --fm_ckpt_path "${FM_CKPT}"
      --resample_t_start "${t_start}"
      --resample_num_steps "${RESAMPLE_NUM_STEPS}"
      --fm_num_steps "${FM_NUM_STEPS}"
      --fm_solver "${FM_SOLVER}"
    )
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local py_rc=${PIPESTATUS[0]}
  set -e
  if [[ "${py_rc}" -ne 0 ]]; then
    echo "ERROR: python exited ${py_rc}" | tee -a "${log_file}"
    exit "${py_rc}"
  fi
  echo "DONE ${tag}" | tee -a "${log_file}"
  ls -1dt "${exp_dir}"/simulation_* 2>/dev/null | head -5 | tee -a "${log_file}"
}

cmd_diff() {
  run_one es "diff_gt_baseline_pop${POPULATION_SIZE}_it${NUM_ES_ITERATIONS}_rs${RESAMPLE_TIMESTEPS}"
}

cmd_fm() {
  if [[ -z "${ARG2}" ]]; then
    echo "Usage: sbatch slurm_fm_hparam_probe.sh fm <t_start>"
    echo "  e.g.  sbatch slurm_fm_hparam_probe.sh fm t0.2"
    echo "        sbatch slurm_fm_hparam_probe.sh fm 0.5"
    exit 1
  fi
  local t
  t="$(parse_tstart "${ARG2}")"
  local tag
  tag="$(tag_from_tstart "${t}")"
  run_one es_fm "${tag}" "${t}"
}

cmd_sweep() {
  require_subset
  echo "Submitting Diff GT baseline + FM t_start sweep to Slurm"
  echo "  subset: ${SUBSET_DIR}"
  echo "  runs:   ${RUNS_DIR}"
  echo "  T_STARTS=${T_STARTS}"
  echo

  sbatch --job-name="fm_probe_diff" \
    --output="${LOG_DIR}/fm_hparam_probe_%j.out" \
    --error="${LOG_DIR}/fm_hparam_probe_%j.err" \
    "${ROOT}/legacy/sim/slurm_fm_hparam_probe.sh" diff

  _submit_fm_starts
  echo
  echo "Submitted. Monitor: squeue -u \$USER"
  echo "Summarize later:"
  echo "  python scripts/summarize_fm_hparam_probe.py --subset_dir ${SUBSET_DIR} --runs_dir ${RUNS_DIR}"
}

_submit_fm_starts() {
  local t
  for t in ${T_STARTS}; do
    local job="fm_probe_t${t}"
    sbatch --job-name="${job}" \
      --output="${LOG_DIR}/fm_hparam_probe_%j.out" \
      --error="${LOG_DIR}/fm_hparam_probe_%j.err" \
      "${ROOT}/legacy/sim/slurm_fm_hparam_probe.sh" fm "${t}"
  done
}

cmd_sweep_fm() {
  # FM-only: does not re-submit Diff baseline or touch already-running jobs.
  require_subset
  echo "Submitting FM-only t_start sweep to Slurm"
  echo "  subset: ${SUBSET_DIR}"
  echo "  runs:   ${RUNS_DIR}"
  echo "  T_STARTS=${T_STARTS}"
  echo
  _submit_fm_starts
  echo
  echo "Submitted. Monitor: squeue -u \$USER"
  echo "Summarize later:"
  echo "  python scripts/summarize_fm_hparam_probe.py --subset_dir ${SUBSET_DIR} --runs_dir ${RUNS_DIR}"
}

apply_phase2_config() {
  # Sets NUM_ES_ITERATIONS / RESAMPLE_NUM_STEPS / FM_NUM_STEPS from a short name.
  local name="$1"
  case "${name}" in
    it6_rs20)
      NUM_ES_ITERATIONS=6
      RESAMPLE_NUM_STEPS=20
      FM_NUM_STEPS=50
      ;;
    it4_rs40)
      NUM_ES_ITERATIONS=4
      RESAMPLE_NUM_STEPS=40
      FM_NUM_STEPS=50
      ;;
    it6_rs40)
      NUM_ES_ITERATIONS=6
      RESAMPLE_NUM_STEPS=40
      FM_NUM_STEPS=50
      ;;
    it6_rs20_n30)
      # Optional follow-up: cheaper initial sample (not in default Phase-2).
      NUM_ES_ITERATIONS=6
      RESAMPLE_NUM_STEPS=20
      FM_NUM_STEPS=30
      ;;
    *)
      echo "ERROR: unknown Phase-2 config '${name}'"
      echo "Known: it6_rs20 it4_rs40 it6_rs40 it6_rs20_n30"
      exit 1
      ;;
  esac
}

cmd_phase2() {
  # One Phase-2 config: bash|sbatch slurm_fm_hparam_probe.sh phase2 it6_rs20
  if [[ -z "${ARG2}" ]]; then
    echo "Usage: sbatch slurm_fm_hparam_probe.sh phase2 <config>"
    echo "  configs: ${PHASE2_CONFIGS} (or it6_rs20_n30)"
    exit 1
  fi
  apply_phase2_config "${ARG2}"
  local tag
  tag="$(tag_from_tstart "${PHASE2_T_START}")"
  # Distinguish from Phase-1 t0.20 baseline tag if iters/rs match defaults — include config name.
  tag="${tag}_${ARG2}"
  run_one es_fm "${tag}" "${PHASE2_T_START}"
}

cmd_sweep_phase2() {
  require_subset
  echo "Submitting Phase-2 FM configs (aim: beat Diff on n=250 GT)"
  echo "  t_start fixed=${PHASE2_T_START}"
  echo "  configs=${PHASE2_CONFIGS}"
  echo "  subset=${SUBSET_DIR}"
  echo
  local cfg
  for cfg in ${PHASE2_CONFIGS}; do
    sbatch --job-name="fm_p2_${cfg}" \
      --output="${LOG_DIR}/fm_hparam_probe_%j.out" \
      --error="${LOG_DIR}/fm_hparam_probe_%j.err" \
      "${ROOT}/legacy/sim/slurm_fm_hparam_probe.sh" phase2 "${cfg}"
  done
  echo
  echo "Submitted. Monitor: squeue -u \$USER"
  echo "Summarize later:"
  echo "  python scripts/summarize_fm_hparam_probe.py --subset_dir ${SUBSET_DIR} --runs_dir ${RUNS_DIR}"
}

usage() {
  cat <<EOF
Usage: bash|sbatch slurm_fm_hparam_probe.sh {build|diff|fm <t>|sweep|sweep_fm|phase2 <cfg>|sweep_phase2}

  build          Create n=${PROBE_N} subset under ${SUBSET_DIR}
  diff           Diff-ES GT baseline on that subset
  fm <t>         One FM-ES GT run (t like 0.2 or t0.2)
  sweep          sbatch diff + all T_STARTS=${T_STARTS}
  sweep_fm       sbatch FM only for T_STARTS=${T_STARTS} (no Diff)
  phase2 <cfg>   One Phase-2 config (t_start=${PHASE2_T_START}): it6_rs20|it4_rs40|it6_rs40
  sweep_phase2   sbatch all Phase-2 configs: ${PHASE2_CONFIGS}

All artifacts: ${FM_ROOT}
EOF
}

case "${MODE}" in
  build) cmd_build ;;
  diff)  cmd_diff ;;
  fm)    cmd_fm ;;
  sweep) cmd_sweep ;;
  sweep_fm) cmd_sweep_fm ;;
  phase2) cmd_phase2 ;;
  sweep_phase2) cmd_sweep_phase2 ;;
  *)
    usage
    exit 1
    ;;
esac
