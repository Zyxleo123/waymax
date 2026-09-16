#!/bin/bash
#SBATCH --job-name=es_fail1500
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/es_fail1500_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/es_fail1500_%j.err

# Shared FM-failure subset (default n=1500, seed=0) — same scenes for all runs.
#
#   # once:
#   python scripts/build_failure_subset_for_es.py --n 1500 --seed 0
#
#   sbatch --job-name=es_diff_cv_fail1500 slurm_es_failure_subset.sh es cv
#   sbatch --job-name=es_diff_gt_fail1500 slurm_es_failure_subset.sh es gt
#   sbatch --job-name=es_fm_cv_fail1500   slurm_es_failure_subset.sh es_fm cv
#   sbatch --job-name=es_fm_gt_fail1500   slurm_es_failure_subset.sh es_fm gt
#
# World-model forecast (3 refresh experiments):
#   Exp1 refresh=50  Exp2 refresh=20  Exp3 refresh=10 (every replan)
#   sbatch --job-name=es_diff_wm50_fail1500 slurm_es_failure_subset.sh es wm 50
#   sbatch --job-name=es_diff_wm20_fail1500 slurm_es_failure_subset.sh es wm 20
#   sbatch --job-name=es_diff_wm10_fail1500 slurm_es_failure_subset.sh es wm 10
#   sbatch --job-name=es_fm_wm10_fail1500   slurm_es_failure_subset.sh es_fm wm 10
#
set -euo pipefail

PLANNER="${1:-}"
OTHER_AGENT_FORECAST="${2:-}"
WM_REFRESH_STEPS="${3:-${WM_REFRESH_STEPS:-10}}"
PLANNER="$(echo "${PLANNER}" | tr '[:upper:]' '[:lower:]')"
OTHER_AGENT_FORECAST="$(echo "${OTHER_AGENT_FORECAST}" | tr '[:upper:]' '[:lower:]')"
set --

if [[ "${PLANNER}" != "es" && "${PLANNER}" != "es_fm" ]]; then
  echo "Usage: sbatch slurm_es_failure_subset.sh {es|es_fm} {cv|gt|wm} [wm_refresh_steps]"
  exit 1
fi
if [[ "${OTHER_AGENT_FORECAST}" != "cv" && "${OTHER_AGENT_FORECAST}" != "gt" && "${OTHER_AGENT_FORECAST}" != "wm" ]]; then
  echo "Usage: sbatch slurm_es_failure_subset.sh {es|es_fm} {cv|gt|wm} [wm_refresh_steps]"
  exit 1
fi
if ! [[ "${WM_REFRESH_STEPS}" =~ ^[0-9]+$ ]] || [[ "${WM_REFRESH_STEPS}" -le 0 ]]; then
  echo "ERROR: wm_refresh_steps must be a positive integer, got '${WM_REFRESH_STEPS}'"
  exit 1
fi

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
cd "${ROOT}"

CONDA_ROOT="/zfsauton/scratch/sbellad/miniconda3"
WAYMAX_ENV="/zfsauton/scratch/sbellad/conda_envs/waymax_rs"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

SUBSET_DIR="${SUBSET_DIR:-/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0}"
TARGET_JSON="${SUBSET_DIR}/target_scenarios.json"
TFRECORD_DIR="${SUBSET_DIR}/tfrecords"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
FM_CKPT="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt}"
WORLD_MODEL_DEVICE="${WORLD_MODEL_DEVICE:-auto}"

if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
  FORECAST_TAG="wm${WM_REFRESH_STEPS}"
else
  FORECAST_TAG="${OTHER_AGENT_FORECAST}"
fi

if [[ "${PLANNER}" == "es" ]]; then
  RUN_NAME="es_diffusion_${FORECAST_TAG}_fail1500"
else
  RUN_NAME="es_fm_${FORECAST_TAG}_fail1500"
fi
EXP_DIR="/zfsauton/scratch/sbellad/immediate_exp/${RUN_NAME}"

NUM_WORLDS="${NUM_WORLDS:-5}"
POPULATION_SIZE="${POPULATION_SIZE:-64}"
ELITE_SIZE="${ELITE_SIZE:-8}"
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
RESAMPLE_TIMESTEPS="${RESAMPLE_TIMESTEPS:-5}"
RESAMPLE_T_START="${RESAMPLE_T_START:-0.2}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/zfsauton/scratch/sbellad/immediate_exp/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${RUN_STAMP}.log"
mkdir -p "${EXP_DIR}" "${LOG_DIR}"

for p in "${TARGET_JSON}" "${TFRECORD_DIR}/training" "${LANE_GRAPH_DIR}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: missing ${p}. Build subset first:" | tee -a "${LOG_FILE}"
    echo "  python scripts/build_failure_subset_for_es.py --n 1500 --seed 0" | tee -a "${LOG_FILE}"
    exit 1
  fi
done

if [[ "${OTHER_AGENT_FORECAST}" == "wm" && ! -e "${WORLD_MODEL_CKPT}" ]]; then
  echo "ERROR: world-model checkpoint missing: ${WORLD_MODEL_CKPT}" | tee -a "${LOG_FILE}"
  exit 1
fi

{
  echo "============================================================"
  echo " Failure-subset ES"
  echo " RUN_NAME=${RUN_NAME}"
  echo " planner=${PLANNER}  other_agent_forecast=${OTHER_AGENT_FORECAST}"
  if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
    echo " wm_refresh_steps=${WM_REFRESH_STEPS}"
    echo " world_model_ckpt=${WORLD_MODEL_CKPT}"
    echo " world_model_device=${WORLD_MODEL_DEVICE}"
  fi
  echo " subset_dir=${SUBSET_DIR}"
  echo " target_json=${TARGET_JSON}"
  echo " tfrecord_dir=${TFRECORD_DIR}"
  echo " exp_dir=${EXP_DIR}"
  echo " log_file=${LOG_FILE}"
  echo "============================================================"
  nvidia-smi || true
  echo
  python -c "import json; from pathlib import Path; m=json.loads((Path('${SUBSET_DIR}')/'subset_manifest.json').read_text()); print('n_final', m['n_final'], 'shards', m['num_shards'], 'mix', m['failure_type_counts'])"
  echo
} | tee -a "${LOG_FILE}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  python "${ROOT}/simulation/run_simulation.py"
  --planner "${PLANNER}"
  --tfrecord_dir "${TFRECORD_DIR}"
  --lane_graph_dir "${LANE_GRAPH_DIR}"
  --exp_dir "${EXP_DIR}"
  --split training
  --num_worlds "${NUM_WORLDS}"
  --num_scenarios 100000
  --population_size "${POPULATION_SIZE}"
  --elite_size "${ELITE_SIZE}"
  --num_es_iterations "${NUM_ES_ITERATIONS}"
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}"
  --other_agent_forecast "${OTHER_AGENT_FORECAST}"
  --target_scenario_path "${TARGET_JSON}"
  --visualize_mode none
  --tag "${RUN_NAME}"
)

if [[ "${PLANNER}" == "es" ]]; then
  CMD+=(--diffusion_ckpt_path "${DIFFUSION_CKPT}" --resample_timesteps "${RESAMPLE_TIMESTEPS}")
else
  CMD+=(--fm_ckpt_path "${FM_CKPT}" --resample_t_start "${RESAMPLE_T_START}")
fi

if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
  CMD+=(
    --world_model_ckpt "${WORLD_MODEL_CKPT}"
    --world_model_device "${WORLD_MODEL_DEVICE}"
    --wm_refresh_steps "${WM_REFRESH_STEPS}"
  )
fi

set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
py_rc=${PIPESTATUS[0]}
set -e
if [[ "${py_rc}" -ne 0 ]]; then
  echo "ERROR: python exited ${py_rc}" | tee -a "${LOG_FILE}"
  exit "${py_rc}"
fi

echo "DONE ${RUN_NAME}" | tee -a "${LOG_FILE}"
ls -1dt "${EXP_DIR}"/simulation_* 2>/dev/null | head -5 | tee -a "${LOG_FILE}"
