#!/bin/bash
#SBATCH --job-name=es_diffusion
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/es_diffusion_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/es_diffusion_%j.err

# Diffusion-ES closed-loop (--planner es).
# Positional args: other-agent forecast [, wm_refresh_steps]
#   sbatch --job-name=es_diffusion_cv    slurm_es_baseline.sh cv
#   sbatch --job-name=es_diffusion_gt    slurm_es_baseline.sh gt
#   sbatch --job-name=es_diffusion_wm10  slurm_es_baseline.sh wm
#   sbatch --job-name=es_diffusion_wm50  slurm_es_baseline.sh wm 50
#
# WM refresh: 50 = Exp1, 20 = Exp2, 10 = Exp3 / every replan (default).
#
# Outputs / logs:
#   /zfsauton/scratch/sbellad/immediate_exp/es_diffusion_{cv|gt|wm10}
#   logs/es_diffusion_{cv|gt|wm10}_train_val_shards00_01_<stamp>.log
#
set -euo pipefail

# MUST capture args before sourcing conda activate (else conda treats $1 as env name).
OTHER_AGENT_FORECAST="${1:-${OTHER_AGENT_FORECAST:-cv}}"
OTHER_AGENT_FORECAST="$(echo "${OTHER_AGENT_FORECAST}" | tr '[:upper:]' '[:lower:]')"
WM_REFRESH_STEPS="${2:-${WM_REFRESH_STEPS:-10}}"
set --
if [[ "${OTHER_AGENT_FORECAST}" != "cv" && "${OTHER_AGENT_FORECAST}" != "gt" && "${OTHER_AGENT_FORECAST}" != "wm" ]]; then
  echo "ERROR: arg must be 'cv', 'gt', or 'wm', got '${OTHER_AGENT_FORECAST}'"
  echo "Usage: sbatch [--job-name=es_diffusion_wm10] slurm_es_baseline.sh [cv|gt|wm] [wm_refresh_steps]"
  exit 1
fi
if ! [[ "${WM_REFRESH_STEPS}" =~ ^[0-9]+$ ]] || [[ "${WM_REFRESH_STEPS}" -le 0 ]]; then
  echo "ERROR: wm_refresh_steps must be a positive integer, got '${WM_REFRESH_STEPS}'"
  exit 1
fi

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
cd "${ROOT}"
if [[ ! -f "${ROOT}/simulation/run_simulation.py" ]]; then
  echo "ERROR: run_simulation.py not found under ROOT=${ROOT}"
  exit 1
fi

CONDA_ROOT="/zfsauton/scratch/sbellad/miniconda3"
WAYMAX_ENV="/zfsauton/scratch/sbellad/conda_envs/waymax_rs"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt}"
WORLD_MODEL_DEVICE="${WORLD_MODEL_DEVICE:-auto}"
if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
  FORECAST_TAG="wm${WM_REFRESH_STEPS}"
else
  FORECAST_TAG="${OTHER_AGENT_FORECAST}"
fi

RUN_NAME="es_diffusion_${FORECAST_TAG}"
EXP_DIR="/zfsauton/scratch/sbellad/immediate_exp/${RUN_NAME}"
TF_ROOT="/zfsauton/scratch/eshau/womd/tf_example"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"

NUM_WORLDS="${NUM_WORLDS:-5}"
NUM_SCENARIOS="${NUM_SCENARIOS:-100000}"
POPULATION_SIZE="${POPULATION_SIZE:-64}"
ELITE_SIZE="${ELITE_SIZE:-8}"
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
RESAMPLE_TIMESTEPS="${RESAMPLE_TIMESTEPS:-5}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/zfsauton/scratch/sbellad/immediate_exp/logs"
TF_LINK_ROOT="/zfsauton/scratch/sbellad/immediate_exp/tfrecord_shards_00_01_${RUN_NAME}_${RUN_STAMP}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_train_val_shards00_01_${RUN_STAMP}.log"

TRAIN_TAG="${RUN_NAME}_TRAIN_tfrecord00_01"
VAL_TAG="${RUN_NAME}_VAL_tfrecord00_01"

mkdir -p "${EXP_DIR}" "${LOG_DIR}" \
  "${TF_LINK_ROOT}/training" \
  "${TF_LINK_ROOT}/validation"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  {
    echo "============================================================"
    echo " SLURM job ${SLURM_JOB_ID}"
    echo " host=$(/usr/bin/hostname 2>/dev/null || echo unknown)"
    echo " date=$(date -Is)"
    echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo " RUN_NAME=${RUN_NAME}"
    echo " other_agent_forecast=${OTHER_AGENT_FORECAST}"
    if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
      echo " wm_refresh_steps=${WM_REFRESH_STEPS}"
      echo " world_model_ckpt=${WORLD_MODEL_CKPT}"
      echo " world_model_device=${WORLD_MODEL_DEVICE}"
    fi
    echo " exp_dir=${EXP_DIR}"
    echo " log_file=${LOG_FILE}"
    echo "============================================================"
    nvidia-smi || true
    echo
  } | tee -a "${LOG_FILE}"
fi

ln -sfn \
  "${TF_ROOT}/training/training_tfexample.tfrecord-00000-of-01000" \
  "${TF_LINK_ROOT}/training/training_tfexample.tfrecord-00000-of-01000"
ln -sfn \
  "${TF_ROOT}/training/training_tfexample.tfrecord-00001-of-01000" \
  "${TF_LINK_ROOT}/training/training_tfexample.tfrecord-00001-of-01000"
ln -sfn \
  "${TF_ROOT}/validation/validation_tfexample.tfrecord-00000-of-00150" \
  "${TF_LINK_ROOT}/validation/validation_tfexample.tfrecord-00000-of-00150"
ln -sfn \
  "${TF_ROOT}/validation/validation_tfexample.tfrecord-00001-of-00150" \
  "${TF_LINK_ROOT}/validation/validation_tfexample.tfrecord-00001-of-00150"

for p in \
  "${DIFFUSION_CKPT}" \
  "${LANE_GRAPH_DIR}/training_tfexample.tfrecord-00000-of-01000.lanegraph.zip" \
  "${LANE_GRAPH_DIR}/training_tfexample.tfrecord-00001-of-01000.lanegraph.zip" \
  "${LANE_GRAPH_DIR}/validation_tfexample.tfrecord-00000-of-00150.lanegraph.zip" \
  "${LANE_GRAPH_DIR}/validation_tfexample.tfrecord-00001-of-00150.lanegraph.zip" \
  "${TF_LINK_ROOT}/training/training_tfexample.tfrecord-00000-of-01000" \
  "${TF_LINK_ROOT}/training/training_tfexample.tfrecord-00001-of-01000" \
  "${TF_LINK_ROOT}/validation/validation_tfexample.tfrecord-00000-of-00150" \
  "${TF_LINK_ROOT}/validation/validation_tfexample.tfrecord-00001-of-00150"
do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: missing path: ${p}" | tee -a "${LOG_FILE}"
    exit 1
  fi
done

if [[ "${OTHER_AGENT_FORECAST}" == "wm" && ! -e "${WORLD_MODEL_CKPT}" ]]; then
  echo "ERROR: world-model checkpoint missing: ${WORLD_MODEL_CKPT}" | tee -a "${LOG_FILE}"
  exit 1
fi

{
  echo "============================================================"
  echo " Diffusion-ES (${RUN_NAME}): TRAIN shards 0+1 + VAL shards 0+1"
  echo " stamp:                  ${RUN_STAMP}"
  echo " other_agent_forecast:   ${OTHER_AGENT_FORECAST}"
  if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
    echo " wm_refresh_steps:       ${WM_REFRESH_STEPS}"
    echo " world_model_ckpt:       ${WORLD_MODEL_CKPT}"
    echo " world_model_device:     ${WORLD_MODEL_DEVICE}"
  fi
  echo " exp_dir:                ${EXP_DIR}"
  echo " log_file:               ${LOG_FILE}"
  echo " tfrecord_dir:           ${TF_LINK_ROOT}"
  echo " lane_graph_dir:         ${LANE_GRAPH_DIR}"
  echo " diffusion:              ${DIFFUSION_CKPT}"
  echo " planner:                es (DiffusionESPlanner)"
  echo " num_worlds:             ${NUM_WORLDS}"
  echo " num_scenarios:          ${NUM_SCENARIOS}"
  echo " population_size:        ${POPULATION_SIZE}"
  echo " elite_size:             ${ELITE_SIZE}"
  echo " num_es_iterations:      ${NUM_ES_ITERATIONS}"
  echo " resample_timesteps:     ${RESAMPLE_TIMESTEPS}"
  echo "============================================================"
  echo
  echo "[TRAIN symlink dir]"
  ls -la "${TF_LINK_ROOT}/training"
  echo
  echo "[VAL symlink dir]"
  ls -la "${TF_LINK_ROOT}/validation"
  echo
} | tee -a "${LOG_FILE}"

run_one() {
  local split="$1"
  local tag="$2"
  local label="$3"

  {
    echo
    echo "############################################################"
    echo "# START: ${label}"
    echo "# split=${split}  tag=${tag}"
    echo "# planner=es  other_agent_forecast=${OTHER_AGENT_FORECAST}"
    if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
      echo "# wm_refresh_steps=${WM_REFRESH_STEPS}"
    fi
    echo "# RUN_NAME=${RUN_NAME}"
    echo "# time=$(date -Is)"
    echo "############################################################"
  } | tee -a "${LOG_FILE}"

  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  CMD=(
    python "${ROOT}/simulation/run_simulation.py"
    --planner es
    --diffusion_ckpt_path "${DIFFUSION_CKPT}"
    --tfrecord_dir "${TF_LINK_ROOT}"
    --lane_graph_dir "${LANE_GRAPH_DIR}"
    --exp_dir "${EXP_DIR}"
    --split "${split}"
    --num_worlds "${NUM_WORLDS}"
    --num_scenarios "${NUM_SCENARIOS}"
    --population_size "${POPULATION_SIZE}"
    --elite_size "${ELITE_SIZE}"
    --num_es_iterations "${NUM_ES_ITERATIONS}"
    --resample_timesteps "${RESAMPLE_TIMESTEPS}"
    --replan_interval_steps "${REPLAN_INTERVAL_STEPS}"
    --other_agent_forecast "${OTHER_AGENT_FORECAST}"
    --visualize_mode none
    --tag "${tag}"
  )
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
    echo "ERROR: python exited ${py_rc} during ${label}" | tee -a "${LOG_FILE}"
    exit "${py_rc}"
  fi

  {
    echo
    echo "############################################################"
    echo "# DONE: ${label}"
    echo "# time=$(date -Is)"
    echo "############################################################"
    echo
  } | tee -a "${LOG_FILE}"
}

run_one training "${TRAIN_TAG}" "TRAINING set — tfrecord shards 00000 + 00001 (${RUN_NAME})"
run_one validation "${VAL_TAG}" "VALIDATION set — tfrecord shards 00000 + 00001 (${RUN_NAME})"

{
  echo
  echo "============================================================"
  echo " ALL RUNS FINISHED"
  echo " Outputs live under: ${EXP_DIR}"
  echo " Expected run folders:"
  echo "   ${EXP_DIR}/simulation_es_training_${TRAIN_TAG}_*"
  echo "   ${EXP_DIR}/simulation_es_validation_${VAL_TAG}_*"
  echo " Log: ${LOG_FILE}"
  echo "============================================================"
  echo
  echo "Recent experiment dirs:"
  ls -1dt "${EXP_DIR}"/simulation_es_* 2>/dev/null | head -10 || true
} | tee -a "${LOG_FILE}"
