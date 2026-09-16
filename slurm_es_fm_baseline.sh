#!/bin/bash
#SBATCH --job-name=es_fm
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/es_fm_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/es_fm_%j.err

# Flow-matching ES closed-loop (--planner es_fm).
# Positional arg selects other-agent forecast + names everything clearly:
#   sbatch --job-name=es_fm_cv slurm_es_fm_baseline.sh cv
#   sbatch --job-name=es_fm_gt slurm_es_fm_baseline.sh gt
#
# Outputs / logs:
#   /zfsauton/scratch/sbellad/immediate_exp/es_fm_cv
#   /zfsauton/scratch/sbellad/immediate_exp/es_fm_gt
#   logs/es_fm_{cv|gt}_train_val_shards00_01_<stamp>.log
#
set -euo pipefail

# MUST capture $1 before sourcing conda activate (else conda treats $1 as env name).
OTHER_AGENT_FORECAST="${1:-${OTHER_AGENT_FORECAST:-cv}}"
OTHER_AGENT_FORECAST="$(echo "${OTHER_AGENT_FORECAST}" | tr '[:upper:]' '[:lower:]')"
set --
if [[ "${OTHER_AGENT_FORECAST}" != "cv" && "${OTHER_AGENT_FORECAST}" != "gt" ]]; then
  echo "ERROR: arg must be 'cv' or 'gt', got '${OTHER_AGENT_FORECAST}'"
  echo "Usage: sbatch [--job-name=es_fm_cv] slurm_es_fm_baseline.sh [cv|gt]"
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

RUN_NAME="es_fm_${OTHER_AGENT_FORECAST}"
EXP_DIR="/zfsauton/scratch/sbellad/immediate_exp/${RUN_NAME}"
TF_ROOT="/zfsauton/scratch/eshau/womd/tf_example"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
FM_CKPT="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500"

NUM_WORLDS="${NUM_WORLDS:-5}"
NUM_SCENARIOS="${NUM_SCENARIOS:-100000}"
POPULATION_SIZE="${POPULATION_SIZE:-64}"
ELITE_SIZE="${ELITE_SIZE:-8}"
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
RESAMPLE_T_START="${RESAMPLE_T_START:-0.2}"
RESAMPLE_NUM_STEPS="${RESAMPLE_NUM_STEPS:-}"
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
  "${FM_CKPT}" \
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

{
  echo "============================================================"
  echo " Flow-matching ES (${RUN_NAME}): TRAIN shards 0+1 + VAL shards 0+1"
  echo " stamp:                  ${RUN_STAMP}"
  echo " other_agent_forecast:   ${OTHER_AGENT_FORECAST}"
  echo " exp_dir:                ${EXP_DIR}"
  echo " log_file:               ${LOG_FILE}"
  echo " tfrecord_dir:           ${TF_LINK_ROOT}"
  echo " lane_graph_dir:         ${LANE_GRAPH_DIR}"
  echo " flow matching:          ${FM_CKPT}"
  echo " planner:                es_fm (FMESPlanner)"
  echo " num_worlds:             ${NUM_WORLDS}"
  echo " num_scenarios:          ${NUM_SCENARIOS}"
  echo " population_size:        ${POPULATION_SIZE}"
  echo " elite_size:             ${ELITE_SIZE}"
  echo " num_es_iterations:      ${NUM_ES_ITERATIONS}"
  echo " resample_t_start:       ${RESAMPLE_T_START}"
  echo " resample_num_steps:     ${RESAMPLE_NUM_STEPS:-"(policy default)"}"
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
    echo "# planner=es_fm  other_agent_forecast=${OTHER_AGENT_FORECAST}"
    echo "# RUN_NAME=${RUN_NAME}"
    echo "# time=$(date -Is)"
    echo "############################################################"
  } | tee -a "${LOG_FILE}"

  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  CMD=(
    python "${ROOT}/simulation/run_simulation.py"
    --planner es_fm
    --fm_ckpt_path "${FM_CKPT}"
    --tfrecord_dir "${TF_LINK_ROOT}"
    --lane_graph_dir "${LANE_GRAPH_DIR}"
    --exp_dir "${EXP_DIR}"
    --split "${split}"
    --num_worlds "${NUM_WORLDS}"
    --num_scenarios "${NUM_SCENARIOS}"
    --population_size "${POPULATION_SIZE}"
    --elite_size "${ELITE_SIZE}"
    --num_es_iterations "${NUM_ES_ITERATIONS}"
    --resample_t_start "${RESAMPLE_T_START}"
    --replan_interval_steps "${REPLAN_INTERVAL_STEPS}"
    --other_agent_forecast "${OTHER_AGENT_FORECAST}"
    --visualize_mode none
    --tag "${tag}"
  )
  if [[ -n "${RESAMPLE_NUM_STEPS}" ]]; then
    CMD+=(--resample_num_steps "${RESAMPLE_NUM_STEPS}")
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
  echo "   ${EXP_DIR}/simulation_es_fm_training_${TRAIN_TAG}_*"
  echo "   ${EXP_DIR}/simulation_es_fm_validation_${VAL_TAG}_*"
  echo " Log: ${LOG_FILE}"
  echo "============================================================"
  echo
  echo "Recent experiment dirs:"
  ls -1dt "${EXP_DIR}"/simulation_es_fm_* 2>/dev/null | head -10 || true
} | tee -a "${LOG_FILE}"
