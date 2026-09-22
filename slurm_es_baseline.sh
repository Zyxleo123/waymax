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
#SBATCH --output=/zfsauton/scratch/sbellad/world_model_sim/logs/es_diffusion_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/world_model_sim/logs/es_diffusion_%j.err

# Diffusion-ES closed-loop on TRAIN/VAL shards 00+01 (--planner es).
#
#   # baselines
#   sbatch --job-name=es_diffusion_cv       slurm_es_baseline.sh cv
#   sbatch --job-name=es_diffusion_gt       slurm_es_baseline.sh gt
#   sbatch --job-name=es_diffusion_wm_det   slurm_es_baseline.sh wm_det
#   sbatch --job-name=es_diffusion_wm10     slurm_es_baseline.sh wm 10   # alias of wm_det
#
#   # Multifuture expected_independent (primary WM baseline on this data)
#   sbatch --job-name=es_diff_wm_mf_exp_ind slurm_es_baseline.sh wm_mf
#   sbatch --job-name=es_diff_wm_mf_top1    slurm_es_baseline.sh wm_mf 10 top1
#   sbatch --job-name=es_diff_wm_mf_worst   slurm_es_baseline.sh wm_mf 10 worst_case_independent
#
# Args: MODE [wm_refresh_steps] [mf_planning_mode]
#   MODE: cv|gt|wm|wm_det|wm_mf|wm_gau|wm_gau_mean
#   refresh default 10; mf planning default expected_independent
#
# Outputs:
#   cv|gt     -> /zfsauton/scratch/sbellad/immediate_exp/
#   wm_*      -> /zfsauton/scratch/sbellad/world_model_sim/
#                (exp dirs, app logs, tfrecord symlinks, and Slurm .out/.err)
#
set -euo pipefail

# MUST capture args before sourcing conda activate (else conda treats $1 as env name).
MODE="${1:-${OTHER_AGENT_FORECAST:-cv}}"
MODE="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')"
ARG2="${2:-}"
ARG3="${3:-}"
set --

USAGE="Usage: sbatch slurm_es_baseline.sh {cv|gt|wm|wm_det|wm_mf|wm_gau|wm_gau_mean} [wm_refresh_steps] [mf_planning_mode]"

case "${MODE}" in
  cv|gt|wm|wm_det|wm_mf|wm_gau|wm_gau_mean) ;;
  *)
    echo "ERROR: ${USAGE}"
    echo "  got MODE='${MODE}'"
    exit 1
    ;;
esac

# Parse optional refresh + mf planning mode.
WM_REFRESH_STEPS="${WM_REFRESH_STEPS:-10}"
WM_PLANNING_MODE="${WM_PLANNING_MODE:-}"
if [[ -n "${ARG2}" ]]; then
  if [[ "${ARG2}" =~ ^[0-9]+$ ]]; then
    WM_REFRESH_STEPS="${ARG2}"
  else
    # Allow: sbatch ... wm_mf expected_independent
    WM_PLANNING_MODE="${ARG2}"
  fi
fi
if [[ -n "${ARG3}" ]]; then
  WM_PLANNING_MODE="${ARG3}"
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

WORLD_MODEL_DEVICE="${WORLD_MODEL_DEVICE:-auto}"
COLLISION_REDUCE="${COLLISION_REDUCE:-mean}"
COLLISION_SAMPLE_SEED="${COLLISION_SAMPLE_SEED:-0}"
WM_BACKEND="production"
NUM_COLLISION_WORLDS=1
GAU_CKPT_ROOT="/zfsauton/scratch/sbellad/world_model/ckpts_gaussian/gaussian_world_model_20260916_194054"
MF_CKPT_DEFAULT="/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt"
DET_CKPT_DEFAULT="/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt"

case "${MODE}" in
  cv|gt)
    OTHER_AGENT_FORECAST="${MODE}"
    WORLD_MODEL_CKPT=""
    ;;
  wm|wm_det)
    # Legacy "wm" == deterministic WM (refresh tag still applied).
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="production"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-${WORLD_MODEL_CKPT_DET:-${DET_CKPT_DEFAULT}}}"
    NUM_COLLISION_WORLDS=1
    ;;
  wm_mf)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="multifuture"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-${WORLD_MODEL_CKPT_MF:-${MF_CKPT_DEFAULT}}}"
    WM_PLANNING_MODE="${WM_PLANNING_MODE:-expected_independent}"
    case "${WM_PLANNING_MODE}" in
      top1|expected_independent|worst_case_independent|expected|worst_case) ;;
      *)
        echo "ERROR: unknown mf planning mode '${WM_PLANNING_MODE}'"
        echo "${USAGE}"
        exit 1
        ;;
    esac
    NUM_COLLISION_WORLDS=1
    COLLISION_REDUCE="any"
    ;;
  wm_gau)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="production"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-${WORLD_MODEL_CKPT_GAU:-${GAU_CKPT_ROOT}/epoch_0370.pt}}"
    NUM_COLLISION_WORLDS="${NUM_COLLISION_WORLDS_OVERRIDE:-8}"
    COLLISION_REDUCE="mean"
    ;;
  wm_gau_mean)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="production"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-${WORLD_MODEL_CKPT_GAU:-${GAU_CKPT_ROOT}/epoch_0370.pt}}"
    NUM_COLLISION_WORLDS=1
    COLLISION_REDUCE="any"
    ;;
esac

# Self-describing run tag.
WM_EPOCH_TAG=""
if [[ -n "${WORLD_MODEL_CKPT}" ]]; then
  _ckpt_base="$(basename "${WORLD_MODEL_CKPT}" .pt)"
  if [[ "${_ckpt_base}" =~ ^epoch_([0-9]+)$ ]]; then
    WM_EPOCH_TAG="epoch${BASH_REMATCH[1]}"
  else
    WM_EPOCH_TAG="${_ckpt_base}"
  fi
fi

case "${MODE}" in
  cv|gt)
    FORECAST_TAG="${MODE}"
    ;;
  wm|wm_det)
    FORECAST_TAG="wm_det_${WM_EPOCH_TAG}_r${WM_REFRESH_STEPS}"
    ;;
  wm_mf)
    FORECAST_TAG="wm_mf_${WM_EPOCH_TAG}_${WM_PLANNING_MODE}_r${WM_REFRESH_STEPS}"
    ;;
  wm_gau)
    FORECAST_TAG="wm_gau_${WM_EPOCH_TAG}_k${NUM_COLLISION_WORLDS}_r${WM_REFRESH_STEPS}"
    ;;
  wm_gau_mean)
    FORECAST_TAG="wm_gau_mean_${WM_EPOCH_TAG}_k1_r${WM_REFRESH_STEPS}"
    ;;
esac

RUN_NAME="es_diffusion_${FORECAST_TAG}"
if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
  EXP_ROOT="${WM_SIM_ROOT:-/zfsauton/scratch/sbellad/world_model_sim}"
else
  EXP_ROOT="${CV_GT_SIM_ROOT:-/zfsauton/scratch/sbellad/immediate_exp}"
fi
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
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
LOG_DIR="${EXP_ROOT}/logs"
TF_LINK_ROOT="${EXP_ROOT}/tfrecord_shards_00_01_${RUN_NAME}_${RUN_STAMP}"
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
    echo " mode=${MODE}  other_agent_forecast=${OTHER_AGENT_FORECAST}"
    if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
      echo " wm_backend=${WM_BACKEND}"
      echo " wm_refresh_steps=${WM_REFRESH_STEPS}"
      echo " world_model_ckpt=${WORLD_MODEL_CKPT}"
      echo " world_model_device=${WORLD_MODEL_DEVICE}"
      echo " num_collision_worlds=${NUM_COLLISION_WORLDS}"
      echo " collision_reduce=${COLLISION_REDUCE}"
      if [[ -n "${WM_PLANNING_MODE}" ]]; then
        echo " wm_planning_mode=${WM_PLANNING_MODE}"
      fi
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
  echo " mode:                   ${MODE}"
  echo " other_agent_forecast:   ${OTHER_AGENT_FORECAST}"
  if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
    echo " wm_backend:             ${WM_BACKEND}"
    echo " wm_refresh_steps:       ${WM_REFRESH_STEPS}"
    echo " world_model_ckpt:       ${WORLD_MODEL_CKPT}"
    echo " world_model_device:     ${WORLD_MODEL_DEVICE}"
    echo " num_collision_worlds:   ${NUM_COLLISION_WORLDS}"
    echo " collision_reduce:       ${COLLISION_REDUCE}"
    if [[ -n "${WM_PLANNING_MODE}" ]]; then
      echo " wm_planning_mode:       ${WM_PLANNING_MODE}"
    fi
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
    echo "# planner=es  mode=${MODE}  other_agent_forecast=${OTHER_AGENT_FORECAST}"
    if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
      echo "# wm_backend=${WM_BACKEND}  wm_refresh_steps=${WM_REFRESH_STEPS}"
      if [[ -n "${WM_PLANNING_MODE}" ]]; then
        echo "# wm_planning_mode=${WM_PLANNING_MODE}"
      fi
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
      --wm_backend "${WM_BACKEND}"
      --wm_refresh_steps "${WM_REFRESH_STEPS}"
      --num_collision_worlds "${NUM_COLLISION_WORLDS}"
      --collision_reduce "${COLLISION_REDUCE}"
      --collision_sample_seed "${COLLISION_SAMPLE_SEED}"
    )
    if [[ -n "${WM_PLANNING_MODE}" ]]; then
      CMD+=(--wm_planning_mode "${WM_PLANNING_MODE}")
    fi
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
