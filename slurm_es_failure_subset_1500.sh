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
#SBATCH --output=/zfsauton/scratch/sbellad/Flow_matching_sim/logs/es_fail1500_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/Flow_matching_sim/logs/es_fail1500_%j.err

# Shared FM-failure subset (default n=1500, seed=0) — same scenes for all runs.
# Ego replan + WM refresh are fixed at 10 steps.
#
# Submit EXACTLY like this (no --export flags — they break on this cluster):
#   sbatch --job-name=es_fm_gt_fail1500 slurm_es_failure_subset_1500.sh es_fm gt
#   sbatch --job-name=es_fm_cv_fail1500 slurm_es_failure_subset_1500.sh es_fm cv
#
# Or use the wrapper (same thing):
#   bash slurm_submit_es_fm_fail1500.sh
#
# Outputs:
#   Diffusion es + cv|gt  -> /zfsauton/scratch/sbellad/immediate_exp/
#   Flow-matching es_fm + cv|gt -> /zfsauton/scratch/sbellad/Flow_matching_sim/
#   wm_*                  -> /zfsauton/scratch/sbellad/world_model_sim/
#
set -euo pipefail

PLANNER="${1:-}"
FORECAST_MODE="${2:-}"
# Optional 3rd arg: mf planning mode (avoids fragile sbatch --export=ALL).
PLANNING_MODE_ARG="${3:-}"
PLANNER="$(echo "${PLANNER}" | tr '[:upper:]' '[:lower:]')"
FORECAST_MODE="$(echo "${FORECAST_MODE}" | tr '[:upper:]' '[:lower:]')"
PLANNING_MODE_ARG="$(echo "${PLANNING_MODE_ARG}" | tr '[:upper:]' '[:lower:]')"
set --

USAGE="Usage: sbatch slurm_es_failure_subset_1500.sh {es|es_fm} {cv|gt|wm_det|wm_gau|wm_gau_mean|wm_mf|wm_occ|wm_ego} [top1|expected_independent|worst_case_independent]"

if [[ "${PLANNER}" != "es" && "${PLANNER}" != "es_fm" ]]; then
  echo "${USAGE}"
  exit 1
fi
case "${FORECAST_MODE}" in
  cv|gt|wm_det|wm_gau|wm_gau_mean|wm_mf|wm_occ|wm_ego) ;;
  *)
    echo "${USAGE}"
    exit 1
    ;;
esac

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

# Fixed: ego replan every 10 steps; WM refresh every replan.
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
WM_REFRESH_STEPS=10

WORLD_MODEL_DEVICE="${WORLD_MODEL_DEVICE:-auto}"
COLLISION_REDUCE="${COLLISION_REDUCE:-mean}"
COLLISION_SAMPLE_SEED="${COLLISION_SAMPLE_SEED:-0}"
WM_BACKEND="production"
WM_PLANNING_MODE=""
GAU_CKPT_ROOT="/zfsauton/scratch/sbellad/world_model/ckpts_gaussian/gaussian_world_model_20260916_194054"

# Map CLI forecast mode -> run_simulation --other_agent_forecast + WM settings.
# Note: mf/occ/ego trainings hit the 2-day time limit; defaults use latest usable epochs.
# Gaussian defaults to pre-collapse ep370 (best val ADE before ~ep397 blowup).
# Prior MC run used ep470 and is archived as:
#   immediate_exp/es_diffusion_wm_gau_fail1500_epoch0470
case "${FORECAST_MODE}" in
  cv|gt)
    OTHER_AGENT_FORECAST="${FORECAST_MODE}"
    WORLD_MODEL_CKPT=""
    NUM_COLLISION_WORLDS=1
    ;;
  wm_det)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="production"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT_DET:-/zfsauton/scratch/sbellad/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt}"
    NUM_COLLISION_WORLDS=1
    ;;
  wm_gau)
    # MC soft collision: K samples from N(μ,σ), reduce=mean
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="production"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT_GAU:-${GAU_CKPT_ROOT}/epoch_0370.pt}"
    NUM_COLLISION_WORLDS="${NUM_COLLISION_WORLDS:-8}"
    ;;
  wm_gau_mean)
    # Ablation: same ckpt, mean trajectory only (hard SAT, K=1)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="production"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT_GAU:-${GAU_CKPT_ROOT}/epoch_0370.pt}"
    NUM_COLLISION_WORLDS=1
    COLLISION_REDUCE="any"
    ;;
  wm_mf)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="multifuture"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT_MF:-/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt}"
    # Agent-wise multimodal risk (not joint same-k worlds).
    # Prefer 3rd CLI arg; else env WM_PLANNING_MODE; else expected_independent.
    if [[ -n "${PLANNING_MODE_ARG}" ]]; then
      WM_PLANNING_MODE="${PLANNING_MODE_ARG}"
    else
      WM_PLANNING_MODE="${WM_PLANNING_MODE:-expected_independent}"
    fi
    case "${WM_PLANNING_MODE}" in
      top1|expected_independent|worst_case_independent|expected|worst_case) ;;
      *)
        echo "ERROR: unknown mf planning mode '${WM_PLANNING_MODE}'"
        echo "${USAGE}"
        exit 1
        ;;
    esac
    # Independent scorer does not use the gaussian K-world SAT ensemble.
    NUM_COLLISION_WORLDS=1
    COLLISION_REDUCE="any"
    ;;
  wm_occ)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="occupancy"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT_OCC:-/zfsauton/scratch/sbellad/world_model/ckpts_occupancy/occupancy_worldModel_20260916_202827/epoch_0255.pt}"
    # SAT collision still uses mean boxes (occupancy risk not in main ES metrics yet).
    NUM_COLLISION_WORLDS=1
    ;;
  wm_ego)
    OTHER_AGENT_FORECAST="wm"
    WM_BACKEND="ego"
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT_EGO:-/zfsauton/scratch/sbellad/world_model/ckpts_ego_conditioned/egoConditioned_WM_20260916_203857/epoch_0455.pt}"
    NUM_COLLISION_WORLDS=1
    ;;
esac

# Tag WM runs with ckpt epoch (+ k for gaussian) so dirs are self-describing.
WM_EPOCH_TAG=""
if [[ -n "${WORLD_MODEL_CKPT}" ]]; then
  _ckpt_base="$(basename "${WORLD_MODEL_CKPT}" .pt)"
  if [[ "${_ckpt_base}" =~ ^epoch_([0-9]+)$ ]]; then
    WM_EPOCH_TAG="epoch${BASH_REMATCH[1]}"
  else
    WM_EPOCH_TAG="${_ckpt_base}"
  fi
fi

case "${FORECAST_MODE}" in
  wm_gau)
    FORECAST_TAG="wm_gau_${WM_EPOCH_TAG}_k${NUM_COLLISION_WORLDS}"
    ;;
  wm_gau_mean)
    FORECAST_TAG="wm_gau_mean_${WM_EPOCH_TAG}_k1"
    ;;
  wm_mf)
    # Include planning mode so expected_independent ≠ old joint-world mf run.
    FORECAST_TAG="wm_mf_${WM_EPOCH_TAG}_${WM_PLANNING_MODE}"
    ;;
  wm_det|wm_occ|wm_ego)
    FORECAST_TAG="${FORECAST_MODE}_${WM_EPOCH_TAG}"
    ;;
  *)
    FORECAST_TAG="${FORECAST_MODE}"
    ;;
esac

if [[ "${PLANNER}" == "es" ]]; then
  RUN_NAME="es_diffusion_${FORECAST_TAG}_fail1500"
else
  RUN_NAME="es_fm_${FORECAST_TAG}_fail1500"
fi
# Optional disambiguator for hparam variants (set inside job wrapper, not via sbatch --export).
if [[ -n "${RUN_SUFFIX:-}" ]]; then
  RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
fi

# WM sims -> world_model_sim.
# Diffusion CV/GT -> immediate_exp.
# Flow-matching CV/GT -> Flow_matching_sim (keeps FM runs separate).
if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
  EXP_ROOT="${WM_SIM_ROOT:-/zfsauton/scratch/sbellad/world_model_sim}"
elif [[ "${PLANNER}" == "es_fm" ]]; then
  EXP_ROOT="${FM_SIM_ROOT:-/zfsauton/scratch/sbellad/Flow_matching_sim}"
else
  EXP_ROOT="${CV_GT_SIM_ROOT:-/zfsauton/scratch/sbellad/immediate_exp}"
fi
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"

NUM_WORLDS="${NUM_WORLDS:-5}"
POPULATION_SIZE="${POPULATION_SIZE:-64}"
ELITE_SIZE="${ELITE_SIZE:-8}"
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
RESAMPLE_TIMESTEPS="${RESAMPLE_TIMESTEPS:-5}"
RESAMPLE_T_START="${RESAMPLE_T_START:-0.2}"
RESAMPLE_NUM_STEPS="${RESAMPLE_NUM_STEPS:-20}"
FM_NUM_STEPS="${FM_NUM_STEPS:-50}"
FM_SOLVER="${FM_SOLVER:-euler}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EXP_ROOT}/logs"
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
  echo " planner=${PLANNER}  forecast_mode=${FORECAST_MODE}  other_agent_forecast=${OTHER_AGENT_FORECAST}"
  echo " replan_interval_steps=${REPLAN_INTERVAL_STEPS}"
  if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
    echo " wm_backend=${WM_BACKEND}"
    echo " wm_refresh_steps=${WM_REFRESH_STEPS}"
    echo " world_model_ckpt=${WORLD_MODEL_CKPT}"
    echo " world_model_device=${WORLD_MODEL_DEVICE}"
    echo " num_collision_worlds=${NUM_COLLISION_WORLDS}"
    echo " collision_reduce=${COLLISION_REDUCE}"
    echo " collision_sample_seed=${COLLISION_SAMPLE_SEED}"
    if [[ -n "${WM_PLANNING_MODE}" ]]; then
      echo " wm_planning_mode=${WM_PLANNING_MODE}"
    fi
  fi
  echo " subset_dir=${SUBSET_DIR}"
  echo " target_json=${TARGET_JSON}"
  echo " tfrecord_dir=${TFRECORD_DIR}"
  echo " exp_dir=${EXP_DIR}"
  echo " log_file=${LOG_FILE}"
  if [[ "${PLANNER}" == "es_fm" ]]; then
    echo " fm: t_start=${RESAMPLE_T_START} resample_num_steps=${RESAMPLE_NUM_STEPS} fm_num_steps=${FM_NUM_STEPS} solver=${FM_SOLVER}"
    echo " es: pop=${POPULATION_SIZE} elite=${ELITE_SIZE} iters=${NUM_ES_ITERATIONS}"
  else
    echo " diffusion resample_timesteps=${RESAMPLE_TIMESTEPS}"
    echo " es: pop=${POPULATION_SIZE} elite=${ELITE_SIZE} iters=${NUM_ES_ITERATIONS}"
  fi
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
  CMD+=(--fm_ckpt_path "${FM_CKPT}" --resample_t_start "${RESAMPLE_T_START}"
        --resample_num_steps "${RESAMPLE_NUM_STEPS}" --fm_num_steps "${FM_NUM_STEPS}"
        --fm_solver "${FM_SOLVER}")
fi

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
  echo "ERROR: python exited ${py_rc}" | tee -a "${LOG_FILE}"
  exit "${py_rc}"
fi

echo "DONE ${RUN_NAME}" | tee -a "${LOG_FILE}"
ls -1dt "${EXP_DIR}"/simulation_* 2>/dev/null | head -5 | tee -a "${LOG_FILE}"
