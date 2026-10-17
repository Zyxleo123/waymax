#!/bin/bash
#SBATCH --job-name=es_wm_mf_fail1500
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/es_wm_mf_fail1500_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/es_wm_mf_fail1500_%j.err

# Diffusion ES (es) or flow-matching ES (es_fm) on the shared 1500 FM-failure
# subset with the multi-future Gaussian WM (expected_independent). Same settings
# as slurm_es_failure_subset_1500.sh {es|es_fm} wm_mf, plus a choice of non-ego rollout.
#
#   mkdir -p /zfsauton/scratch/ishanj/world_model_sim/logs   # once, before submitting
#   CKPT=/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/latest.pt
#   sbatch --job-name=es_fm_wm_mf_log slurm_es_fail1500_wm_mf.sh es_fm log "${CKPT}"
#   sbatch --job-name=es_diff_wm_mf_idm slurm_es_fail1500_wm_mf.sh es idm "${CKPT}"
#   # Override ES iterations / replan interval (defaults 4 / 10); non-default values add _it{N}_rp{M} to the run name:
#   sbatch --export=ALL,NUM_ES_ITERATIONS=6,REPLAN_INTERVAL_STEPS=40 slurm_es_fail1500_wm_mf.sh es_fm log "${CKPT}"
#   # ES offroad objective (default binary); margin_tiebreak adds _offmargin_tiebreak to the run name:
#   sbatch --export=ALL,OFFROAD_MODE=margin_tiebreak slurm_es_fail1500_wm_mf.sh es_fm log "${CKPT}"
#   # WM mode-classifier temperature (default 1 = as trained); non-default adds _T<v> to the run name:
#   sbatch --export=ALL,WM_MODE_TEMPERATURE=0.1 slurm_es_fail1500_wm_mf.sh es log "${CKPT}"
#   # Only part of the targets (e.g. one chunk per GPU; merge the summaries afterwards):
#   sbatch --export=ALL,TARGET_JSON=/path/chunk0.json,RUN_SUFFIX=_c0 slurm_es_fail1500_wm_mf.sh es log "${CKPT}"
set -euo pipefail

USAGE="Usage: sbatch slurm_es_fail1500_wm_mf.sh {es|es_fm} {log|idm} <world_model_ckpt.pt>"
PLANNER="${1:?${USAGE}}"
SIM_AGENTS="${2:?${USAGE}}"
WORLD_MODEL_CKPT="${3:?${USAGE}}"
case "${PLANNER}" in
  es) PLANNER_TAG="es_diffusion" ;;
  es_fm) PLANNER_TAG="es_fm" ;;
  *) echo "ERROR: planner must be es or es_fm, got '${PLANNER}'"; exit 1 ;;
esac
case "${SIM_AGENTS}" in
  log|idm) ;;
  *) echo "ERROR: sim_agents must be log or idm, got '${SIM_AGENTS}'"; exit 1 ;;
esac
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
ES_TAG=""
if [[ "${NUM_ES_ITERATIONS}" != "4" || "${REPLAN_INTERVAL_STEPS}" != "10" ]]; then
  ES_TAG="_it${NUM_ES_ITERATIONS}_rp${REPLAN_INTERVAL_STEPS}"
fi
OFFROAD_MODE="${OFFROAD_MODE:-binary}"
case "${OFFROAD_MODE}" in
  binary) OFF_TAG="" ;;
  margin_tiebreak) OFF_TAG="_off${OFFROAD_MODE}" ;;
  *) echo "ERROR: OFFROAD_MODE must be binary or margin_tiebreak, got '${OFFROAD_MODE}'"; exit 1 ;;
esac

WM_MODE_TEMPERATURE="${WM_MODE_TEMPERATURE:-1}"
TEMP_TAG=""
if [[ "${WM_MODE_TEMPERATURE}" != "1" ]]; then
  TEMP_TAG="_T${WM_MODE_TEMPERATURE}"
fi

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
EXP_ROOT="/zfsauton/scratch/ishanj/world_model_sim"

SUBSET_DIR="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
TARGET_JSON="${TARGET_JSON:-${SUBSET_DIR}/target_scenarios.json}"
TFRECORD_DIR="${SUBSET_DIR}/tfrecords"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
FM_CKPT="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500"

# Teammate's env (yours has a jaxlib / jax_cuda12_plugin version mismatch).
set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Tag runs with the checkpoint folder + file, e.g. 20261005_092950_latest.
WM_RUN="$(basename "$(dirname "${WORLD_MODEL_CKPT}")")"
WM_TAG="${WM_RUN#multiFuture_GaussianWorldModel_}_$(basename "${WORLD_MODEL_CKPT}" .pt)"
RUN_NAME="${PLANNER_TAG}_wm_mf_${WM_TAG}_expected_independent_${SIM_AGENTS}${ES_TAG}${OFF_TAG}${TEMP_TAG}_fail1500${RUN_SUFFIX:-}"
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
LOG_FILE="${EXP_ROOT}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${EXP_DIR}" "${EXP_ROOT}/logs"

for p in "${TARGET_JSON}" "${TFRECORD_DIR}/training" "${LANE_GRAPH_DIR}" "${WORLD_MODEL_CKPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: missing ${p}" | tee -a "${LOG_FILE}"
    exit 1
  fi
done

echo "RUN_NAME=${RUN_NAME}  planner=${PLANNER}  sim_agents=${SIM_AGENTS}  es_iterations=${NUM_ES_ITERATIONS}  replan_interval=${REPLAN_INTERVAL_STEPS}  offroad_mode=${OFFROAD_MODE}  wm_mode_temperature=${WM_MODE_TEMPERATURE}  targets=${TARGET_JSON}  ckpt=${WORLD_MODEL_CKPT}" | tee -a "${LOG_FILE}"

CMD=(
  python "${ROOT}/simulation/run_simulation.py"
  --planner "${PLANNER}"
  --tfrecord_dir "${TFRECORD_DIR}"
  --lane_graph_dir "${LANE_GRAPH_DIR}"
  --exp_dir "${EXP_DIR}"
  --split training
  --num_worlds 5
  --num_scenarios 100000
  --population_size 64
  --elite_size 8
  --num_es_iterations "${NUM_ES_ITERATIONS}"
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}"
  --other_agent_forecast wm
  --target_scenario_path "${TARGET_JSON}"
  --visualize_mode none
  --tag "${RUN_NAME}"
  --sim_agents "${SIM_AGENTS}"
  --world_model_ckpt "${WORLD_MODEL_CKPT}"
  --world_model_device auto
  --wm_backend multifuture
  --wm_refresh_steps 10
  --num_collision_worlds 1
  --collision_reduce any
  --collision_sample_seed 0
  --wm_planning_mode expected_independent
  --offroad_mode "${OFFROAD_MODE}"
  --wm_mode_temperature "${WM_MODE_TEMPERATURE}"
)

if [[ "${PLANNER}" == "es" ]]; then
  CMD+=(--diffusion_ckpt_path "${DIFFUSION_CKPT}" --resample_timesteps 5)
else
  CMD+=(--fm_ckpt_path "${FM_CKPT}" --resample_t_start 0.2 --resample_num_steps 20
        --fm_num_steps 50 --fm_solver euler)
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
