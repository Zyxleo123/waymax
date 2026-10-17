#!/bin/bash
#SBATCH --job-name=es_wm_temp
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/es_wm_temp_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/es_wm_temp_%A_%a.err

# Diffusion ES with the multi-future Gaussian WM on the 1500-scenario failure subset,
# sweeping the WM mode-classifier temperature: collision scoring uses
# softmax(mode_logits / T). One array task per temperature (TEMPERATURES[task id]).
# Each run also writes <output_dir>/wm_forecast_stats.json: mode-probability mean / std
# (per mode and per rank, raw and tempered), and ADE / FDE of the WM against the logged
# future (all agents and moving agents only).
#
#   sbatch slurm_es_wm_temperature_sweep.sh                      # T = 0.1, 1, 10
#   sbatch --export=ALL,TEMPERATURES="0.3 3" --array=0-1 slurm_es_wm_temperature_sweep.sh
#   sbatch --export=ALL,WORLD_MODEL_CKPT=/path/to/ckpt.pt slurm_es_wm_temperature_sweep.sh
set -euo pipefail

read -r -a TEMPS <<< "${TEMPERATURES:-0.1 1 10}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK >= ${#TEMPS[@]} )); then
  echo "ERROR: array task ${TASK} but only ${#TEMPS[@]} temperatures (${TEMPS[*]})"; exit 1
fi
T="${TEMPS[${TASK}]}"

WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt}"
SIM_AGENTS="${SIM_AGENTS:-log}"      # ADE / FDE are against the log: only the real future under log replay
OFFROAD_MODE="${OFFROAD_MODE:-margin_tiebreak}"
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
EXP_ROOT="/zfsauton/scratch/ishanj/world_model_sim/wm_temperature_sweep"
LOG_ROOT="/zfsauton/scratch/ishanj/world_model_sim/logs"
SUBSET_DIR="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
TARGET_JSON="${TARGET_JSON:-${SUBSET_DIR}/target_scenarios.json}"
TFRECORD_DIR="${SUBSET_DIR}/tfrecords"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WM_RUN="$(basename "$(dirname "${WORLD_MODEL_CKPT}")")"
WM_TAG="${WM_RUN#multiFuture_GaussianWorldModel_}_$(basename "${WORLD_MODEL_CKPT}" .pt)"
RUN_NAME="es_diffusion_wm_mf_${WM_TAG}_expected_independent_${SIM_AGENTS}_off${OFFROAD_MODE}_it${NUM_ES_ITERATIONS}_rp${REPLAN_INTERVAL_STEPS}_T${T}_fail1500${RUN_SUFFIX:-}"
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_ROOT}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${EXP_DIR}" "${LOG_ROOT}"

for p in "${TARGET_JSON}" "${TFRECORD_DIR}/training" "${LANE_GRAPH_DIR}" "${WORLD_MODEL_CKPT}" "${DIFFUSION_CKPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: missing ${p}" | tee -a "${LOG_FILE}"
    exit 1
  fi
done

echo "RUN_NAME=${RUN_NAME}  task=${TASK}  temperature=${T}  sim_agents=${SIM_AGENTS}  offroad_mode=${OFFROAD_MODE}  es_iterations=${NUM_ES_ITERATIONS}  replan_interval=${REPLAN_INTERVAL_STEPS}  targets=${TARGET_JSON}  ckpt=${WORLD_MODEL_CKPT}  commit=$(git -C "${ROOT}" rev-parse --short HEAD)" | tee -a "${LOG_FILE}"

set +e
python "${ROOT}/simulation/run_simulation.py" \
  --planner es \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --resample_timesteps 5 \
  --tfrecord_dir "${TFRECORD_DIR}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --split training \
  --num_worlds 5 \
  --num_scenarios 100000 \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations "${NUM_ES_ITERATIONS}" \
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" \
  --other_agent_forecast wm \
  --target_scenario_path "${TARGET_JSON}" \
  --visualize_mode none \
  --tag "${RUN_NAME}" \
  --sim_agents "${SIM_AGENTS}" \
  --world_model_ckpt "${WORLD_MODEL_CKPT}" \
  --world_model_device auto \
  --wm_backend multifuture \
  --wm_refresh_steps 10 \
  --num_collision_worlds 1 \
  --collision_reduce any \
  --collision_sample_seed 0 \
  --wm_planning_mode expected_independent \
  --offroad_mode "${OFFROAD_MODE}" \
  --wm_mode_temperature "${T}" 2>&1 | tee -a "${LOG_FILE}"
py_rc=${PIPESTATUS[0]}
set -e
if [[ "${py_rc}" -ne 0 ]]; then
  echo "ERROR: python exited ${py_rc}" | tee -a "${LOG_FILE}"
  exit "${py_rc}"
fi
echo "DONE ${RUN_NAME}" | tee -a "${LOG_FILE}"
