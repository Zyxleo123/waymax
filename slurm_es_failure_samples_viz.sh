#!/bin/bash
#SBATCH --job-name=es_fail_viz
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/es_fail_viz_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/es_fail_viz_%j.err

# Re-run diffusion ES (es) or FM-ES (es_fm) with the multi-future Gaussian WM on a
# small hand-picked target list and render a video for every scenario.
# Same settings as slurm_es_fail1500_wm_mf.sh except --num_worlds 1 (target lists
# from failure sampling have one scenario per shard; the runner drops partial
# batches) and --visualize_mode all.
#
#   D=/zfsauton/scratch/ishanj/world_model_sim/es_fm_wm_mf_20261005_092950_latest_expected_independent_idm_fail1500
#   CKPT=/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/latest.pt
#   sbatch slurm_es_failure_samples_viz.sh es idm "${CKPT}" "${D}/failure_samples_25_targets.json" fail25
set -euo pipefail

USAGE="Usage: sbatch slurm_es_failure_samples_viz.sh {es|es_fm} {log|idm} <world_model_ckpt.pt> <target_scenarios.json> <tag>"
PLANNER="${1:?${USAGE}}"
SIM_AGENTS="${2:?${USAGE}}"
WORLD_MODEL_CKPT="${3:?${USAGE}}"
TARGET_JSON="${4:?${USAGE}}"
SAMPLE_TAG="${5:?${USAGE}}"
case "${PLANNER}" in
  es) PLANNER_TAG="es_diffusion" ;;
  es_fm) PLANNER_TAG="es_fm" ;;
  *) echo "ERROR: planner must be es or es_fm, got '${PLANNER}'"; exit 1 ;;
esac
case "${SIM_AGENTS}" in
  log|idm) ;;
  *) echo "ERROR: sim_agents must be log or idm, got '${SIM_AGENTS}'"; exit 1 ;;
esac

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
EXP_ROOT="/zfsauton/scratch/ishanj/world_model_sim"

TFRECORD_DIR="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
FM_CKPT="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500"

# Teammate's env (yours has a jaxlib / jax_cuda12_plugin version mismatch).
set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WM_RUN="$(basename "$(dirname "${WORLD_MODEL_CKPT}")")"
WM_TAG="${WM_RUN#multiFuture_GaussianWorldModel_}_$(basename "${WORLD_MODEL_CKPT}" .pt)"
RUN_NAME="${PLANNER_TAG}_wm_mf_${WM_TAG}_expected_independent_${SIM_AGENTS}_${SAMPLE_TAG}_viz"
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
LOG_FILE="${EXP_ROOT}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${EXP_DIR}" "${EXP_ROOT}/logs"

for p in "${TARGET_JSON}" "${TFRECORD_DIR}/training" "${LANE_GRAPH_DIR}" "${WORLD_MODEL_CKPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: missing ${p}" | tee -a "${LOG_FILE}"
    exit 1
  fi
done

echo "RUN_NAME=${RUN_NAME}  planner=${PLANNER}  sim_agents=${SIM_AGENTS}  ckpt=${WORLD_MODEL_CKPT}  targets=${TARGET_JSON}" | tee -a "${LOG_FILE}"

CMD=(
  python "${ROOT}/simulation/run_simulation.py"
  --planner "${PLANNER}"
  --tfrecord_dir "${TFRECORD_DIR}"
  --lane_graph_dir "${LANE_GRAPH_DIR}"
  --exp_dir "${EXP_DIR}"
  --split training
  --num_worlds 1
  --num_scenarios 100000
  --population_size 64
  --elite_size 8
  --num_es_iterations 4
  --replan_interval_steps 10
  --other_agent_forecast wm
  --target_scenario_path "${TARGET_JSON}"
  --visualize_mode all
  --save_trajectory
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
