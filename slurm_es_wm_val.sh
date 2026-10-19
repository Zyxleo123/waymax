#!/bin/bash
#SBATCH --job-name=es_wm_val
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/es_wm_val_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/es_wm_val_%A_%a.err

# Diffusion-ES with the multi-future Gaussian WM on validation shards 00000-00009, one shard per array
# task. Same planner settings as the fail1500 T=1 run (slurm_es_wm_temperature_sweep.sh, TEMPERATURES=1,
# OFFROAD_MODE=margin_tiebreak). Resubmitting resumes the scenarios a shard has already finished.
#
#   sbatch slurm_es_wm_val.sh
#   sbatch --array=3 slurm_es_wm_val.sh                                  # a single shard
#   OFFROAD_MODE=binary WM_MODE_TEMPERATURE=0.1 sbatch slurm_es_wm_val.sh
set -euo pipefail

SHARD="$(printf '%05d' "${SLURM_ARRAY_TASK_ID:-0}")"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt}"
OFFROAD_MODE="${OFFROAD_MODE:-margin_tiebreak}"
WM_MODE_TEMPERATURE="${WM_MODE_TEMPERATURE:-1}"
SIM_AGENTS="${SIM_AGENTS:-log}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
EXP_ROOT="/zfsauton/scratch/ishanj/world_model_sim"
TF_ROOT="/zfsauton/scratch/mineuih/womd/tf_example"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WM_RUN="$(basename "$(dirname "${WORLD_MODEL_CKPT}")")"
WM_TAG="${WM_RUN#multiFuture_GaussianWorldModel_}_$(basename "${WORLD_MODEL_CKPT}" .pt)"
RUN_NAME="es_diffusion_wm_mf_${WM_TAG}_expected_independent_${SIM_AGENTS}_off${OFFROAD_MODE}_T${WM_MODE_TEMPERATURE}_val0-9"
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
OUT="${EXP_DIR}/shard_${SHARD}"
SHARD_NAME="validation_tfexample.tfrecord-${SHARD}-of-00150"
TF_LINK_ROOT="${EXP_DIR}/tfrecords_${SHARD}"
mkdir -p "${OUT}" "${EXP_ROOT}/logs" "${TF_LINK_ROOT}/validation"
ln -sfn "${TF_ROOT}/validation/${SHARD_NAME}" "${TF_LINK_ROOT}/validation/${SHARD_NAME}"
for p in "${TF_ROOT}/validation/${SHARD_NAME}" "${LANE_GRAPH_DIR}/${SHARD_NAME}.lanegraph.zip" \
         "${WORLD_MODEL_CKPT}" "${DIFFUSION_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

echo "[$(date)] ${RUN_NAME}: shard ${SHARD}, temperature=${WM_MODE_TEMPERATURE}, offroad_mode=${OFFROAD_MODE}, sim_agents=${SIM_AGENTS}, host=$(hostname), commit=$(git -C "${ROOT}" rev-parse --short HEAD), out=${OUT}"
python -u "${ROOT}/simulation/run_simulation.py" \
  --planner es \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --resample_timesteps 5 \
  --tfrecord_dir "${TF_LINK_ROOT}" \
  --split validation \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --start_from "${OUT}" \
  --num_worlds 5 \
  --num_scenarios 100000 \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations 4 \
  --replan_interval_steps 10 \
  --other_agent_forecast wm \
  --visualize_mode none \
  --tag "${RUN_NAME}_val${SHARD}" \
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
  --wm_mode_temperature "${WM_MODE_TEMPERATURE}"
echo "[$(date)] done: $(ls "${OUT}"/*.scenario_*[0-9].json 2>/dev/null | wc -l) scenarios"
