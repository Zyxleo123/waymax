#!/bin/bash
#SBATCH --job-name=pdm_closed_val
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/pdm_closed_val_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/pdm_closed_val_%A_%a.err

# PDM-Closed (planner/pdm_closed_planner.py) closed-loop on validation shards 00000-00009, one shard per
# array task (CPU only). Replans every REPLAN_INTERVAL_STEPS like the other planners; nuPlan runs PDM at
# 10 Hz, i.e. REPLAN_INTERVAL_STEPS=1.
#
#   sbatch slurm_pdm_closed_val.sh [log|idm] [tag]
#   REPLAN_INTERVAL_STEPS=1 sbatch slurm_pdm_closed_val.sh log replan1
#   PDM_CV_VELOCITY_RATIOS=0.8,1.0 sbatch slurm_pdm_closed_val.sh log cv08   # nuplan_rs-fork forecast
#   sbatch --array=3 slurm_pdm_closed_val.sh log rerun3                     # a single shard
set -euo pipefail

SIM_AGENTS="${1:-log}"
TAG="${2:-}"
case "${SIM_AGENTS}" in
  log|idm) ;;
  *) echo "ERROR: sim_agents must be log or idm, got '${SIM_AGENTS}'"; exit 1 ;;
esac
SHARD="$(printf '%05d' "${SLURM_ARRAY_TASK_ID:-0}")"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
PDM_CV_VELOCITY_RATIOS="${PDM_CV_VELOCITY_RATIOS:-1.0}"
NUM_WORLDS="${NUM_WORLDS:-5}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
EXP_ROOT="/zfsauton/scratch/ishanj/world_model_sim"
TF_ROOT="/zfsauton/scratch/mineuih/womd/tf_example"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"

# Teammate's env (yours has a jaxlib / jax_cuda12_plugin version mismatch).
set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu

RUN_NAME="pdm_closed_${SIM_AGENTS}_r${REPLAN_INTERVAL_STEPS}${TAG:+_${TAG}}"
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
SHARD_NAME="validation_tfexample.tfrecord-${SHARD}-of-00150"
TF_LINK_ROOT="${EXP_DIR}/tfrecords_${SHARD}"
mkdir -p "${EXP_DIR}" "${EXP_ROOT}/logs" "${TF_LINK_ROOT}/validation"
ln -sfn "${TF_ROOT}/validation/${SHARD_NAME}" "${TF_LINK_ROOT}/validation/${SHARD_NAME}"
for p in "${TF_ROOT}/validation/${SHARD_NAME}" "${LANE_GRAPH_DIR}/${SHARD_NAME}.lanegraph.zip"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

echo "PDM-Closed ${RUN_NAME}: shard ${SHARD}, sim_agents=${SIM_AGENTS}, replan=${REPLAN_INTERVAL_STEPS}, cv_ratios=${PDM_CV_VELOCITY_RATIOS}, host=$(hostname)"
python "${ROOT}/simulation/run_simulation.py" \
  --planner pdm_closed \
  --tfrecord_dir "${TF_LINK_ROOT}" \
  --split validation \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --num_worlds "${NUM_WORLDS}" \
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" \
  --sim_agents "${SIM_AGENTS}" \
  --pdm_cv_velocity_ratios "${PDM_CV_VELOCITY_RATIOS}" \
  --visualize_mode none \
  --tag "val${SHARD}"
