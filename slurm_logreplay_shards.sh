#!/bin/bash
#SBATCH --job-name=logreplay_shards
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=/home/mineuih/ishan/logs/logreplay_shards_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/logreplay_shards_%A_%a.err

# Log replay of the logged ego (the expert reference) on whole WOMD v1.3.1 shards, on CPU, with the closed-loop
# metrics of the planner runs (log-replay agents, start step 10). One shard per array task:
#   val0-9    validation shards 00000-00009            sbatch --array=0-9 slurm_logreplay_shards.sh val0-9
#   trainval  training 00000-00001, validation 00000-00001   sbatch --array=0-3 slurm_logreplay_shards.sh trainval
# Results: /data/user_data/mineuih/ishan/world_model_sim/log_replay_<set>/<split>_<shard>. Resubmitting resumes a shard.
set -euo pipefail

SET="${1:-}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
case "${SET}" in
  val0-9) SPLIT=validation; SHARD_NAME="validation_tfexample.tfrecord-$(printf '%05d' "${TASK}")-of-00150" ;;
  trainval)
    UNITS=("training training_tfexample.tfrecord-00000-of-01000" "training training_tfexample.tfrecord-00001-of-01000"
           "validation validation_tfexample.tfrecord-00000-of-00150" "validation validation_tfexample.tfrecord-00001-of-00150")
    read -r SPLIT SHARD_NAME <<< "${UNITS[${TASK}]}" ;;
  *) echo "ERROR: set must be val0-9 or trainval, got '${SET}'"; exit 1 ;;
esac

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-/home/mineuih/ishan/waymax_rs}}"
EXP_ROOT="/data/user_data/mineuih/ishan/world_model_sim"
TF_ROOT="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example"
LANE_GRAPH_DIR="/data/user_data/mineuih/waymax_rs/lane_graphs"
RUN_NAME="log_replay_${SET}"
PART="${SPLIT}_${SHARD_NAME##*tfrecord-}"
OUT="${EXP_ROOT}/${RUN_NAME}/${PART}"
TF_LINK="${OUT}/tfrecords"
mkdir -p "${TF_LINK}/${SPLIT}"
ln -sfn "${TF_ROOT}/${SPLIT}/${SHARD_NAME}" "${TF_LINK}/${SPLIT}/${SHARD_NAME}"
[[ -e "${TF_ROOT}/${SPLIT}/${SHARD_NAME}" ]] || { echo "ERROR: missing ${TF_ROOT}/${SPLIT}/${SHARD_NAME}"; exit 1; }

set +u
# shellcheck source=/dev/null
source /home/mineuih/miniconda3/bin/activate /home/mineuih/miniconda3/envs/waymax_rs_ishan
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu

echo "Log replay ${RUN_NAME}: ${PART} (task ${TASK}), code=${ROOT} ($(git -C "${ROOT}" rev-parse --short HEAD)), host=$(hostname), out=${OUT}"
python -u "${ROOT}/simulation/run_simulation.py" \
  --planner log-replay \
  --tfrecord_dir "${TF_LINK}" \
  --split "${SPLIT}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_ROOT}/${RUN_NAME}" \
  --start_from "${OUT}" \
  --tag "${RUN_NAME}_${PART}" \
  --num_worlds 5 \
  --num_scenarios 100000 \
  --seed 0 \
  --replan_interval_steps 10 \
  --sim_agents log \
  --visualize_mode none
echo "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenario results in ${OUT}"
