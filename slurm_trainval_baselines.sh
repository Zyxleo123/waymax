#!/bin/bash
#SBATCH --job-name=trainval_baseline
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-7
#SBATCH --chdir=/home/mineuih/ishan/waymax_rs
#SBATCH --output=/home/mineuih/ishan/logs/trainval_%x_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/trainval_%x_%A_%a.err

# PDM-Closed or Diffusion-ES + PDMS on babel over the train/val set of slurm_es_baseline.sh: every scenario of
# WOMD v1.3.1 training shards 00000 + 00001 and validation shards 00000 + 00001 (455 + 479 + 286 + 309 = 1529).
# Array task k runs half k % 2 (contiguous scenario indices) of shard unit k // 2 (train00, train01, val00, val01).
# Settings as slurm_pdm_closed_valfail1568.sh / slurm_es_pdms_valfail1568.sh; resubmitting resumes a task.
#
#   mkdir -p /home/mineuih/ishan/logs   # once
#   sbatch -J pdm_trainval slurm_trainval_baselines.sh pdm_closed
#   sbatch -J es_pdms_trainval --gres=gpu:A6000:1 slurm_trainval_baselines.sh es_pdms
#   DIFFUSION_CKPT=.../goal_cond_diffusion_policy/latest sbatch ... es_pdms
#   sbatch --array=5 ... slurm_trainval_baselines.sh es_pdms      # rerun / resume one task
set -euo pipefail

USAGE="usage: sbatch slurm_trainval_baselines.sh pdm_closed|es_pdms [log|idm]"
MODE="${1:-}"
SIM_AGENTS="${2:-log}"
case "${MODE}" in pdm_closed|es_pdms) ;; *) echo "ERROR: mode must be pdm_closed or es_pdms, got '${MODE}'; ${USAGE}"; exit 1 ;; esac
case "${SIM_AGENTS}" in log|idm) ;; *) echo "ERROR: sim_agents must be log or idm, got '${SIM_AGENTS}'"; exit 1 ;; esac
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
PDM_CV_VELOCITY_RATIOS="${PDM_CV_VELOCITY_RATIOS:-1.0}"
SEED="${SEED:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

ROOT="/home/mineuih/ishan/waymax_rs"
EXP_ROOT="/data/user_data/mineuih/ishan/world_model_sim"
TF_ROOT="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example"
LANE_GRAPH_DIR="/data/user_data/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-/data/user_data/mineuih/waymax_checkpoints/goal_cond_diffusion_policy/epoch_1000}"
UNITS=("training training_tfexample.tfrecord-00000-of-01000" "training training_tfexample.tfrecord-00001-of-01000"
       "validation validation_tfexample.tfrecord-00000-of-00150" "validation validation_tfexample.tfrecord-00001-of-00150")
read -r SPLIT SHARD <<< "${UNITS[$((TASK / 2))]}"
HALF=$((TASK % 2))

set +u
# shellcheck source=/dev/null
source /home/mineuih/miniconda3/bin/activate /home/mineuih/miniconda3/envs/waymax_rs_ishan
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ "${MODE}" == "pdm_closed" ]]; then
  export JAX_PLATFORMS=cpu
  RUN_NAME="pdm_closed_${SIM_AGENTS}_r${REPLAN_INTERVAL_STEPS}_trainval0001"
else
  CKPT_TAG="$(basename "$(dirname "${DIFFUSION_CKPT}")")_$(basename "${DIFFUSION_CKPT}")"
  RUN_NAME="es_pdms_${CKPT_TAG}_${SIM_AGENTS}_r${REPLAN_INTERVAL_STEPS}_seed${SEED}_trainval0001"
fi
PART="${SPLIT}_${SHARD##*tfrecord-}_h${HALF}"
OUT="${EXP_ROOT}/${RUN_NAME}/${PART}"
TF_LINK="${OUT}/tfrecords"
mkdir -p "${TF_LINK}/${SPLIT}"
ln -sfn "${TF_ROOT}/${SPLIT}/${SHARD}" "${TF_LINK}/${SPLIT}/${SHARD}"
for p in "${TF_ROOT}/${SPLIT}/${SHARD}" "${LANE_GRAPH_DIR}/${SHARD}.lanegraph.zip"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done
[[ "${MODE}" == "pdm_closed" || -e "${DIFFUSION_CKPT}" ]] || { echo "ERROR: missing ${DIFFUSION_CKPT}"; exit 1; }

# This half's scenario indices (from the lane graph manifest's record count, which matches the tfrecord).
TARGETS="${OUT}/target_scenarios.json"
python - "${LANE_GRAPH_DIR}/${SHARD}.lanegraph.zip" "${SHARD}" "${HALF}" "${TARGETS}" <<'EOF'
import json, sys, zipfile
zip_path, shard, half, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
n = json.loads(zipfile.ZipFile(zip_path).read("manifest.json"))["num_records"]
mid = (n + 1) // 2
idx = list(range(0, mid)) if half == 0 else list(range(mid, n))
json.dump({shard: idx}, open(dst, "w"))
print(f"{shard} half {half}: scenarios {idx[0]}..{idx[-1]} ({len(idx)} of {n})")
EOF

COMMON=(--tfrecord_dir "${TF_LINK}" --split "${SPLIT}" --target_scenario_path "${TARGETS}"
        --lane_graph_dir "${LANE_GRAPH_DIR}" --exp_dir "${EXP_ROOT}/${RUN_NAME}" --start_from "${OUT}"
        --tag "${RUN_NAME}_${PART}" --num_worlds 5 --num_scenarios 100000 --seed "${SEED}"
        --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" --sim_agents "${SIM_AGENTS}"
        --pdm_cv_velocity_ratios "${PDM_CV_VELOCITY_RATIOS}" --visualize_mode none)
if [[ "${MODE}" == "pdm_closed" ]]; then
  PLAN=(--planner pdm_closed)
else
  PLAN=(--planner es --es_reward pdms --diffusion_ckpt_path "${DIFFUSION_CKPT}" --resample_timesteps 5
        --population_size 64 --elite_size 8 --num_es_iterations 4)
fi

echo "${MODE} ${RUN_NAME}: ${PART} (task ${TASK}), sim_agents=${SIM_AGENTS}, host=$(hostname), out=${OUT}"
python -u "${ROOT}/simulation/run_simulation.py" "${PLAN[@]}" "${COMMON[@]}"
echo "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenario results in ${OUT}"
