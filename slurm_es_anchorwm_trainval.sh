#!/bin/bash
#SBATCH --job-name=es_awm_trainval
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
#SBATCH --output=/home/mineuih/ishan/logs/es_awm_trainval_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/es_awm_trainval_%A_%a.err

# Diffusion-ES with an AnchorWM checkpoint as both forecaster and reward on babel over the train/val set of
# slurm_trainval_baselines.sh: every scenario of WOMD v1.3.1 training shards 00000 + 00001 and validation shards
# 00000 + 00001 (455 + 479 + 286 + 309 = 1529). Array task k runs half k % 2 (contiguous scenario indices) of shard
# unit k // 2 (train00, train01, val00, val01). Planner settings as slurm_es_anchorwm_valfail1568.sh.
# Resubmitting resumes a task.
#
#   CKPT=.../best.pt sbatch slurm_es_anchorwm_trainval.sh [log|idm] [tag]
#   POSTHOC_REPAIR=1 CKPT=.../best.pt sbatch slurm_es_anchorwm_trainval.sh log     # + post-hoc repair of the ES winner
#   CKPT=.../best.pt sbatch --array=5 slurm_es_anchorwm_trainval.sh log            # rerun / resume one task
#   sbatch --chdir=<checkout> ...   # run another checkout (ROOT follows the submit / chdir directory)
set -euo pipefail

SIM_AGENTS="${1:-log}"
TAG="${2:-}"
case "${SIM_AGENTS}" in
  log|idm) ;;
  *) echo "ERROR: sim_agents must be log or idm, got '${SIM_AGENTS}'"; exit 1 ;;
esac
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
UNITS=("training training_tfexample.tfrecord-00000-of-01000" "training training_tfexample.tfrecord-00001-of-01000"
       "validation validation_tfexample.tfrecord-00000-of-00150" "validation validation_tfexample.tfrecord-00001-of-00150")
read -r SPLIT SHARD_NAME <<< "${UNITS[$((TASK / 2))]}"
HALF=$((TASK % 2))
SEED="${SEED:-0}"

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-/home/mineuih/ishan/waymax_rs}}"
POSTHOC_REPAIR="${POSTHOC_REPAIR:-0}"
EXP_ROOT="/data/user_data/mineuih/ishan/world_model_sim"
TF_ROOT="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example"
LANE_GRAPH_DIR="/data/user_data/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-/data/user_data/mineuih/waymax_checkpoints/goal_cond_diffusion_policy/epoch_1000}"
: "${CKPT:?set CKPT to an AnchorWM checkpoint (forecaster + reward)}"
SCORER_CKPT="${SCORER_CKPT:-${CKPT}}"

set +u
# shellcheck source=/dev/null
source /home/mineuih/miniconda3/bin/activate /home/mineuih/miniconda3/envs/waymax_rs_ishan
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="

ckpt_tag() { echo "$(basename "$(dirname "$1")" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')_$(basename "$1" .pt)"; }
AWM_TAG="$(ckpt_tag "${CKPT}")"
[[ "${SCORER_CKPT}" == "${CKPT}" ]] || AWM_TAG="wm_${AWM_TAG}_sc_$(ckpt_tag "${SCORER_CKPT}")"
RUN_NAME="es_awm_${AWM_TAG}_${SIM_AGENTS}_r${REPLAN_INTERVAL_STEPS}_seed${SEED}_trainval0001${TAG:+_${TAG}}"
[[ "${POSTHOC_REPAIR}" == "1" ]] && RUN_NAME="${RUN_NAME}_posthoc"
EXTRA_ARGS=()
[[ "${POSTHOC_REPAIR}" == "1" ]] && EXTRA_ARGS+=(--posthoc_repair)
PART="${SPLIT}_${SHARD_NAME##*tfrecord-}_h${HALF}"
OUT="${EXP_ROOT}/${RUN_NAME}/${PART}"
TF_LINK="${OUT}/tfrecords"
mkdir -p "${TF_LINK}/${SPLIT}"
ln -sfn "${TF_ROOT}/${SPLIT}/${SHARD_NAME}" "${TF_LINK}/${SPLIT}/${SHARD_NAME}"
for p in "${TF_ROOT}/${SPLIT}/${SHARD_NAME}" "${LANE_GRAPH_DIR}/${SHARD_NAME}.lanegraph.zip" "${DIFFUSION_CKPT}" "${CKPT}" "${SCORER_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

# This half's scenario indices (from the lane graph manifest's record count, which matches the tfrecord).
TARGETS="${OUT}/target_scenarios.json"
python - "${LANE_GRAPH_DIR}/${SHARD_NAME}.lanegraph.zip" "${SHARD_NAME}" "${HALF}" "${TARGETS}" <<'EOF'
import json, sys, zipfile
zip_path, shard, half, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
n = json.loads(zipfile.ZipFile(zip_path).read("manifest.json"))["num_records"]
mid = (n + 1) // 2
idx = list(range(0, mid)) if half == 0 else list(range(mid, n))
json.dump({shard: idx}, open(dst, "w"))
print(f"{shard} half {half}: scenarios {idx[0]}..{idx[-1]} ({len(idx)} of {n})")
EOF


echo "Diffusion-ES + AnchorWM ${RUN_NAME}: ${PART} (task ${TASK}), anchorwm=${CKPT}, scorer=${SCORER_CKPT}, policy=${DIFFUSION_CKPT}, sim_agents=${SIM_AGENTS}, posthoc_repair=${POSTHOC_REPAIR}, code=${ROOT} ($(git -C "${ROOT}" rev-parse --short HEAD)), host=$(hostname), out=${OUT}"
python -u "${ROOT}/simulation/run_simulation.py" \
  --planner es \
  --other_agent_forecast wm --world_model_ckpt "${CKPT}" --wm_backend anchor --wm_refresh_steps 10 \
  --num_collision_worlds 1 --collision_reduce any --wm_planning_mode expected_independent \
  --anchor_scorer_ckpt "${SCORER_CKPT}" \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --resample_timesteps 5 \
  --tfrecord_dir "${TF_LINK}" \
  --split "${SPLIT}" \
  --target_scenario_path "${TARGETS}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_ROOT}/${RUN_NAME}" \
  --start_from "${OUT}" \
  --tag "${RUN_NAME}_${PART}" \
  --num_worlds 5 \
  --num_scenarios 100000 \
  --seed "${SEED}" \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations 4 \
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" \
  --sim_agents "${SIM_AGENTS}" \
  --visualize_mode none \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
echo "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenario results in ${OUT}"
