#!/bin/bash
#SBATCH --job-name=es_awm_valfail1568
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
#SBATCH --output=/home/mineuih/ishan/logs/es_awm_valfail1568_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/es_awm_valfail1568_%A_%a.err

# Diffusion-ES with an AnchorWM checkpoint as both forecaster and reward (--wm_backend anchor,
# --anchor_scorer_ckpt) on babel over the 1568 validation failure scenes. Same scenes, lane graphs, chunking and
# ES settings as slurm_es_pdms_valfail1568.sh (population 64, elite 8, 4 iterations, resample_timesteps 5,
# replan every 10; expected_independent WM forecasts refreshed every 10 steps). Resubmitting resumes each chunk.
#
#   CKPT=.../final.pt sbatch slurm_es_anchorwm_valfail1568.sh [log|idm] [tag]
#   CKPT=.../final.pt sbatch --array=3 slurm_es_anchorwm_valfail1568.sh log    # rerun / resume one chunk
#   CKPT=forecaster.pt SCORER_CKPT=scorer.pt sbatch slurm_es_anchorwm_valfail1568.sh   # mix forecaster / reward
set -euo pipefail

SIM_AGENTS="${1:-log}"
TAG="${2:-}"
case "${SIM_AGENTS}" in
  log|idm) ;;
  *) echo "ERROR: sim_agents must be log or idm, got '${SIM_AGENTS}'"; exit 1 ;;
esac
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
NUM_CHUNKS="${NUM_CHUNKS:-8}"
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEED:-0}"

ROOT="/home/mineuih/ishan/waymax_rs"
EXP_ROOT="/data/user_data/mineuih/ishan/world_model_sim"
DATA="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs"
COMPACT_DIR="${DATA}/womd_compact/valfail1568"      # has validation/ and target_scenarios.json
LANE_GRAPH_DIR="${DATA}/lane_graphs_babel"
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
RUN_NAME="es_awm_${AWM_TAG}_${SIM_AGENTS}_r${REPLAN_INTERVAL_STEPS}_seed${SEED}_valfail1568${TAG:+_${TAG}}"
OUT="${EXP_ROOT}/${RUN_NAME}/chunk_${CHUNK}"
mkdir -p "${OUT}"
for p in "${COMPACT_DIR}/validation" "${COMPACT_DIR}/target_scenarios.json" "${LANE_GRAPH_DIR}" "${DIFFUSION_CKPT}" "${CKPT}" "${SCORER_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

TARGETS="${OUT}/target_scenarios.json"
python - "${COMPACT_DIR}/target_scenarios.json" "${TARGETS}" "${CHUNK}" "${NUM_CHUNKS}" <<'EOF'
import json, sys
src, dst, k, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
targets = json.load(open(src))
names = sorted(t for t, v in targets.items() if v)
chunk = {t: targets[t] for t in names[k::n]}
json.dump(chunk, open(dst, "w"), indent=1)
print(f"chunk {k}/{n}: {len(chunk)} tfrecords, {sum(len(v) for v in chunk.values())} scenarios")
EOF

echo "Diffusion-ES + AnchorWM ${RUN_NAME}: chunk ${CHUNK}/${NUM_CHUNKS}, anchorwm=${CKPT}, scorer=${SCORER_CKPT}, policy=${DIFFUSION_CKPT}, sim_agents=${SIM_AGENTS}, host=$(hostname), out=${OUT}"
python -u "${ROOT}/simulation/run_simulation.py" \
  --planner es \
  --other_agent_forecast wm --world_model_ckpt "${CKPT}" --wm_backend anchor --wm_refresh_steps 10 \
  --num_collision_worlds 1 --collision_reduce any --wm_planning_mode expected_independent \
  --anchor_scorer_ckpt "${SCORER_CKPT}" \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --resample_timesteps 5 \
  --tfrecord_dir "${COMPACT_DIR}" \
  --split validation \
  --target_scenario_path "${TARGETS}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_ROOT}/${RUN_NAME}" \
  --start_from "${OUT}" \
  --tag "${RUN_NAME}_c${CHUNK}" \
  --num_worlds 5 \
  --num_scenarios 100000 \
  --seed "${SEED}" \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations 4 \
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" \
  --sim_agents "${SIM_AGENTS}" \
  --visualize_mode none
echo "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenario results in ${OUT}"
