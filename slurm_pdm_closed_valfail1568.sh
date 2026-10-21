#!/bin/bash
#SBATCH --job-name=pdm_valfail1568
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/home/mineuih/ishan/waymax_rs
#SBATCH --output=/home/mineuih/ishan/logs/pdm_valfail1568_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/pdm_valfail1568_%A_%a.err

# PDM-Closed (planner/pdm_closed_planner.py) closed-loop on babel over the 1568 validation failure scenes
# (yixiz's womd_compact/valfail1568: original record indices, full-shard lane graphs). Same settings as
# slurm_pdm_closed_val.sh. Each array task takes every NUM_CHUNKS-th tfrecord. CPU only; the GPU is
# requested only because babel's general partition requires one (any type, so the job starts sooner).
# Resubmitting resumes each chunk from its existing results.
#
#   mkdir -p /home/mineuih/ishan/logs   # once
#   sbatch slurm_pdm_closed_valfail1568.sh [log|idm] [tag]
#   sbatch --array=2 slurm_pdm_closed_valfail1568.sh log    # rerun / resume one chunk
set -euo pipefail

SIM_AGENTS="${1:-log}"
TAG="${2:-}"
case "${SIM_AGENTS}" in
  log|idm) ;;
  *) echo "ERROR: sim_agents must be log or idm, got '${SIM_AGENTS}'"; exit 1 ;;
esac
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
PDM_CV_VELOCITY_RATIOS="${PDM_CV_VELOCITY_RATIOS:-1.0}"
NUM_WORLDS="${NUM_WORLDS:-5}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"

ROOT="/home/mineuih/ishan/waymax_rs"
EXP_ROOT="/data/user_data/mineuih/ishan/world_model_sim"
DATA="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs"
COMPACT_DIR="${DATA}/womd_compact/valfail1568"      # has validation/ and target_scenarios.json
LANE_GRAPH_DIR="${DATA}/lane_graphs_babel"

set +u
# shellcheck source=/dev/null
source /home/mineuih/miniconda3/bin/activate /home/mineuih/miniconda3/envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu

RUN_NAME="pdm_closed_${SIM_AGENTS}_r${REPLAN_INTERVAL_STEPS}_valfail1568${TAG:+_${TAG}}"
OUT="${EXP_ROOT}/${RUN_NAME}/chunk_${CHUNK}"
mkdir -p "${OUT}"
for p in "${COMPACT_DIR}/validation" "${COMPACT_DIR}/target_scenarios.json" "${LANE_GRAPH_DIR}"; do
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

echo "PDM-Closed ${RUN_NAME}: chunk ${CHUNK}/${NUM_CHUNKS}, sim_agents=${SIM_AGENTS}, replan=${REPLAN_INTERVAL_STEPS}, cv_ratios=${PDM_CV_VELOCITY_RATIOS}, host=$(hostname), out=${OUT}"
python "${ROOT}/simulation/run_simulation.py" \
  --planner pdm_closed \
  --tfrecord_dir "${COMPACT_DIR}" \
  --split validation \
  --target_scenario_path "${TARGETS}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_ROOT}/${RUN_NAME}" \
  --start_from "${OUT}" \
  --num_worlds "${NUM_WORLDS}" \
  --num_scenarios 100000 \
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" \
  --sim_agents "${SIM_AGENTS}" \
  --pdm_cv_velocity_ratios "${PDM_CV_VELOCITY_RATIOS}" \
  --visualize_mode none \
  --tag "valfail1568_c${CHUNK}"
echo "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenario results in ${OUT}"
