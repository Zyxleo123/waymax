#!/bin/bash
#SBATCH --job-name=targets_baseline
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-4
#SBATCH --chdir=/home/mineuih/ishan/waymax_rs
#SBATCH --output=/home/mineuih/ishan/logs/targets_%x_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/targets_%x_%A_%a.err

# Diffusion-ES + PDMS (or log replay of the logged ego) on an arbitrary list of WOMD training scenes, e.g. the
# mining pool of slurm_pdm_screen.sh: TARGETS is a {tfrecord name: [scenario indices]} json. Whole tfrecords are
# dealt into the array's NUM_CHUNKS tasks (= array size), balanced by scene count; every task writes into RUN.
# ES settings as slurm_trainval_baselines.sh. Resubmitting resumes.
#
#   TARGETS=.../pool_r0/flagged.json RUN=.../es_pdms_flagged_r0 sbatch slurm_targets_baseline.sh es_pdms
#   TARGETS=... RUN=... NUM_CHUNKS=1 bash slurm_targets_baseline.sh log_replay     (inside an allocation)
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in es_pdms|log_replay) ;; *) echo "ERROR: mode must be es_pdms or log_replay, got '${MODE}'"; exit 1 ;; esac
: "${TARGETS:?set TARGETS to a {tfrecord: [indices]} json}"
: "${RUN:?set RUN to the output directory}"
NUM_CHUNKS="${NUM_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEED:-0}"

ROOT="/home/mineuih/ishan/waymax_rs"
TF_ROOT="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example"
LANE_GRAPH_DIR="/data/user_data/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-/data/user_data/mineuih/waymax_checkpoints/goal_cond_diffusion_policy/epoch_1000}"
OUT="${RUN}/chunk_${CHUNK}"
mkdir -p "${OUT}/tfrecords/training"

set +u
# shellcheck source=/dev/null
source /home/mineuih/miniconda3/bin/activate /home/mineuih/miniconda3/envs/waymax_rs_ishan
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="

python - "${TARGETS}" "${OUT}/target_scenarios.json" "${CHUNK}" "${NUM_CHUNKS}" "${OUT}/tfrecords/training" "${TF_ROOT}/training" <<'EOF'
import json, os, sys
src, dst, k, n, link_dir, tf_root = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
# Whole tfrecords per task (largest first onto the least-loaded task): each task reads fewer shards and fills its
# --num_worlds batches, instead of a few scenes from every shard.
targets = {t: sorted(int(i) for i in idx) for t, idx in json.load(open(src)).items() if idx}
load, owner = [0] * n, {}
for t in sorted(targets, key=lambda t: (-len(targets[t]), t)):
    j = min(range(n), key=lambda j: (load[j], j))
    owner[t] = j
    load[j] += len(targets[t])
chunk = {t: v for t, v in sorted(targets.items()) if owner[t] == k}
pairs = [(t, i) for t, v in chunk.items() for i in v]
for t in chunk:
    link = os.path.join(link_dir, t)
    if not os.path.islink(link):
        os.symlink(os.path.join(tf_root, t), link)
json.dump(chunk, open(dst, "w"), indent=0)
print(f"chunk {k}/{n}: {len(pairs)} scenes in {len(chunk)} tfrecords")
EOF

COMMON=(--tfrecord_dir "${OUT}/tfrecords" --split training --target_scenario_path "${OUT}/target_scenarios.json"
        --lane_graph_dir "${LANE_GRAPH_DIR}" --exp_dir "${RUN}" --start_from "${OUT}" --tag "$(basename "${RUN}")_c${CHUNK}"
        --num_worlds 5 --num_scenarios 100000 --seed "${SEED}" --replan_interval_steps 10 --sim_agents log
        --visualize_mode none)
if [[ "${MODE}" == "es_pdms" ]]; then
  PLAN=(--planner es --es_reward pdms --pdm_cv_velocity_ratios 1.0 --diffusion_ckpt_path "${DIFFUSION_CKPT}"
        --resample_timesteps 5 --population_size 64 --elite_size 8 --num_es_iterations 4)
else
  PLAN=(--planner log-replay)
fi
echo "${MODE} on ${TARGETS}: chunk ${CHUNK}/${NUM_CHUNKS}, host=$(hostname), out=${OUT}"
python -u "${ROOT}/simulation/run_simulation.py" "${PLAN[@]}" "${COMMON[@]}"
echo "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenario results in ${OUT}"
