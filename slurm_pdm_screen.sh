#!/bin/bash
#SBATCH --job-name=pdm_screen
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=72G
#SBATCH --time=0-03:00:00
#SBATCH --array=0-40
#SBATCH --chdir=/home/mineuih/ishan/waymax_rs
#SBATCH --output=/home/mineuih/ishan/logs/pdm_screen_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/pdm_screen_%A_%a.err

# CPU-only PDM-Closed screen of whole WOMD training shards, to build the mining pool for scorer fine-tuning
# without a precomputed failure list: scenes where PDM-Closed overlaps or goes offroad are ~6x richer in
# Diffusion-ES failures than random ones (train/val: 20.3% vs 3.3%). CPU only, on the preempt partition (general
# needs a GPU), so it does not count against the per-user GPU limit; a preempted task requeues and resumes.
# Array task k screens shard SHARDS[k] (default 00100..00140, ~20k scenes, untouched by the v1 / v2 banks and the
# train/val set); every task writes into one run directory. Resubmitting resumes. 2 CPUs per task: the process
# keeps < 3 cores busy, so 32 small tasks under the 64-CPU QOS cap beat 8 tasks of 8 CPUs.
#
#   sbatch slurm_pdm_screen.sh
#   SHARDS="$(seq -f %05g 120 159 | tr '\n' ' ')" sbatch --array=0-39 slurm_pdm_screen.sh
#   PROCS=1 NUM_WORLDS=5 MAX_SCENES=60 RUN=.../bench sbatch -c 1 --mem=24G --array=0 slurm_pdm_screen.sh   (throughput test)
set -euo pipefail

read -r -a SHARDS <<< "${SHARDS:-$(seq -f %05g 100 140 | tr '\n' ' ')}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
PROCS="${PROCS:-3}"
if (( TASK >= ${#SHARDS[@]} )); then echo "ERROR: task ${TASK} but only ${#SHARDS[@]} shards"; exit 1; fi
SHARD="training_tfexample.tfrecord-${SHARDS[${TASK}]}-of-01000"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"

ROOT="/home/mineuih/ishan/waymax_rs"
TF_ROOT="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example"
LANE_GRAPH_DIR="/data/user_data/mineuih/waymax_rs/lane_graphs"
RUN="${RUN:-/data/user_data/mineuih/ishan/mining/pdm_screen_train_r${REPLAN_INTERVAL_STEPS}}"
OUT="${RUN}/${SHARD##*tfrecord-}"
mkdir -p "${OUT}/tfrecords/training"
ln -sfn "${TF_ROOT}/training/${SHARD}" "${OUT}/tfrecords/training/${SHARD}"
for p in "${TF_ROOT}/training/${SHARD}" "${LANE_GRAPH_DIR}/${SHARD}.lanegraph.zip"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

set +u
# shellcheck source=/dev/null
source /home/mineuih/miniconda3/bin/activate /home/mineuih/miniconda3/envs/waymax_rs_ishan
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

# Piece k of PROCS: a contiguous slice of the shard's scenarios (contiguous, so each process scans the tfrecord once).
python - "${LANE_GRAPH_DIR}/${SHARD}.lanegraph.zip" "${SHARD}" "${OUT}" "${MAX_SCENES:-0}" "${PROCS}" <<'EOF'
import json, sys, zipfile
zip_path, shard, out, cap, procs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
n = json.loads(zipfile.ZipFile(zip_path).read("manifest.json"))["num_records"]
idx = list(range(n))[: cap or None]
step = -(-len(idx) // procs)
for k in range(procs):
    json.dump({shard: idx[k * step:(k + 1) * step]}, open(f"{out}/target_scenarios_p{k}of{procs}.json", "w"))
print(f"{shard}: {len(idx)} of {n} scenarios in {procs} pieces of <= {step}")
EOF

echo "PDM-Closed screen: ${SHARD} (task ${TASK}), ${PROCS} processes, host=$(hostname), out=${OUT}"
pids=()
for k in $(seq 0 $((PROCS - 1))); do
  python -u "${ROOT}/simulation/run_simulation.py" --planner pdm_closed \
    --tfrecord_dir "${OUT}/tfrecords" --split training --target_scenario_path "${OUT}/target_scenarios_p${k}of${PROCS}.json" \
    --lane_graph_dir "${LANE_GRAPH_DIR}" --exp_dir "${RUN}" --start_from "${OUT}" --tag "pdm_screen_${SHARD##*tfrecord-}_p${k}" \
    --num_worlds "${NUM_WORLDS:-10}" --num_scenarios 100000 --seed 0 --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" \
    --sim_agents log --pdm_cv_velocity_ratios 1.0 --visualize_mode none > "${OUT}/screen_p${k}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
echo "done (exit ${status}): $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenario results in ${OUT}"
exit "${status}"
