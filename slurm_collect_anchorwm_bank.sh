#!/bin/bash
#SBATCH --job-name=awm_collect
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-7
#SBATCH --chdir=/home/mineuih/ishan/waymax_rs
#SBATCH --output=/home/mineuih/ishan/logs/awm_collect_%A_%a.out
#SBATCH --error=/home/mineuih/ishan/logs/awm_collect_%A_%a.err

# ES candidate bank for AnchorWM scorer fine-tuning, collected and labelled in one pass on babel: closed-loop
# Diffusion-ES with the AnchorWM as forecaster and reward (--save_es_candidates --label_in_loop), over every
# scenario of the WOMD training shards in SHARDS. Array task k runs half k % 2 of shard SHARDS[k // 2]; all tasks
# write into one run directory:
#   <run>/es_candidates/<batch>.npz + .labels / .labels_v2 / .labels_waymax.npz   (finetune --bank_dir)
#   <run>/anchor_cache/<batch>.anchor_cache.npz                                    (finetune --cache_dir)
#   <run>/<tfrecord>.scenario_<i>.json   closed-loop results of this WM + scorer on the collected scenes
# Resubmitting resumes (finished scenarios and their batches are skipped).
#
#   mkdir -p /home/mineuih/ishan/logs   # once
#   sbatch --array=0-7%4 slurm_collect_anchorwm_bank.sh
#   CKPT=.../final.pt SHARDS="00010 00011" sbatch --array=0-3 slurm_collect_anchorwm_bank.sh
set -euo pipefail

CKPT="${CKPT:-/data/user_data/mineuih/ishan/anchor_wm/runs/v2_scratch_joint_simple_fix55k_20261021_012534/final.pt}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-/data/user_data/mineuih/waymax_checkpoints/goal_cond_diffusion_policy/epoch_1000}"
read -r -a SHARDS <<< "${SHARDS:-00002 00003 00004 00005}"
SEED="${SEED:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK / 2 >= ${#SHARDS[@]} )); then echo "ERROR: task ${TASK} but only ${#SHARDS[@]} shards"; exit 1; fi
SHARD="training_tfexample.tfrecord-${SHARDS[$((TASK / 2))]}-of-01000"
HALF=$((TASK % 2))

ROOT="/home/mineuih/ishan/waymax_rs"
TF_ROOT="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example"
LANE_GRAPH_DIR="/data/user_data/mineuih/waymax_rs/lane_graphs"
CK_TAG="$(basename "$(dirname "${CKPT}")" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')_$(basename "${CKPT}" .pt)"
RUN="${RUN:-/data/user_data/mineuih/ishan/anchorwm_bank/es_${CK_TAG}_seed${SEED}_train$(IFS=_; echo "${SHARDS[*]}")}"
PART="${RUN}/parts/${SHARD##*tfrecord-}_h${HALF}"
mkdir -p "${PART}/tfrecords/training"
ln -sfn "${TF_ROOT}/training/${SHARD}" "${PART}/tfrecords/training/${SHARD}"
for p in "${TF_ROOT}/training/${SHARD}" "${LANE_GRAPH_DIR}/${SHARD}.lanegraph.zip" "${CKPT}" "${DIFFUSION_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

set +u
# shellcheck source=/dev/null
source /home/mineuih/miniconda3/bin/activate /home/mineuih/miniconda3/envs/waymax_rs_ishan
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="

TARGETS="${PART}/target_scenarios.json"
python - "${LANE_GRAPH_DIR}/${SHARD}.lanegraph.zip" "${SHARD}" "${HALF}" "${TARGETS}" <<'EOF'
import json, sys, zipfile
zip_path, shard, half, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
n = json.loads(zipfile.ZipFile(zip_path).read("manifest.json"))["num_records"]
mid = (n + 1) // 2
idx = list(range(0, mid)) if half == 0 else list(range(mid, n))
json.dump({shard: idx}, open(dst, "w"))
print(f"{shard} half {half}: scenarios {idx[0]}..{idx[-1]} ({len(idx)} of {n})")
EOF

echo "AnchorWM bank collection: ${SHARD} half ${HALF} (task ${TASK}), ckpt=${CKPT}, host=$(hostname), run=${RUN}"
python -u "${ROOT}/simulation/run_simulation.py" \
  --planner es --diffusion_ckpt_path "${DIFFUSION_CKPT}" --resample_timesteps 5 \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --replan_interval_steps 10 \
  --tfrecord_dir "${PART}/tfrecords" --split training --target_scenario_path "${TARGETS}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" --exp_dir "${RUN}" --start_from "${RUN}" --tag "$(basename "${RUN}")" \
  --num_worlds 5 --num_scenarios 100000 --seed "${SEED}" --sim_agents log --visualize_mode none \
  --other_agent_forecast wm --world_model_ckpt "${CKPT}" --wm_backend anchor --wm_refresh_steps 10 \
  --num_collision_worlds 1 --collision_reduce any --wm_planning_mode expected_independent \
  --anchor_scorer_ckpt "${CKPT}" --save_es_candidates --label_in_loop
echo "done: $(ls "${RUN}"/${SHARD}.scenario_*.json 2>/dev/null | wc -l) scenario results for ${SHARD}; \
$(ls "${RUN}"/anchor_cache/${SHARD}.*.anchor_cache.npz 2>/dev/null | wc -l) cached batches"
