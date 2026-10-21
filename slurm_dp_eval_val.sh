#!/bin/bash
#SBATCH --job-name=dp_eval_val
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/dp_eval_val_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/dp_eval_val_%A_%a.err

# Diffusion Planner closed loop on validation shards 00000-00009 (one shard per array task), same runner and
# settings as slurm_pdm_closed_val.sh: replan every 10 steps, plan replayed, success = goal reached without
# overlap / offroad.
#
#   sbatch slurm_dp_eval_val.sh <run_dir> [log|idm] [tag]
#     run_dir: .../runs/training_log/dp_womd_h50/<timestamp> (has args.json and latest.pth)
#   CKPT_NAME=model_epoch_500_trainloss_X.pth sbatch slurm_dp_eval_val.sh <run_dir>   # a specific epoch
set -euo pipefail

RUN_DIR="${1:?usage: sbatch slurm_dp_eval_val.sh <run_dir> [log|idm] [tag]}"
SIM_AGENTS="${2:-log}"
TAG="${3:-}"
SHARD="$(printf '%05d' "${SLURM_ARRAY_TASK_ID:-0}")"
CKPT="${RUN_DIR}/${CKPT_NAME:-latest.pth}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
DP_REPO="/zfsauton/scratch/ishanj/Diffusion-Planner"
EXP_ROOT="/zfsauton/scratch/ishanj/world_model_sim"
TF_ROOT="/zfsauton/scratch/mineuih/womd/tf_example"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/ishanj/miniconda3/bin/activate /zfsauton/scratch/ishanj/miniconda3/envs/waymax_rs
set -u
# torch on the GPU; JAX (Waymax rollout) on CPU, which avoids the env's jaxlib / CUDA plugin mismatch
export PYTHONPATH="${ROOT}" JAX_PLATFORMS=cpu

RUN_NAME="diffusion_planner_${SIM_AGENTS}_r10${TAG:+_${TAG}}"
EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
SHARD_NAME="validation_tfexample.tfrecord-${SHARD}-of-00150"
TF_LINK_ROOT="${EXP_DIR}/tfrecords_${SHARD}"
mkdir -p "${TF_LINK_ROOT}/validation" "${EXP_ROOT}/logs"
ln -sfn "${TF_ROOT}/validation/${SHARD_NAME}" "${TF_LINK_ROOT}/validation/${SHARD_NAME}"
for p in "${CKPT}" "${RUN_DIR}/args.json" "${LANE_GRAPH_DIR}/${SHARD_NAME}.lanegraph.zip"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

echo "Diffusion Planner ${RUN_NAME}: shard ${SHARD}, ckpt ${CKPT}, host=$(hostname)"
# run_closed_loop.py = run_simulation.py with torch's CUDA libraries loaded before TensorFlow (see its docstring)
python "${ROOT}/diffusion_planner_womd/run_closed_loop.py" \
  --planner diffusion_planner \
  --dp_repo "${DP_REPO}" \
  --dp_args_file "${RUN_DIR}/args.json" \
  --dp_ckpt "${CKPT}" \
  --tfrecord_dir "${TF_LINK_ROOT}" \
  --split validation \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --num_worlds 5 \
  --replan_interval_steps 10 \
  --sim_agents "${SIM_AGENTS}" \
  --visualize_mode none \
  --tag "val${SHARD}"
