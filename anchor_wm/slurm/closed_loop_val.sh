#!/bin/bash
#SBATCH --job-name=awm_clval
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-9
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/closed_loop_val_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/closed_loop_val_%A_%a.err
#
# Diffusion-ES + AnchorWM on WOMD validation shards 00000-00009 (G3), one shard per array task, with
# the same planner settings as the MF-Gaussian val0-9 run (slurm_es_wm_val.sh: T=1, log replay,
# expected_independent, margin_tiebreak offroad). REWARD=scorer uses the checkpoint's learned scorer.
# Resubmitting resumes a shard; a finished shard gets a DONE marker.
#
#   CKPT=.../best.pt REWARD=scorer sbatch anchor_wm/slurm/closed_loop_val.sh
#   SMOKE=1 CKPT=... REWARD=scorer sbatch --array=0 --time=00:40:00 anchor_wm/slurm/closed_loop_val.sh   # 5 scenes
set -euo pipefail

: "${CKPT:?set CKPT to an AnchorWM checkpoint}"
REWARD="${REWARD:-scorer}"
OFFROAD_MODE="${OFFROAD_MODE:-margin_tiebreak}"
SMOKE="${SMOKE:-0}"
SHARD="$(printf '%05d' "${SLURM_ARRAY_TASK_ID:-0}")"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
OUT_ROOT="/zfsauton/scratch/ishanj/anchor_wm/closed_loop_val"
TF_ROOT="/zfsauton/scratch/mineuih/womd/tf_example"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
case "${REWARD}" in
  rule)   REWARD_ARGS=() ;;
  scorer) REWARD_ARGS=(--anchor_scorer_ckpt "${CKPT}") ;;
  *) echo "ERROR: REWARD must be rule|scorer, got ${REWARD}" >&2; exit 1 ;;
esac
TAG="$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}" .pt)"
RUN_NAME="${RUN_NAME:-val0-9_es_anchorwm_${TAG}_reward_${REWARD}_off${OFFROAD_MODE}}"
[[ "${SMOKE}" == "1" ]] && RUN_NAME="smoke_${RUN_NAME}"
EXP_DIR="${OUT_ROOT}/${RUN_NAME}"
OUT="${EXP_DIR}/shard_${SHARD}"
SHARD_NAME="validation_tfexample.tfrecord-${SHARD}-of-00150"
TF_LINK_ROOT="${EXP_DIR}/tfrecords_${SHARD}"
mkdir -p "${OUT}" "${TF_LINK_ROOT}/validation" /zfsauton/scratch/ishanj/anchor_wm/logs
ln -sfn "${TF_ROOT}/validation/${SHARD_NAME}" "${TF_LINK_ROOT}/validation/${SHARD_NAME}"
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1
for p in "${TF_ROOT}/validation/${SHARD_NAME}" "${LANE_GRAPH_DIR}/${SHARD_NAME}.lanegraph.zip" "${CKPT}" "${DIFFUSION_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done

set +u  # conda activate.d hooks reference unset variables
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
echo "[$(date)] ${RUN_NAME}: shard ${SHARD}, host=$(hostname), job=${SLURM_JOB_ID:-local}, ckpt=${CKPT}, out=${OUT}"

python -u simulation/run_simulation.py \
  --planner es \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --resample_timesteps 5 \
  --tfrecord_dir "${TF_LINK_ROOT}" \
  --split validation \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --start_from "${OUT}" \
  --num_worlds 5 \
  --num_scenarios "$([[ "${SMOKE}" == "1" ]] && echo 5 || echo 100000)" \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations 4 \
  --replan_interval_steps 10 \
  --other_agent_forecast wm \
  --visualize_mode none \
  --tag "${RUN_NAME}_val${SHARD}" \
  --sim_agents log \
  --world_model_ckpt "${CKPT}" \
  --world_model_device auto \
  --wm_backend anchor \
  --wm_refresh_steps 10 \
  --num_collision_worlds 1 \
  --collision_reduce any \
  --collision_sample_seed 0 \
  --wm_planning_mode expected_independent \
  --offroad_mode "${OFFROAD_MODE}" \
  --wm_mode_temperature 1 \
  "${REWARD_ARGS[@]}"
n=$(ls "${OUT}"/*.scenario_*[0-9].json 2>/dev/null | wc -l)
echo "[$(date)] done: ${n} scenarios"
date > "${OUT}/DONE"
