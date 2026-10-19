#!/bin/bash
#SBATCH --job-name=time_es
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/time_es_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/logs/time_es_%j.err

# Planning-time benchmark: Diffusion-ES vs FM-ES, each with CV and MF world model.
# Times only planner.plan_trajectory (1 warm-up scene + N_TIMED timed scenes, num_worlds=1).
#
#   sbatch slurm/sim/time_es_planners.sh
#   FM_IT=6 FM_RS=40 PLANNERS=fm sbatch slurm/sim/time_es_planners.sh   # FM only, other budget
set -euo pipefail

ROOT=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad
SUBSET=${SCRATCH}/immediate_exp/es_failure_subset_n1500_seed0
DIFFUSION_CKPT=${SCRATCH}/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500
FM_CKPT=/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500
MF_WM_CKPT=${SCRATCH}/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt
N_TIMED=${N_TIMED:-3}
NUM_WORLDS=${NUM_WORLDS:-1}
FM_IT=${FM_IT:-4}
FM_RS=${FM_RS:-20}
PLANNERS=${PLANNERS:-diffusion,fm}
OUT=${SCRATCH}/timing_es/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_fm_it${FM_IT}_rs${FM_RS}

set +u
source ${SCRATCH}/miniconda3/bin/activate ${SCRATCH}/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH=${ROOT}
mkdir -p "${OUT}"

COMMON=(--tfrecord_dir ${SUBSET}/tfrecords --target_scenario_path ${SUBSET}/target_scenarios.json
        --lane_graph_dir /data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs --split training
        --num_worlds ${NUM_WORLDS} --num_scenarios $(( (N_TIMED + 1) * NUM_WORLDS )) --warmup_scenarios 1
        --seed 0 --population_size 64 --elite_size 8 --replan_interval_steps 10 --visualize_mode none)
DIFF=(--planner es --diffusion_ckpt_path ${DIFFUSION_CKPT} --num_es_iterations 4 --resample_timesteps 5)
FM=(--planner es_fm --fm_ckpt_path ${FM_CKPT} --num_es_iterations ${FM_IT} --resample_num_steps ${FM_RS}
    --resample_t_start 0.2 --fm_num_steps 50 --fm_solver euler)
CV=(--other_agent_forecast cv)
WM=(--other_agent_forecast wm --world_model_ckpt ${MF_WM_CKPT} --world_model_device auto
    --wm_backend multifuture --wm_planning_mode expected_independent --wm_refresh_steps 10
    --num_collision_worlds 1 --collision_reduce any --collision_sample_seed 0)

echo "host=$(hostname) out=${OUT} planners=${PLANNERS} n_timed=${N_TIMED} num_worlds=${NUM_WORLDS} fm_it=${FM_IT} fm_rs=${FM_RS}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

run() {
  local name=$1; shift
  echo "=== ${name} start $(date -Is)"
  python ${ROOT}/scripts/time_es_planners.py "${COMMON[@]}" "$@" \
    --exp_dir ${OUT}/${name} --tag ${name} > ${OUT}/${name}.log 2>&1 || { echo "${name} FAILED (see ${OUT}/${name}.log)"; return 0; }
  grep '^TIMING' ${OUT}/${name}.log | sed "s/^TIMING/${name}:/"
}
if [[ ",${PLANNERS}," == *",diffusion,"* ]]; then
  run diffusion_cv "${DIFF[@]}" "${CV[@]}"
  run diffusion_wm "${DIFF[@]}" "${WM[@]}"
fi
if [[ ",${PLANNERS}," == *",fm,"* ]]; then
  run fm_cv "${FM[@]}" "${CV[@]}"
  run fm_wm "${FM[@]}" "${WM[@]}"
fi
echo "DONE; logs and timing.json under ${OUT}"
