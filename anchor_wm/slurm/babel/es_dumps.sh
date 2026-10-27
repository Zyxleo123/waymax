#!/bin/bash
#SBATCH --job-name=awm_dumps
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/mineuih/ishan/logs/awm_dumps_%j.out
#SBATCH --error=/home/mineuih/ishan/logs/awm_dumps_%j.err
#
# Re-run Diffusion-ES + AnchorWM on a few valfail1568 scenes with a dump per replan (anchor_wm.es_debug), for
# notebooks/anchor_wm_failures.ipynb. Same ES settings as slurm_es_anchorwm_valfail1568.sh, one world per batch.
#   TARGETS  {"<tfrecord name>": [scenario idx, ...]} json (the notebook writes one)
#   CKPT     AnchorWM checkpoint (forecaster); SCORER_CKPT the reward (default CKPT)
#   OUT      run dir: scenario jsons + dumps/
#
#   CKPT=... TARGETS=... OUT=... sbatch anchor_wm/slurm/babel/es_dumps.sh
#   CKPT=... TARGETS=... OUT=... bash anchor_wm/slurm/babel/es_dumps.sh        (inside a GPU allocation)
set -euo pipefail
source "${ROOT:-/home/mineuih/ishan/waymax_rs}/anchor_wm/slurm/babel/env.sh"
DATA=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs
DIFFUSION_CKPT="${DIFFUSION_CKPT:-/data/user_data/mineuih/waymax_checkpoints/goal_cond_diffusion_policy/epoch_1000}"
: "${CKPT:?}" "${TARGETS:?}" "${OUT:?}"
SCORER_CKPT="${SCORER_CKPT:-${CKPT}}"
mkdir -p "${OUT}"
[[ "${TARGETS}" -ef "${OUT}/targets.json" ]] || cp "${TARGETS}" "${OUT}/targets.json"

set +u; source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"; set -u
export PYTHONPATH="${ROOT}" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_command_buffer="
python -u "${ROOT}/simulation/run_simulation.py" --planner es \
  --other_agent_forecast wm --world_model_ckpt "${CKPT}" --wm_backend anchor --wm_refresh_steps 10 \
  --num_collision_worlds 1 --collision_reduce any --wm_planning_mode expected_independent \
  --anchor_scorer_ckpt "${SCORER_CKPT}" --diffusion_ckpt_path "${DIFFUSION_CKPT}" --resample_timesteps 5 \
  --tfrecord_dir "${DATA}/womd_compact/valfail1568" --split validation --target_scenario_path "${OUT}/targets.json" \
  --lane_graph_dir "${DATA}/lane_graphs_babel" --exp_dir "${OUT}" --start_from "${OUT}" --tag "$(basename "${OUT}")" \
  --num_worlds 1 --num_scenarios 100000 --seed "${SEED:-0}" --population_size 64 --elite_size 8 \
  --num_es_iterations 4 --replan_interval_steps 10 --sim_agents log --visualize_mode none \
  --es_debug_dir "${OUT}/dumps"
