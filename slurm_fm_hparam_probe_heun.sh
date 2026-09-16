#!/bin/bash
#SBATCH --job-name=fm_probe_heun
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/fm_hparam_probe_heun_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/fm_hparam_probe_heun_%j.err

# Same 20-scene GT probe as fm_hparam_probe_runs, but FM uses --fm_solver heun.
# Adds milder resample_t_start={0.1, 0.15} (euler probe favored less noise).
#
#   sbatch slurm_fm_hparam_probe_heun.sh
#   python scripts/summarize_fm_hparam_probe.py \
#     --runs_dir /zfsauton/scratch/sbellad/immediate_exp/fm_hparam_probe_runs_heun
#
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
cd "${ROOT}"

CONDA_ROOT="/zfsauton/scratch/sbellad/miniconda3"
WAYMAX_ENV="/zfsauton/scratch/sbellad/conda_envs/waymax_rs"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

SUBSET_DIR="${SUBSET_DIR:-/zfsauton/scratch/sbellad/immediate_exp/fm_hparam_probe_n20_seed0}"
TARGET_JSON="${SUBSET_DIR}/target_scenarios.json"
TFRECORD_DIR="${SUBSET_DIR}/tfrecords"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
FM_CKPT="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500"
PROBE_ROOT="/zfsauton/scratch/sbellad/immediate_exp/fm_hparam_probe_runs_heun"
LOG_DIR="/zfsauton/scratch/sbellad/immediate_exp/logs"
NUM_WORLDS="${NUM_WORLDS:-1}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
FM_SOLVER="${FM_SOLVER:-heun}"

mkdir -p "${PROBE_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS=ignore

if [[ ! -f "${TARGET_JSON}" ]]; then
  echo "Building probe subset..."
  python scripts/build_fm_hparam_probe_subset.py --n 20 --seed 0 --out_dir "${SUBSET_DIR}" --num_worlds "${NUM_WORLDS}"
fi

{
  echo "============================================================"
  echo " FM hparam probe (solver=${FM_SOLVER})"
  echo " subset=${SUBSET_DIR}"
  echo " probe_root=${PROBE_ROOT}"
  echo " time=$(date -Is)"
  echo "============================================================"
  nvidia-smi || true
} | tee -a "${LOG_DIR}/fm_hparam_probe_heun_meta.log"

run_one() {
  local planner="$1"
  local tag="$2"
  shift 2
  local exp_dir="${PROBE_ROOT}/${tag}"
  mkdir -p "${exp_dir}"
  echo
  echo "============================================================"
  echo " START ${tag}  planner=${planner}  $(date -Is)"
  echo "============================================================"

  local cmd=(
    python "${ROOT}/simulation/run_simulation.py"
    --planner "${planner}"
    --tfrecord_dir "${TFRECORD_DIR}"
    --lane_graph_dir "${LANE_GRAPH_DIR}"
    --exp_dir "${exp_dir}"
    --split training
    --num_worlds "${NUM_WORLDS}"
    --num_scenarios 100000
    --replan_interval_steps "${REPLAN_INTERVAL_STEPS}"
    --other_agent_forecast gt
    --target_scenario_path "${TARGET_JSON}"
    --visualize_mode none
    --tag "${tag}"
  )
  cmd+=("$@")

  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: ${tag} exited ${rc}"
    exit "${rc}"
  fi
  echo "DONE ${tag}"
}

# Diff baseline (solver-irrelevant; kept for a self-contained comparison table).
run_one es "diff_gt_baseline_pop64_it4_rs5" \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_timesteps 5

# FM-ES GT + Heun: same grid as euler probe, plus milder t={0.1, 0.15}.
run_one es_fm "fm_gt_heun_t0.1_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.1

run_one es_fm "fm_gt_heun_t0.15_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.15

run_one es_fm "fm_gt_heun_t0.2_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.2

run_one es_fm "fm_gt_heun_t0.4_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.4

run_one es_fm "fm_gt_heun_t0.5_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.5

run_one es_fm "fm_gt_heun_t0.6_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.6

run_one es_fm "fm_gt_heun_t0.8_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.8

run_one es_fm "fm_gt_heun_t0.5_pop64_it6" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 64 --elite_size 8 --num_es_iterations 6 --resample_t_start 0.5

run_one es_fm "fm_gt_heun_t0.5_pop128_it4" \
  --fm_ckpt_path "${FM_CKPT}" --fm_solver "${FM_SOLVER}" \
  --population_size 128 --elite_size 16 --num_es_iterations 4 --resample_t_start 0.5

echo
echo "All Heun probe runs finished. Summarize with:"
echo "  python scripts/summarize_fm_hparam_probe.py --runs_dir ${PROBE_ROOT}"
python scripts/summarize_fm_hparam_probe.py --runs_dir "${PROBE_ROOT}" || true
