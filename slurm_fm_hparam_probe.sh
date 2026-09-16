#!/bin/bash
#SBATCH --job-name=fm_hparam_probe
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/fm_hparam_probe_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/fm_hparam_probe_%j.err

# Small GT probe (~20 scenes): Diff baseline + FM setting sweep.
#
#   python scripts/build_fm_hparam_probe_subset.py --n 20 --seed 0
#   sbatch slurm_fm_hparam_probe.sh
#
# Optional env overrides:
#   SUBSET_DIR=... NUM_WORLDS=1
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
PROBE_ROOT="/zfsauton/scratch/sbellad/immediate_exp/fm_hparam_probe_runs"
LOG_DIR="/zfsauton/scratch/sbellad/immediate_exp/logs"
NUM_WORLDS="${NUM_WORLDS:-1}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"

mkdir -p "${PROBE_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS=ignore

if [[ ! -f "${TARGET_JSON}" ]]; then
  echo "Building probe subset..."
  python scripts/build_fm_hparam_probe_subset.py --n 20 --seed 0 --out_dir "${SUBSET_DIR}" --num_worlds "${NUM_WORLDS}"
fi

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

# Fixed Diff-ES GT baseline on the same 20 scenes.
run_one es "diff_gt_baseline_pop64_it4_rs5" \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_timesteps 5

# FM-ES GT sweep (conclude from these).
run_one es_fm "fm_gt_t0.2_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.2

run_one es_fm "fm_gt_t0.4_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.4

run_one es_fm "fm_gt_t0.5_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.5

run_one es_fm "fm_gt_t0.6_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.6

run_one es_fm "fm_gt_t0.8_pop64_it4" \
  --fm_ckpt_path "${FM_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 4 --resample_t_start 0.8

run_one es_fm "fm_gt_t0.5_pop64_it6" \
  --fm_ckpt_path "${FM_CKPT}" \
  --population_size 64 --elite_size 8 --num_es_iterations 6 --resample_t_start 0.5

run_one es_fm "fm_gt_t0.5_pop128_it4" \
  --fm_ckpt_path "${FM_CKPT}" \
  --population_size 128 --elite_size 16 --num_es_iterations 4 --resample_t_start 0.5

echo
echo "All probe runs finished. Summarize with:"
echo "  python scripts/summarize_fm_hparam_probe.py"
python scripts/summarize_fm_hparam_probe.py || true
