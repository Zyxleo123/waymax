#!/bin/bash
#SBATCH --job-name=collect_es_cand
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/logs/collect_es_candidates_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/logs/collect_es_candidates_%j.err

# Diffusion-ES + multi-future WM (expected_independent) on the 1,500 failure scenes,
# saving every evaluated ES candidate per replan, then GT-replay labels offline.
# Planner settings are identical to
#   slurm/sim/run_es.sh --subset fail1500 --planner diffusion --forecast wm_mf
# plus --save_es_candidates. Submit from rhea or lov4:
#
#   sbatch slurm/sim/collect_es_candidates_fail1500.sh                       # all 1,500 + labels
#   sbatch slurm/sim/collect_es_candidates_fail1500.sh --smoke               # 10 scenes, separate folder
#   sbatch slurm/sim/collect_es_candidates_fail1500.sh --stage label         # labels only
#
# Resubmitting the same command resumes: finished scenarios (result JSON + candidate
# file) are skipped, and already-labelled batch files are not relabelled.
#
# Output: ${OUT_ROOT}/${RUN_NAME}/
#   <tfrecord>.scenario_NNN.json   per-scenario outcome (runner format), summary.json
#   es_candidates/run_config.json  args, checkpoints, ES settings, git commit
#   es_candidates/<tfrecord>.scen_<i>_..._<m>.npz          candidates + context (one per batch of 5)
#   es_candidates/<tfrecord>.scen_<i>_..._<m>.labels.npz   GT-replay labels + progress metrics
set -euo pipefail

ROOT=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad
OUT_ROOT=${SCRATCH}/es_discriminator/es_candidate_data
SUBSET=${SCRATCH}/Diffusion_ES_sim/es_failure_subset_n1500_seed0
DIFFUSION_CKPT=${SCRATCH}/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500
MF_WM_CKPT=${SCRATCH}/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt
LANE_GRAPH_DIR=/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs
SEED=0
STAGE=all
NUM_SCENARIOS=100000
RUN_NAME=fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed${SEED}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --smoke) NUM_SCENARIOS=10; RUN_NAME="smoke_${RUN_NAME}"; shift ;;
    --num_scenarios) NUM_SCENARIOS="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
case "${STAGE}" in collect|label|all) ;; *) echo "ERROR: --stage must be collect|label|all" >&2; exit 1 ;; esac
set --  # conda's activate script reads "$@"

OUT=${OUT_ROOT}/${RUN_NAME}
mkdir -p "${OUT}" "${SCRATCH}/es_discriminator/logs"
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

set +u  # conda activate.d hooks reference unset variables
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} stage=${STAGE} out=${OUT}"
nvidia-smi || true

if [[ "${STAGE}" == collect || "${STAGE}" == all ]]; then
  echo "[$(date)] collect: WM + Diffusion-ES on ${SUBSET} (num_scenarios=${NUM_SCENARIOS})"
  python -u simulation/run_simulation.py \
    --planner es \
    --tfrecord_dir "${SUBSET}/tfrecords" \
    --split training \
    --target_scenario_path "${SUBSET}/target_scenarios.json" \
    --lane_graph_dir "${LANE_GRAPH_DIR}" \
    --exp_dir "${OUT_ROOT}" \
    --start_from "${OUT}" \
    --tag "${RUN_NAME}" \
    --num_worlds 5 \
    --num_scenarios "${NUM_SCENARIOS}" \
    --seed "${SEED}" \
    --population_size 64 \
    --elite_size 8 \
    --num_es_iterations 4 \
    --resample_timesteps 5 \
    --replan_interval_steps 10 \
    --other_agent_forecast wm \
    --other_agents log \
    --visualize_mode none \
    --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
    --world_model_ckpt "${MF_WM_CKPT}" \
    --world_model_device auto \
    --wm_backend multifuture \
    --wm_refresh_steps 10 \
    --num_collision_worlds 1 \
    --collision_reduce any \
    --collision_sample_seed 0 \
    --wm_planning_mode expected_independent \
    --save_es_candidates
fi

if [[ "${STAGE}" == label || "${STAGE}" == all ]]; then
  echo "[$(date)] label: GT replay collision/offroad + progress for every candidate"
  XLA_PYTHON_CLIENT_PREALLOCATE=false python -u -m es_discriminator.label_es_candidates \
    --data_dir "${OUT}/es_candidates" \
    --tfrecord_dir "${SUBSET}/tfrecords/training"
fi

echo "[$(date)] done; $(ls "${OUT}"/es_candidates/*.labels.npz 2>/dev/null | wc -l) labelled batch files, $(du -sh "${OUT}/es_candidates" | cut -f1)"
