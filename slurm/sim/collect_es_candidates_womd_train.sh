#!/bin/bash
#SBATCH --job-name=collect_womd
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=2-11%2
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/es_discriminator/logs/collect_womd_%A_%a.out
#SBATCH --error=/zfsauton/scratch/sbellad/es_discriminator/logs/collect_womd_%A_%a.err

# Discriminator training data on whole WOMD training shards 00002-00011 (one array task per
# shard, 2 at a time). Same planner as collect_es_candidates_fail1500.sh: Diffusion-ES with
# the usual diffusion checkpoint + multi-future WM (expected_independent), --save_es_candidates.
# Scenes already in the fail1500 set are skipped via target_scenarios.json
# (es_discriminator/make_collection_targets.py). Shards 00000/00001 are the trainval benchmark.
# Each task runs the full pipeline for its shard:
#   collect -> labels (.labels.npz) -> relabel (.labels_v2.npz) + scenes.npz -> Waymax labels
#   (.labels_waymax.npz) -> WM + road-edge caches
# Afterwards, es_discriminator.build_training_set links fail1500 + these shards into one
# training set (see slurm/train/train_wm_discriminator.sh --data).
# Submit from rhea or lov4:
#
#   sbatch --array=2 slurm/sim/collect_es_candidates_womd_train.sh --smoke   # 1 batch of 5 scenes
#   sbatch slurm/sim/collect_es_candidates_womd_train.sh                     # shards 2-11
#   sbatch --gres=gpu:a5000:1 slurm/sim/collect_es_candidates_womd_train.sh  # A5000s instead
#   sbatch --array=5 slurm/sim/collect_es_candidates_womd_train.sh           # redo one shard
#
# Resubmitting resumes: finished scenarios, labelled batches and cached batches are skipped.
# A task stops before each stage if scratch has less than MIN_FREE_GB free.
#
# Output: ${OUT_ROOT}/${RUN_NAME}/shard_NNNNN/
#   <tfrecord>.scenario_NNN.json, summary.json      per-scenario outcomes (runner format)
#   es_candidates/*.npz (+ .labels / .labels_v2 / .labels_waymax)
#   wm_predictions/*.wm.npz, road_edges/*.edges.npz, analysis/scenes.npz, job_<id>.log
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
OUT_ROOT=${SCRATCH}/es_discriminator/es_candidate_data
WOMD_TRAIN=/zfsauton/datasets/womd/tf_example/training
DIFFUSION_CKPT=${SCRATCH}/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500
MF_WM_CKPT=${SCRATCH}/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt
LANE_GRAPH_DIR=/zfsauton/scratch/mineuih/waymax_rs/lane_graphs
SEED=0
RUN_NAME=womd_train_s02_11_es_diffusion_wm_mf_epoch0450_expected_independent_seed${SEED}
TARGETS=${OUT_ROOT}/${RUN_NAME}/target_scenarios.json
NUM_SCENARIOS=100000
VALIDATE_FILES=6
MIN_FREE_GB=${MIN_FREE_GB:-150}
SHARD=${SLURM_ARRAY_TASK_ID:-}
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --shard) SHARD="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
[[ -n "${SHARD}" ]] || { echo "ERROR: no shard (submit as an array or pass --shard N)" >&2; exit 1; }
[[ -f "${TARGETS}" ]] || { echo "ERROR: ${TARGETS} missing; build it with es_discriminator.make_collection_targets" >&2; exit 1; }
set --  # conda's activate script reads "$@"

SHARD_NAME=$(printf "training_tfexample.tfrecord-%05d-of-01000" "${SHARD}")
RUN_DIR=${OUT_ROOT}/${RUN_NAME}
if [[ ${SMOKE} == 1 ]]; then
  RUN_DIR=${OUT_ROOT}/smoke_${RUN_NAME}; NUM_SCENARIOS=5; VALIDATE_FILES=1
fi
OUT=$(printf "%s/shard_%05d" "${RUN_DIR}" "${SHARD}")
DATA=${OUT}/es_candidates
TFRECORDS=${OUT}/tfrecords/training
mkdir -p "${TFRECORDS}" "${SCRATCH}/es_discriminator/logs"
ln -sfn "${WOMD_TRAIN}/${SHARD_NAME}" "${TFRECORDS}/${SHARD_NAME}"
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

set +u  # conda activate.d hooks reference unset variables
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
log() { echo "[$(date)] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-local} shard=${SHARD_NAME} smoke=${SMOKE} out=${OUT}"
nvidia-smi || true

# The runner treats an unreadable shard as finished, so a node without the dataset would
# silently produce nothing.
if ! head -c 16 "${TFRECORDS}/${SHARD_NAME}" > /dev/null 2>&1; then
  log "ERROR: ${WOMD_TRAIN}/${SHARD_NAME} unreadable on $(hostname); resubmit with --exclude $(hostname -s)"
  exit 1
fi
check_disk() {
  local free_gb
  free_gb=$(df -BG --output=avail "${SCRATCH}" | tail -1 | tr -dc '0-9')
  log "scratch free: ${free_gb} GB (minimum ${MIN_FREE_GB} GB)"
  if (( free_gb < MIN_FREE_GB )); then log "ERROR: stopping before $1, scratch nearly full"; exit 1; fi
}

check_disk collect
log "collect: Diffusion-ES + MF WM on ${SHARD_NAME} ($(python -c "import json;print(len(json.load(open('${TARGETS}')).get('${SHARD_NAME}',[])))") targets)"
python -u simulation/run_simulation.py \
  --planner es \
  --tfrecord_dir "${OUT}/tfrecords" \
  --split training \
  --target_scenario_path "${TARGETS}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${RUN_DIR}" \
  --start_from "${OUT}" \
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

export XLA_PYTHON_CLIENT_PREALLOCATE=false
check_disk labels
log "labels: GT replay collision/offroad + progress (.labels.npz)"
python -u -m es_discriminator.label_es_candidates --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}"

log "relabel: Waymax-exact offroad (.labels_v2.npz), then per-scene log replay (analysis/scenes.npz)"
rm -f "${OUT}/analysis/scenes.npz"  # the scenes stage never refreshes an existing file
for stage in relabel scenes; do
  python -u -m es_discriminator.analyze_es_decisions "${stage}" \
    --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" --out_dir "${OUT}/analysis"
done

log "Waymax labels (.labels_waymax.npz)"
python -u -m es_discriminator.real_es.waymax_labels --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
  --summary_out "${OUT}/analysis/waymax_labels_summary.json"

check_disk wm_cache
log "WM + road-edge caches"
python -u -m es_discriminator.real_es.wm_cache --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
  --checkpoint "${MF_WM_CKPT}" --wm_dir "${OUT}/wm_predictions" --edge_dir "${OUT}/road_edges" \
  --validate_files "${VALIDATE_FILES}" --summary_out "${OUT}/wm_predictions/summary_${SLURM_JOB_ID:-local}.json"

n_ready=$(ls "${DATA}"/*.labels_waymax.npz 2>/dev/null | wc -l)
log "done: ${n_ready} batch files ready for training, $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenarios, $(du -sh "${OUT}" | cut -f1)"
