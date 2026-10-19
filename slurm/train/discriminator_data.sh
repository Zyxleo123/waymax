#!/bin/bash
#SBATCH --job-name=disc_data
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/es_discriminator/logs/disc_data_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/es_discriminator/logs/disc_data_%j.err

# Discriminator training data: Diffusion-ES + multi-future WM (expected_independent) with every
# evaluated ES candidate saved per replan, then labels and the WM / road-edge input caches.
# Afterwards `python -m es_discriminator.build_training_set` links fail1500 + the WOMD shards
# into one training set for slurm/train/train_wm_discriminator.sh --data.
# Submit from rhea or lov4:
#
#   sbatch slurm/train/discriminator_data.sh --source fail1500                 # the 1,500 failure scenes
#   sbatch slurm/train/discriminator_data.sh --source fail1500 --stage wm_cache
#   sbatch --array=2-11%2 --gres=gpu:a6000:1 --time=16:00:00 \
#          slurm/train/discriminator_data.sh --source womd                     # WOMD training shards 2-11
#   sbatch --array=5 slurm/train/discriminator_data.sh --source womd           # redo one shard
#   ... --smoke                                                                # a few scenes, separate folder
#
# Stages (--stage S runs one, default all; resubmitting resumes, finished work is skipped):
#   collect   simulation with --save_es_candidates
#   label     GT-replay collision/offroad + progress            (.labels.npz)
#   decisions Waymax-exact offroad relabel (.labels_v2.npz) + per-scene log replay (scenes.npz);
#             fail1500 also writes the decision report and invalid_at_start.json
#   waymax    Waymax-exact overlap/offroad labels                (.labels_waymax.npz);
#             fail1500 also checks time/frame alignment and the scenario split
#   wm_cache  frozen WM forward + road-edge tokens per replan    (.wm.npz, .edges.npz)
#
# Output: es_candidate_data/<run>/ (fail1500) or es_candidate_data/<run>/shard_NNNNN/ (womd);
# fail1500 keeps its caches in es_discriminator/{wm_predictions,road_edges}/<run>/ and its reports
# in es_discriminator/{decision_analysis,real_es_labels}/.
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
DISC=${SCRATCH}/es_discriminator
OUT_ROOT=${DISC}/es_candidate_data
FAIL1500_SUBSET=${SCRATCH}/Diffusion_ES_sim/es_failure_subset_n1500_seed0
WOMD_TRAIN=/zfsauton/datasets/womd/tf_example/training
DIFFUSION_CKPT=${SCRATCH}/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500
MF_WM_CKPT=${SCRATCH}/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt
LANE_GRAPH_DIR=/zfsauton/scratch/mineuih/waymax_rs/lane_graphs
SEED=0
MIN_FREE_GB=${MIN_FREE_GB:-150}
SOURCE="" STAGE=all SMOKE=0 SHARD=${SLURM_ARRAY_TASK_ID:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --shard) SHARD="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
case "${STAGE}" in all|collect|label|decisions|waymax|wm_cache) ;;
  *) echo "ERROR: --stage must be all|collect|label|decisions|waymax|wm_cache" >&2; exit 1 ;; esac
set --  # conda's activate script reads "$@"

case "${SOURCE}" in
  fail1500)
    RUN=fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed${SEED}
    NUM_SCENARIOS=100000 VALIDATE_FILES=6
    [[ ${SMOKE} == 1 ]] && RUN=smoke_${RUN} NUM_SCENARIOS=10
    OUT=${OUT_ROOT}/${RUN}
    TFRECORD_ROOT=${FAIL1500_SUBSET}/tfrecords
    TARGETS=${FAIL1500_SUBSET}/target_scenarios.json
    WM_DIR=${DISC}/wm_predictions/${RUN} EDGE_DIR=${DISC}/road_edges/${RUN}
    ANALYSIS=${DISC}/decision_analysis
    LABEL_REPORTS=${DISC}/real_es_labels
    COLLECT_EXTRA=(--exp_dir "${OUT_ROOT}" --tag "${RUN}")
    ;;
  womd)
    [[ -n "${SHARD}" ]] || { echo "ERROR: no shard (submit as an array or pass --shard N)" >&2; exit 1; }
    RUN=womd_train_s02_11_es_diffusion_wm_mf_epoch0450_expected_independent_seed${SEED}
    TARGETS=${OUT_ROOT}/${RUN}/target_scenarios.json
    [[ -f "${TARGETS}" ]] || { echo "ERROR: ${TARGETS} missing; build it with es_discriminator.make_collection_targets" >&2; exit 1; }
    RUN_DIR=${OUT_ROOT}/${RUN} NUM_SCENARIOS=100000 VALIDATE_FILES=6
    [[ ${SMOKE} == 1 ]] && RUN_DIR=${OUT_ROOT}/smoke_${RUN} NUM_SCENARIOS=5 VALIDATE_FILES=1
    SHARD_NAME=$(printf "training_tfexample.tfrecord-%05d-of-01000" "${SHARD}")
    OUT=$(printf "%s/shard_%05d" "${RUN_DIR}" "${SHARD}")
    TFRECORD_ROOT=${OUT}/tfrecords
    WM_DIR=${OUT}/wm_predictions EDGE_DIR=${OUT}/road_edges
    ANALYSIS=${OUT}/analysis
    LABEL_REPORTS=${OUT}/analysis
    COLLECT_EXTRA=(--exp_dir "${RUN_DIR}")
    mkdir -p "${TFRECORD_ROOT}/training"
    ln -sfn "${WOMD_TRAIN}/${SHARD_NAME}" "${TFRECORD_ROOT}/training/${SHARD_NAME}"
    ;;
  *) echo "ERROR: --source must be fail1500|womd" >&2; exit 1 ;;
esac
DATA=${OUT}/es_candidates
TFRECORDS=${TFRECORD_ROOT}/training
mkdir -p "${OUT}" "${DISC}/logs"
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

set +u  # conda activate.d hooks reference unset variables
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
log() { echo "[$(date)] $*"; }
run_stage() { [[ "${STAGE}" == all || "${STAGE}" == "$1" ]]; }
check_disk() {
  local free_gb
  free_gb=$(df -BG --output=avail "${SCRATCH}" | tail -1 | tr -dc '0-9')
  log "scratch free: ${free_gb} GB (minimum ${MIN_FREE_GB} GB)"
  if (( free_gb < MIN_FREE_GB )); then log "ERROR: stopping before $1, scratch nearly full"; exit 1; fi
}
log "host=$(hostname) job=${SLURM_JOB_ID:-local} source=${SOURCE} stage=${STAGE} smoke=${SMOKE} out=${OUT}"
nvidia-smi || true

# The runner treats an unreadable shard as finished, so a node without the dataset would
# silently produce nothing.
for shard in "${TFRECORDS}"/*; do
  head -c 16 "${shard}" > /dev/null 2>&1 || {
    log "ERROR: ${shard} unreadable on $(hostname); resubmit with --exclude $(hostname -s)"; exit 1; }
done

if run_stage collect; then
  check_disk collect
  log "collect: Diffusion-ES + MF WM (num_scenarios=${NUM_SCENARIOS})"
  python -u simulation/run_simulation.py \
    --planner es \
    --tfrecord_dir "${TFRECORD_ROOT}" \
    --split training \
    --target_scenario_path "${TARGETS}" \
    --lane_graph_dir "${LANE_GRAPH_DIR}" \
    "${COLLECT_EXTRA[@]}" \
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
    --wm_planning_mode expected_independent \
    --save_es_candidates
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
if run_stage label; then
  check_disk label
  log "label: GT replay collision/offroad + progress (.labels.npz)"
  python -u -m es_discriminator.label_es_candidates --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}"
fi

if run_stage decisions; then
  mkdir -p "${ANALYSIS}"
  if [[ "${SOURCE}" == fail1500 ]]; then
    log "decisions: relabel + scenes + report (${ANALYSIS})"
    python -u -m es_discriminator.analyze_es_decisions all \
      --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" --out_dir "${ANALYSIS}"
  else
    log "decisions: relabel (.labels_v2.npz) + per-scene log replay (scenes.npz)"
    rm -f "${ANALYSIS}/scenes.npz"  # the scenes stage never refreshes an existing file
    for stage in relabel scenes; do
      python -u -m es_discriminator.analyze_es_decisions "${stage}" \
        --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" --out_dir "${ANALYSIS}"
    done
  fi
fi

if run_stage waymax; then
  mkdir -p "${LABEL_REPORTS}"
  log "waymax: Waymax-exact labels (.labels_waymax.npz)"
  python -u -m es_discriminator.real_es.waymax_labels --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
    --summary_out "${LABEL_REPORTS}/waymax_labels_summary.json"
  if [[ "${SOURCE}" == fail1500 && ${SMOKE} == 0 ]]; then
    log "waymax: time/frame/context alignment check on 12 batches"
    python -u -m es_discriminator.real_es.check_alignment --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
      --max_files 12 --out "${LABEL_REPORTS}/alignment.json"
    log "waymax: scenario split + ES-unsafe scene check"
    python -u - "${DATA}" "${DISC}/splits/fail1500_scenario_split_seed0.json" <<'EOF'
import json, sys
from es_discriminator.real_es.dataset import (all_scenario_ids, es_unsafe_scenes, invalid_scene_keys, load_bank,
                                              load_or_create_split)
data, split_path = sys.argv[1:3]
split = load_or_create_split(split_path, all_scenario_ids(data))
bank = load_bank(data, verbose=False)
mask, _ = es_unsafe_scenes(bank, invalid_scene_keys())
counts = {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}
in_test = int(sum(split[s] == "test" for s in bank.scenes["scen_id"][mask]))
print(json.dumps({"batch_files": len(bank.files), "scenes_per_split": counts,
                  "es_unsafe_scenes": int(mask.sum()), "es_unsafe_scenes_in_test": in_test}))
if len(bank.files) != 300 or int(mask.sum()) != 285:
    raise SystemExit("expected 300 labelled batch files and 285 ES-unsafe scenes")
EOF
  fi
fi

if run_stage wm_cache; then
  check_disk wm_cache
  mkdir -p "${WM_DIR}" "${EDGE_DIR}"
  log "wm_cache: WM + road-edge caches (${WM_DIR}, ${EDGE_DIR})"
  python -u -m es_discriminator.real_es.wm_cache --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
    --checkpoint "${MF_WM_CKPT}" --wm_dir "${WM_DIR}" --edge_dir "${EDGE_DIR}" \
    --validate_files "${VALIDATE_FILES}" --summary_out "${WM_DIR}/summary_${SLURM_JOB_ID:-local}.json"
  n_wm=$(find "${WM_DIR}" -name '*.wm.npz' | wc -l)
  n_edge=$(find "${EDGE_DIR}" -name '*.edges.npz' | wc -l)
  log "cached ${n_wm} WM files, ${n_edge} edge files"
  if [[ "${SOURCE}" == fail1500 && ${SMOKE} == 0 ]]; then
    [[ ${n_wm} -eq 300 && ${n_edge} -eq 300 ]] || { log "ERROR: incomplete cache (expected 300)"; exit 1; }
  fi
fi

log "done: $(ls "${DATA}"/*.labels_waymax.npz 2>/dev/null | wc -l) batch files ready for training, $(du -sh "${OUT}" | cut -f1)"
