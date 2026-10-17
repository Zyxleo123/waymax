#!/bin/bash
#SBATCH --job-name=es_disc_collect
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/es_discriminator/logs/collect_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/es_discriminator/logs/collect_%A_%a.err

# Discriminator training data: Diffusion-ES + multi-future Gaussian WM (expected_independent)
# on the 1500-scenario failure subset, recording every evaluated ES candidate per replan
# (--save_es_candidates), then labelling and caching them for es_discriminator training.
# Same ES settings as sbellad's collection (population 64, elite 8, 4 iterations,
# resample_timesteps 5, replan every 10 steps, log replay), but with OUR world model and
# the binary offroad reward. The targets are split into NUM_CHUNKS chunks by tfrecord
# (one array task each), so batches of 5 scenes are the same as in a single run.
# Each task, for its chunk:
#   collect -> labels (.labels.npz) -> relabel (.labels_v2.npz) + scenes (analysis/scenes.npz)
#   -> Waymax labels (.labels_waymax.npz) -> WM + road-edge caches
# Training on all chunks: slurm_es_disc_train.sh (submit both with slurm_es_disc_pipeline.sh).
# Resubmitting resumes: finished scenarios, labelled batches and cached batches are skipped.
#
#   sbatch slurm_es_disc_collect.sh                 # all chunks
#   sbatch --array=0 slurm_es_disc_collect.sh --smoke   # 20 scenes of chunk 0, separate folder
#
# Output: ${RUN_DIR}/chunk_<k>/{es_candidates,wm_predictions,road_edges,analysis}/
set -euo pipefail

WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt}"
OFFROAD_MODE="${OFFROAD_MODE:-binary}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
SEED=0
SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
OUT_ROOT="/zfsauton/scratch/ishanj/es_discriminator/es_candidate_data"
SUBSET_DIR="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
TFRECORDS="${SUBSET_DIR}/tfrecords/training"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"

WM_RUN="$(basename "$(dirname "${WORLD_MODEL_CKPT}")")"
WM_TAG="${WM_RUN#multiFuture_GaussianWorldModel_}_$(basename "${WORLD_MODEL_CKPT}" .pt)"
RUN_NAME="${RUN_NAME:-fail1500_es_diffusion_wm_mf_${WM_TAG}_expected_independent_off${OFFROAD_MODE}_seed${SEED}}"
NUM_SCENARIOS=100000
VALIDATE_FILES=6
if [[ ${SMOKE} == 1 ]]; then
  RUN_NAME="smoke_${RUN_NAME}"; NUM_SCENARIOS=20; VALIDATE_FILES=1
fi
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
OUT="${RUN_DIR}/chunk_${CHUNK}"
DATA="${OUT}/es_candidates"
mkdir -p "${OUT}" /zfsauton/scratch/ishanj/es_discriminator/logs
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

for p in "${SUBSET_DIR}/target_scenarios.json" "${TFRECORDS}" "${LANE_GRAPH_DIR}" "${WORLD_MODEL_CKPT}" "${DIFFUSION_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done

set +u  # conda activate.d hooks reference unset variables
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
log() { echo "[$(date)] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-local} chunk=${CHUNK}/${NUM_CHUNKS} smoke=${SMOKE} wm=${WORLD_MODEL_CKPT} offroad_mode=${OFFROAD_MODE} commit=$(git -C "${ROOT}" rev-parse --short HEAD) out=${OUT}"
nvidia-smi || true

# Chunk k = every NUM_CHUNKS-th tfrecord (sorted) of the fail1500 targets.
TARGETS="${OUT}/target_scenarios.json"
python - "${SUBSET_DIR}/target_scenarios.json" "${TARGETS}" "${CHUNK}" "${NUM_CHUNKS}" <<'EOF'
import json, sys
src, dst, k, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
targets = json.load(open(src))
names = sorted(t for t, v in targets.items() if v)
chunk = {t: targets[t] for t in names[k::n]}
json.dump(chunk, open(dst, "w"), indent=1)
print(f"chunk {k}/{n}: {len(chunk)} tfrecords, {sum(len(v) for v in chunk.values())} scenarios")
EOF

log "collect: Diffusion-ES + MF WM, --save_es_candidates (num_scenarios=${NUM_SCENARIOS})"
python -u simulation/run_simulation.py \
  --planner es \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --resample_timesteps 5 \
  --tfrecord_dir "${SUBSET_DIR}/tfrecords" \
  --split training \
  --target_scenario_path "${TARGETS}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${RUN_DIR}" \
  --start_from "${OUT}" \
  --tag "${RUN_NAME}" \
  --num_worlds 5 \
  --num_scenarios "${NUM_SCENARIOS}" \
  --seed "${SEED}" \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations 4 \
  --replan_interval_steps 10 \
  --other_agent_forecast wm \
  --sim_agents log \
  --visualize_mode none \
  --world_model_ckpt "${WORLD_MODEL_CKPT}" \
  --world_model_device auto \
  --wm_backend multifuture \
  --wm_refresh_steps 10 \
  --num_collision_worlds 1 \
  --collision_reduce any \
  --collision_sample_seed 0 \
  --wm_planning_mode expected_independent \
  --offroad_mode "${OFFROAD_MODE}" \
  --save_es_candidates

export XLA_PYTHON_CLIENT_PREALLOCATE=false
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

log "WM + road-edge caches (same WM as the planner)"
python -u -m es_discriminator.real_es.wm_cache --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
  --checkpoint "${WORLD_MODEL_CKPT}" --wm_dir "${OUT}/wm_predictions" --edge_dir "${OUT}/road_edges" \
  --validate_files "${VALIDATE_FILES}" --summary_out "${OUT}/wm_predictions/summary_${SLURM_JOB_ID:-local}.json"

n_batch=$(ls "${DATA}"/*.labels_waymax.npz 2>/dev/null | wc -l)
n_wm=$(ls "${OUT}"/wm_predictions/*.wm.npz 2>/dev/null | wc -l)
n_edge=$(ls "${OUT}"/road_edges/*.edges.npz 2>/dev/null | wc -l)
log "done: ${n_batch} labelled batches, ${n_wm} WM / ${n_edge} edge caches, $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenarios"
[[ ${n_batch} -gt 0 && ${n_wm} -eq ${n_batch} && ${n_edge} -eq ${n_batch} ]] || { log "ERROR: incomplete chunk"; exit 1; }
