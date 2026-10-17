#!/bin/bash
#SBATCH --job-name=es_disc_cl
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
#SBATCH --output=/zfsauton/scratch/ishanj/es_discriminator/logs/closed_loop_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/es_discriminator/logs/closed_loop_%A_%a.err

# Closed-loop Diffusion-ES on the 1500-scenario failure subset with a learned discriminator as the
# ES reward (--discriminator_ckpt; the handcrafted scorer is not used). Same ES settings as the
# discriminator's training collection (slurm_es_disc_collect.sh): population 64, elite 8,
# 4 iterations, resample_timesteps 5, replan every 10 steps, log replay. The planner and the
# discriminator share WORLD_MODEL_CKPT unless DISC_WM_CKPT is set. The targets are split into
# NUM_CHUNKS chunks by tfrecord (one array task each); resubmitting resumes finished scenarios.
#
#   sbatch slurm_es_disc_closed_loop.sh
#   sbatch --export=ALL,DISC_CKPT=/path/best.pt,DISC_TAG=mine slurm_es_disc_closed_loop.sh
#
# Output: ${RUN_DIR}/chunk_<k>/  (pair chunks with es_discriminator.real_es.compare_closed_loop)
set -euo pipefail

DISC_CKPT="${DISC_CKPT:-/zfsauton/scratch/sbellad/es_discriminator/safety_progress/safety_progress_v3_20261016_155456_86103/best.pt}"
DISC_TAG="${DISC_TAG:-sbellad_v3}"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt}"
DISC_WM_CKPT="${DISC_WM_CKPT:-${WORLD_MODEL_CKPT}}"
OFFROAD_MODE="${OFFROAD_MODE:-binary}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
SEED=0
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
OUT_ROOT="/zfsauton/scratch/ishanj/es_discriminator/closed_loop"
SUBSET_DIR="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"

wm_tag() { local run; run="$(basename "$(dirname "$1")")"; echo "${run#multiFuture_GaussianWorldModel_}_$(basename "$1" .pt)"; }
WM_TAG="$(wm_tag "${WORLD_MODEL_CKPT}")"
RUN_NAME="${RUN_NAME:-fail1500_es_disc_${DISC_TAG}_discwm_$(wm_tag "${DISC_WM_CKPT}")_wm_mf_${WM_TAG}_off${OFFROAD_MODE}_seed${SEED}}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
OUT="${RUN_DIR}/chunk_${CHUNK}"
mkdir -p "${OUT}" /zfsauton/scratch/ishanj/es_discriminator/logs
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

for p in "${SUBSET_DIR}/target_scenarios.json" "${SUBSET_DIR}/tfrecords/training" "${LANE_GRAPH_DIR}" \
         "${DISC_CKPT}" "${WORLD_MODEL_CKPT}" "${DISC_WM_CKPT}" "${DIFFUSION_CKPT}"; do
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
log "host=$(hostname) job=${SLURM_JOB_ID:-local} chunk=${CHUNK}/${NUM_CHUNKS} disc=${DISC_CKPT} disc_wm=${DISC_WM_CKPT} wm=${WORLD_MODEL_CKPT} offroad_mode=${OFFROAD_MODE} commit=$(git -C "${ROOT}" rev-parse --short HEAD) out=${OUT}"
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

log "closed loop: Diffusion-ES + discriminator reward"
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
  --num_scenarios 100000 \
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
  --discriminator_ckpt "${DISC_CKPT}" \
  --discriminator_wm_ckpt "${DISC_WM_CKPT}"
log "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenarios"
