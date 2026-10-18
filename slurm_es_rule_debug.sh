#!/bin/bash
#SBATCH --job-name=es_rule_dbg
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu12,gpu13
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/es_rule_dbg_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/es_rule_dbg_%j.err

# Rule-based Diffusion-ES (slurm_es_wm_temperature_sweep.sh, one temperature) on the debug
# partition: 1 h limit, so the fail1500 targets are split into NUM_CHUNKS chunks by tfrecord and
# each chunk runs as a chain of 1 h jobs that resume (--start_from) where the previous one stopped.
# gpu12/13 (RTX 2080 Ti, 11 GB) are excluded. Chain CHAIN_LEN jobs per chunk:
#
#   for k in 0 1; do prev=""; for i in $(seq 1 10); do
#     prev=$(sbatch --parsable ${prev:+--dependency=afterany:${prev}} --export=ALL,CHUNK=${k} slurm_es_rule_debug.sh)
#   done; done
#
# Output: ${EXP_ROOT}/${RUN_NAME}/chunk_<k>/
set -euo pipefail

T="${TEMPERATURE:-1}"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt}"
SIM_AGENTS="${SIM_AGENTS:-log}"
OFFROAD_MODE="${OFFROAD_MODE:-margin_tiebreak}"
NUM_ES_ITERATIONS="${NUM_ES_ITERATIONS:-4}"
REPLAN_INTERVAL_STEPS="${REPLAN_INTERVAL_STEPS:-10}"
NUM_CHUNKS="${NUM_CHUNKS:-2}"
CHUNK="${CHUNK:-0}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
EXP_ROOT="/zfsauton/scratch/ishanj/world_model_sim/wm_temperature_sweep"
SUBSET_DIR="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
TFRECORD_DIR="${SUBSET_DIR}/tfrecords"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"

WM_RUN="$(basename "$(dirname "${WORLD_MODEL_CKPT}")")"
WM_TAG="${WM_RUN#multiFuture_GaussianWorldModel_}_$(basename "${WORLD_MODEL_CKPT}" .pt)"
RUN_NAME="es_diffusion_wm_mf_${WM_TAG}_expected_independent_${SIM_AGENTS}_off${OFFROAD_MODE}_it${NUM_ES_ITERATIONS}_rp${REPLAN_INTERVAL_STEPS}_T${T}_fail1500${RUN_SUFFIX:-_waymaxz}"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"
OUT="${RUN_DIR}/chunk_${CHUNK}"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

for p in "${SUBSET_DIR}/target_scenarios.json" "${TFRECORD_DIR}/training" "${LANE_GRAPH_DIR}" "${WORLD_MODEL_CKPT}" "${DIFFUSION_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} chunk=${CHUNK}/${NUM_CHUNKS} temperature=${T} offroad_mode=${OFFROAD_MODE} commit=$(git -C "${ROOT}" rev-parse --short HEAD) out=${OUT}"
nvidia-smi -L || true

# Chunk k = every NUM_CHUNKS-th tfrecord (sorted) of the fail1500 targets.
TARGETS="${OUT}/target_scenarios.json"
rc=0
python - "${SUBSET_DIR}/target_scenarios.json" "${TARGETS}" "${CHUNK}" "${NUM_CHUNKS}" "${OUT}" <<'EOF' || rc=$?
import glob, json, sys
src, dst, k, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
targets = json.load(open(src))
names = sorted(t for t, v in targets.items() if v)
chunk = {t: targets[t] for t in names[k::n]}
json.dump(chunk, open(dst, "w"), indent=1)
total, done = sum(len(v) for v in chunk.values()), len(glob.glob(f"{out}/*.scenario_*[0-9].json"))
print(f"chunk {k}/{n}: {len(chunk)} tfrecords, {total} scenarios, {done} done")
sys.exit(3 if done >= total else 0)
EOF
if [[ ${rc} -eq 3 ]]; then echo "chunk complete, nothing to do"; exit 0; fi

python -u simulation/run_simulation.py \
  --planner es \
  --diffusion_ckpt_path "${DIFFUSION_CKPT}" \
  --resample_timesteps 5 \
  --tfrecord_dir "${TFRECORD_DIR}" \
  --lane_graph_dir "${LANE_GRAPH_DIR}" \
  --exp_dir "${RUN_DIR}" \
  --start_from "${OUT}" \
  --split training \
  --num_worlds 5 \
  --num_scenarios 100000 \
  --population_size 64 \
  --elite_size 8 \
  --num_es_iterations "${NUM_ES_ITERATIONS}" \
  --replan_interval_steps "${REPLAN_INTERVAL_STEPS}" \
  --other_agent_forecast wm \
  --target_scenario_path "${TARGETS}" \
  --visualize_mode none \
  --tag "${RUN_NAME}" \
  --sim_agents "${SIM_AGENTS}" \
  --world_model_ckpt "${WORLD_MODEL_CKPT}" \
  --world_model_device auto \
  --wm_backend multifuture \
  --wm_refresh_steps 10 \
  --num_collision_worlds 1 \
  --collision_reduce any \
  --collision_sample_seed 0 \
  --wm_planning_mode expected_independent \
  --offroad_mode "${OFFROAD_MODE}" \
  --wm_mode_temperature "${T}"
echo "[$(date)] done: $(ls "${OUT}"/*.scenario_*[0-9].json 2>/dev/null | wc -l) scenarios"
