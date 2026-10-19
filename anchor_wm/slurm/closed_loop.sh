#!/bin/bash
#SBATCH --job-name=awm_cl
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/closed_loop_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/closed_loop_%A_%a.err
#
# Closed-loop Diffusion-ES on fail1500 with AnchorWM, same settings as the baseline runs
# (slurm_es_disc_collect.sh = MF WM + rule reward, slurm_es_disc_closed_loop.sh = MF WM + discriminator):
# population 64, elite 8, 4 iterations, resample_timesteps 5, replan every 10 steps, log replay,
# expected_independent collision, binary offroad, 4 chunks by tfrecord. Resubmitting resumes.
#   REWARD=rule    AnchorWM forecaster + handcrafted ES reward      (the WM's contribution)
#   REWARD=scorer  AnchorWM forecaster + its scorer head as reward  (SCORER_CKPT, default CKPT)
#
#   CKPT=/zfsauton/scratch/ishanj/anchor_wm/runs/joint_<ts>/final.pt REWARD=rule sbatch --exclude=gpu27 anchor_wm/slurm/closed_loop.sh
#   CKPT=.../joint_<ts>/final.pt REWARD=scorer sbatch --exclude=gpu27 anchor_wm/slurm/closed_loop.sh
#   SMOKE=1 CKPT=... REWARD=scorer sbatch --partition=debug --qos=qos_debug --gres=gpu:a6000:1 --time=00:50:00 --array=0 anchor_wm/slurm/closed_loop.sh
#
# Output: /zfsauton/scratch/ishanj/anchor_wm/closed_loop/<RUN_NAME>/chunk_<k>/ (compare with
# es_discriminator.real_es.compare_closed_loop against the baseline chunk dirs).
set -euo pipefail

: "${CKPT:?set CKPT to an AnchorWM checkpoint (or an MF-Gaussian one with WM_BACKEND=multifuture)}"
# WM_BACKEND=multifuture runs the MF-Gaussian baseline through this same script (REWARD=rule only).
WM_BACKEND="${WM_BACKEND:-anchor}"
REWARD="${REWARD:-rule}"
SCORER_CKPT="${SCORER_CKPT:-${CKPT}}"
OFFROAD_MODE="${OFFROAD_MODE:-binary}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
SMOKE="${SMOKE:-0}"
SEED=0
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
OUT_ROOT="/zfsauton/scratch/ishanj/anchor_wm/closed_loop"
SUBSET_DIR="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT="/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"

RULE_LAMBDA="${RULE_LAMBDA:-5}"
# SCENE_SPLIT=train|val|test keeps only that split of the fail1500 scenes (B0 discriminator split),
# padded per tfrecord to full batches of 5 with other fail1500 scenes (ignore those in the analysis).
SCENE_SPLIT="${SCENE_SPLIT:-}"
BANK_DIR="/zfsauton/scratch/ishanj/es_discriminator/es_candidate_data/fail1500_es_diffusion_wm_mf_20261005_092950_epoch_0450_expected_independent_offbinary_seed0/train_set"
case "${REWARD}" in
  rule)   REWARD_ARGS=() ;;
  scorer) REWARD_ARGS=(--anchor_scorer_ckpt "${SCORER_CKPT}") ;;
  hybrid) REWARD_ARGS=(--anchor_scorer_ckpt "${SCORER_CKPT}" --anchor_scorer_rule_lambda "${RULE_LAMBDA}"
                       --anchor_scorer_rule_terms "${RULE_TERMS:-collision,offroad}") ;;
  *) echo "ERROR: REWARD must be rule|scorer|hybrid, got ${REWARD}" >&2; exit 1 ;;
esac
case "${WM_BACKEND}" in
  anchor) WM_LABEL=anchorwm ;;
  multifuture) WM_LABEL=mfwm; [[ "${REWARD}" == rule ]] || { echo "ERROR: WM_BACKEND=multifuture needs REWARD=rule" >&2; exit 1; } ;;
  *) echo "ERROR: WM_BACKEND must be anchor|multifuture, got ${WM_BACKEND}" >&2; exit 1 ;;
esac
TAG="$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}" .pt)"
RTAG="${REWARD}"
[[ "${REWARD}" == hybrid ]] && RTAG="hybrid_l${RULE_LAMBDA}"
STAG=""
[[ -n "${SCENE_SPLIT}" ]] && STAG="_${SCENE_SPLIT}"
RUN_NAME="${RUN_NAME:-fail1500${STAG}_es_${WM_LABEL}_${TAG}_reward_${RTAG}_off${OFFROAD_MODE}_seed${SEED}}"
[[ "${SMOKE}" == "1" ]] && RUN_NAME="smoke_${RUN_NAME}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
OUT="${RUN_DIR}/chunk_${CHUNK}"
mkdir -p "${OUT}" /zfsauton/scratch/ishanj/anchor_wm/logs
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

for p in "${SUBSET_DIR}/target_scenarios.json" "${SUBSET_DIR}/tfrecords/training" "${LANE_GRAPH_DIR}" \
         "${CKPT}" "${SCORER_CKPT}" "${DIFFUSION_CKPT}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done

# Same environment as the baseline closed-loop runs (JAX planner on the GPU; no JAX_PLATFORMS=cpu here).
set +u  # conda activate.d hooks reference unset variables
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
log() { echo "[$(date)] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-local} chunk=${CHUNK}/${NUM_CHUNKS} ckpt=${CKPT} reward=${REWARD} scorer=${SCORER_CKPT} offroad_mode=${OFFROAD_MODE} out=${OUT}"
nvidia-smi || true

# Chunk k = every NUM_CHUNKS-th tfrecord (sorted) of the fail1500 targets (as the baseline runs).
TARGETS="${OUT}/target_scenarios.json"
# TARGETS_FILE=<json>: run exactly these {tfrecord: [scenario indices]} instead of a chunk of fail1500
# (e.g. re-running chosen scenes' original batches of 5 for debugging).
if [[ -n "${TARGETS_FILE:-}" ]]; then
  cp "${TARGETS_FILE}" "${TARGETS}"
  log "targets from ${TARGETS_FILE}: $(python -c "import json; d=json.load(open('${TARGETS}')); print(len(d), 'tfrecords,', sum(map(len, d.values())), 'scenarios')")"
else
python - "${SUBSET_DIR}/target_scenarios.json" "${TARGETS}" "${CHUNK}" "${NUM_CHUNKS}" "${SMOKE}" "${SCENE_SPLIT}" "${BANK_DIR}" <<'EOF'
import glob, json, re, sys
src, dst, k, n, smoke = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
scene_split, bank = sys.argv[6], sys.argv[7]
targets = json.load(open(src))
if scene_split:
    import numpy as np
    assign = json.load(open(bank + "/scenario_split_seed0.json"))["assignment"]
    sid = {}
    for f in glob.glob(bank + "/es_candidates/*.npz"):
        if re.search(r"\.scen_[\d_]+\.npz$", f):
            z = np.load(f)
            for tf, i, s in zip(z["scen_tfrecord"], z["scen_index"], z["scen_id"]):
                sid[(str(tf), int(i))] = str(s)
    kept, n_split = {}, 0
    for tf, idxs in targets.items():
        want = [i for i in idxs if assign.get(sid.get((tf, int(i)), "")) == scene_split]
        if not want:
            continue
        n_split += len(want)
        pad = [i for i in idxs if i not in want][: (-len(want)) % 5]
        if (len(want) + len(pad)) % 5:
            raise SystemExit(f"cannot pad {tf} to a multiple of 5")
        kept[tf] = sorted(want + pad)
    print(f"split {scene_split}: {n_split} scenes, {sum(map(len, kept.values()))} with padding")
    targets = kept
names = sorted(t for t, v in targets.items() if v)
chunk = {t: targets[t] for t in names[k::n]}
if smoke:  # one full batch of num_worlds = 5 scenes (the runner drops incomplete batches)
    picked, left = {}, 5
    for t, v in chunk.items():
        take = v[:left]
        picked[t], left = take, left - len(take)
        if not left:
            break
    chunk = picked
json.dump(chunk, open(dst, "w"), indent=1)
print(f"chunk {k}/{n}: {len(chunk)} tfrecords, {sum(len(v) for v in chunk.values())} scenarios")
EOF
fi

log "closed loop: Diffusion-ES + ${WM_BACKEND} WM (${REWARD} reward)"
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
  --world_model_ckpt "${CKPT}" \
  --world_model_device auto \
  --wm_backend "${WM_BACKEND}" \
  --wm_refresh_steps 10 \
  --num_collision_worlds 1 \
  --collision_reduce any \
  --collision_sample_seed 0 \
  --wm_planning_mode expected_independent \
  --offroad_mode "${OFFROAD_MODE}" \
  "${REWARD_ARGS[@]}" ${EXTRA_SIM_ARGS:-}
log "done: $(ls "${OUT}"/*.scenario_*.json 2>/dev/null | wc -l) scenarios"
