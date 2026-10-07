#!/bin/bash
#SBATCH --job-name=run_es
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/immediate_exp/logs/run_es_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/immediate_exp/logs/run_es_%j.err

# One closed-loop ES simulation job (Diffusion-ES or FM-ES) on one subset.
# Normally submitted by slurm/sim/{trainval,fail1500,probe250}.sh; see README.md.
#
#   sbatch slurm/sim/run_es.sh --subset fail1500 --planner fm --forecast wm_mf
#   bash   slurm/sim/run_es.sh ... --print_cmd     # show run name, dirs, command; run nothing
#
# Run names and output dirs are identical to the pre-reorganisation scripts
# (legacy/sim/), so old results, resume and comparison scripts keep working.
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
SCRATCH="/zfsauton/scratch/sbellad"
IMMEDIATE_ROOT="${SCRATCH}/immediate_exp"
WM_SIM_ROOT="${SCRATCH}/world_model_sim"
FM_SIM_ROOT="${SCRATCH}/Flow_matching_sim"
WOMD_TF_ROOT="/zfsauton/datasets/womd/tf_example"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
FAIL1500_SUBSET_DIR="${IMMEDIATE_ROOT}/es_failure_subset_n1500_seed0"
DIFFUSION_CKPT="${SCRATCH}/diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
FM_CKPT="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500"
DET_WM_CKPT="${SCRATCH}/world_model/ckpts/world_model_20260912_170741/epoch_0500.pt"
MF_WM_CKPT="${SCRATCH}/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt"
GAU_WM_CKPT="${SCRATCH}/world_model/ckpts_gaussian/gaussian_world_model_20260916_194054/epoch_0370.pt"
CONDA_ROOT="${SCRATCH}/miniconda3"
WAYMAX_ENV="${SCRATCH}/conda_envs/waymax_rs"

usage() {
  cat <<'EOF'
Usage: run_es.sh --subset {trainval|fail1500|probe250} --planner {diffusion|fm} --forecast F [options]

  --forecast F          cv | gt | wm_det | wm_mf | wm_gau | wm_gau_mean   (wm = wm_det)
  --mf_mode M           wm_mf scoring: expected_independent (default) | top1 | worst_case_independent
  --mf_collision C      wm_mf collision ablation: mean_uniform | mean_learned | gaussian_uniform | gaussian_learned
  --M N                 samples per mode for gaussian_* (default 8)
  --mf_seed S           gaussian sampling seed (default = --seed)
  --ablation_subdir D   e.g. _smoke_n10 (ablation runs only)
  --seed S              planner seed (default 0)
  --num_scenarios N     cap scenes (default 100000 = all)
  --num_worlds N        batch size (default 5; probe250 default 1)
  --it N                ES iterations (default 4)
  --pop N / --elite N   population / elite size (default 64 / 8)
  --rs N                FM resample ODE steps (default 20)
  --t_start T           FM resample t_start (default 0.2; written into names verbatim)
  --fm_steps N          FM full-sample ODE steps (default 50)
  --solver S            FM ODE solver (default euler)
  --resample_timesteps N  Diffusion resample timesteps (default 5)
  --wm_refresh N        WM refresh steps (default 10)
  --wm_ckpt PATH        override the world-model checkpoint
  --num_collision_worlds N  wm_gau Monte-Carlo samples (default 8)
  --tag_suffix S        probe250: appended to the run tag (Phase-2 config name)
  --run_suffix S        appended to the run name
  --probe_n N / --probe_seed S   probe250 subset (default 250 / 0)
  --print_cmd           print run name, dirs and simulation command(s); run nothing
  --print_paths         print run name and dirs only (used by the submit scripts)
EOF
}

need() { [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }; }

SUBSET="" PLANNER="" FORECAST="" MF_MODE="" MF_COLLISION="" MF_M=8 MF_SEED="" ABLATION_SUBDIR=""
SEED=0 NUM_SCENARIOS=100000 NUM_WORLDS="" NUM_ES_ITERATIONS=4 POPULATION_SIZE=64 ELITE_SIZE=8
RESAMPLE_NUM_STEPS=20 RESAMPLE_T_START=0.2 FM_NUM_STEPS=50 FM_SOLVER=euler RESAMPLE_TIMESTEPS=5
WM_REFRESH_STEPS=10 WM_CKPT_OVERRIDE="" NUM_COLLISION_WORLDS_OVERRIDE="" TAG_SUFFIX="" RUN_SUFFIX=""
PROBE_N=250 PROBE_SEED=0 PRINT_MODE=""
REPLAN_INTERVAL_STEPS=10 COLLISION_SAMPLE_SEED=0 WORLD_MODEL_DEVICE=auto MAX_RETRIES=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset) need "$@"; SUBSET="$2"; shift 2 ;;
    --planner) need "$@"; PLANNER="$2"; shift 2 ;;
    --forecast) need "$@"; FORECAST="$2"; shift 2 ;;
    --mf_mode) need "$@"; MF_MODE="$2"; shift 2 ;;
    --mf_collision) need "$@"; MF_COLLISION="$2"; shift 2 ;;
    --M) need "$@"; MF_M="$2"; shift 2 ;;
    --mf_seed) need "$@"; MF_SEED="$2"; shift 2 ;;
    --ablation_subdir) need "$@"; ABLATION_SUBDIR="$2"; shift 2 ;;
    --seed) need "$@"; SEED="$2"; shift 2 ;;
    --num_scenarios) need "$@"; NUM_SCENARIOS="$2"; shift 2 ;;
    --num_worlds) need "$@"; NUM_WORLDS="$2"; shift 2 ;;
    --it) need "$@"; NUM_ES_ITERATIONS="$2"; shift 2 ;;
    --pop) need "$@"; POPULATION_SIZE="$2"; shift 2 ;;
    --elite) need "$@"; ELITE_SIZE="$2"; shift 2 ;;
    --rs) need "$@"; RESAMPLE_NUM_STEPS="$2"; shift 2 ;;
    --t_start) need "$@"; RESAMPLE_T_START="$2"; shift 2 ;;
    --fm_steps) need "$@"; FM_NUM_STEPS="$2"; shift 2 ;;
    --solver) need "$@"; FM_SOLVER="$2"; shift 2 ;;
    --resample_timesteps) need "$@"; RESAMPLE_TIMESTEPS="$2"; shift 2 ;;
    --wm_refresh) need "$@"; WM_REFRESH_STEPS="$2"; shift 2 ;;
    --wm_ckpt) need "$@"; WM_CKPT_OVERRIDE="$2"; shift 2 ;;
    --num_collision_worlds) need "$@"; NUM_COLLISION_WORLDS_OVERRIDE="$2"; shift 2 ;;
    --tag_suffix) need "$@"; TAG_SUFFIX="$2"; shift 2 ;;
    --run_suffix) need "$@"; RUN_SUFFIX="$2"; shift 2 ;;
    --probe_n) need "$@"; PROBE_N="$2"; shift 2 ;;
    --probe_seed) need "$@"; PROBE_SEED="$2"; shift 2 ;;
    --print_cmd) PRINT_MODE=cmd; shift ;;
    --print_paths) PRINT_MODE=paths; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
  esac
done
# Conda's activate script reads "$@"; it must be empty when we source it.
set --

case "${SUBSET}" in
  trainval|fail1500|probe250) ;;
  *) echo "ERROR: --subset must be trainval|fail1500|probe250 (got '${SUBSET}')" >&2; exit 1 ;;
esac
case "${PLANNER}" in
  diffusion|diff|es) PLANNER=es ;;
  fm|flow_matching|es_fm) PLANNER=es_fm ;;
  *) echo "ERROR: --planner must be diffusion|fm (got '${PLANNER}')" >&2; exit 1 ;;
esac
[[ "${FORECAST}" == "wm" ]] && FORECAST=wm_det

# ---- forecast / world model --------------------------------------------------
WM_BACKEND=production WORLD_MODEL_CKPT="" NUM_COLLISION_WORLDS=1 COLLISION_REDUCE=mean WM_PLANNING_MODE=""
case "${FORECAST}" in
  cv|gt) OTHER_AGENT_FORECAST="${FORECAST}" ;;
  wm_det)
    OTHER_AGENT_FORECAST=wm WORLD_MODEL_CKPT="${DET_WM_CKPT}" ;;
  wm_mf)
    OTHER_AGENT_FORECAST=wm WM_BACKEND=multifuture WORLD_MODEL_CKPT="${MF_WM_CKPT}" COLLISION_REDUCE=any
    WM_PLANNING_MODE="${MF_MODE:-expected_independent}"
    case "${WM_PLANNING_MODE}" in
      top1|expected_independent|worst_case_independent|expected|worst_case) ;;
      *) echo "ERROR: unknown --mf_mode '${WM_PLANNING_MODE}'" >&2; exit 1 ;;
    esac ;;
  wm_gau)
    OTHER_AGENT_FORECAST=wm WORLD_MODEL_CKPT="${GAU_WM_CKPT}" NUM_COLLISION_WORLDS="${NUM_COLLISION_WORLDS_OVERRIDE:-8}" ;;
  wm_gau_mean)
    OTHER_AGENT_FORECAST=wm WORLD_MODEL_CKPT="${GAU_WM_CKPT}" COLLISION_REDUCE=any ;;
  *) echo "ERROR: --forecast must be cv|gt|wm_det|wm_mf|wm_gau|wm_gau_mean (got '${FORECAST}')" >&2; exit 1 ;;
esac
if [[ "${OTHER_AGENT_FORECAST}" == "wm" && -n "${WM_CKPT_OVERRIDE}" ]]; then
  WORLD_MODEL_CKPT="${WM_CKPT_OVERRIDE}"
fi
# MF-only options are ignored for other forecasts so comma lists can mix forecasts.
if [[ "${FORECAST}" != "wm_mf" ]]; then
  MF_COLLISION=""
fi

WM_EPOCH_TAG=""
if [[ -n "${WORLD_MODEL_CKPT}" ]]; then
  ckpt_base="$(basename "${WORLD_MODEL_CKPT}" .pt)"
  if [[ "${ckpt_base}" =~ ^epoch_([0-9]+)$ ]]; then
    WM_EPOCH_TAG="epoch${BASH_REMATCH[1]}"
  else
    WM_EPOCH_TAG="${ckpt_base}"
  fi
fi
case "${FORECAST}" in
  cv|gt) FORECAST_TAG="${FORECAST}" ;;
  wm_det) FORECAST_TAG="wm_det_${WM_EPOCH_TAG}" ;;
  wm_mf) FORECAST_TAG="wm_mf_${WM_EPOCH_TAG}_${WM_PLANNING_MODE}" ;;
  wm_gau) FORECAST_TAG="wm_gau_${WM_EPOCH_TAG}_k${NUM_COLLISION_WORLDS}" ;;
  wm_gau_mean) FORECAST_TAG="wm_gau_mean_${WM_EPOCH_TAG}_k1" ;;
esac
# trainval always tags the refresh; the other subsets only when it differs from 10.
if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]] && [[ "${SUBSET}" == "trainval" || "${WM_REFRESH_STEPS}" != "10" ]]; then
  FORECAST_TAG="${FORECAST_TAG}_r${WM_REFRESH_STEPS}"
fi

# Budget tag appended to trainval/fail1500 names only when it differs from the
# defaults those runs were made with (FM fail1500 names always carry it).
diff_budget_tag() {
  if [[ "${NUM_ES_ITERATIONS}" != 4 || "${RESAMPLE_TIMESTEPS}" != 5 || "${POPULATION_SIZE}" != 64 || "${ELITE_SIZE}" != 8 ]]; then
    echo "_it${NUM_ES_ITERATIONS}_rt${RESAMPLE_TIMESTEPS}_p${POPULATION_SIZE}_e${ELITE_SIZE}"
  fi
}
fm_budget_tag() {
  local tag="_it${NUM_ES_ITERATIONS}_rs${RESAMPLE_NUM_STEPS}_p${POPULATION_SIZE}_e${ELITE_SIZE}"
  [[ "${RESAMPLE_T_START}" != "0.2" ]] && tag="${tag}_t${RESAMPLE_T_START}"
  [[ "${FM_NUM_STEPS}" != 50 ]] && tag="${tag}_n${FM_NUM_STEPS}"
  [[ "${FM_SOLVER}" != euler ]] && tag="${tag}_${FM_SOLVER}"
  echo "${tag}"
}

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TARGET_JSON=""
SPLITS=(training)

# ---- per-subset naming / layout ---------------------------------------------
case "${SUBSET}" in
  trainval)
    NUM_WORLDS="${NUM_WORLDS:-5}"
    if [[ "${PLANNER}" == "es" ]]; then
      RUN_NAME="es_diffusion_${FORECAST_TAG}$(diff_budget_tag)"
    else
      RUN_NAME="es_fm_${FORECAST_TAG}"
      [[ "$(fm_budget_tag)" != "_it4_rs20_p64_e8" ]] && RUN_NAME="${RUN_NAME}$(fm_budget_tag)"
    fi
    [[ -n "${RUN_SUFFIX}" ]] && RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
    if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then EXP_ROOT="${WM_SIM_ROOT}"; else EXP_ROOT="${IMMEDIATE_ROOT}"; fi
    EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
    LOG_DIR="${EXP_ROOT}/logs"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}_train_val_shards00_01_${RUN_STAMP}.log"
    TFRECORD_DIR="${EXP_ROOT}/tfrecord_shards_00_01_${RUN_NAME}_${RUN_STAMP}"
    SPLITS=(training validation)
    ;;
  fail1500)
    NUM_WORLDS="${NUM_WORLDS:-5}"
    if [[ "${PLANNER}" == "es" ]]; then
      RUN_NAME="es_diffusion_${FORECAST_TAG}_fail1500$(diff_budget_tag)"
    else
      RUN_NAME="es_fm_${FORECAST_TAG}_fail1500$(fm_budget_tag)"
    fi
    [[ -n "${RUN_SUFFIX}" ]] && RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
    if [[ "${PLANNER}" == "es_fm" ]]; then EXP_ROOT="${FM_SIM_ROOT}"
    elif [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then EXP_ROOT="${WM_SIM_ROOT}"
    else EXP_ROOT="${IMMEDIATE_ROOT}"; fi
    EXP_DIR="${EXP_ROOT}/${RUN_NAME}"
    LOG_DIR="${EXP_ROOT}/logs"
    if [[ -n "${MF_COLLISION}" ]]; then
      case "${MF_COLLISION}" in
        mean_uniform|mean_learned) MF_VARIANT="${MF_COLLISION}" MF_M=8 ;;
        gaussian_uniform|gaussian_learned) MF_VARIANT="$(printf '%s_M%02d' "${MF_COLLISION}" "${MF_M}")" ;;
        *) echo "ERROR: unknown --mf_collision '${MF_COLLISION}'" >&2; exit 1 ;;
      esac
      MF_SEED="${MF_SEED:-${SEED}}"
      if [[ "${PLANNER}" == "es_fm" ]]; then ABL_ROOT="${FM_SIM_ROOT}/mf_multiSampling_collision_ablation"
      else ABL_ROOT="${IMMEDIATE_ROOT}/mf_multiSampling_collision_ablation"; fi
      [[ -n "${ABLATION_SUBDIR}" ]] && ABL_ROOT="${ABL_ROOT}/${ABLATION_SUBDIR}"
      RUN_NAME="${RUN_NAME/_${WM_PLANNING_MODE}_fail1500/_${MF_VARIANT}_fail1500}_seed${SEED}"
      EXP_DIR="${ABL_ROOT}/${MF_VARIANT}/seed_${SEED}"
      LOG_DIR="${EXP_DIR}/logs"
    fi
    LOG_FILE="${LOG_DIR}/${RUN_NAME}_${RUN_STAMP}.log"
    TFRECORD_DIR="${FAIL1500_SUBSET_DIR}/tfrecords"
    TARGET_JSON="${FAIL1500_SUBSET_DIR}/target_scenarios.json"
    ;;
  probe250)
    NUM_WORLDS="${NUM_WORLDS:-1}"
    PROBE_SUBSET_DIR="${FM_SIM_ROOT}/fm_hparam_probe_n${PROBE_N}_seed${PROBE_SEED}"
    if [[ "${PLANNER}" == "es" ]]; then
      RUN_NAME="diff_${FORECAST_TAG}_baseline_pop${POPULATION_SIZE}_it${NUM_ES_ITERATIONS}_rs${RESAMPLE_TIMESTEPS}"
    else
      RUN_NAME="fm_${FORECAST_TAG}_t${RESAMPLE_T_START}_pop${POPULATION_SIZE}_it${NUM_ES_ITERATIONS}_rs${RESAMPLE_NUM_STEPS}_n${FM_NUM_STEPS}_${FM_SOLVER}"
    fi
    [[ -n "${TAG_SUFFIX}" ]] && RUN_NAME="${RUN_NAME}_${TAG_SUFFIX}"
    [[ -n "${RUN_SUFFIX}" ]] && RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
    EXP_DIR="${FM_SIM_ROOT}/fm_hparam_probe_runs/${RUN_NAME}"
    LOG_DIR="${FM_SIM_ROOT}/logs"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}_${RUN_STAMP}.log"
    TFRECORD_DIR="${PROBE_SUBSET_DIR}/tfrecords"
    TARGET_JSON="${PROBE_SUBSET_DIR}/target_scenarios.json"
    ;;
esac
if [[ -n "${MF_COLLISION}" && "${SUBSET}" != "fail1500" ]]; then
  echo "ERROR: --mf_collision is only wired for --subset fail1500" >&2
  exit 1
fi

split_tag() {
  if [[ "${SUBSET}" != "trainval" ]]; then
    echo "${RUN_NAME}"
  elif [[ "$1" == "training" ]]; then
    echo "${RUN_NAME}_TRAIN_tfrecord00_01"
  else
    echo "${RUN_NAME}_VAL_tfrecord00_01"
  fi
}

build_cmd() {
  local split="$1"
  CMD=(
    python "${ROOT}/simulation/run_simulation.py"
    --planner "${PLANNER}"
    --tfrecord_dir "${TFRECORD_DIR}"
    --lane_graph_dir "${LANE_GRAPH_DIR}"
    --exp_dir "${EXP_DIR}"
    --split "${split}"
    --num_worlds "${NUM_WORLDS}"
    --num_scenarios "${NUM_SCENARIOS}"
    --seed "${SEED}"
    --population_size "${POPULATION_SIZE}"
    --elite_size "${ELITE_SIZE}"
    --num_es_iterations "${NUM_ES_ITERATIONS}"
    --replan_interval_steps "${REPLAN_INTERVAL_STEPS}"
    --other_agent_forecast "${OTHER_AGENT_FORECAST}"
    --visualize_mode none
    --tag "$(split_tag "${split}")"
  )
  [[ -n "${TARGET_JSON}" ]] && CMD+=(--target_scenario_path "${TARGET_JSON}")
  if [[ "${PLANNER}" == "es" ]]; then
    CMD+=(--diffusion_ckpt_path "${DIFFUSION_CKPT}" --resample_timesteps "${RESAMPLE_TIMESTEPS}")
  else
    CMD+=(--fm_ckpt_path "${FM_CKPT}" --resample_t_start "${RESAMPLE_T_START}"
          --resample_num_steps "${RESAMPLE_NUM_STEPS}" --fm_num_steps "${FM_NUM_STEPS}"
          --fm_solver "${FM_SOLVER}")
  fi
  if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
    CMD+=(--world_model_ckpt "${WORLD_MODEL_CKPT}" --world_model_device "${WORLD_MODEL_DEVICE}"
          --wm_backend "${WM_BACKEND}" --wm_refresh_steps "${WM_REFRESH_STEPS}"
          --num_collision_worlds "${NUM_COLLISION_WORLDS}" --collision_reduce "${COLLISION_REDUCE}"
          --collision_sample_seed "${COLLISION_SAMPLE_SEED}")
    [[ -n "${WM_PLANNING_MODE}" ]] && CMD+=(--wm_planning_mode "${WM_PLANNING_MODE}")
    if [[ -n "${MF_COLLISION}" ]]; then
      CMD+=(--mf_collision_method "${MF_COLLISION}" --mf_gaussian_num_samples "${MF_M}"
            --mf_gaussian_seed "${MF_SEED}")
    fi
  fi
}

if [[ -n "${PRINT_MODE}" ]]; then
  echo "RUN_NAME=${RUN_NAME}"
  echo "EXP_DIR=${EXP_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
  if [[ "${PRINT_MODE}" == "cmd" ]]; then
    for split in "${SPLITS[@]}"; do
      build_cmd "${split}"
      printf 'CMD[%s]:' "${split}"; printf ' %q' "${CMD[@]}"; echo
    done
  fi
  exit 0
fi

# ---- run ---------------------------------------------------------------------
cd "${ROOT}"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi
# XLA CUDA command buffers (CUDA graphs) intermittently abort long runs with
# CUDA_ERROR_STREAM_CAPTURE_INVALIDATED; run without them.
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${EXP_DIR}" "${LOG_DIR}"

log() { echo "$*" | tee -a "${LOG_FILE}"; }

if [[ "${SUBSET}" == "trainval" ]]; then
  mkdir -p "${TFRECORD_DIR}/training" "${TFRECORD_DIR}/validation"
  for shard in training/training_tfexample.tfrecord-00000-of-01000 training/training_tfexample.tfrecord-00001-of-01000 \
               validation/validation_tfexample.tfrecord-00000-of-00150 validation/validation_tfexample.tfrecord-00001-of-00150; do
    ln -sfn "${WOMD_TF_ROOT}/${shard}" "${TFRECORD_DIR}/${shard}"
  done
fi

required=("${LANE_GRAPH_DIR}" "${TFRECORD_DIR}/training")
[[ -n "${TARGET_JSON}" ]] && required+=("${TARGET_JSON}")
if [[ "${PLANNER}" == "es" ]]; then required+=("${DIFFUSION_CKPT}"); else required+=("${FM_CKPT}"); fi
[[ "${OTHER_AGENT_FORECAST}" == "wm" ]] && required+=("${WORLD_MODEL_CKPT}")
for p in "${required[@]}"; do
  if [[ ! -e "${p}" ]]; then
    log "ERROR: missing ${p}"
    [[ "${SUBSET}" == "probe250" && "${p}" == "${TARGET_JSON}" ]] && log "  build it first: bash slurm/sim/probe250.sh build"
    exit 1
  fi
done

# The loader treats unreadable shards as "done", so a node without the data
# mount would finish with 0 scenes and still print DONE. Fail fast instead.
n_unreadable=0
for shard in "${TFRECORD_DIR}"/*/*; do
  if ! head -c 16 "${shard}" > /dev/null 2>&1; then
    log "ERROR: unreadable on $(hostname): ${shard} -> $(readlink "${shard}")"
    n_unreadable=$((n_unreadable + 1))
  fi
done
if (( n_unreadable > 0 )); then
  log "ERROR: ${n_unreadable} tfrecord shards unreadable on $(hostname); aborting"
  exit 2
fi

{
  echo "============================================================"
  echo " ES simulation  subset=${SUBSET}  planner=${PLANNER}  forecast=${FORECAST}"
  echo " RUN_NAME=${RUN_NAME}"
  echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}  date=$(date -Is)"
  echo " exp_dir=${EXP_DIR}"
  echo " log_file=${LOG_FILE}"
  echo " tfrecord_dir=${TFRECORD_DIR}"
  [[ -n "${TARGET_JSON}" ]] && echo " target_json=${TARGET_JSON}"
  echo " es: pop=${POPULATION_SIZE} elite=${ELITE_SIZE} iters=${NUM_ES_ITERATIONS} num_worlds=${NUM_WORLDS} seed=${SEED} num_scenarios=${NUM_SCENARIOS}"
  if [[ "${PLANNER}" == "es_fm" ]]; then
    echo " fm: t_start=${RESAMPLE_T_START} resample_num_steps=${RESAMPLE_NUM_STEPS} fm_num_steps=${FM_NUM_STEPS} solver=${FM_SOLVER}"
  else
    echo " diffusion: resample_timesteps=${RESAMPLE_TIMESTEPS}"
  fi
  if [[ "${OTHER_AGENT_FORECAST}" == "wm" ]]; then
    echo " wm: backend=${WM_BACKEND} ckpt=${WORLD_MODEL_CKPT} refresh=${WM_REFRESH_STEPS} collision_worlds=${NUM_COLLISION_WORLDS} reduce=${COLLISION_REDUCE}"
    [[ -n "${WM_PLANNING_MODE}" ]] && echo " wm_planning_mode=${WM_PLANNING_MODE}"
    [[ -n "${MF_COLLISION}" ]] && echo " mf_collision=${MF_COLLISION} M=${MF_M} mf_seed=${MF_SEED}"
  fi
  echo "============================================================"
  nvidia-smi || true
} | tee -a "${LOG_FILE}"

# On a crash, resume into the same simulation_* dir; finished scenes are skipped.
for split in "${SPLITS[@]}"; do
  build_cmd "${split}"
  tag="$(split_tag "${split}")"
  log "# START split=${split} tag=${tag} time=$(date -Is)"
  resume_args=()
  attempt=0
  while true; do
    set +e
    "${CMD[@]}" "${resume_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
    py_rc=${PIPESTATUS[0]}
    set -e
    [[ "${py_rc}" -eq 0 ]] && break
    attempt=$((attempt + 1))
    resume_dir="$(ls -1dt "${EXP_DIR}/simulation_${PLANNER}_${split}_${tag}_"* 2>/dev/null | head -1)"
    if (( attempt > MAX_RETRIES )) || [[ -z "${resume_dir}" ]]; then
      log "ERROR: python exited ${py_rc} (split=${split})"
      exit "${py_rc}"
    fi
    log "WARN: python exited ${py_rc}; resuming from ${resume_dir} (retry ${attempt}/${MAX_RETRIES})"
    resume_args=(--start_from "${resume_dir}")
  done
  log "# DONE split=${split} time=$(date -Is)"
done

log "DONE ${RUN_NAME}"
ls -1dt "${EXP_DIR}"/simulation_* 2>/dev/null | head -5 | tee -a "${LOG_FILE}"
