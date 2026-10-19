#!/bin/bash
#SBATCH --job-name=diffusion_only_fail1500
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/diffusion_sim/logs/diffusion_only_fail1500_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/diffusion_sim/logs/diffusion_only_fail1500_%j.err

# Plain diffusion or flow-matching planner (one sample per replan, no ES, no
# other-agent forecast) on the fail1500 subset (default) or on WOMD validation
# shards 00000-00009 (--subset val10). Submit from rhea or lov4 (sbatch is denied on lov1):
#
#   sbatch slurm/sim/run_diffusion_fail1500.sh                         # other agents replay the log
#   sbatch slurm/sim/run_diffusion_fail1500.sh --other_agents idm      # reactive IDM agents
#   sbatch slurm/sim/run_diffusion_fail1500.sh --planner fm --other_agents idm   # flow matching only
#   sbatch slurm/sim/run_diffusion_fail1500.sh --subset val10 --planner fm       # validation shards 0-9
#   sbatch slurm/sim/run_diffusion_fail1500.sh --other_agents idm --num_scenarios 10   # smoke test
#   bash   slurm/sim/run_diffusion_fail1500.sh ... --print_paths      # show run name and dirs only
set -euo pipefail

ROOT=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad
FAIL1500_DIR=${SCRATCH}/Diffusion_ES_sim/es_failure_subset_n1500_seed0
VAL10_TFRECORD_DIR=${SCRATCH}/womd_val_shards00_09
DIFFUSION_CKPT=${SCRATCH}/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500
FM_CKPT=/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_flow_matching/pretrain_fm_policy_20260828_011255/epoch_0500
LANE_GRAPH_DIR=/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs
PLANNER=${PLANNER:-diffusion}
SUBSET=fail1500
PRINT_PATHS=0
SEED=${SEED:-0}
NUM_WORLDS=${NUM_WORLDS:-5}
NUM_SCENARIOS=${NUM_SCENARIOS:-100000}
OTHER_AGENTS=${OTHER_AGENTS:-log}
EXP_ROOT_OVERRIDE=""
MAX_RETRIES=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --planner) PLANNER="$2"; shift 2 ;;
    --subset) SUBSET="$2"; shift 2 ;;
    --print_paths) PRINT_PATHS=1; shift ;;
    --other_agents) OTHER_AGENTS="$2"; shift 2 ;;
    --exp_root) EXP_ROOT_OVERRIDE="$2"; shift 2 ;;
    --num_scenarios) NUM_SCENARIOS="$2"; shift 2 ;;
    --num_worlds) NUM_WORLDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
set --
case "${OTHER_AGENTS}" in
  log|idm) ;;
  *) echo "ERROR: --other_agents must be log|idm (got '${OTHER_AGENTS}')" >&2; exit 1 ;;
esac
case "${PLANNER}" in
  diffusion) OUT_ROOT=${SCRATCH}/diffusion_sim; CKPT=${DIFFUSION_CKPT}
             PLANNER_ARGS=(--diffusion_ckpt_path "${CKPT}") ;;
  fm) OUT_ROOT=${SCRATCH}/Flow_matching_sim; CKPT=${FM_CKPT}
      PLANNER_ARGS=(--fm_ckpt_path "${CKPT}" --population_size 1 --fm_num_steps 50 --fm_solver euler) ;;
  *) echo "ERROR: --planner must be diffusion|fm (got '${PLANNER}')" >&2; exit 1 ;;
esac

case "${SUBSET}" in
  fail1500) SPLIT=training; TFRECORD_DIR=${FAIL1500_DIR}/tfrecords
            TARGET_ARGS=(--target_scenario_path "${FAIL1500_DIR}/target_scenarios.json")
            LOG_DIR=${OUT_ROOT}/logs ;;
  val10) SPLIT=validation; TFRECORD_DIR=${VAL10_TFRECORD_DIR}; TARGET_ARGS=()
         LOG_DIR=${OUT_ROOT}/logs_val10 ;;
  *) echo "ERROR: --subset must be fail1500|val10 (got '${SUBSET}')" >&2; exit 1 ;;
esac

IDM_TAG=""
[[ "${OTHER_AGENTS}" == idm ]] && IDM_TAG=_IDM && LOG_DIR=${LOG_DIR}_IDM
RUN_NAME=${PLANNER}_only${IDM_TAG}_${SUBSET}_seed${SEED}
[[ "${NUM_SCENARIOS}" != 100000 ]] && RUN_NAME=${RUN_NAME}_n${NUM_SCENARIOS}
if [[ -n "${EXP_ROOT_OVERRIDE}" ]]; then
  OUT_ROOT="${EXP_ROOT_OVERRIDE}"
  LOG_DIR="${OUT_ROOT}/logs"
  [[ "${OTHER_AGENTS}" == idm ]] && LOG_DIR="${OUT_ROOT}/logs_IDM"
fi
EXP_DIR=${OUT_ROOT}/${RUN_NAME}
LOG_FILE=${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log

if (( PRINT_PATHS )); then
  echo "RUN_NAME=${RUN_NAME}"
  echo "EXP_DIR=${EXP_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
  exit 0
fi

set +u
source ${SCRATCH}/miniconda3/bin/activate ${SCRATCH}/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export PYTHONPATH=${ROOT}
mkdir -p "${EXP_DIR}" "${LOG_DIR}"

log() { echo "$*" | tee -a "${LOG_FILE}"; }

for p in "${CKPT}" "${LANE_GRAPH_DIR}" "${TFRECORD_DIR}/${SPLIT}"; do
  [[ -e "${p}" ]] || { log "ERROR: missing ${p}"; exit 1; }
done
[[ ${#TARGET_ARGS[@]} -eq 0 || -e "${TARGET_ARGS[1]}" ]] || { log "ERROR: missing ${TARGET_ARGS[1]}"; exit 1; }
n_unreadable=0
for shard in "${TFRECORD_DIR}/${SPLIT}"/*; do
  head -c 16 "${shard}" > /dev/null 2>&1 || { log "ERROR: unreadable ${shard}"; n_unreadable=$((n_unreadable + 1)); }
done
(( n_unreadable == 0 )) || { log "ERROR: ${n_unreadable} unreadable shards; aborting"; exit 2; }

{
  echo "============================================================"
  echo " ${PLANNER}-only (no ES) on ${SUBSET}   RUN_NAME=${RUN_NAME}"
  echo " host=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is)"
  echo " exp_dir=${EXP_DIR}"
  echo " ckpt=${CKPT}"
  echo " num_worlds=${NUM_WORLDS} num_scenarios=${NUM_SCENARIOS} seed=${SEED} other_agents=${OTHER_AGENTS}"
  echo "============================================================"
  nvidia-smi || true
} | tee -a "${LOG_FILE}"

CMD=(python ${ROOT}/simulation/run_simulation.py
     --planner "${PLANNER}"
     "${PLANNER_ARGS[@]}"
     --tfrecord_dir "${TFRECORD_DIR}"
     "${TARGET_ARGS[@]}"
     --lane_graph_dir "${LANE_GRAPH_DIR}"
     --exp_dir "${EXP_DIR}"
     --split "${SPLIT}"
     --num_worlds "${NUM_WORLDS}"
     --num_scenarios "${NUM_SCENARIOS}"
     --seed "${SEED}"
     --replan_interval_steps 10
     --other_agents "${OTHER_AGENTS}"
     --visualize_mode none
     --tag "${RUN_NAME}")
{ printf 'CMD:'; printf ' %q' "${CMD[@]}"; echo; } | tee -a "${LOG_FILE}"

# On a crash, resume into the same simulation_* dir; finished scenes are skipped.
resume_args=()
attempt=0
while true; do
  set +e
  "${CMD[@]}" "${resume_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
  rc=${PIPESTATUS[0]}
  set -e
  [[ ${rc} -eq 0 ]] && break
  attempt=$((attempt + 1))
  resume_dir="$(ls -1dt "${EXP_DIR}"/simulation_${PLANNER}_${SPLIT}_${RUN_NAME}_* 2>/dev/null | head -1 || true)"
  if (( attempt > MAX_RETRIES )) || [[ -z "${resume_dir}" ]]; then
    log "ERROR: python exited ${rc}"
    exit "${rc}"
  fi
  log "WARN: python exited ${rc}; resuming from ${resume_dir} (retry ${attempt}/${MAX_RETRIES})"
  resume_args=(--start_from "${resume_dir}")
done

log "DONE ${RUN_NAME}"
ls -1dt "${EXP_DIR}"/simulation_* | head -3 | tee -a "${LOG_FILE}"
