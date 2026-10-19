#!/bin/bash
#SBATCH --job-name=es_disc
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/es_disc/slurm-%x-%A_%a.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/es_disc/slurm-%x-%A_%a.out
#
# Diffusion-ES (+ world model, + o_am repair, + es_WM's learned discriminator) on one or
# more (arm, view, shard group) configs, scored by es_WM's evaluators (log replay, or the
# closed-loop IDM of simulation/idm_closed_loop.py). Submitted by slurm/sim/submit_es_disc.sh;
# each array task's config is <env dir>/<task id>.env.
#
# A task env may name several config files in ES_SUBTASKS; they share this GPU,
# concurrently (ES_SUBTASK_MODE=par, each with its own log under ES_LOG_DIR) or one after
# another (seq, the smoke). Each config is resumable on its own (finished scenes are
# skipped), so a time limit, preemption or requeue is a resubmit, not a restart.
#
# GPU: a6000 / a5000 only (the env dies on v100s, 2080 Tis OOM). Config arrives as a
# positional env FILE or array DIR. Never `sbatch --export`.
set -euo pipefail

ES_COMMON="${SLURM_SUBMIT_DIR:-/nonexistent}/slurm/sim/es_disc_common.sh"
[[ -r "$ES_COMMON" ]] || ES_COMMON=/home/yixiz/waymax_rs_es_disc/slurm/sim/es_disc_common.sh
[[ -r "$ES_COMMON" ]] || { echo "FATAL: shared preamble $ES_COMMON missing" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ES_COMMON"
es_load_args "$@"

echo "host=${SLURMD_NODENAME:-unknown} ($(hostname)) gpu=${CUDA_VISIBLE_DEVICES:-unset} preamble=$ES_COMMON"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: unavailable"
echo "request: partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?} array_task=${SLURM_ARRAY_TASK_ID:-none} restarts=${SLURM_RESTART_COUNT:-0}"
mkdir -p "$ES_LOG_DIR"
es_monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

if [[ -z "${ES_SUBTASKS:-}" ]]; then
  es_run_one
  exit 0
fi

mode="${ES_SUBTASK_MODE:-par}"
read -r -a cfgs <<< "$ES_SUBTASKS"
echo "subtasks ($mode, ${#cfgs[@]}): ${cfgs[*]}"
rc=0
if [[ "$mode" == seq ]]; then
  for cfg in "${cfgs[@]}"; do
    t0=$(date +%s)
    echo "=== subtask $cfg ==="
    if ( set -euo pipefail; source "$cfg"; es_run_one ); then
      echo "=== ok ($(( $(date +%s) - t0 )) s): $cfg"
    else
      echo "=== FAILED rc=$? ($(( $(date +%s) - t0 )) s): $cfg"; rc=1
    fi
  done
else
  pids=()
  for cfg in "${cfgs[@]}"; do
    log="$ES_LOG_DIR/$(basename "$cfg" .env).${SLURM_JOB_ID:-local}.log"
    ( set -euo pipefail; source "$cfg"; es_run_one ) >> "$log" 2>&1 &
    pids+=("$!")
    echo "started pid $! : $cfg -> $log"
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "ok: ${cfgs[$i]}"; else echo "FAILED rc=$?: ${cfgs[$i]} (see its log)"; rc=1; fi
  done
fi
exit "$rc"
