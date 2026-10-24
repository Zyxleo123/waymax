#!/bin/bash
#SBATCH --job-name=lanegraph
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/lane_graphs/slurm-%x-%A_%a.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/lane_graphs/slurm-%x-%A_%a.out
#
# Rebuild the ES runner's lane-graph zips on Babel (scripts/rebuild_lane_graphs.py; CPU only).
#   sbatch scripts/rebuild_lane_graphs.sbatch MODE=targets
#   sbatch --array=0-39 scripts/rebuild_lane_graphs.sbatch MODE=scan NPARTS=40   (part = array task id)
#   sbatch scripts/rebuild_lane_graphs.sbatch MODE=merge      (then check)
# Config: positional VAR=value arguments. Each mode is resumable (scan skips files with a .done marker).
set -euo pipefail
for arg in "$@"; do [[ "$arg" == *=* ]] && export "${arg?}"; done
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
[[ -f "$ROOT/scripts/rebuild_lane_graphs.py" ]] || ROOT=/home/yixiz/waymax_rs
PY=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs/bin/python
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M restarts=${SLURM_RESTART_COUNT:-0} mode=${MODE:?set MODE}"
nvidia-smi -L 2>/dev/null || echo "gpu: none"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user_data/yixiz/tools/wod_protos/gen${PYTHONPATH:+:$PYTHONPATH}"
case "$MODE" in
  scan) "$PY" scripts/rebuild_lane_graphs.py scan --part "${SLURM_ARRAY_TASK_ID:?array job}" --nparts "${NPARTS:?}" ;;
  merge) "$PY" scripts/rebuild_lane_graphs.py merge && "$PY" scripts/rebuild_lane_graphs.py check ;;
  *) "$PY" scripts/rebuild_lane_graphs.py "$MODE" ;;
esac
echo "[$(date)] $MODE done"
