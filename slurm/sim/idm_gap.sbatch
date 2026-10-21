#!/bin/bash
#SBATCH --job-name=idm_gap
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/es_idmfix/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/es_idmfix/slurm-%x-%j.out
# Replay saved ES ego paths under several NPC models (scripts/diagnose_idm_gap.py); one JSONL per
# (view, arm) under $IDM_GAP_OUT. Resumable: a requeued job skips the rows already written.
#
#   sbatch slurm/sim/idm_gap.sbatch <run root> view:arm:npc1,npc2 [view:arm:npcs ...]   (expert::npcs = the human log; delay:D:npcs = it D steps late)
#   e.g. ... $SCR/runs/nm_val10_20261018_034021_870941 idm:gt:log,idm,idm_faithful log:gt:idm,idm_faithful
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
nvidia-smi -L || true
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB gres=${SLURM_JOB_GPUS:-?}"
ROOT=/home/yixiz/waymax_rs_es_idmfix
PY=/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python
OUT="${IDM_GAP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/es_posthoc/idm_gap}"
RUN="${1:?run root}"; shift
cd "$ROOT"
echo "repo=$ROOT @ $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
mkdir -p "$OUT"
for spec in "$@"; do
  IFS=: read -r view arm npcs <<< "$spec"
  echo "=== $view/$arm npc=$npcs $(date)"
  case "$view" in
    expert) ego=expert; out="$OUT/expert.jsonl" ;;
    delay) ego="delay:$arm"; out="$OUT/delay.jsonl" ;;   # delay:D:npcs = the human's path D steps late
    *) ego="$RUN/$view/$arm"; out="$OUT/${view}_${arm}.jsonl" ;;
  esac
  set +e
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 "$PY" scripts/diagnose_idm_gap.py --ego "$ego" --npc "$npcs" \
    --shards 0-9 --batch 16 --out "$out" 2>&1 \
    | grep --line-buffered -v '^WARNING\|^I0000\|^E10\|oneDNN\|cuda_\|TF-TRT\|computation placer\|absl::'
  rc=${PIPESTATUS[0]}
  set -e
  (( rc == 0 )) || { echo "FATAL: diagnose_idm_gap.py exited $rc on $spec"; exit "$rc"; }
done
echo "=== done $(date)"
