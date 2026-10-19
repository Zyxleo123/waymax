#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs

# VLA pre-training / fine-tuning (train_vla/). Submits itself.
#   bash slurm/train/vla.sh pretrain [--subgoal] [--tag T]
#   bash slurm/train/vla.sh finetune --scene_tokenizer_ckpt P [--subgoal] [--tag T]
# Logs: <repo>/logs/{pretrain,finetune}_vla_<jobid>.{out,err}
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
  pretrain|finetune) shift ;;
  *) echo "Usage: bash slurm/train/vla.sh {pretrain|finetune} [options]" >&2; exit 1 ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  REPO="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
  mkdir -p "${REPO}/logs"
  exec sbatch --job-name="${MODE}_vla" \
    --output="${REPO}/logs/${MODE}_vla_%j.out" --error="${REPO}/logs/${MODE}_vla_%j.err" \
    "$(readlink -f "$0")" "${MODE}" "$@"
fi

SUBGOAL=false TAG="" SCENE_TOKENIZER_CKPT=""
[[ "${MODE}" == "pretrain" ]] && TAG=None
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subgoal) SUBGOAL=true; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --scene_tokenizer_ckpt) SCENE_TOKENIZER_CKPT="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
set --

set +u
# shellcheck source=/dev/null
source ~/.bashrc
conda activate waymax_rs
set -u

if [[ "${MODE}" == "pretrain" ]]; then
  args=(--tag "${TAG}")
  [[ "${SUBGOAL}" == "true" ]] && args=(--predict_subgoal "${args[@]}")
  echo "VLA pretraining subgoal=${SUBGOAL} tag=${TAG}"
  python -m train_vla.pretrain_vla "${args[@]}"
else
  args=(--add_eos --scene_tokenizer_ckpt "${SCENE_TOKENIZER_CKPT}")
  [[ "${SUBGOAL}" == "true" ]] && args+=(--predict_subgoal)
  args+=(--tag "${TAG}")
  echo "VLA finetuning subgoal=${SUBGOAL} tag=${TAG}"
  python -m train_vla.finetuning_vla "${args[@]}"
fi
