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

# Diffusion policy pre-training / training (train_diffusion/). Submits itself.
#   bash slurm/train/diffusion.sh pretrain [--subgoal] [--tag T] [--resume_path P]
#   bash slurm/train/diffusion.sh train --pretrained_ckpt P [--tag T]
# Logs: <repo>/logs/{pretrain,train}_diffusion_<jobid>.{out,err}
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
  pretrain|train) shift ;;
  *) echo "Usage: bash slurm/train/diffusion.sh {pretrain|train} [options]" >&2; exit 1 ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  REPO="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
  mkdir -p "${REPO}/logs"
  exec sbatch --job-name="${MODE}_diffusion" \
    --output="${REPO}/logs/${MODE}_diffusion_%j.out" --error="${REPO}/logs/${MODE}_diffusion_%j.err" \
    "$(readlink -f "$0")" "${MODE}" "$@"
fi

SUBGOAL=false TAG="" RESUME_PATH="" PRETRAINED_CKPT=""
[[ "${MODE}" == "pretrain" ]] && TAG=None
[[ "${MODE}" == "train" ]] && TAG=without_subgoal
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subgoal) SUBGOAL=true; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --resume_path) RESUME_PATH="$2"; shift 2 ;;
    --pretrained_ckpt) PRETRAINED_CKPT="$2"; shift 2 ;;
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
  args=(--tag "${TAG}" --resume_path "${RESUME_PATH}")
  [[ "${SUBGOAL}" == "true" ]] && args=(--subgoal_conditioned "${args[@]}")
  echo "Diffusion pretraining subgoal=${SUBGOAL} tag=${TAG}"
  python -m train_diffusion.pretrain_diffusion_policy "${args[@]}"
else
  python -m train_diffusion.train_diffusion_policy --pretrained_ckpt "${PRETRAINED_CKPT}" --tag "${TAG}"
fi
