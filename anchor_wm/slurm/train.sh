#!/bin/bash
#SBATCH --job-name=awm_train
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-13:00:00
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/train_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/train_%j.err
#
# Step 2: train one arm, warm-started from the B0 world model's encoder.
#   ARM=separate  A1, scorer on detached WM outputs        (--joint_grad_scale 0)
#   ARM=joint     A2, scorer gradients reach the WM         (--joint_grad_scale ${JOINT_SCALE:-1})
#   ARM=wm_only   forecasting only
# PROFILE=1 instead profiles wm_only / separate / joint for 30 steps each (s/step, peak memory).
#
#   PROFILE=1 sbatch --time=0-00:45:00 anchor_wm/slurm/train.sh
#   ARM=separate sbatch anchor_wm/slurm/train.sh
#   ARM=joint sbatch anchor_wm/slurm/train.sh
#   ARM=joint MAX_HOURS=10 MAX_STEPS=90000 sbatch anchor_wm/slurm/train.sh
set -euo pipefail
source "$(dirname "$0")/common.sh" 2>/dev/null || source /zfsauton/scratch/ishanj/waymax_rs/anchor_wm/slurm/common.sh

ARM="${ARM:-joint}"
MAX_STEPS="${MAX_STEPS:-75000}"
MAX_HOURS="${MAX_HOURS:-11.5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_CANDS="${TRAIN_CANDS:-48}"
COMMON=(--cache_dir "${CACHE_DIR}" --anchors "${ANCHORS}" --init_ckpt "${INIT_CKPT}"
        --save_dir "${OUT_ROOT}/runs" --batch_size "${BATCH_SIZE}" --train_cands "${TRAIN_CANDS}"
        --num_workers "${NUM_WORKERS:-8}" --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}")

if [[ "${PROFILE:-0}" == "1" ]]; then
  for cfg in "wm_only:--no_scorer" "separate:--joint_grad_scale 0" "joint:--joint_grad_scale 1 --joint_warmup_steps 0 --joint_ramp_steps 1"; do
    echo "== profile ${cfg%%:*}"
    # shellcheck disable=SC2086
    python -m anchor_wm.train "${COMMON[@]}" --run_name "profile_${cfg%%:*}" --profile_steps 30 \
      --wandb_mode disabled ${cfg#*:}
  done
  exit 0
fi

case "${ARM}" in
  separate) ARM_ARGS=(--joint_grad_scale 0) ;;
  joint)    ARM_ARGS=(--joint_grad_scale "${JOINT_SCALE:-1}") ;;
  wm_only)  ARM_ARGS=(--no_scorer) ;;
  *) echo "ERROR: ARM must be separate|joint|wm_only, got ${ARM}" >&2; exit 1 ;;
esac
python -m anchor_wm.train "${COMMON[@]}" "${ARM_ARGS[@]}" --run_name "${RUN_NAME:-${ARM}}" \
  --max_steps "${MAX_STEPS}" --max_hours "${MAX_HOURS}" --wandb_mode "${WANDB_MODE}" ${EXTRA_ARGS:-}
