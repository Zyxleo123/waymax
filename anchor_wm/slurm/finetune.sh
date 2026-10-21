#!/bin/bash
#SBATCH --job-name=awm_ft
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/finetune_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/finetune_%j.err
#
# Scorer fine-tuning on the v3 ES bank (anchor_wm/finetune.py), then the offline check against v3's
# saved test predictions on the same groups (anchor_wm/eval_bank_cached.py, "G1").
#
#   sbatch --exclude=gpu27 anchor_wm/slurm/finetune.sh
#   RUN_NAME=ft_joint EXTRA_ARGS="--unfreeze_decoder" sbatch --exclude=gpu27 anchor_wm/slurm/finetune.sh
#   RUN_NAME=ft_nogoal EXTRA_ARGS="--no_goal_features --no_route_features" sbatch ... (ablations)
#   SMOKE=1 sbatch --time=00:30:00 --exclude=gpu27 anchor_wm/slurm/finetune.sh
#   RESUME=.../latest.pt ...      continue a run (same EXTRA_ARGS / MAX_STEPS); SKIP_G1=1 skips the G1 check
set -euo pipefail
source /zfsauton/scratch/ishanj/waymax_rs/anchor_wm/slurm/common.sh

SET="${SET:-fail1500_plus_womd_train_s02_11}"
V3_SET="/zfsauton/scratch/sbellad/es_discriminator/training_sets/${SET}"
REF_DIR="${REF_DIR:-/zfsauton/scratch/sbellad/es_discriminator/safety_progress/safety_progress_v3_20261016_155456_86103}"
INIT="${INIT:-${OUT_ROOT}/runs/joint_20261018_011644/final.pt}"
BANK_CACHE="${BANK_CACHE:-${OUT_ROOT}/bank_cache/${SET}}"
RUN_NAME="${RUN_NAME:-ft_v1}"
SAVE="${OUT_ROOT}/finetune"
ARGS=(--max_steps "${MAX_STEPS:-30000}" --max_hours "${MAX_HOURS:-6.5}")
EVAL_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="smoke_${RUN_NAME}"; ARGS=(--max_steps 60 --val_every 30 --val_groups 40 --log_every 10); EVAL_ARGS=(--max_files 3 --n_boot 50)
fi
python -m anchor_wm.finetune --init_ckpt "${INIT}" --bank_dir "${V3_SET}/es_candidates" --cache_dir "${BANK_CACHE}" \
  --split_json "${V3_SET}/scenario_split_seed0.json" --invalid_json "${V3_SET}/invalid_at_start.json" \
  --self_cache_dir "${CACHE_DIR}" --save_dir "${SAVE}" --run_name "${RUN_NAME}" --num_workers 8 \
  --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}" --wandb_mode "${WANDB_MODE}" \
  ${RESUME:+--resume "${RESUME}"} "${ARGS[@]}" ${EXTRA_ARGS:-}
RUN_DIR="$(ls -td "${SAVE}/${RUN_NAME}"_2*/ | head -1)"
if [[ "${SKIP_G1:-0}" == "1" ]]; then echo "results in ${RUN_DIR} (G1 skipped)"; exit 0; fi
# A resumed run writes best.pt only if it beats the best carried over from the run it resumed.
CKPT="${RUN_DIR}best.pt"; [[ -f "${CKPT}" ]] || CKPT="${RUN_DIR}final.pt"
python -m anchor_wm.eval_bank_cached --ckpt "${CKPT}" --ref_dir "${REF_DIR}" --cache_dir "${BANK_CACHE}" \
  --out_dir "${RUN_DIR}g1_test" "${EVAL_ARGS[@]}"
echo "results in ${RUN_DIR}"
