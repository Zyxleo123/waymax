#!/bin/bash
# Submitter for the W&B backfill. Only calls sbatch. CPU job.
#
# Pushes an existing train_log.jsonl into W&B. Use it for any safety run started
# before train_safety.py grew --wandb (2026-09-02), including one still running:
# the run id lands in <out_dir>/wandb_run_id.txt, and a later resumed trainer
# with --wandb continues that same run rather than opening a second chart.
#
# Usage (from repo root, on rhea):
#   DFT_OUT_DIR=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_train/20260901_234654 \
#     bash rl/diffusion_ft/slurm/submit_wandb_backfill.sh
#
# Knobs: DFT_WANDB_PROJECT DFT_WANDB_GROUP DFT_WANDB_NAME DFT_BACKFILL_DRY
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
# `$$` so two sibling backfills in the same second do not share one env file.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"; ENV_FILE="$ENV_DIR/wbfill_${STAMP}.env"
: "${DFT_OUT_DIR:?set DFT_OUT_DIR to a train_safety out_dir}"
{
  echo "DFT_OUT_DIR=$DFT_OUT_DIR"
  echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
  [[ -n "${DFT_WANDB_GROUP:-}" ]] && echo "DFT_WANDB_GROUP=${DFT_WANDB_GROUP}"
  [[ -n "${DFT_WANDB_NAME:-}" ]]  && echo "DFT_WANDB_NAME=${DFT_WANDB_NAME}"
  echo "DFT_BACKFILL_DRY=${DFT_BACKFILL_DRY:-0}"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"
JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable rl/diffusion_ft/slurm/wandb_backfill.sbatch "$ENV_FILE")
echo "backfill job: $JID"
echo "watch: tail -f $ROOT/logs/dft/slurm-dft_wbfill-${JID}.out"
