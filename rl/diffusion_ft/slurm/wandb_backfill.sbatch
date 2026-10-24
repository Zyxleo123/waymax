#!/bin/bash
#SBATCH --job-name=dft_wbfill
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Replay a safety fine-tune's train_log.jsonl into Weights & Biases. Reads saved
# rows only -- no checkpoint, no corpus, no GPU -- so it belongs on `cpu`, and it
# can be rerun by hand against any out_dir, including one whose training job is
# still running.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_WANDB_PROJECT="${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
DFT_WANDB_GROUP="${DFT_WANDB_GROUP:-}"
DFT_WANDB_NAME="${DFT_WANDB_NAME:-}"
DFT_BACKFILL_DRY="${DFT_BACKFILL_DRY:-0}"

echo "host=${SLURMD_NODENAME:-unknown} out_dir=$DFT_OUT_DIR project=$DFT_WANDB_PROJECT"

ARGS=(--out_dir "$DFT_OUT_DIR" --project "$DFT_WANDB_PROJECT" --dry_run "$DFT_BACKFILL_DRY")
[[ -n "$DFT_WANDB_GROUP" ]] && ARGS+=(--group "$DFT_WANDB_GROUP")
[[ -n "$DFT_WANDB_NAME" ]]  && ARGS+=(--name "$DFT_WANDB_NAME")

"$PY" -m rl.diffusion_ft.tools.wandb_backfill "${ARGS[@]}"
