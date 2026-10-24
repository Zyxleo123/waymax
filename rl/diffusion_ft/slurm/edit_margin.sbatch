#!/bin/bash
#SBATCH --job-name=dft_margin
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# What margin each repair arm actually left at its switches, and how big the
# optimized edits were (rl/diffusion_ft/tools/edit_margin.py). Pure re-read of a
# select run's feat/*.npz: no GPU, no re-roll, free to re-run.
#
# Config arrives as a positional env FILE. Never `sbatch --export`.
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
SEL_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR}"
SPLITS="${DFT_EM_SPLITS:-failure_heldout,intermittent_heldout}"
echo "host=$(hostname) request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?} (CPU)"
echo "select_dir=$SEL_DIR splits=$SPLITS"

# Gate inside the job: an upstream select that wrote nothing (e.g. it gated out)
# is explained here and exits 0 rather than stranding anything.
if ! compgen -G "$SEL_DIR/tasks/*/feat/*.npz" > /dev/null; then
  echo "EDIT-MARGIN NOT RUN: no feat/*.npz under $SEL_DIR (did the select job gate out?)"
  exit 0
fi
IFS=',' read -r -a SPLIT_ARR <<< "$SPLITS"
for sp in "${SPLIT_ARR[@]}"; do
  echo
  "$PY" -u -m rl.diffusion_ft.tools.edit_margin --select_dir "$SEL_DIR" --split "$sp" \
    --json "$SEL_DIR/edit_margin_${sp}.json"
done
