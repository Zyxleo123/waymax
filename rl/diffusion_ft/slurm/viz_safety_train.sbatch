#!/bin/bash
#SBATCH --job-name=dft_safeviz
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
# Redraw the safety fine-tune figures from train_log.jsonl. Reads saved rows
# only -- no checkpoint, no corpus, no GPU -- so it is cheap to rerun by hand
# against any finished out_dir. Asking for a GPU to draw a figure would just
# queue behind real work.
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

echo "host=${SLURMD_NODENAME:-unknown} out_dir=$DFT_OUT_DIR"

# Gate INSIDE the job: a missing log means the train stage produced nothing.
# Say so and exit 0 -- a non-zero exit here would strand any dependent job in
# DependencyNeverSatisfied with no explanation.
if [[ ! -s "$DFT_OUT_DIR/train_log.jsonl" ]]; then
  echo "GATE: $DFT_OUT_DIR/train_log.jsonl missing or empty -- the train stage"
  echo "GATE: wrote no rows. Nothing to plot; check the dft_safetrain .out."
  exit 0
fi

JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tools.plot_safety_train \
  --out_dir "$DFT_OUT_DIR"
