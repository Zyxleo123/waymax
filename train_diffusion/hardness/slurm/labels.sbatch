#!/bin/bash
#SBATCH --job-name=hrd_labels
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Stage 1/3: join the closed-loop eval JSONs into one label table.
# CPU-only (no --gres=gpu): this is pure file I/O over ~84k small JSONs, so it
# packs onto any node with free cores instead of queueing behind GPU work.
#
# Config via a positional env-file path (never `sbatch --export`).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/hardness"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1

HRD_POLICY_DIR="${HRD_POLICY_DIR:-/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_diffusion_training_20260816_141401}"
HRD_EXPERT_DIR="${HRD_EXPERT_DIR:-/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_log-replay_training_20260716_025213}"
HRD_OUT="${HRD_OUT:-$ROOT/runs/hardness}"

echo "host=${SLURMD_NODENAME:-unknown} policy=$HRD_POLICY_DIR expert=$HRD_EXPERT_DIR out=$HRD_OUT"

"$PYTHON" -m train_diffusion.hardness.labels \
  --policy_dir "$HRD_POLICY_DIR" \
  --expert_dir "$HRD_EXPERT_DIR" \
  --out_dir "$HRD_OUT"
