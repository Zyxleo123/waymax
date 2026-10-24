#!/bin/bash
#SBATCH --job-name=hrd_viz
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Hardness diagnostics. Pure artefact read -- no GPU, no simulator -- so it runs
# on the cpu partition and never queues behind training. Config arrives as a
# positional env-file argument (never --export; see the workflow notes).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
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

export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1

HRD_OUT="${HRD_OUT:-$ROOT/runs/hardness}"
HRD_DESC_DIR="${HRD_DESC_DIR:-$HRD_OUT/desc}"
HRD_CACHE_DIR="${HRD_CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/}"
HRD_VIZ_DIR="${HRD_VIZ_DIR:-$HRD_OUT/viz}"
HRD_N_PER_CELL="${HRD_N_PER_CELL:-6}"
HRD_SEED="${HRD_SEED:-0}"

if [[ ! -f "$HRD_OUT/risk.npz" || ! -f "$HRD_OUT/labels.npz" ]]; then
  echo "GATE: risk.npz/labels.npz missing under $HRD_OUT -- run submit_phase2.sh first. Nothing to do."
  exit 0
fi

echo "host=${SLURMD_NODENAME:-unknown}  desc=$HRD_DESC_DIR  out=$HRD_VIZ_DIR"
"$PYTHON" -m train_diffusion.hardness.viz \
  --risk "$HRD_OUT/risk.npz" \
  --labels "$HRD_OUT/labels.npz" \
  --desc_dir "$HRD_DESC_DIR" \
  --cache_dir "$HRD_CACHE_DIR" \
  --out_dir "$HRD_VIZ_DIR" \
  --n_per_cell "$HRD_N_PER_CELL" \
  --seed "$HRD_SEED"
