#!/bin/bash
#SBATCH --job-name=hrd_rank
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Stage 3/3: rank the heuristic families against the failure labels.
# Reads only saved artefacts (labels.npz + desc/*.npz), so the ranking and any
# figures drawn from it can be recomputed without touching the cache shards.
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

HRD_OUT="${HRD_OUT:-$ROOT/runs/hardness}"
HRD_GATE_AUC="${HRD_GATE_AUC:-0.60}"
HRD_FOLDS="${HRD_FOLDS:-5}"

if [[ ! -f "$HRD_OUT/labels.npz" ]] || ! compgen -G "$HRD_OUT/desc/*.desc.npz" > /dev/null; then
  echo "GATE: labels or descriptors missing under $HRD_OUT -- upstream stage produced nothing."
  exit 0
fi

echo "host=${SLURMD_NODENAME:-unknown} shards=$(ls "$HRD_OUT"/desc/*.desc.npz | wc -l)"

"$PYTHON" -m train_diffusion.hardness.rank \
  --labels "$HRD_OUT/labels.npz" \
  --desc_dir "$HRD_OUT/desc" \
  --out_dir "$HRD_OUT" \
  --n_folds "$HRD_FOLDS" \
  --gate_auc "$HRD_GATE_AUC"
