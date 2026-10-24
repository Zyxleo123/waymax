#!/bin/bash
#SBATCH --job-name=hrd_risk
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Phase 2, stage 2: fit the risk model on the expert-clean labelled scenes and
# score every scored shard. Fit and score read the SAME descriptor directory so
# the feature definitions cannot drift between them.
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
HRD_FIT_SLICE="${HRD_FIT_SLICE:-expert_clean}"

if [[ ! -f "$HRD_OUT/labels.npz" ]] || ! compgen -G "$HRD_OUT/desc_all/*.desc.npz" > /dev/null; then
  echo "GATE: need labels.npz and desc_all/*.desc.npz under $HRD_OUT -- upstream produced nothing."
  exit 0
fi

N_SHARDS="$(ls "$HRD_OUT"/desc_all/*.desc.npz | wc -l)"
echo "host=${SLURMD_NODENAME:-unknown} scoring $N_SHARDS shards fit_slice=$HRD_FIT_SLICE"

"$PYTHON" -m train_diffusion.hardness.risk_model \
  --labels "$HRD_OUT/labels.npz" \
  --fit_desc_dir "$HRD_OUT/desc_all" \
  --score_desc_dir "$HRD_OUT/desc_all" \
  --fit_slice "$HRD_FIT_SLICE" \
  --out_dir "$HRD_OUT"
