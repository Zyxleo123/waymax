#!/bin/bash
#SBATCH --job-name=dft_rebuildsplits
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Stage B of the split rebuild: read the verification rows and write a new
# eval_splits JSON keeping only scenes the base VERIFIABLY fails (and, for the
# do-no-harm set, verifiably solves). CPU only -- reads saved rows, so it is
# re-runnable at any threshold without a GPU.
#
# The gate is INSIDE the job: a missing rows file prints why and exits 0 rather
# than stranding every dependent job in DependencyNeverSatisfied.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
DFT_VERIFY_OUT="${DFT_VERIFY_OUT:-${DFT_OUT_DIR}/verify_rows.jsonl}"
DFT_VERIFIED_JSON="${DFT_VERIFIED_JSON:-${DFT_OUT_DIR}/eval_splits_verified.json}"
DFT_HELDOUT_FAILURE="${DFT_HELDOUT_FAILURE:--1}"   # -1 = half the verified failures
DFT_HELDOUT_SUCCESS="${DFT_HELDOUT_SUCCESS:-1000}"
DFT_SPLIT_SEED="${DFT_SPLIT_SEED:-0}"

echo "host=${SLURMD_NODENAME:-unknown} rows=$DFT_VERIFY_OUT"

if [[ ! -s "$DFT_VERIFY_OUT" ]]; then
  echo "GATE: no verification rows at $DFT_VERIFY_OUT -- the verify stage"
  echo "GATE: produced nothing. Nothing to rebuild; read the dft_verify .out."
  exit 0
fi

JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tools.rebuild_verified_splits \
  --rows "$DFT_VERIFY_OUT" --in_json "$DFT_SPLITS_JSON" \
  --out_json "$DFT_VERIFIED_JSON" \
  --heldout_failure="$DFT_HELDOUT_FAILURE" \
  --heldout_success "$DFT_HELDOUT_SUCCESS" --seed "$DFT_SPLIT_SEED"
echo "verified splits -> $DFT_VERIFIED_JSON"
