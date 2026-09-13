#!/bin/bash
#SBATCH --job-name=dft_select
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Stage B: read the scan rows, gate on expert-clean, keep the base policy's
# failures, write the pool + a summary table. Reads saved artifacts only -- no
# model, no GPU -- so the summary re-renders without re-scanning.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

DFT_SCAN_OUT="${DFT_SCAN_OUT:?set DFT_SCAN_OUT}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:?set DFT_SHARD_GLOB}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_MIN_POOL="${DFT_MIN_POOL:-64}"
DFT_INCLUDE_MARGINAL="${DFT_INCLUDE_MARGINAL:-0}"

MARGINAL_ARGS=(--no-include_marginal)
[[ "$DFT_INCLUDE_MARGINAL" != "0" ]] && MARGINAL_ARGS=(--include_marginal)

echo "host=${SLURMD_NODENAME:-unknown} scan_out=$DFT_SCAN_OUT"

# Gate, not a crash: if the scan produced too thin a pool the chain should say so
# and stop cleanly. A non-zero exit here would strand any dependent job in
# DependencyNeverSatisfied with no explanation, so report and exit 0.
if ! "$PYTHON" -m rl.diffusion_ft.tools.select_hard_womd \
      --scan_dir "$DFT_SCAN_OUT" --shard_glob "$DFT_SHARD_GLOB" \
      --index_dir "$DFT_INDEX_DIR" --min_pool "$DFT_MIN_POOL" \
      "${MARGINAL_ARGS[@]}"; then
  echo "GATE FAILED: no usable pool from $DFT_SCAN_OUT (see the message above)."
  echo "Scan more scenes (DFT_SCAN_SCENES) or set DFT_INCLUDE_MARGINAL=1, then rerun."
  exit 0
fi

echo
echo "Pool ready. Train on it with:"
echo "  DFT_SCENE_IDS_JSON=$DFT_SCAN_OUT/hard_womd.json \\"
echo "    bash rl/diffusion_ft/slurm/submit_dppo_entryclean_womd.sh"
