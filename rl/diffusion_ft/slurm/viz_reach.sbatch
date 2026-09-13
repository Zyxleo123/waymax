#!/bin/bash
#SBATCH --job-name=dft_reachviz
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
# Render the post-hoc reach eval (reach_summary.png + reach_paths.png) from the
# dump reach_eval writes. Reads saved artifacts only -- no checkpoint, no corpus,
# no GPU -- so it re-renders cheaply against any finished dump dir.
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
DFT_DUMP_DIR="${DFT_DUMP_DIR:-${DFT_OUT_DIR}/eval}"
DFT_VIZ_SCENES="${DFT_VIZ_SCENES:-16}"

echo "host=${SLURMD_NODENAME:-unknown} dump=$DFT_DUMP_DIR"

# Gate INSIDE the job: a missing npz means the GPU eval stage produced nothing.
# Say so and exit 0 -- a non-zero exit would strand nothing here (leaf job) but
# keeps the failure legible.
if [[ ! -s "$DFT_DUMP_DIR/reach_rollouts.npz" ]]; then
  echo "GATE: $DFT_DUMP_DIR/reach_rollouts.npz missing -- the GPU eval stage"
  echo "GATE: wrote no dump. Nothing to plot; check the dft_evalreach .out."
  exit 0
fi

JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tools.viz_reach \
  "$DFT_DUMP_DIR" --scenes "$DFT_VIZ_SCENES"
