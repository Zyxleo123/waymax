#!/bin/bash
#SBATCH --job-name=dft_dragviz
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Render the draggable page from the dumped JSON. Pure text transform -- rerun it
# alone whenever the page changes, without re-reading the corpus.
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
OUT_DIR="${DFT_DRAG_DIR:?set DFT_DRAG_DIR}"

# Gate INSIDE the job: a missing dump means the upstream stage did no work. Say
# so and exit 0 rather than stranding anything behind DependencyNeverSatisfied.
if [[ ! -s "$OUT_DIR/safety_scenes.json" ]]; then
  echo "GATE: $OUT_DIR/safety_scenes.json missing -- the dump stage wrote nothing."
  exit 0
fi

JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tools.viz_safety_drag \
  --scenes "$OUT_DIR/safety_scenes.json" --out_dir "$OUT_DIR" \
  --title "${DFT_DRAG_TITLE:-What the safety loss charges, pose by pose}"
