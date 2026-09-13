#!/bin/bash
#SBATCH --job-name=dft_dragdump
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Dump the geometry a handful of scenes need for the draggable safety-loss page.
# Reads log_trajectory only -- no policy, no checkpoint, no GPU -- so it belongs
# on `cpu` and must not queue behind training.
#
# Config arrives as a positional env FILE (never `sbatch --export`, which trips
# SLURM_GET_USER_ENV and requeue-holds the job).
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
TFR="${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"

echo "host=${SLURMD_NODENAME:-unknown} out=$OUT_DIR tfrecord=$TFR"

JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tools.dump_safety_scenes \
  --tfrecord "$TFR" \
  --indices "${DFT_DRAG_INDICES:-0,1,2,3,4,5}" \
  --out_dir "$OUT_DIR" \
  --radius "${DFT_DRAG_RADIUS:-70}" \
  --n_probes "${DFT_DRAG_PROBES:-48}" \
  --edge_margin "${DFT_EDGE_MARGIN:-0.3}" \
  --agent_margin "${DFT_AGENT_MARGIN:-0.75}" \
  --edge_geometry "${DFT_EDGE_GEOMETRY:-point}"
