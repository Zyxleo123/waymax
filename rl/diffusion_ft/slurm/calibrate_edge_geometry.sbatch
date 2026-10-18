#!/bin/bash
#SBATCH --job-name=dft_edgecal
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# The edge margin for edge_geometry="segment", by the rule that set 0.3 under
# "point": the human's p01 road-edge clearance. Runs tests/calibrate_expert.py
# twice over the SAME scenes (the point run is the control and must reproduce the
# 0.314 on record), then tools/decide_edge_margin.py, which writes
# $DFT_EDGECAL_OUT/edge_margin.env only if its gates pass. Pure geometry: no
# checkpoint, no GPU. The decision is data, so this job exits 0 either way; a
# failed gate shows up downstream as train_safety's "TRAINING NOT RUN".
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
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu

# Defaults = the calibration that set the WOMD margins (job 40089, 2026-09-01).
TFR="${DFT_TFRECORD:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"
N="${DFT_CAL_SCENES:-64}"
START="${DFT_CAL_START:-0}"
OUT="${DFT_EDGECAL_OUT:?set DFT_EDGECAL_OUT}"
echo "host=${SLURMD_NODENAME:-unknown}  scenes $START..$((START + N - 1)) of $TFR"
mkdir -p "$OUT"
rm -f "$OUT/edge_margin.env"      # never let a previous run's answer stand in

for g in point segment; do
  echo "=== calibrate_expert edge_geometry=$g ==="
  "$PY" -m rl.diffusion_ft.tests.calibrate_expert \
    --tfrecord "$TFR" --n_scenes "$N" --start "$START" --chunk 8 \
    --edge_margin_m 0.3 --edge_tau_m 0.15 --agent_margin_m 0.75 --agent_tau_m 0.25 \
    --agent_geometry exact --edge_geometry "$g" --out_dir "$OUT/$g"
done

echo "=== decide ==="
"$PY" -m rl.diffusion_ft.tools.decide_edge_margin \
  --point "$OUT/point" --segment "$OUT/segment" --out_dir "$OUT" \
  --ref_p01 "${DFT_EDGECAL_REF_P01:-0.314}" --floor "${DFT_EDGECAL_FLOOR:-0.10}"
