#!/bin/bash
#SBATCH --job-name=dft_fixpairs
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Base vs repair-arm side-by-side PNGs on fixed scenes
# (rl/diffusion_ft/tools/plot_fix_pairs.py). CPU only: geometry from an existing
# scene bundle, paths from the select run's feat npz. Config arrives as a
# positional env FILE. Never sbatch --export.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
echo "host=$(hostname)  job=${SLURM_JOB_ID:-}  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=8G  (CPU job, no GPU)"
PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
SCENES="${DFT_FIX_SCENES_JSON:?set DFT_FIX_SCENES_JSON}"
SEL_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR}"
OUT_DIR="${DFT_FIX_OUT_DIR:?set DFT_FIX_OUT_DIR}"

if [[ ! -s "$SCENES" ]] || ! compgen -G "$SEL_DIR/tasks/*/feat/*.npz" > /dev/null; then
  echo "FIX PAIRS NOT RUN: missing bundle ($SCENES) or feat npz under $SEL_DIR"; exit 0
fi
"$PY" -u -m rl.diffusion_ft.tools.plot_fix_pairs --scenes_json "$SCENES" \
  --select_dir "$SEL_DIR" --arm "${DFT_FIX_ARM:-o_lpA3}" --ids "${DFT_FIX_IDS:-}" \
  --rep "${DFT_FIX_REP:-0}" --mode "${DFT_FIX_MODE:-fixed}" --out_dir "$OUT_DIR"
echo "pngs -> $OUT_DIR"
