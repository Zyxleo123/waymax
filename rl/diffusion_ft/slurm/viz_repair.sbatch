#!/bin/bash
#SBATCH --job-name=dft_ride
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Ride quality of the executed repairs (rl/diffusion_ft/tools/viz_repair.py).
# Pure re-read of a select run scored with --dump_traj: no GPU, no re-roll, so
# it is free to re-run and the figures redraw from the npz already on disk.
#
# Config arrives as a positional env FILE. Never `sbatch --export`.
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
SEL_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR}"
SPLITS="${DFT_RIDE_SPLITS:-failure_heldout}"
ARM="${DFT_RIDE_ARM:-}"
TOP="${DFT_RIDE_TOP:-6}"
SMOOTH_W="${DFT_RIDE_SMOOTH_W:-9}"
WORLD_DT="${DFT_RESID_WORLD_DT:-0.1}"

echo "host=${SLURMD_NODENAME:-unknown}  select_dir=$SEL_DIR  splits=$SPLITS  arm=${ARM:-<all>}  top=$TOP"

# Gate inside the job: explain and exit 0 rather than strand anything downstream.
if ! compgen -G "$SEL_DIR/tasks/*/feat/*.npz" > /dev/null; then
  echo "RIDE ANALYSIS NOT RUN: no feat/*.npz under $SEL_DIR"; exit 0
fi
if ! "$PY" - "$SEL_DIR" <<'PYEOF'
import glob, sys, numpy as np
for f in glob.glob(f"{sys.argv[1]}/tasks/*/feat/*.npz"):
    if any(k.endswith("/traj") for k in np.load(f).keys()):
        sys.exit(0)
sys.exit(1)
PYEOF
then
  echo "RIDE ANALYSIS NOT RUN: $SEL_DIR has no '*/traj' arrays"
  echo "  (it was scored without --dump_traj; re-score a capped subset with DFT_DUMP_TRAJ=1)"
  exit 0
fi

IFS=',' read -r -a SPLIT_ARR <<< "$SPLITS"
for sp in "${SPLIT_ARR[@]}"; do
  echo; echo "########## $sp ##########"
  "$PY" -u -m rl.diffusion_ft.tools.viz_repair \
    --select_dir "$SEL_DIR" --split "$sp" --arm "$ARM" \
    --world_dt "$WORLD_DT" --top "$TOP" --smooth_w "$SMOOTH_W" --json "$SEL_DIR/ride_${sp}.json"
done
echo; echo "out -> $SEL_DIR/ride_*.json and $SEL_DIR/ride_*/"
