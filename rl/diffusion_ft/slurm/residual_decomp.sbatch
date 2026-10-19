#!/bin/bash
#SBATCH --job-name=dft_resid
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Residual decomposition: WHY the cons filter did not save each scene it failed
# to save (rl/diffusion_ft/tools/residual_decomp.py). Pure re-read of an existing
# selection run's rows.jsonl + feat/*.npz -- no GPU, no re-rolling, so it can be
# re-run on any past select dir. The figure is drawn FROM residual_<split>.json,
# so it is redrawable without touching the npz again:
#   python -m rl.diffusion_ft.tools.residual_decomp --json <j> --png <p> --plot_only
#
# Config arrives as a positional env FILE. Never `sbatch --export`.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
SEL_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR (a select run directory)}"
SPLITS="${DFT_RESID_SPLITS:-failure_heldout,intermittent_heldout}"
ARMS="${DFT_RESID_ARMS:-}"
# World step in seconds; sets the ramp duration behind the executed-edit
# acceleration figures. NOT recoverable from meta, so it is an explicit knob.
WORLD_DT="${DFT_RESID_WORLD_DT:-0.1}"

echo "host=${SLURMD_NODENAME:-unknown}  select_dir=$SEL_DIR  splits=$SPLITS  arms=${ARMS:-<all>}  world_dt=$WORLD_DT"

# Gate INSIDE the job: an upstream scoring array that wrote nothing is a reason
# to explain and exit 0, not to strand anything downstream on a failed dependency.
if ! compgen -G "$SEL_DIR/tasks/*/rows.jsonl" > /dev/null; then
  echo "DECOMPOSITION NOT RUN: no tasks/*/rows.jsonl under $SEL_DIR"
  echo "  (the scoring array wrote no rows -- read its .out before re-running)"
  exit 0
fi
if ! compgen -G "$SEL_DIR/tasks/*/feat/*.npz" > /dev/null; then
  echo "DECOMPOSITION NOT RUN: rows exist but no feat/*.npz under $SEL_DIR"
  echo "  (per-replan features are what this job reads; the run predates them)"
  exit 0
fi

IFS=',' read -r -a SPLIT_ARR <<< "$SPLITS"
for sp in "${SPLIT_ARR[@]}"; do
  echo; echo "########## $sp ##########"
  "$PY" -u -m rl.diffusion_ft.tools.residual_decomp \
    --select_dir "$SEL_DIR" --split "$sp" --arms "$ARMS" \
    --world_dt "$WORLD_DT" \
    --json "$SEL_DIR/residual_${sp}.json" --png "$SEL_DIR/residual_${sp}.png"
done
echo; echo "out -> $SEL_DIR/residual_*.{json,png}"
