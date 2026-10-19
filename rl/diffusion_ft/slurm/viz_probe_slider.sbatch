#!/bin/bash
#SBATCH --job-name=dft_probeviz
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Render the optimization-step slider from the probe dumps two training runs
# wrote. Reads saved npz/jsonl only -- no checkpoint, no corpus, no GPU -- so it
# re-renders in seconds against any finished (or still-running) pair of runs.
#
# Config arrives as a positional env FILE (never `sbatch --export`, which
# triggers SLURM_GET_USER_ENV and requeue-holds the job).
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
# Space-separated NAME=OUT_DIR, one per row of the figure.
DFT_PROBE_ARMS="${DFT_PROBE_ARMS:?set DFT_PROBE_ARMS='name=dir name=dir'}"
DFT_PROBE_FIG_DIR="${DFT_PROBE_FIG_DIR:-}"
DFT_PROBE_TITLE="${DFT_PROBE_TITLE:-Does the safety loss fix the trajectory?}"

echo "host=${SLURMD_NODENAME:-unknown} arms=$DFT_PROBE_ARMS"

# Gate INSIDE the job. A missing static.npz means the training stage never
# reached its first probe point; say so and exit 0 rather than stranding any
# dependent job in DependencyNeverSatisfied with no explanation.
ARGS=()
MISSING=0
for A in $DFT_PROBE_ARMS; do
  D="${A#*=}"
  if [[ ! -s "$D/probe/static.npz" ]]; then
    echo "GATE: $D/probe/static.npz missing -- that arm wrote no probe dumps."
    echo "GATE: check DFT_PROBE_PER_SPLIT > 0 and DFT_SPLITS_JSON in its .out."
    MISSING=1
  fi
  ARGS+=(--arm "$A")
done
if [[ "$MISSING" == "1" ]]; then
  echo "GATE: nothing to render."; exit 0
fi
[[ -n "$DFT_PROBE_FIG_DIR" ]] && ARGS+=(--out_dir "$DFT_PROBE_FIG_DIR")

JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tools.viz_probe_slider \
  "${ARGS[@]}" --title "$DFT_PROBE_TITLE"
