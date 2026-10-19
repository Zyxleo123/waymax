#!/bin/bash
#SBATCH --job-name=dft_sigagg
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Read the sigma sweep's artifacts and write the table + figure. No model and no
# GPU -- asking for one would only queue behind real work -- so the threshold can
# be re-read with a different tolerance for the cost of a CPU slot.
#
# Pass the sweep dir as $1 (or DFT_SWEEP_OUT via the env file).
#
# No `--qos` line on purpose: the `cpu` partition applies qos_cpu as its default,
# and naming a QOS explicitly is what killed the first gpu2 submission -- an
# association this account lacks fails with a bare "Invalid qos specification".
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

SWEEP_ARG=""
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ -d "$arg" ]]; then
    SWEEP_ARG="$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1

SWEEP_DIR="${SWEEP_ARG:-${DFT_SWEEP_OUT:-}}"
DFT_SIGMA_TOL="${DFT_SIGMA_TOL:-0.10}"

# A gate, not a failure: exit 0 with the reason printed rather than stranding
# anything downstream on an unexplained DependencyNeverSatisfied.
if [[ -z "$SWEEP_DIR" || ! -d "$SWEEP_DIR" ]]; then
  echo "[sigagg] GATE: no sweep dir (positional dir arg or DFT_SWEEP_OUT); got '$SWEEP_DIR'."
  echo "[sigagg] The GPU sweep either never ran or wrote nowhere. Nothing to aggregate."
  exit 0
fi

echo "[sigagg] sweep=$SWEEP_DIR tolerance=$DFT_SIGMA_TOL"
"$PYTHON" -m rl.diffusion_ft.tests.aggregate_sigma_sweep "$SWEEP_DIR" \
  --tolerance "$DFT_SIGMA_TOL"
