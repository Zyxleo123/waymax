#!/bin/bash
#SBATCH --job-name=dft_rewagg
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
# Read the reward sweep's train_log.jsonl files and write the ranking table plus
# the figure. No model and no GPU -- asking for one would only queue behind the
# training arms it is summarizing -- so the table re-renders for the cost of a
# CPU slot whenever the tail window needs changing.
#
# Pass the sweep dir as $1 (or DFT_SWEEP_OUT via the env file).
#
# No `--qos` line on purpose: the `cpu` partition applies qos_cpu as its default,
# and naming a QOS this account has no association for fails the submission with
# a bare "Invalid qos specification".
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
DFT_AGG_TAIL="${DFT_AGG_TAIL:-5}"

# A gate, not a failure: exit 0 with the reason printed rather than stranding
# anything downstream on an unexplained DependencyNeverSatisfied. This job runs
# `afterany` on the arms, so it also fires when every arm died -- and then the
# useful output is the reason, not a non-zero exit.
if [[ -z "$SWEEP_DIR" || ! -d "$SWEEP_DIR" ]]; then
  echo "[rewardagg] GATE: no sweep dir (positional dir arg or DFT_SWEEP_OUT); got '$SWEEP_DIR'."
  echo "[rewardagg] The training arms either never ran or wrote nowhere."
  exit 0
fi

echo "[rewardagg] sweep=$SWEEP_DIR tail=$DFT_AGG_TAIL"
"$PYTHON" -m rl.diffusion_ft.tests.aggregate_reward_sweep "$SWEEP_DIR" \
  --tail "$DFT_AGG_TAIL"
