#!/bin/bash
#SBATCH --job-name=dft_rank
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# The reward-ORDERING gate. Run this before any DPPO launch whose reward flags
# changed: it asks whether the reward ranks a clean solve above every other
# episode archetype, not whether the coefficients are big enough.
#
# Pure reward algebra on synthetic episodes -- no simulator, no checkpoint, no
# GPU, seconds. Asking for a GPU would only queue it behind the training runs it
# is supposed to gate.
#
# No `--qos` line on purpose: the `cpu` partition applies qos_cpu as its default,
# and naming a QOS this account has no association for fails the submission with
# a bare "Invalid qos specification".
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

echo "=== reward ranking gate: $(date) on $(hostname) ==="
# `set -e` would abort before $? could be read, so capture the failure inline.
rc=0
"$PYTHON" -m rl.diffusion_ft.tests.test_reward_ranking || rc=$?
echo "=== exit $rc ==="
exit "$rc"
