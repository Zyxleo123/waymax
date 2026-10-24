#!/bin/bash
#SBATCH --job-name=hrd_reeval_agg
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Read the per-scenario JSONs written by the five reeval_arm jobs and print the
# comparison table. CPU-only: it only reads saved artefacts, so it must not queue
# behind GPU work for a card it never uses. Runs afterok on the eval siblings.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/hardness"

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
RE_OUT="${RE_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/hardness_reeval}"

"$PYTHON" -m train_diffusion.hardness.aggregate_reeval --reeval_dir "$RE_OUT" --epoch epoch_0010
