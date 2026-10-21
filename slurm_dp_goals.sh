#!/bin/bash
#SBATCH --job-name=dp_goals
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0-24
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/diffusion_planner_womd/logs/goals_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/diffusion_planner_womd/logs/goals_%A_%a.err

# Episode goals of the converted Diffusion Planner training samples (diffusion_planner_womd/goals_womd.py), for
# goal-conditioned training (GOAL_CONDITIONED=1 sbatch slurm_dp_train.sh). 25 array tasks x 40 shards (~25 s
# each), which leaves room under the 50-job submit limit; shards with a goals file are skipped.
set -euo pipefail

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
DATA_ROOT="${DATA_ROOT:-/zfsauton/scratch/ishanj/diffusion_planner_womd/data}"
TF_DIR="/zfsauton/scratch/mineuih/womd/tf_example/training"
SHARDS_PER_TASK=40

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/ishanj/miniconda3/bin/activate /zfsauton/scratch/ishanj/miniconda3/envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}" JAX_PLATFORMS=cpu

TASK="${SLURM_ARRAY_TASK_ID:-0}"
for ((k = 0; k < SHARDS_PER_TASK; k++)); do
  NAME="training_tfexample.tfrecord-$(printf '%05d' $((TASK * SHARDS_PER_TASK + k)))-of-01000"
  if [[ -f "${DATA_ROOT}/goals/${NAME}.json" ]]; then
    echo "${NAME}: goals exist"
    continue
  fi
  python "${ROOT}/diffusion_planner_womd/goals_womd.py" --tfrecord "${TF_DIR}/${NAME}" --data_root "${DATA_ROOT}"
done
