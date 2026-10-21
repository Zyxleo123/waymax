#!/bin/bash
#SBATCH --job-name=dp_convert
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-49
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/diffusion_planner_womd/logs/convert_%A_%a.out
#SBATCH --error=/zfsauton/scratch/ishanj/diffusion_planner_womd/logs/convert_%A_%a.err

# WOMD training shards -> Diffusion Planner samples (diffusion_planner_womd/convert_womd.py). 50 array tasks
# (the per-user submit limit) x 20 shards each = all 1000 training shards; then merge the per-shard lists:
#
#   sbatch slurm_dp_convert.sh
#   python diffusion_planner_womd/merge_lists.py /zfsauton/scratch/ishanj/diffusion_planner_womd/data
#
# ANCHORS_PER_SCENARIO (default 2) samples per scenario -> ~1M samples, the size of their nuPlan set.
set -euo pipefail

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
DATA_ROOT="${DATA_ROOT:-/zfsauton/scratch/ishanj/diffusion_planner_womd/data}"
TF_DIR="/zfsauton/scratch/mineuih/womd/tf_example/training"
LANE_GRAPH_DIR="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
ANCHORS_PER_SCENARIO="${ANCHORS_PER_SCENARIO:-2}"
SHARDS_PER_TASK=20

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/ishanj/miniconda3/bin/activate /zfsauton/scratch/ishanj/miniconda3/envs/waymax_rs
set -u
export PYTHONPATH="${ROOT}" JAX_PLATFORMS=cpu
mkdir -p "${DATA_ROOT}/lists"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
for ((k = 0; k < SHARDS_PER_TASK; k++)); do
  SHARD="$(printf '%05d' $((TASK * SHARDS_PER_TASK + k)))"
  NAME="training_tfexample.tfrecord-${SHARD}-of-01000"
  if [[ -f "${DATA_ROOT}/lists/${NAME}.json" ]]; then
    echo "${NAME}: already converted"
    continue
  fi
  python "${ROOT}/diffusion_planner_womd/convert_womd.py" \
    --tfrecord "${TF_DIR}/${NAME}" \
    --lane_graph_dir "${LANE_GRAPH_DIR}" \
    --out_dir "${DATA_ROOT}" \
    --anchors_per_scenario "${ANCHORS_PER_SCENARIO}"
done
