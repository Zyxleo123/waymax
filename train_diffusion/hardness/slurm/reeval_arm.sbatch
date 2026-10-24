#!/bin/bash
#SBATCH --job-name=hrd_reeval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Re-evaluate one alpha-sweep checkpoint with the FIXED eval (verify job 35289):
#   A) the planner drives with the weights it is handed, and
#   B) the 218-scene failure set parses under num_sdc_paths rather than 45x800.
# The 2026-08-20 eval_logs are invalid for both reasons, so this reruns the
# closed-loop eval offline from the saved epoch_0010 checkpoints -- no retrain.
#
# Scores TWO sets, exactly as the in-training eval did:
#   failures  the 218-scene ScenarioMax failure set (the metric that never loaded)
#   womd      a 200-scene held-out WOMD slice (general-quality guardrail)
#
# simulation.runner.run writes one JSON per scenario plus a summary.json under a
# timestamped subdir; aggregate_reeval.py reads those, so the comparison table is
# redrawable without touching the GPU again.
#
# gpu29 (a6000, 48G, general) picked off `myfree` on 2026-08-25: the only
# AVAILABLE general a6000, and 48G clears the offroad-metric allocation
# (f32[num_worlds,512,30000]) that OOM'd the 24G retries in run 35150/35199.
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
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

RE_ARM="${RE_ARM:?set RE_ARM to the arm label, e.g. control}"
RE_CKPT="${RE_CKPT:?set RE_CKPT to the epoch_0010 checkpoint directory}"
RE_OUT="${RE_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/hardness_reeval}"
RE_FAIL_TFRECORD_DIR="${RE_FAIL_TFRECORD_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures}"
RE_FAIL_NUM="${RE_FAIL_NUM:-218}"
RE_WOMD_TFRECORD_DIR="${RE_WOMD_TFRECORD_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training}"
RE_WOMD_NUM="${RE_WOMD_NUM:-200}"
RE_LANE_GRAPH_DIR="${RE_LANE_GRAPH_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"
RE_NUM_WORLDS="${RE_NUM_WORLDS:-10}"

if [[ ! -f "$RE_CKPT/metadata.json" ]]; then
  echo "ERROR: no metadata.json under $RE_CKPT -- not a checkpoint dir." >&2
  exit 1
fi

echo "host=${SLURMD_NODENAME:-unknown} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "arm=$RE_ARM ckpt=$RE_CKPT"

# The failure set has no lanegraph.zip; the sidecar redirects each scene to the
# WOMD shard entry it was copied from. Cheap to rebuild; --verify refuses a map
# whose scenes do not line up. (verify job 35289, section 0.)
LG_MAP="$RE_FAIL_TFRECORD_DIR/failures.tfrecord.lanegraph_map.json"
if [[ -f "$LG_MAP" ]]; then
  echo "lanegraph map present: $LG_MAP"
else
  "$PYTHON" -m rl.build_failure_lanegraph_map \
    --tfrecord "$RE_FAIL_TFRECORD_DIR/failures.tfrecord" --verify 5
fi

echo
echo "=== failures ($RE_FAIL_NUM scenarios) ==="
"$PYTHON" -m train_diffusion.tests.eval_failure_set \
  --ckpt "$RE_CKPT" \
  --tfrecord_dir "$RE_FAIL_TFRECORD_DIR" \
  --lane_graph_dir "$RE_LANE_GRAPH_DIR" \
  --output_dir "$RE_OUT/$RE_ARM/failures" \
  --num "$RE_FAIL_NUM" \
  --num_worlds "$RE_NUM_WORLDS"

echo
echo "=== womd ($RE_WOMD_NUM scenarios) ==="
"$PYTHON" -m train_diffusion.tests.eval_failure_set \
  --ckpt "$RE_CKPT" \
  --tfrecord_dir "$RE_WOMD_TFRECORD_DIR" \
  --lane_graph_dir "$RE_LANE_GRAPH_DIR" \
  --output_dir "$RE_OUT/$RE_ARM/womd" \
  --num "$RE_WOMD_NUM" \
  --num_worlds "$RE_NUM_WORLDS"

echo
echo "DONE arm=$RE_ARM -> $RE_OUT/$RE_ARM"
