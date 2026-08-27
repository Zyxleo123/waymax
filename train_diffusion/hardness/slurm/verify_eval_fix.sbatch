#!/bin/bash
#SBATCH --job-name=hrd_verify
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nodelist=gpu28
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/hardness/slurm-%x-%j.out
#
# Verifies the two fixes that invalidated the 2026-08-20 alpha sweep, before any
# arm is resubmitted:
#
#   A) the planner drives with the weights it is handed (was: pinned to whatever
#      the first eval traced, so epochs 5/10/15 produced identical rollouts)
#   B) the failure set parses at all (was: 10x300 routes read under the 45x800
#      WOD_1_3_1_TRAINING contract, so every [eval:failures] pass loaded zero
#      scenarios and reported nothing)
#
# gpu28 (a5000, general) picked off `myfree` -- gpu29 went backfill-reserved, and
# its contention already showed up as CUDA_ERROR_OUT_OF_MEMORY retries in run
# 35150. 24G still clears the 2080_ti allocation death that kills the process
# with no Python traceback.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
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

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Belt and braces with data/scenario_loader.py hiding the GPU from TF: if some
# other import gets TF onto the device first, at least it grows instead of
# claiming the card and starving the waymax rollout.
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

HRD_CKPT="${HRD_CKPT:?set HRD_CKPT to the pretrained diffusion checkpoint}"
HRD_EVAL_TFRECORD_DIR="${HRD_EVAL_TFRECORD_DIR:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures}"
HRD_EVAL_LANE_GRAPH_DIR="${HRD_EVAL_LANE_GRAPH_DIR:-/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"
HRD_VERIFY_NUM="${HRD_VERIFY_NUM:-20}"
# Parameterised so a memory retry is one variable, not an edit: the offroad
# metric allocates f32[num_worlds,512,30000], ~61MB per world per intermediate.
HRD_VERIFY_WORLDS="${HRD_VERIFY_WORLDS:-10}"
HRD_VERIFY_OUT="${HRD_VERIFY_OUT:-/zfsauton/scratch/yixiz/waymax_rs/hardness_verify}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# Run 35199 died in waymax's offroad metric with "Allocator (GPU_0_bfc) ran out
# of memory" -- TF's allocator, holding the card it only ever needed for
# tfrecord I/O. Print what TF can actually see, so the next failure does not
# need to be inferred from an allocator name in a stack trace.
"$PYTHON" -c "import data.scenario_loader, tensorflow as tf; \
print('TF visible GPUs:', tf.config.get_visible_devices('GPU'), \
'of', len(tf.config.list_physical_devices('GPU')), 'physical')" 2>/dev/null
echo "ckpt=$HRD_CKPT"

echo
echo "=== 0. lane graph redirect for the failure set ==="
# failures.tfrecord has no lanegraph.zip of its own; the sidecar points each
# scene at the WOMD shard entry it was copied from. Cheap to rebuild, and
# --verify refuses to write a map whose scenes do not line up.
LG_MAP="$HRD_EVAL_TFRECORD_DIR/failures.tfrecord.lanegraph_map.json"
if [[ -f "$LG_MAP" ]]; then
  echo "present: $LG_MAP"
else
  "$PYTHON" -m rl.build_failure_lanegraph_map \
    --tfrecord "$HRD_EVAL_TFRECORD_DIR/failures.tfrecord" --verify 5
fi

echo
echo "=== A. planner uses the weights it is handed ==="
"$PYTHON" -m train_diffusion.tests.test_eval_params_live --ckpt "$HRD_CKPT"

echo
echo "=== B. the failure set parses and drives ($HRD_VERIFY_NUM scenarios) ==="
"$PYTHON" -m train_diffusion.tests.eval_failure_set \
  --ckpt "$HRD_CKPT" \
  --tfrecord_dir "$HRD_EVAL_TFRECORD_DIR" \
  --lane_graph_dir "$HRD_EVAL_LANE_GRAPH_DIR" \
  --output_dir "$HRD_VERIFY_OUT/failures" \
  --num "$HRD_VERIFY_NUM" \
  --num_worlds "$HRD_VERIFY_WORLDS"

echo
echo "Both checks passed (set -e would have stopped the job otherwise)."
