#!/bin/bash
#SBATCH --job-name=vmh_smoke
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#
# Optional pre-flight: verify the hardness generator yields 64-object states with
# the right batch prefix and that a real V-Max env resets on one. Builds offset
# indexes for just two shards into a throwaway dir. Uses the existing risk.npz
# only to exercise plumbing (values irrelevant here).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/vmax_hardness"
# shellcheck source=/dev/null
source "$ROOT/rl/vmax_hardness/slurm/_preamble.sh" "$@"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

SMOKE_GLOB="/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-0000[01]-of-01000"
SMOKE_INDEX="${VMH_OUT}/smoke_offsets"
RISK="${VMH_RISK_OUT}/risk.npz"
[[ -f "$RISK" ]] || RISK="$ROOT/runs/hardness/risk.npz"   # fall back to the diffusion table for plumbing

echo "host=${SLURMD_NODENAME:-unknown} glob=$SMOKE_GLOB risk=$RISK"
"$PYTHON" -m rl.vmax_hardness.tests.smoke_generator \
  --shard-glob "$SMOKE_GLOB" \
  --index-dir "$SMOKE_INDEX" \
  --risk-table "$RISK" \
  --max-num-objects "$VMH_MAX_NUM_OBJECTS"
