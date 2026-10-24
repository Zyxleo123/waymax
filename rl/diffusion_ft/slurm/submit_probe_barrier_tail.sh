#!/bin/bash
# Submitter for the barrier-tail probe. Only calls sbatch.
#
# The question: does the safety term charge scenes where nothing is wrong? The
# softplus barrier never reaches zero, and it is SUMMED over 4 ego corners, over
# every valid agent, and over every step -- so a scene is charged partly for its
# traffic density and episode length rather than for its geometry. The probe
# splits loss AND gradient into the part inside the margin and the part already
# clear of it, and re-scores the same clearances under the compact-support
# shapes so the size of the fix is on the same numbers.
#
# Two submissions, because they answer different halves of it and are siblings
# (no dependency, they queue concurrently):
#   * failures.tfrecord -- the scenes the fine-tune is graded on. Few of these
#     are trivial, so this says how much of a HARD scene's loss is tail.
#   * a WOMD training shard -- ordinary driving, which is what the corpus-level
#     regression is measured over and where the trivial scene is the common
#     case. This is the one that says whether the term misfires.
#
# CPU-only and checkpoint-free, so it does not queue behind GPU work.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_probe_barrier_tail.sh
#
# Knobs: DFT_TAIL_SCENES (default 64 per corpus), DFT_EDGE_MARGIN/TAU,
# DFT_AGENT_MARGIN/TAU, DFT_AGENT_GEOMETRY (exact|disc).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${DFT_TAIL_OUT_ROOT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/barrier_tail/${STAMP}}"

FAILURES="${DFT_FAILURES_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
WOMD="${DFT_WOMD_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"

submit_one () {   # $1 = tag, $2 = tfrecord
  local tag="$1" rec="$2"
  local env_file="$ENV_DIR/btail_${tag}_${STAMP}.env"
  {
    echo "DFT_TFRECORD=$rec"
    echo "DFT_TAIL_SCENES=${DFT_TAIL_SCENES:-64}"
    echo "DFT_TAIL_START=${DFT_TAIL_START:-0}"
    echo "DFT_TAIL_CHUNK=${DFT_TAIL_CHUNK:-8}"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_EDGE_TAU=${DFT_EDGE_TAU:-0.15}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_AGENT_TAU=${DFT_AGENT_TAU:-0.25}"
    echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    echo "DFT_TAIL_OUT=$OUT_ROOT/$tag"
  } > "$env_file"
  local jid
  jid=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable rl/diffusion_ft/slurm/probe_barrier_tail.sbatch "$env_file")
  echo "  $tag: job $jid  env $env_file  -> $OUT_ROOT/$tag"
}

echo "barrier-tail probe (cpu partition):"
submit_one failures "$FAILURES"
submit_one womd     "$WOMD"
echo
echo "artifacts -> $OUT_ROOT/{failures,womd}/barrier_tail.{json,jsonl}"
echo "read the '=== per-scene loss on scenes where NOTHING is inside a margin ===' block"
