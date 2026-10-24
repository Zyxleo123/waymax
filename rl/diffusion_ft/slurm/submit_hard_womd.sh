#!/bin/bash
# Build the hard-WOMD training pool. Only calls sbatch.
#
# Chain: GPU scan array (Stage A) -> CPU select (Stage B), afterok.
#
# ---- why ------------------------------------------------------------------
# The WOMD re-run of `entry_clean` reported eval_solve_deterministic = 0.979 at
# iteration 0 (jobs 40469/40470): on a UNIFORM WOMD draw the base checkpoint
# already solves ~95-98% of scenes, so no reward change can be read there. The
# failure set had headroom because it was selected for it -- but it is the corpus
# whose degenerate timestamps invalidated every DPPO run taken on it.
#
# This keeps WOMD and recovers the headroom by selecting on THIS policy's own
# measured failures, gated on the expert log being clean so the pool is hard and
# still winnable.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_hard_womd.sh
#   DFT_SCAN_SCENES=8192 DFT_ARRAY=0-15 bash rl/diffusion_ft/slurm/submit_hard_womd.sh
#
# Resumable: rerun the same command with the same DFT_SCAN_OUT and DFT_SCAN_SEED
# to fill in whatever a task did not finish (rows already written are skipped).
# To EXTEND a scan, scan a second seed into a new dir rather than growing
# DFT_SCAN_SCENES -- a larger n redraws the whole set.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/hardwomd_${STAMP}.env"

# Default to a NAMED dir, not a timestamped one: the scan is resumable and
# extendable, and a fresh timestamp per submission would silently start over.
DFT_SCAN_OUT="${DFT_SCAN_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed${DFT_SCAN_SEED:-0}}"

{
  echo "DFT_SHARD_GLOB='${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*-of-01000}'"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  echo "DFT_SCAN_OUT=$DFT_SCAN_OUT"
  echo "DFT_SCAN_SCENES=${DFT_SCAN_SCENES:-4096}"
  echo "DFT_SCAN_BATCH=${DFT_SCAN_BATCH:-32}"
  echo "DFT_SCAN_SEED=${DFT_SCAN_SEED:-0}"
  # The collection sigma whose per-scene behaviour the scan also records. Keep it
  # equal to whatever the fine-tune will collect at, or the sto_* columns
  # describe a policy nothing will run.
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.03}"
  echo "DFT_NUM_PATHS=${DFT_NUM_PATHS:-45}"
  echo "DFT_NUM_POINTS_PER_PATH=${DFT_NUM_POINTS_PER_PATH:-800}"
  echo "DFT_CKPT=${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_MIN_POOL=${DFT_MIN_POOL:-64}"
  echo "DFT_INCLUDE_MARGINAL=${DFT_INCLUDE_MARGINAL:-0}"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# gpu28 is the 24 GB a5000; the rollout batch is the same shape the trainer
# evaluates at, which is where an arm died there before (job 37156).
#
# Scheduling overrides (plain sbatch flags, NOT --export):
#   DFT_TIME       per-task wall clock. The sbatch asks 8h, but a task is ~30 min
#                  (2048 scenes / 32 per batch x 3 rollouts). An honest, short
#                  limit is what makes a task BACKFILL-eligible into a reserved
#                  node's gap, which on a saturated `general` is the difference
#                  between starting now and starting tomorrow.
#   DFT_PARTITION  e.g. preempt (a6000s on gpu1/27/30/31), paired with DFT_QOS.
#                  Every partition mandates its own QOS or sbatch dies on
#                  "Invalid qos specification": general->qos_general (default,
#                  no flag), preempt->qos_preempt, legacy->qos_legacy. NOT
#                  project: this account holds no association for either project
#                  QOS however idle myfree claims the h200 is.
#   DFT_QOS        the QOS matching DFT_PARTITION.
# Preemption is cheap here BECAUSE the scan is resumable: a killed task's written
# rows are skipped on requeue, so it resumes rather than restarts. The sbatch
# carries --requeue and --open-mode=append for exactly that.
SB_FLAGS=()
[[ -n "${DFT_EXCLUDE-}" ]] && SB_FLAGS+=(--exclude="${DFT_EXCLUDE-}")
[[ -n "${DFT_NODELIST:-}" ]]    && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_TIME:-}" ]]        && SB_FLAGS+=(--time="$DFT_TIME")
[[ -n "${DFT_PARTITION:-}" ]]   && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]         && SB_FLAGS+=(--qos="$DFT_QOS")
echo "sbatch flags: ${SB_FLAGS[*]}"

# Scrub SLURM_* so submitting from inside an allocation cannot leak this job's
# environment into the new one.
SCAN=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --array="${DFT_ARRAY:-0-7}" "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/scan_womd_headroom.sbatch "$ENV_FILE")
echo "scan array job:  $SCAN"

# afterok on an array waits for EVERY task. The selector reads whatever rows
# exist, so a partial scan still yields a pool -- but only a clean array means
# the pool covers the whole scan set.
SEL=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency=afterok:"$SCAN" \
  rl/diffusion_ft/slurm/select_hard_womd.sbatch "$ENV_FILE")
echo "select job:      $SEL  (afterok:$SCAN)"

echo
echo "Read the FIRST scan .out before trusting the chain:"
echo "  * host= is an a6000 (a v100 dies in cuDNN init)"
echo "  * '[scan] corpus: 1000 shards' and a nonzero 'to do'"
echo "  * the per-batch line's det_solve should be ~0.95 -- if it is ~0, the"
echo "    route dims or the checkpoint are wrong, not the corpus"
echo
echo "Then: $DFT_SCAN_OUT/hard_womd_summary.md  and  hard_womd.json"
