#!/bin/bash
# Submitter for the H3 termination-attribution diagnosis: which flag absorbs
# every world in the first replan under the V-Max parity reward?
#
# Only calls sbatch. Chains: GPU rollout+record (one pass per reward config) ->
# one CPU figure job per pass dir (afterok).
#
# Passes: legacy (regression anchor), parity (job 34866), perstep (the candidate
# fix -- offroad charged every off-road step, non-terminating). Select a subset
# with DFT_TERM_PASSES, e.g. DFT_TERM_PASSES=perstep to re-read just the new arm.
#
# Writes a timestamped env file and passes its path as a positional arg to the
# batch script (no `sbatch --export`, which requeue-holds on this cluster).
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_diagnose_term.sh
#
# Node: defaults to general/gpu28 (a5000 24 GB, 436 G free host RAM), which is
# where job 34866 itself ran. Do NOT retarget this at project/gpu2: `myfree` and
# `scontrol show partition project` both advertise it (AllowQos lists
# qos_radio_freq), but this account holds no association for either project QOS,
# so sbatch dies on "Invalid qos specification" -- measured 2026-08-21.
# Override with DFT_PARTITION + DFT_QOS + DFT_NODELIST. Do NOT let this land on
# an 11 GB 2080 Ti (the `legacy`/`debug` partitions): it holds a B=32 batched
# Waymax state and an XLA OOM kills the process with no traceback.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/term_${STAMP}.env"

PARITY_RUN="/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo/20260820_203119"
DFT_ARMS="${DFT_ARMS:-base=,parity500=${PARITY_RUN}/ckpt_0500}"
DFT_DIAG_OUT="${DFT_DIAG_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/term/${STAMP}}"
DFT_TERM_PASSES="${DFT_TERM_PASSES:-legacy parity perstep}"

{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
  echo "DFT_ARMS=$DFT_ARMS"
  echo "DFT_DIAG_SEEDS=${DFT_DIAG_SEEDS:-4}"
  echo "DFT_DIAG_STOCHASTIC=${DFT_DIAG_STOCHASTIC:-1}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  # Job 34866 ran sigma_floor=0.1; the collapse happened during stochastic
  # collection, so measuring at 0.03 would measure a policy it never ran.
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.1}"
  echo "DFT_REPLAN=${DFT_REPLAN:-10}"
  # Parity recipe, verbatim from logs/dft/env/dppo_dppo_vmax_parity_20260820_203024.env
  echo "DFT_REWARD_PROGRESS=${DFT_REWARD_PROGRESS:-0.2}"
  echo "DFT_REWARD_OFFROAD=${DFT_REWARD_OFFROAD:--1.0}"
  echo "DFT_REWARD_COLLISION=${DFT_REWARD_COLLISION:--1.0}"
  echo "DFT_REWARD_GOAL_BONUS=${DFT_REWARD_GOAL_BONUS:-5.0}"
  echo "DFT_REWARD_OFF_ROUTE=${DFT_REWARD_OFF_ROUTE:--0.2}"
  echo "DFT_GOAL_SHAPING_COEF=${DFT_GOAL_SHAPING_COEF:-1.0}"
  # Paid on EVERY off-road step in the perstep pass, so 5x smaller than the
  # terminating variant's -1.0 (a 24-step excursion costs -4.8 vs a +5 goal).
  echo "DFT_REWARD_OFFROAD_PERSTEP=${DFT_REWARD_OFFROAD_PERSTEP:--0.2}"
  echo "DFT_NUM_PATHS=${DFT_NUM_PATHS:-10}"
  echo "DFT_NUM_POINTS_PER_PATH=${DFT_NUM_POINTS_PER_PATH:-300}"
  echo "DFT_GAMMA=${DFT_GAMMA:-0.99}"
  echo "DFT_VIZ_SCENES=${DFT_VIZ_SCENES:-16}"
  echo "DFT_TERM_PASSES=\"$DFT_TERM_PASSES\""
  echo "DFT_DIAG_OUT=$DFT_DIAG_OUT"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# Every partition on this cluster mandates its own QOS, and sbatch rejects the
# job with a bare "Invalid qos specification" if the two disagree:
#   general -> qos_general (the default; no flag needed)
#   legacy  -> qos_legacy      debug -> qos_debug       cpu -> qos_cpu
#   project -> qos_radio_freq | qos_project_preempt   (h200 gpu2)
#   preempt -> qos_preempt
# `general` runs qos_general as its default QOS, so no --qos flag is needed (and
# passing one this account does not hold is what killed the first attempt).
DFT_PARTITION="${DFT_PARTITION:-general}"
DFT_QOS="${DFT_QOS:-}"
DFT_NODELIST="${DFT_NODELIST:-gpu28}"
SB_FLAGS=(--partition="$DFT_PARTITION")
[[ -n "$DFT_QOS" ]]      && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "$DFT_NODELIST" ]] && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]] && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
echo "sbatch flags: ${SB_FLAGS[*]}"

# Scrub SLURM_* so submitting from inside an allocation cannot leak this job's
# environment into the new one.
JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/diagnose_term.sbatch "$ENV_FILE")
echo "termination-attribution job: $JID"

for cfg in $DFT_TERM_PASSES; do
  VID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable --dependency=afterok:"$JID" \
    rl/diffusion_ft/slurm/viz_drive.sbatch "$DFT_DIAG_OUT/$cfg" "$ENV_FILE")
  echo "viz job ($cfg):              $VID  (afterok:$JID)"
done

echo
echo "read the TAIL of logs/dft/slurm-dft_term-${JID}.out:"
echo "  1. legacy  base/det must match diag/20260819_021146 (absorb@1 ~ 0.29)"
echo "  2. parity  base/det absorb@1 -- and the first-prefix offroad curve"
echo "  3. perstep base/sto nrep must recover toward legacy (parity: 1.02 at"
echo "     sigma 0.1) while offFr stays BELOW legacy -- otherwise -0.2 is too weak"
echo "artifacts -> $DFT_DIAG_OUT/{$(echo $DFT_TERM_PASSES | tr ' ' ',')}/term_attribution.json"
