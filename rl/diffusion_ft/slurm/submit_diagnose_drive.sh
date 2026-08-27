#!/bin/bash
# Submitter for the "how does the fine-tuned policy drive?" diagnosis.
# Only calls sbatch. Chains: GPU rollout+record -> CPU figures (afterok).
#
# Writes a timestamped env file and passes its path as a positional arg to the
# batch script (no `sbatch --export`, which requeue-holds on this cluster).
#
# Usage (from repo root, on rhea):
#   DFT_ARMS="base=,dppo500=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo/\
# baseline_resume_20260818_220750/dppo_B32_baseline_resume/20260818_220815/ckpt_0250" \
#   bash rl/diffusion_ft/slurm/submit_diagnose_drive.sh
#
# Optional: DFT_PARTITION / DFT_NODELIST to place the GPU job. NOT `project`/gpu2:
# this account holds no association for either project QOS, so sbatch dies on
# "Invalid qos specification" however idle `myfree` claims the h200 is (measured
# 2026-08-21). When the `general` a6000s are full, use DFT_NODELIST=gpu28.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/diag_${STAMP}.env"

: "${DFT_ARMS:?set DFT_ARMS, e.g. DFT_ARMS='base=,dppo500=/path/to/ckpt_0250'}"
DFT_DIAG_OUT="${DFT_DIAG_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/diag/${STAMP}}"

{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
  echo "DFT_ARMS=$DFT_ARMS"
  echo "DFT_DIAG_SEEDS=${DFT_DIAG_SEEDS:-4}"
  echo "DFT_DIAG_STOCHASTIC=${DFT_DIAG_STOCHASTIC:-1}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  # Must match the run being diagnosed: the sigma floor defines the stochastic
  # policy, so a mismatched floor measures a policy the trainer never ran.
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.03}"
  echo "DFT_VIZ_SCENES=${DFT_VIZ_SCENES:-16}"
  echo "DFT_DIAG_OUT=$DFT_DIAG_OUT"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# Every partition on this cluster mandates its own QOS, and sbatch rejects the
# job with a bare "Invalid qos specification" if the two disagree:
#   general -> qos_general (the default; no flag needed)
#   legacy  -> qos_legacy
#   debug   -> qos_debug
#   cpu     -> qos_cpu
#   project -> qos_radio_freq | qos_project_preempt   (h200 gpu2)
#   preempt -> qos_preempt
# So DFT_PARTITION alone is not enough -- set DFT_QOS with it.
SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
if [[ -n "${DFT_PARTITION:-}" && -z "${DFT_QOS:-}" && "$DFT_PARTITION" != "general" ]]; then
  echo "WARNING: DFT_PARTITION=$DFT_PARTITION set without DFT_QOS; sbatch will"
  echo "WARNING: likely fail with 'Invalid qos specification'." >&2
fi
echo "sbatch flags: ${SB_FLAGS[*]:-<none>}"

# Scrub SLURM_* so submitting from inside an allocation cannot leak this job's
# environment into the new one.
JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/diagnose_drive.sbatch "$ENV_FILE")
echo "diagnose job: $JID"

VID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency=afterok:"$JID" \
  rl/diffusion_ft/slurm/viz_drive.sbatch "$DFT_DIAG_OUT" "$ENV_FILE")
echo "viz job:      $VID  (afterok:$JID)"
echo
echo "artifacts -> $DFT_DIAG_OUT/{rollouts.jsonl,rollouts.npz}"
echo "figures   -> $DFT_DIAG_OUT/{drive_diagnosis.png,drive_paths.png,drive_summary.md}"
