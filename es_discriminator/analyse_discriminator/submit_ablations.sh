#!/bin/bash
# Train discriminator ablations A1-A8 with A0's exact pipeline (slurm/train/train_wm_discriminator.sh):
# same combined data set, split, invalid list, frozen WM caches (epoch_0450), loss, hyper-parameters,
# 30 epochs, seed 0. Only --ablation differs. Each run writes a new directory
#   /zfsauton/scratch/sbellad/es_discriminator/safety_progress/safety_progress_v3_<AX>_<timestamp>_<jobid>/
# so A0 (safety_progress_v3_20261016_155456_86103) is never touched. Submit from rhea / lov4:
#
#   bash es_discriminator/analyse_discriminator/submit_ablations.sh --dry_run
#   bash es_discriminator/analyse_discriminator/submit_ablations.sh            # A1 A2 A3 A4
#   bash es_discriminator/analyse_discriminator/submit_ablations.sh A2 A4      # a subset
#   bash es_discriminator/analyse_discriminator/submit_ablations.sh A5 A6 A7 A8   # second round (A4 changes)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_NAME=fail1500_plus_womd_train_s02_11
DRY_RUN=0
NAMES=()
for arg in "$@"; do
  case "${arg}" in
    --dry_run) DRY_RUN=1 ;;
    A[1-8]) NAMES+=("${arg}") ;;
    *) echo "ERROR: unknown argument '${arg}' (A1..A8 or --dry_run)" >&2; exit 1 ;;
  esac
done
[[ ${#NAMES[@]} -gt 0 ]] || NAMES=(A1 A2 A3 A4)

for name in "${NAMES[@]}"; do
  cmd=(sbatch --job-name="disc_abl_${name}" slurm/train/train_wm_discriminator.sh
       --data "${DATA_NAME}" --run_name "safety_progress_v3_${name}" -- --ablation "${name}")
  if (( DRY_RUN )); then
    echo "[dry-run] ${cmd[*]}"
  else
    echo "${name}: $("${cmd[@]}")"
  fi
done
