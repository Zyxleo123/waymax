#!/bin/bash
#SBATCH --job-name=awm_eval
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/eval_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/eval_%j.err
#
# Step 3: the decision readout.
#   (a) ES-bank picking on B0's held-out test groups, for each checkpoint, next to B0
#   (b) forecasting on WOMD validation shards: MF-Gaussian (B0's WM) vs each checkpoint
#
#   CKPTS="/zfsauton/scratch/ishanj/anchor_wm/runs/separate_<ts>/final.pt /zfsauton/scratch/ishanj/anchor_wm/runs/joint_<ts>/final.pt" \
#     sbatch anchor_wm/slurm/eval.sh
#   SMOKE=1 CKPTS=... sbatch --time=0-00:30:00 anchor_wm/slurm/eval.sh    # 5 bank files, 50 WM batches
set -euo pipefail
source "$(dirname "$0")/common.sh" 2>/dev/null || source /zfsauton/scratch/ishanj/waymax_rs/anchor_wm/slurm/common.sh

: "${CKPTS:?set CKPTS to one or more AnchorWM checkpoints}"
SPLIT="${SPLIT:-test}"
EVAL_DIR="${OUT_ROOT}/eval/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
BANK_ARGS=()
WM_BATCHES=400
if [[ "${SMOKE:-0}" == "1" ]]; then BANK_ARGS=(--max_files 5 --n_boot 50); WM_BATCHES=50; fi
mkdir -p "${EVAL_DIR}"

for ck in ${CKPTS}; do
  name="$(basename "$(dirname "${ck}")")"
  python -m anchor_wm.eval_bank --ckpt "${ck}" --b0_dir "${B0_DIR}" --tfrecord_dir "${TFRECORDS}" \
    --out_dir "${EVAL_DIR}/bank_${name}" --split "${SPLIT}" "${BANK_ARGS[@]}"
done
# shellcheck disable=SC2086
python -m anchor_wm.eval_wm --cache_dir "${CACHE_DIR}" --mf_ckpt "${INIT_CKPT}" --ckpts ${CKPTS} \
  --max_batches "${WM_BATCHES}" --out "${EVAL_DIR}/forecasting.json"
echo "results in ${EVAL_DIR}"
