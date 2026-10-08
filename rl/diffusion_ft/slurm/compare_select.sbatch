#!/bin/bash
#SBATCH --job-name=dft_selcmp
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Paired B-minus-A between the fine-tune arms of two select_eval runs that share
# draws (tools/compare_select.py). CPU only; re-runnable from its env file:
#   sbatch rl/diffusion_ft/slurm/compare_select.sbatch <compare.env>
# Gate inside the job: if either run left no rows, say which and exit 0.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
A="${DFT_CMP_A:?set DFT_CMP_A (select dir of the reference run)}"
B="${DFT_CMP_B:?set DFT_CMP_B (select dir of the candidate run)}"
SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"

missing=0
for d in "$A" "$B"; do
  for t in "$d"/tasks/*/; do
    [[ -s "$t/rows.jsonl" ]] || {
      echo "COMPARISON NOT RUN: no rows in $t -- check that task's .out"; missing=1; }
  done
done
[[ "$missing" == 0 ]] || exit 0

"$PY" -m rl.diffusion_ft.tools.compare_select --splits_json "$SPLITS_JSON" \
  --a "$A/tasks/*/rows.jsonl" --b "$B/tasks/*/rows.jsonl" \
  --label_a "${DFT_CMP_LABEL_A:-a}" --label_b "${DFT_CMP_LABEL_B:-b}" \
  --arm_a "${DFT_CMP_ARM_A:-}" --arm_b "${DFT_CMP_ARM_B:-}" \
  --ckpt_a "${DFT_CMP_CKPT_A:-}" --ckpt_b "${DFT_CMP_CKPT_B:-}" \
  --repeats "${DFT_SELECT_REPEATS:-4}" --out "${DFT_CMP_OUT:-$B/compare.json}" \
  ${DFT_CMP_TERM:+--term}
