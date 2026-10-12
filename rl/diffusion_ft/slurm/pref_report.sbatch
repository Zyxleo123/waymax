#!/bin/bash
#SBATCH --job-name=dft_pref_report
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# One Diffusion-DPO arm's paired comparisons, then the run-wide summary table
# (submit_pref_dpo.sh step 5). Reads select rows only -> CPU partition.
#
# For each scored iteration, the arm's `ft` checkpoints (seed mean) minus:
#   base    the frozen base, same eval dir            -> compare_it<N>_vs_base.json
#   bank    the teacher b_l24_r20 (bank-BC run 1)     -> compare_it<N>_vs_bank.json
#   bcd     plain repaired-trajectory BC @150         -> compare_it<N>_vs_bcd.json
#   bcsw50  BC with half-edit batches @150            -> compare_it<N>_vs_bcsw50.json
# All four share the select_eval draws (seed 4242, 2 shards, chunk 32), which
# compare_select verifies on the base rows before it pairs anything.
#
# Config arrives as a positional env FILE (never --export).
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
RUN="${DFT_PREF_RUN:?set DFT_PREF_RUN (submit_pref_dpo.sh output dir)}"
EVAL="${DFT_PREF_EVAL:?set DFT_PREF_EVAL (the select dir of this arm)}"
ITERS="${DFT_PREF_ITERS:?set DFT_PREF_ITERS (scored iterations)}"
BCD="${DFT_BCD_RUN:?set DFT_BCD_RUN}"
BCSW="${DFT_BCSW_RUN:?set DFT_BCSW_RUN}"
SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
REPS="${DFT_SELECT_REPEATS:-4}"
echo "host=${SLURMD_NODENAME:-unknown} job=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "mem=${SLURM_MEM_PER_NODE:-?}M"
echo "eval=$EVAL iters=$ITERS"

missing=0
for t in "$EVAL"/tasks/*/; do
  [[ -s "$t/rows.jsonl" ]] || { echo "REPORT NOT RUN: no rows in $t -- check that task's .out"; missing=1; }
done
[[ "$missing" == 0 ]] || exit 0

cmp() {  # name  A_dir  arm_a  label_a  iter
  local out="$EVAL/compare_it$(printf '%03d' "$5")_vs_$1.json"
  echo; echo "===== it $5 vs $1 ====="
  "$PY" -m rl.diffusion_ft.tools.compare_select --splits_json "$SPLITS_JSON" \
    --a "$2/tasks/*/rows.jsonl" --b "$EVAL/tasks/*/rows.jsonl" \
    --arm_a "$3" --arm_b ft --label_a "$4" --label_b dpo \
    --ckpt_b "/iter_$(printf '%06d' "$5")\$" --repeats "$REPS" --out "$out" \
    || echo "compare $1 @ $5 FAILED (kept going)"
}
for it in $ITERS; do
  cmp base   "$EVAL"      base      base   "$it"
  cmp bank   "$BCD/eval"  b_l24_r20 bank   "$it"
  cmp bcd    "$BCD/eval"  ft        bcd    "$it"
  cmp bcsw50 "$BCSW/eval" ft        bcsw50 "$it"
done
echo; echo "===== run summary ====="
"$PY" -m rl.diffusion_ft.tools.pref_summary --run "$RUN"
