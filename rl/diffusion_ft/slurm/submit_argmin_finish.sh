#!/bin/bash
# Finish the no-rule repair A/B (o_am / o_amc vs o_lpA3) that job 71188 left
# incomplete, and render its before/after figures. Only calls sbatch.
#
# 71188 shard 3 died on 2026-10-02 in intermittent_heldout (69 scenes) when the
# eshau shards vanished; every other split and shard finished. select_eval.sbatch
# now defaults to /zfsauton/datasets/womd (same 1000 files, offsets verified
# 10-03), and the runner skips finished chunks, so re-running array task 3 with
# its ORIGINAL env file resumes it: same seed, same fixed chunk slices, same arms.
#
#   now:      analysis on the rows already there (complete scenes only)
#             + figures for the two failure scenes already scored (33174, 59208)
#   resume:   array task 3 (GPU, a6000; gpu28 excluded as in submit_select_eval.sh)
#   afterok:  analysis again, the six paired compares, figures for all five scenes
#
# Usage (from repo root; on lov4 sbatch is local):
#   bash rl/diffusion_ft/slurm/submit_argmin_finish.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
unset "${!SLURM_@}" 2>/dev/null || true
RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_argmin_20261002_142129_3029435
SEL="$RUN/select"
BUNDLE=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/bank_honest_20260925_160329_3074201/scenes/scenes_20260925_161856.json
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ARM="${DFT_FIX_ARM:-o_am}"
mkdir -p "$ROOT/logs/dft"
sb() { echo "+ sbatch $*" >&2; env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

[[ -s "$SEL/env/3.env" && -s "$SEL/analyze.env" ]] || { echo "FATAL: $SEL is missing env/3.env or analyze.env" >&2; exit 1; }
grep -q '^PASS' "$RUN/gate/verdict.txt" || { echo "FATAL: gate verdict is not PASS" >&2; exit 1; }

fix_env() {  # <out_dir> <ids>
  mkdir -p "$1"
  { kv DFT_FIX_SCENES_JSON "$BUNDLE"; kv DFT_SELECT_DIR "$SEL"; kv DFT_FIX_OUT_DIR "$1"
    kv DFT_FIX_ARM "$ARM"; kv DFT_FIX_IDS "$2"; kv DFT_FIX_REP "${DFT_FIX_REP:-0}"; } > "$1/fix.env"
  echo "$1/fix.env"
}

# ---- now: partial analysis + the two figures whose rows exist -----------------
A0=$(sb --parsable rl/diffusion_ft/slurm/select_analysis.sbatch "$SEL/analyze.env")
F0ENV=$(fix_env "$SEL/fix_pairs_${ARM}_partial_${STAMP}" "33174,59208")
F0=$(sb --parsable rl/diffusion_ft/slurm/plot_fix_pairs.sbatch "$F0ENV")

# ---- resume shard 3 -------------------------------------------------------------
R=$(sb --parsable --array=3 --exclude=gpu28 --job-name=dft_select \
      rl/diffusion_ft/slurm/select_eval.sbatch "$SEL/env")
R="${R%%;*}"

# ---- after the resume -------------------------------------------------------------
A1=$(sb --parsable --dependency="afterok:$R" rl/diffusion_ft/slurm/select_analysis.sbatch "$SEL/analyze.env")
CMPS=()
for c in o_am_vs_o_lpA3 o_amc_vs_o_lpA3 o_amc_vs_o_am; do
  for t in "" _term; do
    CMPS+=("$(sb --parsable --dependency="afterok:$R" rl/diffusion_ft/slurm/compare_select.sbatch "$RUN/cmp_${c}${t}.env")")
  done
done
F1ENV=$(fix_env "$SEL/fix_pairs_${ARM}_${STAMP}" "33174,59208,1236,74274,122120")
F1=$(sb --parsable --dependency="afterok:$R" rl/diffusion_ft/slurm/plot_fix_pairs.sbatch "$F1ENV")

{ echo "stamp=$STAMP"
  echo "now:     analysis $A0, figures $F0 -> $(dirname "$F0ENV")"
  echo "resume:  select array task 3 -> $R  (log logs/dft/slurm-dft_select-${R}.out)"
  echo "afterok: analysis $A1, compares ${CMPS[*]}, figures $F1 -> $(dirname "$F1ENV")"
} | tee "$RUN/finish_${STAMP}.txt"
