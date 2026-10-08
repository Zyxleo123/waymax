#!/bin/bash
# Base vs best arm (o_lpA3) side-by-side figures on scenes the arm fixed in all 4
# paired repeats. Only calls sbatch (one CPU job).
#
# Scenes: the intersection of o_lpA3's always-fixed set (proj run, term-aware) with
# the bank_honest geometry bundle, so no GPU geometry dump is needed:
#   off-road: 33174 (failure), 59208 (failure), 1236, 74274 (intermittent)
#   collision: 122120 (intermittent)
#
# Usage (from repo root; on lov4 sbatch is local):
#   bash rl/diffusion_ft/slurm/submit_plot_fix_pairs.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
unset "${!SLURM_@}" 2>/dev/null || true
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SEL=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_proj_20260930_171718_1665097/select
OUT="${DFT_FIX_OUT_DIR:-$SEL/fix_pairs_${STAMP}}"
mkdir -p "$OUT" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
{ kv DFT_FIX_SCENES_JSON "/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/bank_honest_20260925_160329_3074201/scenes/scenes_20260925_161856.json"
  kv DFT_SELECT_DIR "$SEL"
  kv DFT_FIX_OUT_DIR "$OUT"
  kv DFT_FIX_ARM "${DFT_FIX_ARM:-o_lpA3}"
  kv DFT_FIX_IDS "${DFT_FIX_IDS:-33174,59208,1236,74274,122120}"
  kv DFT_FIX_REP "${DFT_FIX_REP:-0}"; } > "$OUT/fix.env"
JID=$(sb --parsable rl/diffusion_ft/slurm/plot_fix_pairs.sbatch "$OUT/fix.env")
echo "fix pairs -> job $JID (CPU) -> $OUT"
