#!/bin/bash
# The fixed-clock repair bank, scene by scene: recovered / still failing /
# regressed / still success, in the layout of the 09-18 scene page. Only calls sbatch.
#
# Reads the honest-bank A/B run (submit_bank_honest.sh), which dumps executed paths
# for base, b_oracle (the bank as measured, reading the other cars' recorded
# future), b_cv (the bank predicting cars by constant velocity) and b_edge (road
# edges only). Kinds are judged on DFT_SOLVED_ARM (default b_oracle) vs base; every
# card draws all four paths, toggleable, so each scene also shows what the bank
# does WITHOUT the oracle.
#
# Scenes are picked automatically (dump_reach_from_select --auto_pick): up to N per
# kind, diversified round-robin over (split, failure cause), deterministic. On the
# 09-24 fixed-clock eval the populations were 139 recovered / 20 still failing /
# 2 regressed / 1081 still success.
#
# CHAIN: dump (GPU, afterok on the honest-bank select array) -> page (CPU).
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_viz_bank_honest.sh <honest-bank run dir>
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
RUN="${1:?usage: submit_viz_bank_honest.sh <select/bank_honest_* run dir>}"
SEL="$RUN/select"
[[ -s "$SEL/jobid.txt" ]] || { echo "FATAL: no $SEL/jobid.txt" >&2; exit 1; }
grep -q "^DFT_DUMP_TRAJ='1'$" "$SEL/env/0.env" || {
  echo "FATAL: $SEL was not configured with DFT_DUMP_TRAJ=1 -- no paths to draw" >&2; exit 1; }
AID="$(cat "$SEL/jobid.txt")"
OUT_DIR="$RUN/scenes"; mkdir -p "$OUT_DIR" "$ROOT/logs/dft"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="$OUT_DIR/scenes_${STAMP}.env"
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
{ kv DFT_SELECT_DIR "$SEL"
  kv DFT_SPLITS_JSON /zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
  kv DFT_OUT_JSON "$OUT_DIR/scenes_${STAMP}.json"
  kv DFT_OUT_HTML "$OUT_DIR/scenes_${STAMP}.html"
  kv DFT_VIZ_ARMS "base,b_oracle,b_cv,b_edge"
  kv DFT_SHOW_ARMS "${DFT_SHOW_ARMS:-}"
  kv DFT_SOLVED_ARM "${DFT_SOLVED_ARM:-b_oracle}"
  kv DFT_AUTO_PICK "${DFT_AUTO_PICK:-6}"; } > "$ENVF"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
DID=$(sb --parsable --dependency="afterok:${AID}" --exclude=gpu28 \
  rl/diffusion_ft/slurm/dump_reach_from_select.sbatch "$ENVF")
VID=$(sb --parsable --dependency="afterok:${DID}" rl/diffusion_ft/slurm/viz_scenes.sbatch "$ENVF")
echo "dump -> job $DID (GPU, afterok:$AID)"
echo "page -> job $VID (CPU, afterok:$DID) -> $OUT_DIR/scenes_${STAMP}.html"
