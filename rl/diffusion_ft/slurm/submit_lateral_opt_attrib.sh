#!/bin/bash
# Why does the loss-optimized repair beat the ramp-20 bank? SHAPE vs MARGIN, as a
# paired 2x2 in one job. Only calls sbatch.
#
# WHAT IS KNOWN (run lateral_opt_20260928_123606_44989, 4 repeats, cv neighbours):
# o_0 (one optimized edit) beats b_l24 (the ship bank) by +0.110 [0.075, 0.146]
# on failure_heldout. Two things differ at once. SHAPE: the edit is solved per
# replan instead of being one fixed curve at 24 sizes. MARGIN: the loss pushes the
# clearance out to 0.3 m edge / 0.75 m agent, while the bank executes its SMALLEST
# candidate that is clean at 0. tools/edit_margin.py on that run (job 68971) says
# margin is the live suspect: at its switches the bank's chosen clearance is p50
# 0.09 m (37% under 0.1 m to the edge), o_0's is p50 0.31 m (2% under 0.1 m); the
# optimizer's edits are modest (peak |d| p50 0.50 m, only 4% past the bank's 1.6 m),
# so RANGE is not it. That is correlational; this is the causal test.
#
# THE 2x2 (+ controls), paired in ONE job, same draws, cv neighbours:
#                 no margin (just clean)          margin (0.3 edge / 0.75 agent)
#   rigid r20     b_l24   (the ship bank)         b_l24m  (same 24 candidates;
#                                                          among clean ones, prefer
#                                                          those clearing the margin)
#   optimized     o_nm    (loss margins 0.1 m)    o_0     (loss margins 0.3 / 0.75)
#   plus base, and b_l24om = b_l24o with the margin preference (does the union's
#   deficit vanish once it stops picking barely-clean rigid shifts?).
# The preference changes only WHICH clean candidate runs, never WHETHER one runs,
# so b_l24m has exactly b_l24's switch opportunities and do-no-harm.
# o_nm uses 0.1 m, not 0: under hinge2 the comfort pull settles an edit slightly
# inside the margin, so 0 would land violating and lose coverage for a reason that
# is not margin. 0.1 m is about the bank's typical realised margin (p50 0.09 m).
#
# PREDICTIONS, written before the run (failure_heldout solve; last run: base 0.286,
# b_l24 0.708, o_0 0.818):
#   margin explains it  -> b_l24m ~0.78-0.82 and o_nm ~0.71-0.75;
#   shape explains it   -> b_l24m ~0.71 and o_nm ~0.81.
#   My call: mostly margin -- b_l24m 0.76-0.81, o_nm 0.72-0.77, b_l24om ~ b_l24m.
#   If b_l24m ties o_0, the shipping choice is the SIMPLER one (bank + preference,
#   no optimizer, same compute as today's bank).
#
# CHAIN: gate (CPU, gate_lateral_opt.sbatch: pure tests incl. prefer_k + real-scene
# plumbing) -> select (GPU array, afterok, reads the verdict) -> select_analysis +
# ride quality + residual + edit_margin (CPU, each afterok on the array).
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   DFT_LOPT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_lateral_opt_attrib.sh
#   bash rl/diffusion_ft/slurm/submit_lateral_opt_attrib.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
# Submitting from inside an allocation must not leak its SLURM_* into the jobs.
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_LOPT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_attrib_${STAMP}}"
SPLITS3=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
NB="${DFT_LOPT_NEIGHBORS:-cv}"
OPT_CFG="${DFT_SELECT_OPT_CFG:-}"
NM="${DFT_LOPT_NOMARGIN_M:-0.1}"
PREF="${DFT_LOPT_PREFER:-0.3,0.75}"
case "$NB" in log|cv|none) ;; *) echo "FATAL: DFT_LOPT_NEIGHBORS=$NB not log|cv|none" >&2; exit 1;; esac
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_LOPT_JSON "$OUT/gate/gate_lateral_opt.json"
  kv DFT_LOPT_NEIGHBORS "$NB"; kv DFT_SELECT_OPT_CFG "$OPT_CFG"
  kv DFT_LOPT_GATE_SCENES "$([[ -n "${DFT_LOPT_SMOKE:-}" ]] && echo 4 || echo 16)"
} > "$OUT/gate/gate.env"
echo "+ sbatch --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch $OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
BANKS=(l24 l24m o0 onm l24om)
per() { local s=""; for b in "${BANKS[@]}"; do s+="${s:+;}$b=$1"; done; echo "$s"; }
# No fine-tune arm, so this checkpoint dir only names the task dirs (its
# tparams.npz went in the 09-28 cleanup; nothing loads it).
OLD=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,b_l24=bank@l24:cons:20,b_l24m=bank@l24m:cons:20,o_0=bank@o0:cons:20,o_nm=bank@onm:cons:20,b_l24om=bank@l24om:cons:20" \
DFT_SELECT_BANKS="l24=${L24};l24m=${L24};o0=;onm=;l24om=${L24}" \
DFT_SELECT_BANK_OPTS="o0=0;onm=0;l24om=best,0" \
DFT_SELECT_OPT_CFG="$OPT_CFG" \
DFT_SELECT_BANK_OPT_CFGS="onm=edge_margin_m=${NM},agent_margin_m=${NM}" \
DFT_SELECT_BANK_PREFER="l24m=${PREF};l24om=${PREF}" \
DFT_SELECT_BANK_RAMPS="$(per 20)" \
DFT_SELECT_BANK_BLENDS="$(per cubic)" \
DFT_SELECT_BANK_NEIGHBORS="$(per "$NB")" \
DFT_DUMP_TRAJ=1 DFT_SELECT_SMOKE="${DFT_LOPT_SMOKE:-}" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-4}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh

AID="$(cat "$OUT/select/jobid.txt")"
{ kv DFT_SELECT_DIR "$OUT/select"; kv DFT_RIDE_SPLITS "failure_heldout,intermittent_heldout"
  kv DFT_RESID_SPLITS "failure_heldout,intermittent_heldout"
  kv DFT_EM_SPLITS "failure_heldout,intermittent_heldout"; } > "$OUT/post.env"
RID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/viz_repair.sbatch "$OUT/post.env")
DID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/residual_decomp.sbatch "$OUT/post.env")
MID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/edit_margin.sbatch "$OUT/post.env")
echo "ride    -> job $RID (CPU, afterok:$AID) -> $OUT/select/ride_*.json"
echo "resid   -> job $DID (CPU, afterok:$AID) -> $OUT/select/residual_*.json"
echo "margin  -> job $MID (CPU, afterok:$AID) -> $OUT/select/edit_margin_*.json"
echo "all under $OUT; read $OUT/select/select_summary.json and edit_margin_failure_heldout.json"
