#!/bin/bash
# What o_0 still fails, and two repairs aimed at it: a 2x2 of PREV x LON, paired
# in one job. Only calls sbatch.
#
# WHAT IS LEFT (tools/failure_anatomy.py on lateral_opt_attrib_20260930_000641,
# o_0, failure_heldout, 692 episodes; counted only up to the step the env
# terminated the row -- see the ACCOUNTING note below): 118 unsolved =
#   63 overlap -- collider REAR 31 (ego lags its own log by p50 2.9 m / 7 steps,
#               the log-replayed follower does 6.7 m/s to the ego's 4.3), side 21,
#               front 11; 45 of the 63 were predicted and o_0 had no clean edit,
#               18 were never predicted (constant-velocity misses);
#   44 off-road -- shallow: at every swept onset an on-road pose is p50 0.10 m
#               (p90 0.23 m) away, never trapped; but onsets bunch in the first
#               1-3 steps after a replan (a fresh sample jumps back into the
#               hazard), where a smooth edit pinned at the anchor cannot move
#               10 cm in time. In 19 of the 44 the PREVIOUS replan's executed plan
#               was certified clean over exactly those steps;
#   11 no goal.
# Intermittent is the same shape (117: 66 overlap, 40 of them rear; 36 off-road).
#
# THE 2x2 (+ base), paired draws, cv neighbours, all `cons`. Every arm keeps o_0's
# own edit as its FIRST choice; the additions are extra candidates ranked behind it
# (proposal_select.OPT_TIER_M), so wherever o_0 fires they execute o_0's edit:
#                   o_0's edit only        + lon edit of the fresh sample
#   fresh only      o_0  ("0")             o_l  ("0,lon0")
#   + prev plan     o_p  ("0,prev")        o_lp ("0,lon0,prev,lonprev")
# `prev` = the same optimizer run on the tail of the plan executed at the previous
# replan (lateral_opt.parse_inits). `lon` = a second profile s(t), metres along the
# path (branching.path_retime), no more than lon_back_m = 0.9 behind the base so a
# braking edit stays inside cons's 1 m progress tolerance; speeding up is free of
# that bound. lon is NOT a replacement for the lateral edit: in the real-scene gate
# (09-30) lon alone repaired fewer lateral distortions (0.58 vs 0.74) by braking
# into the hazard later instead of clearing it.
#
# PREDICTIONS (failure_heldout, legacy solve; last two runs o_0 0.818 / 0.819):
#   o_p  +0.01..+0.03 (<= 19 off-road episodes addressable, some re-fail later);
#   o_l  +0.02..+0.05 (45 predicted collisions, rear-ends need a speed-up of
#        ~2 m/s held over the window, which the comfort terms charge);
#   o_lp roughly additive. Do-no-harm: success brk stays ~0 (both are still cons).
# A REGRESSION CHECK comes free: o_0 here runs the old code path with the same
# seed, shards and chunks as the attrib run, so its rows should match that run's
# o_0 rows (up to GPU nondeterminism).
#
# ACCOUNTING. The runner keeps driving a terminated row (goal reached, or a
# collision) with the base proposal until every row in its batch is done, and ORs
# the flags from all of it into `solved`. Rows now also carry `*_term` flags that
# stop at termination; select_analysis.sbatch writes select_summary_term.json from
# them. The legacy summary stays the headline for comparability.
#
# CHAIN: gate (CPU: pure tests incl. lon, then the real-scene gate with the
# slow/fast re-timing distortions and the L1-L4 lon checks) -> select (GPU array,
# a6000, afterok, reads the verdict) -> analysis (legacy + term) + ride + residual
# + edit_margin (CPU, afterok on the array).
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   DFT_LOPT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_lateral_opt_lonprev.sh
#   bash rl/diffusion_ft/slurm/submit_lateral_opt_lonprev.sh
#   DFT_LOPT_ARMS="o_l o_lp" bash ...   # a subset (base and o_0 always run, for
#                                       # pairing and the regression check)
#
# 2026-09-30: the first full run (lateral_opt_lonprev_20260930_143816) had a NaN
# lon gradient (branching.lateral_offset's norm at zero-length steps) -- its o_l /
# o_lp lon candidates never moved; its o_0 / o_p are valid. Fixed; rerun o_l o_lp.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
# Submitting from inside an allocation must not leak its SLURM_* into the jobs.
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_LOPT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_lonprev_${STAMP}}"
SPLITS3=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
NB="${DFT_LOPT_NEIGHBORS:-cv}"
OPT_CFG="${DFT_SELECT_OPT_CFG:-}"
# Optional overrides for the lon candidates' weights (w_lon_acc=..,lon_back_m=..);
# NOT lon=1, which would switch every candidate of the bank to lon.
LON_CFG="${DFT_LOPT_LON_CFG:-}"
[[ "$LON_CFG" != *lon=1* ]] || { echo "FATAL: DFT_LOPT_LON_CFG must not set lon=1 (use the lon0/lonprev tokens)" >&2; exit 1; }
case "$NB" in log|cv|none) ;; *) echo "FATAL: DFT_LOPT_NEIGHBORS=$NB not log|cv|none" >&2; exit 1;; esac
[[ "$OPT_CFG" != *lon=1* ]] || { echo "FATAL: lon goes per bank (DFT_LOPT_LON_CFG), not in the global opt cfg" >&2; exit 1; }
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

declare -A OPTS=([o_p]="op=0,prev" [o_l]="ol=0,lon0" [o_lp]="olp=0,lon0,prev,lonprev")
ARMS_EXTRA=(${DFT_LOPT_ARMS:-o_p o_l o_lp})
BANKS=(o0); ARM_SPEC="base,o_0=bank@o0:cons:20"; OPT_SPEC="o0=0"; BANK_SPEC="o0="
for a in "${ARMS_EXTRA[@]}"; do
  [[ -n "${OPTS[$a]:-}" ]] || { echo "FATAL: unknown arm '$a' (o_p o_l o_lp)" >&2; exit 1; }
  b="${OPTS[$a]%%=*}"
  BANKS+=("$b"); ARM_SPEC+=",$a=bank@$b:cons:20"; OPT_SPEC+=";${OPTS[$a]}"; BANK_SPEC+=";$b="
done
per() { local s=""; for b in "${BANKS[@]}"; do s+="${s:+;}$b=$1"; done; echo "$s"; }
# No fine-tune arm, so this checkpoint dir only names the task dirs (its
# tparams.npz went in the 09-28 cleanup; nothing loads it).
OLD=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="$ARM_SPEC" \
DFT_SELECT_BANKS="$BANK_SPEC" \
DFT_SELECT_BANK_OPTS="$OPT_SPEC" \
DFT_SELECT_OPT_CFG="$OPT_CFG" \
DFT_SELECT_BANK_OPT_CFGS="$(if [[ -n "$LON_CFG" ]]; then for b in "${BANKS[@]:1}"; do printf '%s' "${sep:-}$b=$LON_CFG"; sep=';'; done; fi)" \
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
# Paired arm-vs-o_0 contrasts inside this run (tools/compare_select.py, A == B),
# on the legacy flags and on the termination-aware ones.
CIDS=()
for arm in "${ARMS_EXTRA[@]}"; do
  for term in "" 1; do
    f="$OUT/cmp_${arm}${term:+_term}.env"
    { kv DFT_CMP_A "$OUT/select"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
      kv DFT_CMP_ARM_A o_0; kv DFT_CMP_ARM_B "$arm"; kv DFT_CMP_LABEL_A o_0; kv DFT_CMP_LABEL_B "$arm"
      kv DFT_CMP_TERM "$term"
      kv DFT_SELECT_REPEATS "$([[ -n "${DFT_LOPT_SMOKE:-}" ]] && echo 1 || echo 4)"
      kv DFT_CMP_OUT "$OUT/select/compare_${arm}_vs_o_0${term:+_term}.json"; } > "$f"
    CIDS+=("$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")")
  done
done
echo "compare -> jobs ${CIDS[*]} (CPU, afterok:$AID) -> $OUT/select/compare_*_vs_o_0*.json"
# REGRESSION: this run's o_0 against the attrib run's o_0 (same seed, shards and
# chunks; compare_select refuses if their base rows are not the same draws).
if [[ -z "${DFT_LOPT_SMOKE:-}" ]]; then
  REF="${DFT_LOPT_REF:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_attrib_20260930_000641_1279237/select}"
  f="$OUT/cmp_regress.env"
  { kv DFT_CMP_A "$REF"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_CMP_ARM_A o_0; kv DFT_CMP_ARM_B o_0; kv DFT_CMP_LABEL_A attrib; kv DFT_CMP_LABEL_B lonprev
    kv DFT_CMP_OUT "$OUT/select/compare_regress_o_0.json"; } > "$f"
  XID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")
  echo "regress -> job $XID (CPU) -> $OUT/select/compare_regress_o_0.json (want delta ~0)"
fi
echo "ride    -> job $RID (CPU, afterok:$AID) -> $OUT/select/ride_*.json"
echo "resid   -> job $DID (CPU, afterok:$AID) -> $OUT/select/residual_*.json"
echo "margin  -> job $MID (CPU, afterok:$AID) -> $OUT/select/edit_margin_*.json"
echo "all under $OUT; read $OUT/select/select_summary.json and select_summary_term.json"
