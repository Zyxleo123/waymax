#!/bin/bash
# Always-on projection: run o_lp's optimizer on CLEAN base proposals too, not only
# where one is predicted to violate. Only calls sbatch.
#
# THE IDEA (user, 2026-09-30). The loss is a hinge barrier: exactly zero -- no
# gradient, the optimizer returns the base line bit for bit -- wherever every
# corner is beyond 0.3 m of the road edge and every car beyond 0.75 m. So running
# it everywhere leaves comfortable plans alone and pushes TIGHT ones outward, which
# might catch the collisions `cons` never sees because constant-velocity
# prediction called them clear.
#
# MEASURED BEFORE BUILDING (rerun lateral_opt_lonprev_20260930_151623, o_lp):
#   catch -- o_lp's never-predicted collisions: at the fatal replan the predicted
#            car gap was < 0.75 m in 15/16 (failure) and 20/30 (intermittent),
#            < 0.3 m in 10 and 11. The barrier IS active there.
#   cost  -- 7.1% of clean success_heldout replans are inside the margin (5.7% for
#            cars), and 17.3% of success episodes have one. Those get edited, so
#            do-no-harm is no longer structural; `broken` is the number to watch.
#
# ARMS, one shared bank `olp` ("0,lon0,prev,lonprev"), paired draws, cv neighbours:
#   o_lp    cons              -- the reference (the current best arm)
#   o_lpA3  proj, theta 0.3   -- fresh-sample optimizer also on clean rows whose
#                                predicted clearance is < 0.3 m
#   o_lpA   proj, no theta    -- ... on every alive row (loss decides)
# `proj` executes an edit on a clean row only if it moved, is predicted clean,
# keeps progress within tol, and is >= PROJ_MIN_GAIN_M (1 cm) clearer than the
# base; the `prev` candidates stay violation-only, so a zero-loss plan is never
# swapped for the old one (proposal_select.select_switch / _optimized).
#
# PREDICTIONS (vs o_lp): failure +0.00..+0.02, intermittent +0.01..+0.03 (up to
# ~35 catchable episodes across the two); success broken 0.000-0.003 per episode
# -- and since successes are ~98.5% of the corpus, m3 turns on that: 0.001 broken
# on success costs about as much as the whole failure-split gain buys.
#
# CHAIN: gate (CPU: pure tests incl. the proj rule, real-scene gate) -> select
# (GPU array, a6000) -> analysis (legacy + term) + compares vs o_lp + regression of
# o_lp vs the 151623 rerun + ride + residual + edit_margin (CPU).
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   DFT_LOPT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_lateral_opt_proj.sh
#   bash rl/diffusion_ft/slurm/submit_lateral_opt_proj.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_LOPT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_proj_${STAMP}}"
SPLITS3=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
NB="${DFT_LOPT_NEIGHBORS:-cv}"
THETA="${DFT_PROJ_THETA:-0.3}"
case "$NB" in log|cv|none) ;; *) echo "FATAL: DFT_LOPT_NEIGHBORS=$NB not log|cv|none" >&2; exit 1;; esac
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
SMOKE="${DFT_LOPT_SMOKE:-}"

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_LOPT_JSON "$OUT/gate/gate_lateral_opt.json"
  kv DFT_LOPT_NEIGHBORS "$NB"; kv DFT_SELECT_OPT_CFG ""
  kv DFT_LOPT_GATE_SCENES "$([[ -n "$SMOKE" ]] && echo 4 || echo 16)"
} > "$OUT/gate/gate.env"
echo "+ sbatch --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch $OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

OLD=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,o_lp=bank@olp:cons:20,o_lpA3=bank@olp:proj:20:${THETA},o_lpA=bank@olp:proj:20" \
DFT_SELECT_BANKS="olp=" \
DFT_SELECT_BANK_OPTS="olp=0,lon0,prev,lonprev" \
DFT_SELECT_OPT_CFG="" DFT_SELECT_BANK_OPT_CFGS="" \
DFT_SELECT_BANK_RAMPS="olp=20" DFT_SELECT_BANK_BLENDS="olp=cubic" \
DFT_SELECT_BANK_NEIGHBORS="olp=$NB" \
DFT_DUMP_TRAJ=1 DFT_SELECT_SMOKE="$SMOKE" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-4}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh

AID="$(cat "$OUT/select/jobid.txt")"
{ kv DFT_SELECT_DIR "$OUT/select"; kv DFT_RIDE_SPLITS "failure_heldout,intermittent_heldout,success_heldout"
  kv DFT_RESID_SPLITS "failure_heldout,intermittent_heldout"
  kv DFT_EM_SPLITS "failure_heldout,intermittent_heldout"; } > "$OUT/post.env"
RID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/viz_repair.sbatch "$OUT/post.env")
DID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/residual_decomp.sbatch "$OUT/post.env")
MID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/edit_margin.sbatch "$OUT/post.env")
CIDS=()
for arm in o_lpA3 o_lpA; do
  for term in "" 1; do
    f="$OUT/cmp_${arm}${term:+_term}.env"
    { kv DFT_CMP_A "$OUT/select"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
      kv DFT_CMP_ARM_A o_lp; kv DFT_CMP_ARM_B "$arm"; kv DFT_CMP_LABEL_A o_lp; kv DFT_CMP_LABEL_B "$arm"
      kv DFT_CMP_TERM "$term"; kv DFT_SELECT_REPEATS "$([[ -n "$SMOKE" ]] && echo 1 || echo 4)"
      kv DFT_CMP_OUT "$OUT/select/compare_${arm}_vs_o_lp${term:+_term}.json"; } > "$f"
    CIDS+=("$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")")
  done
done
echo "compare -> jobs ${CIDS[*]} (CPU, afterok:$AID) -> $OUT/select/compare_*_vs_o_lp*.json"
if [[ -z "$SMOKE" ]]; then
  # REGRESSION: o_lp here (a shared bank, new selector code) vs o_lp in the 151623 rerun.
  REF="${DFT_LOPT_REF:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_lonprev_20260930_151623_1595722/select}"
  f="$OUT/cmp_regress.env"
  { kv DFT_CMP_A "$REF"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_CMP_ARM_A o_lp; kv DFT_CMP_ARM_B o_lp; kv DFT_CMP_LABEL_A rerun; kv DFT_CMP_LABEL_B proj
    kv DFT_CMP_OUT "$OUT/select/compare_regress_o_lp.json"; } > "$f"
  XID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")
  echo "regress -> job $XID (CPU) -> $OUT/select/compare_regress_o_lp.json (want delta ~0)"
fi
echo "ride    -> job $RID   resid -> job $DID   margin -> job $MID (CPU, afterok:$AID)"
echo "all under $OUT; read select_summary{,_term}.json and compare_*_vs_o_lp*.json"
