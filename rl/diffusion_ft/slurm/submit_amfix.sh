#!/bin/bash
# o_am failure-anatomy fixes, as an A/B against o_am in one job. Only calls sbatch.
#
# WHY (2026-10-11, memory project_o_am_pitfalls). Measured on the 10-02 o_am run:
#   * initprev never wins: 0 of 5067 switches. The comfort term pads the edit with
#     zeros at the anchor, so continuing the previous plan is billed a ~10 m/s^2
#     jump at step 1, and the fresh plan's own jump away from the car's motion is
#     free.                                                     -> join=1
#   (join=1 was the fix for this; the smoke amfix_smoke_20261011_203116_1629400
#   refuted it: it charged 81% of fully clean bases, edits jumped 24% -> 83% of
#   replans, broke success scene 91, and initprev still won 0/57. Not in this run;
#   initprev is dropped from the new arms instead -- it never wins, so dropping it
#   changes no outcome and halves the optimizer cost.)
#   * the barrier is linear in depth, and the metric is not: 171055 bought 0.7 m
#     less predicted off-road with a predicted COLLISION; 22 switches introduced a
#     violation type the base did not have.          -> w_contact_edge/agent=1
#   * the yaw the edit adds, atan(dd / max(ds, 5 cm)), lets a parked car slide and
#     spin (171055: 18 deg in one 3.6 cm step, into the parked car beside it).
#                                                               -> w_curv=0.02
#   * 80 of 107 failing episodes executed a plan their own score predicted to
#     violate: the optimizer did not reach a clean plan from its start.
#                                   -> starts initbrake, initleft, initright
# Every one is a loss term or a start of the same argmin: 0 decision rules added.
#
# ARMS (one job, paired draws, cv neighbours, base checkpoint s0 iter 110):
#   o_am    bank am   init0,initprev                          -- the shipped arm
#   o_am3   bank am3  init0                 + contact, curvature
#   o_am3s  bank am3s init0,initbrake,initleft,initright, same loss as o_am3
#
# PREDICTIONS vs o_am (term-aware): failure +0.00..+0.04 (contact and starts reach
# rows o_am executed predicted-violating); success broken <= o_am's (contact stops
# trades; curvature stops the parked-car spins). Switch rate ~o_am's: both terms
# are exactly 0 at x = 0 on a base clear of contact.
#
# CHAIN: gate (CPU: pure tests incl. the join / curvature / start claims, then the
# real-scene gate with A3-A4 for the new loss) -> select (GPU array, a6000) ->
# analysis + compares (o_am3 vs o_am, o_am3s vs o_am, o_am3s vs o_am3; legacy and
# term) + regression of o_am against the 10-02 run (want ~0: the new code is off
# by default) + ride + residual.
#
# Usage (from this repo -- a worktree is fine; on lov4 sbatch is local):
#   DFT_AMFIX_SMOKE=1 bash rl/diffusion_ft/slurm/submit_amfix.sh
#   bash rl/diffusion_ft/slurm/submit_amfix.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SMOKE="${DFT_AMFIX_SMOKE:-}"
OUT="${DFT_AMFIX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/amfix${SMOKE:+_smoke}_${STAMP}}"
VDIR=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777
SPLITS3="$VDIR/eval_splits_3strata_womd.json"
SHARD_GLOB=/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*
NB=cv
AM_CFG="lon=1,w_lon_prox=0.05,lon_back_m=1000"
FIX_CFG="${DFT_AMFIX_CFG:-w_contact_edge=1,w_contact_agent=1,w_curv=0.02}"
STARTS_S="init0,initbrake,initleft,initright"
OLD_RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_argmin_20261002_142129_3029435/select
[[ -s "$SPLITS3" ]] || { echo "FATAL: missing $SPLITS3" >&2; exit 1; }
mkdir -p "$OUT/gate" "/zfsauton2/home/yixiz/waymax_rs/logs/dft"
sb() { echo "+ sbatch $*" >&2; env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
echo "code: $ROOT @ $(git rev-parse --short HEAD) (+ $(git status --porcelain | wc -l) uncommitted paths)"
echo "out:  $OUT"

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_SHARD_GLOB "$SHARD_GLOB"
  kv DFT_LOPT_JSON "$OUT/gate/gate_lateral_opt.json"
  kv DFT_LOPT_NEIGHBORS "$NB"; kv DFT_SELECT_OPT_CFG ""
  kv DFT_LOPT_AM2_CFG "$FIX_CFG"; kv DFT_LOPT_AM2_STARTS "init0,initbrake,initleft,initright"
  kv DFT_LOPT_GATE_SCENES "$([[ -n "$SMOKE" ]] && echo 4 || echo 16)"
} > "$OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

DFT_SPLITS3_JSON="$SPLITS3" DFT_SHARD_GLOB="$SHARD_GLOB" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947/base_s0/iter_000110" \
DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,o_am=bank@am:argmin:20,o_am3=bank@am3:argmin:20,o_am3s=bank@am3s:argmin:20" \
DFT_SELECT_BANKS="am=;am3=;am3s=" \
DFT_SELECT_BANK_OPTS="am=init0,initprev;am3=init0;am3s=$STARTS_S" \
DFT_SELECT_OPT_CFG="" \
DFT_SELECT_BANK_OPT_CFGS="am=$AM_CFG;am3=$AM_CFG,$FIX_CFG;am3s=$AM_CFG,$FIX_CFG" \
DFT_SELECT_BANK_RAMPS="am=20;am3=20;am3s=20" DFT_SELECT_BANK_BLENDS="am=cubic;am3=cubic;am3s=cubic" \
DFT_SELECT_BANK_NEIGHBORS="am=$NB;am3=$NB;am3s=$NB" \
DFT_DUMP_TRAJ=1 DFT_SELECT_SMOKE="$SMOKE" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-4}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh

AID="$(cat "$OUT/select/jobid.txt")"
CIDS=()
for pair in "o_am3:o_am" "o_am3s:o_am" "o_am3s:o_am3"; do
  arm="${pair%%:*}"; ref="${pair##*:}"
  for term in "" 1; do
    f="$OUT/cmp_${arm}_vs_${ref}${term:+_term}.env"
    { kv DFT_CMP_A "$OUT/select"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
      kv DFT_CMP_ARM_A "$ref"; kv DFT_CMP_ARM_B "$arm"; kv DFT_CMP_LABEL_A "$ref"; kv DFT_CMP_LABEL_B "$arm"
      kv DFT_CMP_TERM "$term"; kv DFT_SELECT_REPEATS "$([[ -n "$SMOKE" ]] && echo 1 || echo 4)"
      kv DFT_CMP_OUT "$OUT/select/compare_${arm}_vs_${ref}${term:+_term}.json"; } > "$f"
    CIDS+=("$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")")
  done
done
echo "compare -> jobs ${CIDS[*]} (CPU, afterok:$AID) -> $OUT/select/compare_*.json"
if [[ -z "$SMOKE" ]]; then
  f="$OUT/cmp_regress.env"
  { kv DFT_CMP_A "$OLD_RUN"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_CMP_ARM_A o_am; kv DFT_CMP_ARM_B o_am; kv DFT_CMP_LABEL_A o_am_1002; kv DFT_CMP_LABEL_B o_am_now
    kv DFT_CMP_TERM 1
    kv DFT_CMP_OUT "$OUT/select/compare_regress_o_am_term.json"; } > "$f"
  XID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")
  { kv DFT_SELECT_DIR "$OUT/select"; kv DFT_RIDE_SPLITS "failure_heldout,intermittent_heldout,success_heldout"
    kv DFT_RESID_SPLITS "failure_heldout,intermittent_heldout"; } > "$OUT/post.env"
  RID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/viz_repair.sbatch "$OUT/post.env")
  DID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/residual_decomp.sbatch "$OUT/post.env")
  echo "regress -> job $XID (CPU) -> $OUT/select/compare_regress_o_am_term.json (want delta ~0)"
  echo "ride    -> job $RID   resid -> job $DID (CPU, afterok:$AID)"
fi
echo "all under $OUT; read select_summary{,_term}.json and compare_*.json"
