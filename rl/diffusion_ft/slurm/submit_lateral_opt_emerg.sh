#!/bin/bash
# What o_lpA3 still fails, and two levers aimed at it: a 2x2 of EMERGENCY TIERS x
# TRIGGER THETA, paired in one job. Only calls sbatch.
#
# WHAT IS LEFT (lateral_opt_proj_20260930_171718, o_lpA3, failure_heldout, term-aware,
# tools/failure_anatomy.py jobs 69429/69436): 71 episodes from ~30 scenes --
#   off-road, no clean candidate 20: the optimizer KNOWS it has not cleared (loss
#     0.63 -> 0.23, never < 0.01) and stops at peak |d| 0.69 m (p90 2.9 m = its
#     whole reach): comfort outbids the saturating barrier. Exit is 0.10 m away at
#     every onset, but the plan runs ~1 m off over the window.
#   side / rear / front collisions, no clean candidate 11 / 10 / 5: 90% of the rear
#     cases' lon edits sit at the 0.9 m braking cap (a car closing from behind-side;
#     the fix is to let it pass); intermittent's front cases are a STOPPED car ahead
#     at 6 m/s -- ~5 m of braking, not 0.9.
#   unpredicted collisions 9 (18 intermittent): predicted gap p50 0.35-0.47 m, just
#     above theta 0.3.
#   no goal 13 (4 scenes): pace, not edit space -- out of scope here.
#
# THE 2x2 (+ base), paired draws, cv neighbours, all `proj`:
#                     bank olp (o_lpA3's)          + emergency tiers (olpe)
#   theta 0.3         o_lpA3  (the arm to beat)    o_lpA3e
#   theta 0.5         o_lpA5                       o_lpA5e
# Emergency tiers (lateral_opt.variant_cfg), ranked below every ordinary candidate
# and run only where the base is predicted to VIOLATE:
#   emerg -- the lon optimizer with comfort weights x0.1, step 0.5 m, 48 iters;
#   brake -- the same, allowed 4 m behind the plan with the progress tolerance
#            widened to match, and only where the violation is a car's and the road
#            edge is clean (so braking cannot slide a static edge past the window).
#
# PREDICTIONS vs o_lpA3: emergency tiers failure +0.01..+0.03 (off-road + front/rear
# no-clean), intermittent +0.005..+0.02; rougher ride on the episodes they fire in;
# no-goal may tick up (brake). theta 0.5: failure +0.00..+0.01, intermittent
# +-0.01, more success edits (the "always" arm lost intermittent).
#
# CHAIN: gate (CPU: pure tests incl. the emergency tokens + E1-E3 real-scene
# checks) -> select (GPU array, a6000) -> analysis (legacy + term) + compares vs
# o_lpA3 + regression of o_lpA3 vs the proj run + ride (3 splits) + residual + margin.
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   DFT_LOPT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_lateral_opt_emerg.sh
#   bash rl/diffusion_ft/slurm/submit_lateral_opt_emerg.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_LOPT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_emerg_${STAMP}}"
SPLITS3=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
NB="${DFT_LOPT_NEIGHBORS:-cv}"
case "$NB" in log|cv|none) ;; *) echo "FATAL: DFT_LOPT_NEIGHBORS=$NB not log|cv|none" >&2; exit 1;; esac
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
SMOKE="${DFT_LOPT_SMOKE:-}"
REF_ARM=o_lpA3

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_LOPT_JSON "$OUT/gate/gate_lateral_opt.json"
  kv DFT_LOPT_NEIGHBORS "$NB"; kv DFT_SELECT_OPT_CFG ""
  kv DFT_LOPT_GATE_SCENES "$([[ -n "$SMOKE" ]] && echo 4 || echo 16)"
} > "$OUT/gate/gate.env"
echo "+ sbatch --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch $OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

OLD=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,o_lpA3=bank@olp:proj:20:0.3,o_lpA3e=bank@olpe:proj:20:0.3,o_lpA5=bank@olp:proj:20:0.5,o_lpA5e=bank@olpe:proj:20:0.5" \
DFT_SELECT_BANKS="olp=;olpe=" \
DFT_SELECT_BANK_OPTS="olp=0,lon0,prev,lonprev;olpe=0,lon0,prev,lonprev,emerg,brake" \
DFT_SELECT_OPT_CFG="" DFT_SELECT_BANK_OPT_CFGS="" \
DFT_SELECT_BANK_RAMPS="olp=20;olpe=20" DFT_SELECT_BANK_BLENDS="olp=cubic;olpe=cubic" \
DFT_SELECT_BANK_NEIGHBORS="olp=$NB;olpe=$NB" \
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
for arm in o_lpA3e o_lpA5 o_lpA5e; do
  for term in "" 1; do
    f="$OUT/cmp_${arm}${term:+_term}.env"
    { kv DFT_CMP_A "$OUT/select"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
      kv DFT_CMP_ARM_A "$REF_ARM"; kv DFT_CMP_ARM_B "$arm"; kv DFT_CMP_LABEL_A "$REF_ARM"; kv DFT_CMP_LABEL_B "$arm"
      kv DFT_CMP_TERM "$term"; kv DFT_SELECT_REPEATS "$([[ -n "$SMOKE" ]] && echo 1 || echo 4)"
      kv DFT_CMP_OUT "$OUT/select/compare_${arm}_vs_${REF_ARM}${term:+_term}.json"; } > "$f"
    CIDS+=("$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")")
  done
done
echo "compare -> jobs ${CIDS[*]} (CPU, afterok:$AID) -> $OUT/select/compare_*_vs_${REF_ARM}*.json"
if [[ -z "$SMOKE" ]]; then
  REF="${DFT_LOPT_REF:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_proj_20260930_171718_1665097/select}"
  f="$OUT/cmp_regress.env"
  { kv DFT_CMP_A "$REF"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_CMP_ARM_A "$REF_ARM"; kv DFT_CMP_ARM_B "$REF_ARM"; kv DFT_CMP_LABEL_A proj; kv DFT_CMP_LABEL_B emerg
    kv DFT_CMP_OUT "$OUT/select/compare_regress_${REF_ARM}.json"; } > "$f"
  XID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")
  echo "regress -> job $XID (CPU) -> $OUT/select/compare_regress_${REF_ARM}.json (want delta ~0)"
fi
echo "ride    -> job $RID   resid -> job $DID   margin -> job $MID (CPU, afterok:$AID)"
echo "all under $OUT; read select_summary{,_term}.json and compare_*_vs_${REF_ARM}*.json"
