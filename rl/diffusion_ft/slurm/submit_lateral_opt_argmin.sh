#!/bin/bash
# One objective, no decision rules: the argmin arm against o_lpA3's decision tree,
# and the certificate as the single rule under test. Only calls sbatch.
#
# WHY (user, 2026-09-30: "we don't want to make too many rules. I think no rule is
# preferred"). o_lpA3 decides through: a cons/proj trigger + theta, a 1 cm min
# gain, a "moved" check, tier offsets (fresh lateral < lon < prev lateral < prev
# lon), a 0.9 m braking cap tied to a 1 m progress tolerance. The argmin arm
# replaces all of it with ONE optimization per replan on every alive row:
#   objective -- o_lpA3's own: safety barrier (hinge2, 0.3 m edge / 0.75 m agent)
#                + comfort of the edit (lateral AND along-path, together), plus
#                w_lon_prox 0.05 * mean(s_t^2): a cost on leaving the policy's
#                timing that REPLACES the braking cap and the progress tolerance;
#   starts    -- the fresh plan (init0) and the previous plan's tail projected
#                onto it (initprev, lateral_opt.project_onto_path): the same loss
#                from two inits, so their final losses compare directly;
#   decision  -- execute the per-row argmin wherever it moved. A plan with zero
#                loss returns x = 0 exactly and is left alone with no rule.
#
# ARMS (one job, paired draws, cv neighbours):
#   o_lpA3  bank olp "0,lon0,prev,lonprev", proj theta 0.3   -- the arm to beat
#   o_am    bank am  "init0,initprev", argmin                -- 0 decision rules
#   o_amc   bank am, argmin_c                                -- + the certificate:
#           never replace a base predicted clean by a plan predicted to violate
#
# PREDICTIONS vs o_lpA3: failure -0.02..+0.03 (argmin can trade a predicted
# violation for a lower loss where the tiers insisted on a certified edit, and it
# can brake past 0.9 m where needed); intermittent -0.01..+0.02; success broken is
# the risk for o_am (no certificate). o_amc - o_am >= 0 on success; if o_am ties
# o_amc, the certificate goes too.
#
# CHAIN: gate (CPU: pure tests incl. projection round trip + A1-A2 real-scene) ->
# select (GPU array, a6000) -> analysis (legacy + term) + compares (o_am, o_amc vs
# o_lpA3; o_amc vs o_am) + regression of o_lpA3 vs the proj run + ride (3 splits)
# + residual + edit_margin.
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   DFT_LOPT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_lateral_opt_argmin.sh
#   bash rl/diffusion_ft/slurm/submit_lateral_opt_argmin.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_LOPT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_argmin_${STAMP}}"
SPLITS3=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
NB="${DFT_LOPT_NEIGHBORS:-cv}"
AM_CFG="${DFT_ARGMIN_CFG:-lon=1,w_lon_prox=0.05,lon_back_m=1000}"
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
DFT_BASE_ARMS="base,o_lpA3=bank@olp:proj:20:0.3,o_am=bank@am:argmin:20,o_amc=bank@am:argmin_c:20" \
DFT_SELECT_BANKS="olp=;am=" \
DFT_SELECT_BANK_OPTS="olp=0,lon0,prev,lonprev;am=init0,initprev" \
DFT_SELECT_OPT_CFG="" DFT_SELECT_BANK_OPT_CFGS="am=${AM_CFG}" \
DFT_SELECT_BANK_RAMPS="olp=20;am=20" DFT_SELECT_BANK_BLENDS="olp=cubic;am=cubic" \
DFT_SELECT_BANK_NEIGHBORS="olp=$NB;am=$NB" \
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
for pair in "o_am:$REF_ARM" "o_amc:$REF_ARM" "o_amc:o_am"; do
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
  REF="${DFT_LOPT_REF:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_proj_20260930_171718_1665097/select}"
  f="$OUT/cmp_regress.env"
  { kv DFT_CMP_A "$REF"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_CMP_ARM_A "$REF_ARM"; kv DFT_CMP_ARM_B "$REF_ARM"; kv DFT_CMP_LABEL_A proj; kv DFT_CMP_LABEL_B argmin
    kv DFT_CMP_OUT "$OUT/select/compare_regress_${REF_ARM}.json"; } > "$f"
  XID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")
  echo "regress -> job $XID (CPU) -> $OUT/select/compare_regress_${REF_ARM}.json (want delta ~0)"
fi
echo "ride    -> job $RID   resid -> job $DID   margin -> job $MID (CPU, afterok:$AID)"
echo "all under $OUT; read select_summary{,_term}.json and compare_*.json"
