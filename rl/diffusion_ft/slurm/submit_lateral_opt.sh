#!/bin/bash
# The ramp-20 bank with its profile SOLVED FOR instead of picked. Only calls sbatch.
#
# WHY. The ship bank (l24 @ ramp 20, cubic) offers 24 rigid lateral shifts
# A * smoothstep(t/20) and executes the smallest one the exact clearance test
# certifies. Its residual is a SHAPE limit: at coverage-limited replans the
# clearest candidate falls ~0.7 m short and that is flat in K (4 -> 24), because
# every candidate is the same curve, scaled. rl/diffusion_ft/lateral_opt.py keeps
# everything else (trigger, certificate, cons rule, neighbour source) and replaces
# the fixed curve with a per-step offset d(t) minimizing
#     training barrier (hinge2, segment edges) + lateral accel + jerk of the edit,
# by CHOMP-style preconditioned descent. Minimizing accel alone subject to a held
# offset IS the cubic smoothstep, so the ramp is the special case; the optimizer
# departs from it only where the hazard asks for a different shape.
#
# THE ARMS, paired in ONE job (same draws), all with the SAME neighbour source
# (DFT_LOPT_NEIGHBORS, default cv -- the deployable one; `log` is the oracle):
#   base      the policy, no repair
#   b_l24     the ship bank: l24 @ ramp 20, cubic                     (24 fixed)
#   o_0       ONE optimized edit, started on the base's own line        (1 opt)
#   o_ms      optimized from 0, +/-0.75, +/-1.6 (the r20 profiles)      (5 opt)
#   b_l24o    l24 + optimized from the bank's clearest candidate and 0  (24 + 2)
# o_0 vs b_l24 is the direct question (a loss instead of a fixed function);
# o_ms says whether the loss has bad local minima that a multi-start fixes;
# b_l24o can only ADD eligible candidates to b_l24, so it isolates the coverage
# the loss buys on top of the bank.
#
# PREDICTIONS, written before the run (failure_heldout solve; b_cv was 0.717 on
# the 09-25 fixed-clock run, base 0.288):
#   b_l24o >= b_l24 by +0.01-0.04: the ~30% of coverage failures within 0.5 m of
#          clean are reachable by bending, the rest mostly are not;
#   o_0    within +/-0.03 of b_l24 -- a single local descent loses some of the
#          wide swerves the bank's +/-1.6 entries find, gains the bumps;
#   o_ms   ~ b_l24o;
#   ride:  the optimized edits' peak jerk BELOW r20 cubic's (no accel step at the
#          join), peak accel similar. `broken` stays ~0 (same trigger).
#   no-goal is untouched by construction (a lateral edit cannot add progress).
#
# CHAIN: gate (CPU, gate_lateral_opt.sbatch: pure tests + real-scene plumbing,
# writes a verdict) -> select (GPU array, afterok:gate, reads the verdict) ->
# select_analysis (CPU, m2/m3) + ride quality (CPU, viz_repair) + residual
# decomposition (CPU), each afterok on the select array.
#
# Usage (from repo root, ON RHEA):
#   DFT_LOPT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_lateral_opt.sh   # 4 scenes/split
#   bash rl/diffusion_ft/slurm/submit_lateral_opt.sh
#   DFT_SELECT_OPT_CFG="w_jerk=0.1,iters=48" bash ...    # optimizer overrides
#   DFT_LOPT_NEIGHBORS=log bash ...                      # the oracle variant
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_LOPT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_${STAMP}}"
SPLITS3=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
NB="${DFT_LOPT_NEIGHBORS:-cv}"
OPT_CFG="${DFT_SELECT_OPT_CFG:-}"
case "$NB" in log|cv|none) ;; *) echo "FATAL: DFT_LOPT_NEIGHBORS=$NB not log|cv|none" >&2; exit 1;; esac
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_LOPT_JSON "$OUT/gate/gate_lateral_opt.json"
  kv DFT_LOPT_NEIGHBORS "$NB"; kv DFT_SELECT_OPT_CFG "$OPT_CFG"
  kv DFT_LOPT_GATE_SCENES "$([[ -n "${DFT_LOPT_SMOKE:-}" ]] && echo 4 || echo 16)"
} > "$OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/gate_lateral_opt.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
# No fine-tune arm, so this checkpoint dir only names the task dirs (its
# tparams.npz went in the 09-28 cleanup; nothing loads it).
OLD=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,b_l24=bank@l24:cons:20,o_0=bank@o0:cons:20,o_ms=bank@oms:cons:20,b_l24o=bank@l24o:cons:20" \
DFT_SELECT_BANKS="l24=${L24};o0=;oms=;l24o=${L24}" \
DFT_SELECT_BANK_OPTS="o0=0;oms=0,lat0.75,lat-0.75,lat1.6,lat-1.6;l24o=best,0" \
DFT_SELECT_OPT_CFG="$OPT_CFG" \
DFT_SELECT_BANK_RAMPS="l24=20;o0=20;oms=20;l24o=20" \
DFT_SELECT_BANK_BLENDS="l24=cubic;o0=cubic;oms=cubic;l24o=cubic" \
DFT_SELECT_BANK_NEIGHBORS="l24=$NB;o0=$NB;oms=$NB;l24o=$NB" \
DFT_DUMP_TRAJ=1 DFT_SELECT_SMOKE="${DFT_LOPT_SMOKE:-}" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-4}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh

AID="$(cat "$OUT/select/jobid.txt")"
{ kv DFT_SELECT_DIR "$OUT/select"; kv DFT_RIDE_SPLITS "failure_heldout,intermittent_heldout"
  kv DFT_RESID_SPLITS "failure_heldout,intermittent_heldout"; } > "$OUT/post.env"
RID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/viz_repair.sbatch "$OUT/post.env")
DID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/residual_decomp.sbatch "$OUT/post.env")
echo "ride    -> job $RID (CPU, afterok:$AID) -> $OUT/select/ride_*.json"
echo "resid   -> job $DID (CPU, afterok:$AID) -> $OUT/select/residual_*.json"
echo "all under $OUT; read $OUT/gate/gate_lateral_opt.json first, then $OUT/select/select_summary.json"
