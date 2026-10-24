#!/bin/bash
# Population search projected by the LOSS-OPTIMIZED lateral repair. Only calls sbatch.
#
# WHY THIS AND NOT submit_es_population.sh. That experiment projected each ES
# member with the l24 ramp-20 bank and then re-noised/denoised it "for
# smoothness". Its smoke (jobs 68920-68924, 2026-09-29) refuted the smoothing
# reading: the gentlest denoise this model has (one deterministic step at t=0,
# whose noise level is already 2.5 m per waypoint) kept 12-30% of each repair and
# 0-11% of the projected-feasible members stayed feasible. So here:
#   * smoothness comes from the PROJECTION itself -- lateral_opt minimizes the
#     training barrier plus the edit's lateral accel and jerk (the `o_0` arm:
#     failure 0.818 vs the bank's 0.708, broken 0, run lateral_opt_20260928_...);
#   * re-noise/denoise, where used, only MUTATES elites (exploration) and the
#     projection runs after it (`order:sp`), so every executed trajectory is a raw
#     model sample or an optimizer output.
#
# THE ARMS, paired in ONE job (same draws; the pop-16 arms share one population):
#   base        the policy
#   o_0         lateral_opt on the base proposal alone -- the number to beat
#   es_o1       pop 1, opt projection: MUST reproduce o_0 row for row
#   es_opt      pop 16, every violating member optimizer-projected, no diffusion
#               pass, no generations: does a population of starts beat one?
#   es_optmut   pop 16, 2 generations x 4 elites: elites mutated by the stock ES
#               re-noise (t 3, scale 2), mutants projected: does search on top of
#               the population pay?
# All run only where the base proposal is predicted to violate (cons's trigger),
# and the winner still has to pass cons with the exact scorer. Constant-velocity
# neighbours throughout; nothing reads the recorded future.
#
# PREDICTIONS, written before the run:
#   es_o1 == o_0 (agreement >= 0.99)
#   es_opt > o_0 on failure_heldout by a few points (o_0's residual is ~100
#     coverage failures of 692; other starts give the optimizer other basins)
#   es_optmut ~ es_opt (the mutation re-draws at 11.6 m; its mutants mostly need
#     projecting again, so generations add little beyond more starts)
#   broken ~0 for all (cons); ride quality of es_* ~ o_0 (same optimizer)
#
# CHAIN: gate (CPU) -> select array (GPU, afterok:gate, reads the verdict) ->
# {select_analysis, es_analysis, ride quality} (CPU, afterok:select).
#
# COST: pop-1 = 15 extra full samples plus up to 16 (es_opt) or ~48 (es_optmut)
# optimizer calls per triggered replan. Run the smoke first; es_summary.json
# reports wall_s per triggered replan.
#
# Usage (from repo root; on lov4 sbatch is local):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_es_opt.sh
#   bash rl/diffusion_ft/slurm/submit_es_opt.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
unset "${!SLURM_@}" 2>/dev/null || true
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_ES_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/es_opt_${STAMP}}"
SPLITS3=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
POP="${DFT_ES_POP:-16}"
OPT_CFG="${DFT_SELECT_OPT_CFG:-}"
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; } > "$OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/test_es_population.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

ES="o1=pop:1,iters:0,opt:1,t:0,nb:cv"
ES+=";opt=pop:${POP},iters:0,opt:1,t:0,nb:cv"
ES+=";optmut=pop:${POP},iters:2,elite:4,opt:1,order:sp,t:3,scale:2,nb:cv"
ARMS="base,o_0=bank@o0:cons:20,es_o1=es@o1:cons:20,es_opt=es@opt:cons:20"
ARMS+=",es_optmut=es@optmut:cons:20"
# The task tag only; no arm here loads a fine-tune.
OLD=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="$ARMS" \
DFT_SELECT_BANKS="o0=" \
DFT_SELECT_BANK_OPTS="o0=0" \
DFT_SELECT_OPT_CFG="$OPT_CFG" \
DFT_SELECT_BANK_RAMPS="o0=20" \
DFT_SELECT_BANK_BLENDS="o0=cubic" \
DFT_SELECT_BANK_NEIGHBORS="o0=cv" \
DFT_SELECT_ES="$ES" \
DFT_DUMP_TRAJ="${DFT_DUMP_TRAJ:-1}" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-8}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh

AID="$(cat "$OUT/select/jobid.txt")"
{ kv DFT_SELECT_DIR "$OUT/select"; kv DFT_ES_REF o_0; kv DFT_ES_IDENTITY "es_o1:o_0"; } \
  > "$OUT/select/es_analysis.env"
EID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/es_analysis.sbatch \
  "$OUT/select/es_analysis.env")
echo "es read -> job $EID (CPU, afterok:$AID) -> $OUT/select/es_summary.json"
{ kv DFT_SELECT_DIR "$OUT/select"; kv DFT_RIDE_SPLITS "failure_heldout,intermittent_heldout"
  kv DFT_RIDE_ARM "o_0,es_opt,es_optmut"; } > "$OUT/select/ride.env"
RID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/viz_repair.sbatch \
  "$OUT/select/ride.env")
echo "ride    -> job $RID (CPU, afterok:$AID) -> $OUT/select/ride_*.json"
echo "all under $OUT; read es_summary.json first (identity es_o1 == o_0), then select_summary.json"
