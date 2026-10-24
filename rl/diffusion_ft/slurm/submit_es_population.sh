#!/bin/bash
# Population search with the repair bank as its projection. Only calls sbatch.
#
# QUESTION. Diffusion-ES mutates a sampled population by re-noise/denoise and
# keeps elites; the mutation has to FIND a clean trajectory by chance. The honest
# repair bank (l24 lateral, ramp 20, cubic, constant-velocity neighbours) finds
# one geometrically: b_cv took failure_heldout 0.288 -> 0.717, m2 +0.0091
# (2026-09-25). So project every population member with that bank first --
# feasible right off the bat -- and let re-noise/denoise only SMOOTH the edit
# back onto the policy's manifold. Does that beat b_cv, and does the smoothing
# keep the repair? (rl/diffusion_ft/es_population.py)
#
# THE CATCH, computed before the run: the model's lowest noise level is 2.5 m per
# waypoint (sqrt(1-abar_0) x ego_range 100 m); the stock ES mutation (t=3,
# scale 2) injects 11.6 m. The 0.1-1.6 m repairs sit far below both, so
# `survival` in es_summary.json is the first number to read.
#
# THE ARMS, paired in ONE job (same sampler draws; every ES arm with pop 16 also
# starts from the SAME population, so they differ only in their operators):
#   base        the policy
#   b_cv        the bank on the base proposal alone -- the number to beat
#   es_p1       pop 1, no smoothing, same bank: MUST reproduce b_cv row for row
#               (the plumbing check es_analysis enforces)
#   es_proj     THE PROPOSAL: pop 16, project, smooth with one deterministic
#               Tweedie step (t 1, scale 0 -- the gentlest the model allows),
#               2 generations of 4 elites
#   es_projns   pop 16, project, NO smoothing, no generations: the bank over a
#               population. es_proj - es_projns is what smoothing costs/buys
#   es_projrn   pop 16, project, then the STOCK ES mutation (t 3, scale 2): does
#               the ordinary re-noise undo the projection?
#   es_raw      pop 16, stock mutation, NO projection: Diffusion-ES itself (cv
#               scorer, cons gate) -- es_proj - es_raw is what projecting buys
# All ES arms run only where the base proposal is predicted to violate (cons's
# trigger) and their winner still has to pass cons, so do-no-harm is structural
# exactly as for the bank. Neighbours are constant velocity everywhere: nothing
# reads the recorded future.
#
# PREDICTIONS, written before the run:
#   es_p1 == b_cv (agreement >= 0.99; exact up to float ties)
#   survival p50: es_proj ~0.5-0.9, es_projrn ~0-0.3 -- the stock mutation is a
#     re-draw at this noise scale, not a smoother
#   retention: es_proj well above es_projrn
#   solve on failure_heldout: es_projns >= b_cv (a superset of its candidates);
#     es_proj between es_projns and b_cv; es_raw < b_cv
#   comfort (ride_*.json): es_proj gentler than es_projns / b_cv on jerk and
#     yaw rate at intervened episodes -- the reason to smooth at all
#
# CHAIN: gate (CPU, pure rules) -> select array (GPU, afterok:gate, reads the
# verdict) -> {select_analysis, es_analysis, ride quality} (CPU, afterok:select).
#
# COST: an ES arm pays pop-1 = 15 extra full samples on every replan where any
# scene in the chunk triggers (most failure/intermittent chunks, ~15% of success
# chunks): roughly 7 plain-eval arm units each, ~25-30 a6000-hours in all at 4
# repeats. Run the smoke first -- it also prints wall_s per triggered replan.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_es_population.sh
#   bash rl/diffusion_ft/slurm/submit_es_population.sh
#   DFT_ES_POP=32 DFT_SELECT_REPEATS=2 bash ...     # bigger population, fewer repeats
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
unset "${!SLURM_@}" 2>/dev/null || true
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_ES_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/es_population_${STAMP}}"
SPLITS3=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
POP="${DFT_ES_POP:-16}"
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; } > "$OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/test_es_population.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
ES="p1=pop:1,iters:0,t:0,bank:l24cv,nb:cv"
ES+=";proj=pop:${POP},iters:2,elite:4,bank:l24cv,t:1,scale:0,nb:cv"
ES+=";projns=pop:${POP},iters:0,bank:l24cv,t:0,nb:cv"
ES+=";projrn=pop:${POP},iters:2,elite:4,bank:l24cv,t:3,scale:2,nb:cv"
ES+=";raw=pop:${POP},iters:2,elite:4,t:3,scale:2,nb:cv"
ARMS="base,b_cv=bank@l24cv:cons:20,es_p1=es@p1:cons:20,es_proj=es@proj:cons:20"
ARMS+=",es_projns=es@projns:cons:20,es_projrn=es@projrn:cons:20,es_raw=es@raw:cons:20"
# The task tag only; no arm here loads a fine-tune.
OLD=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="$ARMS" \
DFT_SELECT_BANKS="l24cv=${L24}" \
DFT_SELECT_BANK_RAMPS="l24cv=20" \
DFT_SELECT_BANK_BLENDS="l24cv=cubic" \
DFT_SELECT_BANK_NEIGHBORS="l24cv=cv" \
DFT_SELECT_ES="$ES" \
DFT_DUMP_TRAJ="${DFT_DUMP_TRAJ:-1}" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-8}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh

AID="$(cat "$OUT/select/jobid.txt")"
{ kv DFT_SELECT_DIR "$OUT/select"; kv DFT_ES_REF b_cv; kv DFT_ES_IDENTITY "es_p1:b_cv"; } \
  > "$OUT/select/es_analysis.env"
EID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/es_analysis.sbatch \
  "$OUT/select/es_analysis.env")
echo "es read -> job $EID (CPU, afterok:$AID) -> $OUT/select/es_summary.json"
{ kv DFT_SELECT_DIR "$OUT/select"; kv DFT_RIDE_SPLITS "failure_heldout,intermittent_heldout"
  kv DFT_RIDE_ARM "b_cv,es_proj,es_projns,es_projrn,es_raw"; } > "$OUT/select/ride.env"
RID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/viz_repair.sbatch \
  "$OUT/select/ride.env")
echo "ride    -> job $RID (CPU, afterok:$AID) -> $OUT/select/ride_*.json"
echo "all under $OUT; read es_summary.json first (identity, then survival), then select_summary.json"
