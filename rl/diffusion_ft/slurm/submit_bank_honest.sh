#!/bin/bash
# Does the repair bank still win without knowing the future? Only calls sbatch.
#
# WHY. The bank's selector scores candidates against the other cars' boxes over
# the next 20 steps (2 s), read from the RECORDED future. Under log replay that is
# exactly where every car will be -- an oracle no deployed planner has; the policy
# itself only sees their current state. Attribution on the 09-24 eval: 25% of the
# bank's switches were TRIGGERED by a future-car conflict, 31% of the failures it
# fixed involved one, and 26% of chosen candidates passed within 0.5 m of a future
# car (the oracle deciding WHICH fix). The road-edge half reads only the static map.
#
# THE ARMS, paired in ONE job (same draws; only the neighbour source differs --
# for the trigger AND the certificate):
#   base        the policy, no bank
#   b_oracle    the ship bank (l24, ramp 20, cubic) reading the recorded future
#   b_cv        the same bank, other cars predicted by CONSTANT VELOCITY from
#               their state at t0 (1-step error 4 cm, 2 s error ~1 m p50 / 3 m p90)
#   b_edge      the same bank ignoring other cars entirely (road edges only)
# b_cv is the deployable number; b_edge is the floor (it can switch INTO a car).
#
# PREDICTIONS, written before the run (m2 vs base; the oracle was +0.0097):
#   b_cv   ~ +0.006-0.009: most switches are edge-triggered (map, legitimate);
#          the agent-triggered third degrades with a 1-3 m prediction error;
#   b_edge ~ +0.005-0.007 with broken > 0: it loses every agent-triggered repair
#          and can steer into traffic it cannot see.
#   If b_cv lands near the oracle, the headline survives as a deployable result.
#
# CHAIN: gate (CPU, test_neighbor_pred.sbatch) -> select (GPU, afterok:gate, reads
# the verdict) -> select_analysis (CPU). Fixed clock (median), rails by default;
# DFT_SIM_AGENTS=idm_brake re-runs all four arms against reactive traffic.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_bank_honest.sh
#   bash rl/diffusion_ft/slurm/submit_bank_honest.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_HONEST_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/bank_honest_${STAMP}}"
SPLITS3=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; kv DFT_SPLITS_JSON "$SPLITS3"; } > "$OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/test_neighbor_pred.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
OLD=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" \
DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,b_oracle=bank@l24log:cons:20,b_cv=bank@l24cv:cons:20,b_edge=bank@l24none:cons:20" \
DFT_SELECT_BANKS="l24log=${L24};l24cv=${L24};l24none=${L24}" \
DFT_SELECT_BANK_RAMPS="l24log=20;l24cv=20;l24none=20" \
DFT_SELECT_BANK_BLENDS="l24log=cubic;l24cv=cubic;l24none=cubic" \
DFT_SELECT_BANK_NEIGHBORS="l24log=log;l24cv=cv;l24none=none" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-4}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh
echo "all under $OUT; read $OUT/select/select_summary.json"
