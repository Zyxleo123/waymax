#!/bin/bash
# The shipping safety fine-tune, retrained on the corrected clock -- and every
# shipping policy re-scored on it. Only calls sbatch.
#
# WHY. Until 2026-09-24 the env read each scene's world step off its FIRST
# timestamp pair, so 44/173 failure_heldout scenes (and presumably a similar
# share of failure_train) ran the plan at ~0.75x. The base policy alone goes
# 0.01 -> 0.95 solved on them once the clock is right (jobs 62629-62634). The
# shipping fine-tune (stopping/20260911_160658_962947/base_s*/iter_000110) was
# TRAINED on those rollouts and SCORED on them, so both its weights and every
# rate quoted for it carry the bug.
#
# THREE SIBLINGS, all on DFT_WORLD_DT_METHOD=median, all on the 3-stratum splits:
#   train/     the 09-11 recipe EXACTLY (sweep_safety_stopping.sh, arms
#              base_s0..s2: safety-only, base anchor 100 success-only, 1:1 pool,
#              150 iters, eval+keep every 10, eval seed 12345) -- only the clock
#              differs. Its own eval curve (failure/success heldout + marginal)
#              says whether the stopping point moved.
#   eval_old/  RIGHT AWAY: base, the ship bank (l24, ramp 20, cubic) and the OLD
#              fine-tune at iter 110, 3 seeds -- the current policies' real rates.
#   eval_new/  afterok:train: base and the NEW fine-tune at iter 110 (the same,
#              pre-registered stopping point), 3 seeds.
# The two evals share seed 4242, chunking and shards, so their `base` arms are
# the same draws (common random numbers) and old-vs-new is read as each one's
# paired delta against it.
#
# COST. Train ~2.3 h per seed (3 at once). eval_old 4 ckpt-units x 1450 x 4 at 2
# shards; eval_new 3. Each eval's select_analysis chains itself.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_retrain_worlddt.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_RETRAIN_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/retrain_worlddt/${STAMP}}"
OLD=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
SPLITS3=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
ITER="${DFT_EVAL_ITER:-000110}"
mkdir -p "$OUT"
export DFT_WORLD_DT_METHOD=median

# ---- 1. retrain -------------------------------------------------------------- #
echo "=== train: the 09-11 recipe, world_dt_method=median ==="
DFT_ARMS="base_s0 base_s1 base_s2" DFT_SWEEP_OUT="$OUT/train" \
DFT_WANDB_GROUP="worlddt_retrain_${STAMP}" \
  bash rl/diffusion_ft/slurm/sweep_safety_stopping.sh
TID="$(cat "$OUT/train/jobid.txt")"
grep -q "^DFT_WORLD_DT_METHOD=median$" "$OUT/train/env/0.env" || {
  echo "FATAL: train env files do not carry DFT_WORLD_DT_METHOD=median" >&2; exit 1; }
echo

# ---- 2. the current policies on the right clock (no dependency) -------------- #
L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
export DFT_SPLITS3_JSON="$SPLITS3"
export DFT_FT_ARMS=ft
export DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}"
echo "=== eval_old: base | ship bank l24 ramp 20 | old ft @ $ITER ==="
DFT_SELECT_DIR="$OUT/eval_old" DFT_SELECT_DEP="" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_$ITER $OLD/base_s1/iter_$ITER $OLD/base_s2/iter_$ITER" \
DFT_BASE_ARMS="base,b_l24_r20=bank@l24r20:cons:20" \
DFT_SELECT_BANKS="l24r20=${L24}" DFT_SELECT_BANK_RAMPS="l24r20=20" \
DFT_SELECT_BANK_BLENDS="l24r20=cubic" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh
echo

# ---- 3. the retrained policies, after training -------------------------------- #
echo "=== eval_new: base | new ft @ $ITER (afterok:$TID) ==="
DFT_SELECT_DIR="$OUT/eval_new" DFT_SELECT_DEP="afterok:$TID" \
DFT_SELECT_CKPTS="$OUT/train/base_s0/iter_$ITER $OUT/train/base_s1/iter_$ITER $OUT/train/base_s2/iter_$ITER" \
DFT_BASE_ARMS="base" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh
echo
echo "all under $OUT"
echo "read: train/base_s*/train_log.jsonl (curves), eval_old/ and eval_new/select_summary.json"
