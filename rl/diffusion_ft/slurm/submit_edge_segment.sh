#!/bin/bash
# The safety fine-tune with the road-edge clearance measured to the polyline
# SEGMENT instead of the nearest SAMPLE -- one A/B against the shipping recipe on
# the fixed clock. Only calls sbatch.
#
# WHY. Under the default edge_geometry="point" the barrier charges the distance
# to a roadgraph sample (~1 m apart), so it overstates depth by ~0.10 m and its
# descent points at the nearer exit 89.9% of the time; "segment" gets 97.1% with
# the identical offroad boundary (failure anatomy, 2026-09-19/23). A minority
# effect, and it does not explain the 2.1 s excursions -- this measures whether
# it is worth anything once trained.
#
# THE MARGIN IS RE-MEASURED, NOT CARRIED OVER. 0.3 m is the human's p01 clearance
# UNDER "point". "segment" clearances are never larger, so 0.3 there is a
# stricter barrier and the A/B would confound geometry with strictness. The arm
# keeps the RULE (margin = expert p01, floored to the cm) instead:
#   cal/     (CPU) calibrate_edge_geometry.sbatch: point control + segment on the
#            09-01 scenes; writes cal/edge_margin.env only if its gates pass.
#   train/   (GPU, afterok:cal) the 09-11 recipe EXACTLY as retrain_worlddt ran
#            it -- base_s0..s2, median clock, rails -- plus DFT_EDGE_GEOMETRY=
#            segment and the measured margin. The env files are diffed against
#            the reference's before anything is submitted: only the edge keys,
#            the out dir and the wandb group may differ.
#   eval/    (GPU, afterok:train) base + ft @ iter 110, seed 4242, same shards
#            and chunking as retrain_worlddt/eval_new -> the same base draws.
#   compare  (CPU, afterok:eval) tools/compare_select.py: segment minus point,
#            3-seed means paired per (scene, repeat) against eval_new's rows.
# DFT_SEG_MARGIN=0.3 skips the re-measurement (the literal flag flip); the
# calibration still runs, for the record.
#
# READ: train/base_s*/train_log.jsonl against the reference's (same eval seed
# 12345, same splits -- the curves are directly comparable), then
# eval/select_summary.json and eval/compare.json. The pre-registered comparison
# is iter 110; if segment's curve peaks elsewhere, that is a finding, not a
# reason to move the eval.
#
# COST. cal ~5 min CPU. train ~2.3 h/seed, 3 at once. eval 4 arm-units at 2
# shards (~2.5 h GPU). compare ~1 min CPU.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_edge_segment.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_EDGESEG_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/edge_segment/${STAMP}}"
REF=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/retrain_worlddt/20260924_235925_2645687
SPLITS3=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json
ITER="${DFT_EVAL_ITER:-000110}"
ARMS="base_s0 base_s1 base_s2"
mkdir -p "$OUT" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }

for f in "$REF/eval_new/select_summary.json" "$REF/train/env/0.env" "$SPLITS3"; do
  [[ -s "$f" ]] || { echo "FATAL: reference artefact missing: $f" >&2; exit 1; }
done

# ---- the training env files, checked before anything is submitted ----------- #
MARGIN_KEYS='DFT_EDGE_GEOMETRY|DFT_EDGE_MARGIN_FILE'
if [[ -n "${DFT_SEG_MARGIN:-}" ]]; then
  MFILE=""; export DFT_EDGE_MARGIN="$DFT_SEG_MARGIN"; MARGIN_KEYS+='|DFT_EDGE_MARGIN'
else
  MFILE="$OUT/cal/edge_margin.env"
fi
train_sweep() {
  DFT_ARMS="$ARMS" DFT_SWEEP_OUT="$OUT/train" DFT_WORLD_DT_METHOD=median \
  DFT_SIM_AGENTS=none DFT_NEIGHBOR_SOURCE=log \
  DFT_EDGE_GEOMETRY=segment DFT_EDGE_MARGIN_FILE="$MFILE" \
  DFT_WANDB_GROUP="edge_segment_${STAMP}" \
    bash rl/diffusion_ft/slurm/sweep_safety_stopping.sh
}
DFT_DRY_RUN=1 train_sweep > /dev/null
norm() {
  grep -E '^DFT_[A-Z_]*=' "$1" \
    | grep -vE "^(DFT_OUT_DIR|DFT_WANDB_GROUP|${MARGIN_KEYS})=" | sort
}
for i in 0 1 2; do
  # Keys the reference predates, at the values its job fell back to.
  diffs="$(diff <(norm "$REF/train/env/$i.env") <(norm "$OUT/train/env/$i.env") \
    | grep -E '^[<>]' | grep -vxE '> DFT_(SIM_AGENTS=none|NEIGHBOR_SOURCE=log)' || true)"
  if [[ -n "$diffs" ]]; then
    echo "FATAL: train/env/$i.env differs from the point reference beyond the edge keys:" >&2
    echo "$diffs" >&2; exit 1
  fi
  grep -qx "DFT_EDGE_GEOMETRY=segment" "$OUT/train/env/$i.env" \
    || { echo "FATAL: train/env/$i.env is not DFT_EDGE_GEOMETRY=segment" >&2; exit 1; }
done
echo "train env files match $REF/train/env except the edge keys"
echo

# ---- 1. calibration (CPU) ---------------------------------------------------- #
CENV="$OUT/cal.env"
echo "DFT_EDGECAL_OUT=$OUT/cal" > "$CENV"
CID=$(sb --parsable rl/diffusion_ft/slurm/calibrate_edge_geometry.sbatch "$CENV")
echo "cal     -> job $CID (CPU) -> $OUT/cal/{decision.json,edge_margin.env}"

# ---- 2. train (GPU) ---------------------------------------------------------- #
DEP=""; [[ -n "$MFILE" ]] && DEP="afterok:$CID"
DFT_DEPENDENCY="$DEP" train_sweep
TID="$(cat "$OUT/train/jobid.txt")"
echo

# ---- 3. eval: same draws as eval_new (GPU) ----------------------------------- #
CKPTS=""; for a in $ARMS; do CKPTS+="$OUT/train/$a/iter_$ITER "; done
DFT_SELECT_DIR="$OUT/eval" DFT_SELECT_DEP="afterok:$TID" DFT_SELECT_CKPTS="${CKPTS% }" \
DFT_SPLITS3_JSON="$SPLITS3" DFT_BASE_ARMS=base DFT_FT_ARMS=ft \
DFT_SELECT_SEED=4242 DFT_SELECT_REPEATS=4 DFT_NUM_SHARDS=2 DFT_EVAL_CHUNK=32 \
DFT_SIM_AGENTS=none DFT_EGO_VEL_SOURCE=commanded DFT_WORLD_DT_METHOD=median \
DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh
SID="$(cat "$OUT/eval/jobid.txt")"
echo

# ---- 4. compare against eval_new (CPU) --------------------------------------- #
XENV="$OUT/compare.env"
{ echo "DFT_CMP_A=$REF/eval_new"; echo "DFT_CMP_B=$OUT/eval"
  echo "DFT_CMP_LABEL_A=point"; echo "DFT_CMP_LABEL_B=segment"
  echo "DFT_SPLITS_JSON=$SPLITS3"; echo "DFT_SELECT_REPEATS=4"
  echo "DFT_CMP_OUT=$OUT/eval/compare.json"; } > "$XENV"
XID=$(sb --parsable --dependency="afterok:$SID" rl/diffusion_ft/slurm/compare_select.sbatch "$XENV")
echo "compare -> job $XID (CPU, afterok:$SID) -> $OUT/eval/compare.json"
echo
echo "all under $OUT"
echo "  cal/decision.json               the margin, and what it asks of the human"
echo "  train/base_s*/train_log.jsonl   curves; reference: $REF/train/base_s*/"
echo "  eval/select_summary.json        segment ft vs base"
echo "  eval/compare.json               segment minus point, paired"
