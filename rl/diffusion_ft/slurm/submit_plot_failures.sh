#!/bin/bash
# Figures of the episodes the no-rule repair (o_am) still FAILS: base vs o_am on the
# same paired draw, zoomed on o_am's own violation (tools/plot_fix_pairs.py
# --mode failing). Only calls sbatch.
#
#   now:     CPU plot on the existing bank_honest bundle (18 scenes; 6 of them fail
#            under o_am) -- a smoke test of the failing mode before the GPU job lands
#   dump:    GPU, geometry only (no denoiser): roadgraph/agents/goal for every scene
#            where o_am leaves a repeat unsolved (rows of 71188 shards 0-3, 2026-10-07)
#   afterok: CPU plot of all of them -> fail_<sid>_r<rep>.png
#
# The 69 intermittent scenes of shard 3 are not scored yet (resume 75393); their
# failures are not in the lists below.
#
# Usage (from repo root; on lov4 sbatch is local):
#   bash rl/diffusion_ft/slurm/submit_plot_failures.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
unset "${!SLURM_@}" 2>/dev/null || true
SEL=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/lateral_opt_argmin_20261002_142129_3029435/select
OLD_BUNDLE=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/bank_honest_20260925_160329_3074201/scenes/scenes_20260925_161856.json
SPLITS=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata_womd.json
ARM="${DFT_FIX_ARM:-o_am}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$SEL/failing_${ARM}_${STAMP}"
mkdir -p "$OUT/smoke" "$OUT/scenes" "$OUT/png" "$ROOT/logs/dft"
sb() { echo "+ sbatch $*" >&2; env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

# o_am unsolved in some repeat; "regressed" = the base solved that same draw.
UNREC="5757,6360,12956,17880,32138,37149,46703,61937,75179,75517,85946,123317,123579,125447,128567,129464,131027,142996,152402,156297,175167,196406,218220,218345,276728,285511,302334,304125,309297,327150,371344,371662,375312,380955,387865,403335,408620,430035,463303,464928,469912"
REGR="32062,34551,38905,49484,51019,81859,139243,171055,249301,400269"

plot_env() {  # <env path> <bundle> <out dir>
  { kv DFT_FIX_SCENES_JSON "$2"; kv DFT_SELECT_DIR "$SEL"; kv DFT_FIX_OUT_DIR "$3"
    kv DFT_FIX_ARM "$ARM"; kv DFT_FIX_MODE failing; kv DFT_FIX_IDS ""; } > "$1"
}

plot_env "$OUT/smoke/fix.env" "$OLD_BUNDLE" "$OUT/smoke"
P0=$(sb --parsable rl/diffusion_ft/slurm/plot_fix_pairs.sbatch "$OUT/smoke/fix.env")

{ kv DFT_SPLITS_JSON "$SPLITS"; kv DFT_SELECT_DIR "$SEL"; kv DFT_OUT_JSON "$OUT/scenes/scenes.json"
  kv DFT_SHARD_GLOB "/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*"
  kv DFT_VIZ_ARMS "base,$ARM"; kv DFT_SOLVED_ARM "$ARM"
  kv DFT_UNRECOVERED "$UNREC"; kv DFT_REGRESSED "$REGR"; } > "$OUT/scenes/dump.env"
D=$(sb --parsable rl/diffusion_ft/slurm/dump_reach_from_select.sbatch "$OUT/scenes/dump.env")

plot_env "$OUT/png/fix.env" "$OUT/scenes/scenes.json" "$OUT/png"
P1=$(sb --parsable --dependency="afterok:$D" rl/diffusion_ft/slurm/plot_fix_pairs.sbatch "$OUT/png/fix.env")

{ echo "stamp=$STAMP"
  echo "smoke plot (CPU, old bundle): $P0 -> $OUT/smoke"
  echo "geometry dump (GPU):          $D  -> $OUT/scenes/scenes.json"
  echo "plot all (CPU, afterok):      $P1 -> $OUT/png"
} | tee "$OUT/submit.txt"
