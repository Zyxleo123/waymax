#!/bin/bash
# The world-dt fix, A/B. Only calls sbatch.
#
# QUESTION. `compute_world_dt_seconds` took each scene's world step from its FIRST
# timestamp pair, ~0.075 s in 44/173 failure_heldout scenes, so the plan ran at
# ~0.75x there -- and 43 of those 44 never reach the goal. Does executing the plan
# on the scene's own clock (`median`) give them the goal back, and does it touch
# anything else?
#
# DESIGN. Two sibling select jobs, identical but for DFT_WORLD_DT_METHOD:
#   first/   the old grid; reproduces every earlier select result
#   median/  the fix; depends on the gate and reads its verdict (DFT_REQUIRE_GATE)
# Same seed / chunk / shards, so the cells are paired by common random numbers
# (pace_analysis checks that from the metas). Arm `base` only by default: this
# measures the simulator fix, not a policy. DFT_WDT_ARMS adds arms to both.
#
# PREDICTIONS, written before the run (pace_summary.json, control first:base):
#   * failure_heldout `dt_short`: reached ~0.02 -> near the dt_normal rate (~0.97),
#     and `world_dt` used there ~0.075 -> ~0.100;
#   * failure_heldout `dt_normal`: unchanged within noise;
#   * success_heldout / intermittent: unchanged -- neither has a short-dt scene,
#     and `median` moves a normal scene's dt by < 0.2%.
# `solved` on dt_short is the open question: those scenes were put in the failure
# split by the bug, so what they fail on once they can arrive is new information.
#
# CHAIN
#   gate     (CPU) test_world_dt.sbatch -> gate/verdict.txt, always exits 0
#   first    (GPU) submit_select_eval.sh, no dependency
#   median   (GPU) submit_select_eval.sh, afterok:gate + DFT_REQUIRE_GATE
#   analysis (CPU) pace_analysis.sbatch, afterok on both arrays
#
# COST. One arm x 1450 scenes x 4 repeats per sibling ~ one plain-eval arm-unit
# (~37 min on an a6000) -> ~20 min per task at 2 shards, 4 tasks.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_world_dt.sh
#   bash rl/diffusion_ft/slurm/submit_world_dt.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="${DFT_WDT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/worlddt_${STAMP}}"
SPLITS3="${DFT_SPLITS3_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json}"
RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
REPEATS="${DFT_SELECT_REPEATS:-4}"
[[ -n "${DFT_SELECT_SMOKE:-}" ]] && REPEATS=1
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

# ---- gate ----------------------------------------------------------------- #
{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"
  kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_TRACK_TOL "${DFT_TRACK_TOL:-0.25}"; } > "$OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/test_world_dt.sbatch "$OUT/gate/gate.env")
echo "gate      -> job $GID (CPU) -> $OUT/gate/verdict.txt"
echo

# ---- the two siblings ------------------------------------------------------ #
export DFT_SPLITS3_JSON="$SPLITS3"
export DFT_BASE_ARMS="${DFT_WDT_ARMS:-base}"
export DFT_FT_ARMS=""                 # no fine-tune arms (the dir is required, never loaded)
export DFT_SELECT_CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110}"
export DFT_DUMP_TRAJ="${DFT_DUMP_TRAJ:-1}"   # failure_anatomy can re-read these later
AIDS=()
for method in first median; do
  echo "=== world_dt_method=$method ==="
  if [[ "$method" == median ]]; then
    DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
    DFT_WORLD_DT_METHOD="$method" DFT_SELECT_DIR="$OUT/$method" \
      bash rl/diffusion_ft/slurm/submit_select_eval.sh
  else
    DFT_SELECT_DEP="" DFT_REQUIRE_GATE="" \
    DFT_WORLD_DT_METHOD="$method" DFT_SELECT_DIR="$OUT/$method" \
      bash rl/diffusion_ft/slurm/submit_select_eval.sh
  fi
  AIDS+=("$(cat "$OUT/$method/jobid.txt")")
  echo
done

# ---- cross-cell readout ---------------------------------------------------- #
{ kv DFT_PACE_DIR "$OUT"
  kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_SELECT_REPEATS "$REPEATS"
  kv DFT_PACE_CELLS "first median"
  kv DFT_PACE_CONTROL "first:base"; } > "$OUT/analyze.env"
PID=$(sb --parsable --dependency="afterok:$(IFS=:; echo "${AIDS[*]}")" \
  rl/diffusion_ft/slurm/pace_analysis.sbatch "$OUT/analyze.env")
echo "analysis  -> job $PID (CPU, afterok:${AIDS[*]})"
echo "out       -> $OUT/pace_summary.{json,png}"
echo "first read: $OUT/gate/verdict.txt, then the median array's first .out"
