#!/bin/bash
# The pace experiment: fix 1 (a deploy-time speed-up) x fix 2a (the observation
# shows the ego's REALISED speed). Only calls sbatch.
#
# QUESTION. The no-goal residual is trip length x pace: positions under-travel the
# human by ~7-10% while the velocity the policy writes -- and is then shown as its
# own ego_state -- reads 1.13x what it drives. Does (1) driving 5-10% faster, or
# (2a) an honest speed in the observation, buy back the long trips, and what does
# it cost the scenes the base already solves?
#
# WHY A 2x2 AND NOT ONE ARM. The pace arm is `bank@<p>:always` with one time-warp
# candidate, and `branching.time_warp` rewrites the velocity field from finite
# differences of the warped positions. So under `commanded` the pace arm ALSO
# feeds the policy a self-consistent speed -- it is "faster + honest observation",
# not "faster". Running the same arms under `realised` separates them:
#   commanded:pace - commanded:base   = the arm as built (confounded)
#   realised:base  - commanded:base   = fix 2a alone
#   realised:pace  - realised:base    = the speed-up alone, observation held fixed
# and commanded:pace ~ realised:pace is itself a check (both observe FD speeds).
#
# THE ARMS. `pace10` = tw1.1, `pace05` = tw1.05 (a dose point), both at ramp 1:
# with the bank's default ramp of 10 and replans every 10 steps the factor eases
# in 1.0 -> 1.1 inside every executed window and resets at the next anchor, so
# "tw1.1" would execute an average of only 5.5%. At ramp 1 every executed step is
# the nominal factor. `always` is not restriction-shaped: `broken` will not be
# 0.0000, and success_heldout is the cost measurement, not a side note.
#
# CHAIN
#   gate     (CPU)  test_ego_vel.sbatch -> gate/verdict.txt, always exits 0
#   commanded (GPU) submit_select_eval.sh, no dependency
#   realised  (GPU) submit_select_eval.sh, afterok:gate, and DFT_REQUIRE_GATE so
#                   its tasks exit 0 with the reason unless the verdict is PASS
#   analysis (CPU)  pace_analysis.sbatch, afterok on both arrays
# Each sibling also chains its own select_analysis (m2/m3 inside that cell).
# Same seed / chunk / shards in both, so the cells are paired by common random
# numbers; pace_analysis checks that from the metas before pairing.
#
# COST. 3 arms x 1450 scenes x 4 repeats per sibling ~ 3 plain-eval arm-units
# (~37 min each on an a6000) -> ~1 h per task at 2 shards, 4 tasks total.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_pace.sh   # 4 scenes/split, 1 repeat
#   bash rl/diffusion_ft/slurm/submit_pace.sh
#   DFT_PACE_ARMS="base,pace10=bank@p10:always:20" bash ...        # drop the dose point
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
PACE_DIR="${DFT_PACE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/pace_${STAMP}}"
SPLITS3="${DFT_SPLITS3_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json}"
RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
REPEATS="${DFT_SELECT_REPEATS:-4}"
[[ -n "${DFT_SELECT_SMOKE:-}" ]] && REPEATS=1
mkdir -p "$PACE_DIR/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

# ---- gate ----------------------------------------------------------------- #
{ kv DFT_GATE_VERDICT "$PACE_DIR/gate/verdict.txt"
  kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_CALIB_TOL "${DFT_CALIB_TOL:-0.01}"; } > "$PACE_DIR/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/test_ego_vel.sbatch "$PACE_DIR/gate/gate.env")
echo "gate      -> job $GID (CPU) -> $PACE_DIR/gate/verdict.txt"
echo

# ---- the two siblings ------------------------------------------------------ #
export DFT_SPLITS3_JSON="$SPLITS3"
export DFT_SELECT_BANKS="${DFT_PACE_BANKS:-p10=tw1.1;p05=tw1.05}"
export DFT_SELECT_BANK_RAMPS="${DFT_PACE_BANK_RAMPS:-p10=1;p05=1}"
export DFT_BASE_ARMS="${DFT_PACE_ARMS:-base,pace10=bank@p10:always:20,pace05=bank@p05:always:20}"
export DFT_FT_ARMS=""                 # no fine-tune arms (the dir is required, never loaded)
export DFT_SELECT_CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110}"
export DFT_DUMP_TRAJ="${DFT_DUMP_TRAJ:-1}"   # failure_anatomy can re-read these later
AIDS=()
for src in commanded realised; do
  echo "=== ego_vel_source=$src ==="
  if [[ "$src" == realised ]]; then
    DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$PACE_DIR/gate/verdict.txt" \
    DFT_EGO_VEL_SOURCE="$src" DFT_SELECT_DIR="$PACE_DIR/$src" \
      bash rl/diffusion_ft/slurm/submit_select_eval.sh
  else
    DFT_SELECT_DEP="" DFT_REQUIRE_GATE="" \
    DFT_EGO_VEL_SOURCE="$src" DFT_SELECT_DIR="$PACE_DIR/$src" \
      bash rl/diffusion_ft/slurm/submit_select_eval.sh
  fi
  AIDS+=("$(cat "$PACE_DIR/$src/jobid.txt")")
  echo
done

# ---- cross-cell readout ---------------------------------------------------- #
{ kv DFT_PACE_DIR "$PACE_DIR"
  kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_SELECT_REPEATS "$REPEATS"
  kv DFT_PACE_CELLS "commanded realised"; } > "$PACE_DIR/analyze.env"
PID=$(sb --parsable --dependency="afterok:$(IFS=:; echo "${AIDS[*]}")" \
  rl/diffusion_ft/slurm/pace_analysis.sbatch "$PACE_DIR/analyze.env")
echo "analysis  -> job $PID (CPU, afterok:${AIDS[*]})"
echo "out       -> $PACE_DIR/pace_summary.{json,png}"
echo "first read: $PACE_DIR/gate/verdict.txt, then the realised array's first .out"
