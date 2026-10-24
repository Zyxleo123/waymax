#!/bin/bash
# Ride-comfort A/B of the repair edits, paired in one job. Only calls sbatch.
#
# QUESTION. The l24 bank takes failure_heldout from 0.046 to 0.555 and breaks
# nothing -- but `solved` is reached & !offroad & !overlap and says nothing about
# HOW. The env executes through StateDynamics, which teleports the ego to the
# states it is handed, so nothing rejects a kinematically absurd edit. Peak
# lateral acceleration of the ease-in goes as |A|/T^2 and jerk as |A|/T^3 with
# T = ramp_steps * world_dt, measured on a straight path at the shipped ramp:
#
#   ramp  T     blend     A=1.6: peak|acc|   peak|jerk|
#     10  1.0s  cubic            7.68          44.8      <- SHIPPED
#     10  1.0s  quintic          8.93          51.6      (worse: too coarse to resolve C2)
#     20  2.0s  cubic            2.16          11.6
#     20  2.0s  quintic          2.29           8.1
#     30  3.0s  quintic          1.02           2.6
#
# So the RAMP is the lever and the easing is second-order until the ramp is long
# enough to resolve it. This runs all four settings as paired arms over the SAME
# bank (l24) and the same draws, so any solve difference is the ramp's.
#
# THE CATCH, and the reason this is an experiment and not a config change: the
# filter re-decides every --replan_interval_steps (10), so a ramp longer than 10
# is TRUNCATED by the next replan and the executed offset never reaches A. The
# verifier still scores the full ramped trajectory over its 20-step window, so
# it may certify clearance the car does not actually achieve in the 10 steps it
# drives. If solve drops from r10 to r20, that is why -- and it is the real
# comfort/efficacy trade, not a bug.
#
# --dump_traj is ON, so the executed states land in feat/*.npz and the chained
# CPU job measures peak lateral acceleration, jerk and yaw rate per episode
# against the base counterfactual, plus per-scene figures.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_comfort_ab.sh
#   bash rl/diffusion_ft/slurm/submit_comfort_ab.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"

L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"

# Four copies of the SAME candidate set; only the ease-in differs.
export DFT_SELECT_BANKS="c_r10=${L24};c_r20=${L24};c_r20q=${L24};c_r30q=${L24}"
export DFT_SELECT_BANK="$L24"
export DFT_SELECT_BANK_RAMPS="c_r10=10;c_r20=20;c_r20q=20;c_r30q=30"
export DFT_SELECT_BANK_BLENDS="c_r10=cubic;c_r20=cubic;c_r20q=quintic;c_r30q=quintic"
export DFT_BASE_ARMS="base,c_r10=bank@c_r10:cons:20,c_r20=bank@c_r20:cons:20,\
c_r20q=bank@c_r20q:cons:20,c_r30q=bank@c_r30q:cons:20"
export DFT_FT_ARMS="${DFT_FT_ARMS:-ft}"
export DFT_DUMP_TRAJ=1
RUN=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
export DFT_SELECT_CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110}"
SEL_DIR="${DFT_SELECT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/$(date +%Y%m%d_%H%M%S)_$$}"
export DFT_SELECT_DIR="$SEL_DIR"

echo "comfort A/B over one bank (K=24), ramp x easing:"
echo "  c_r10   ramp 10 (1.0s) cubic    <- shipped reference"
echo "  c_r20   ramp 20 (2.0s) cubic"
echo "  c_r20q  ramp 20 (2.0s) quintic"
echo "  c_r30q  ramp 30 (3.0s) quintic"
echo "  (ramps > replan_interval_steps=10 are truncated by the next replan)"
echo
bash rl/diffusion_ft/slurm/submit_select_eval.sh

# Ride-quality read-out, gated inside the job if no trajectories were dumped.
RENV="$SEL_DIR/ride.env"
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
{ kv DFT_SELECT_DIR "$SEL_DIR"
  kv DFT_RIDE_SPLITS "${DFT_RIDE_SPLITS:-failure_heldout}"
  kv DFT_RIDE_ARM "${DFT_RIDE_ARM:-}"
  kv DFT_RIDE_TOP "${DFT_RIDE_TOP:-6}"; } > "$RENV"
# The scoring submitter records its array id; chain on THAT, never on a squeue
# guess (which would pick up an unrelated dft_select job from another run).
AID="$(cat "$SEL_DIR/jobid.txt" 2>/dev/null || true)"
[[ -n "$AID" ]] || { echo "FATAL: no $SEL_DIR/jobid.txt -- scoring did not submit" >&2; exit 1; }
# afterANY, not afterok: a time-limit kill still leaves usable dumped states, and
# the ride job gates on their absence itself.
DEP=(--dependency="afterany:${AID}")
RID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch --parsable \
  ${DEP[@]+"${DEP[@]}"} rl/diffusion_ft/slurm/viz_repair.sbatch "$RENV")
echo "ride    -> job $RID (CPU${AID:+, afterany:$AID})"
echo "out     -> $SEL_DIR/ride_*.json, $SEL_DIR/ride_*/"
