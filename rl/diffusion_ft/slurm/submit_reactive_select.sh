#!/bin/bash
# Does the shipping repair bank survive reactive NPCs? Only calls sbatch.
#
# QUESTION. Every selection number this project has -- `b_l24`'s +0.0103 m2, and
# `cons`'s structural `broken 0.0000` -- was measured against log-replayed agents
# that cannot see the ego. Job 58232 put a number on what that is worth: an ego
# running 40 world steps late collides 0.719 under rails and 0.219 under braking
# NPCs, so up to 0.500 of the measured collision rate is charged for a rear-end
# the ego did not cause.
#
# WHY THE ORDERING IS AT RISK, NOT JUST THE LEVEL. The inflation grows with how
# far behind schedule an arm is, and the arms lag by very different amounts
# (failure_anatomy, jobs 58135/58136: ft is 2.58 m / 6 steps behind at first
# contact, `c_r10` is 14.07 m / 11 steps, 87% of its colliders dead astern,
# |bearing| p50 176 deg). The arm that lags most is penalised most, so removing
# the rails should help `c_r10` MORE than `base`. That is this run's falsifiable
# prediction: the bank's margin over base should GROW, not shrink.
#
# DESIGN. Two sibling jobs, identical in everything but the simulator:
#   rails/  DFT_SIM_AGENTS=none       -- reproduces the shipping measurement
#   react/  DFT_SIM_AGENTS=idm_brake  -- the same arms under braking NPCs
# Arms are paired draw-for-draw WITHIN a job and not across one, so read arm
# gaps inside each simulator and `base` for the level shift between them. They
# are siblings, not a chain, so they queue concurrently.
#
# COST: each job is the l24 ladder's shape (~1 h per task, 2 shards), so budget
# roughly double a single submit_select_eval.sh run. Use the smoke first.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_reactive_select.sh
#   bash rl/diffusion_ft/slurm/submit_reactive_select.sh
#   DFT_REACT_SIMS=idm_brake bash ...    # skip the rails arm (already measured)
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"

# The shipping bank, verbatim from submit_ft_parity.sh so the two runs are the
# same candidate set and any difference belongs to the simulator.
L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"

export DFT_SELECT_BANKS="l24=${L24}"
export DFT_SELECT_BANK="$L24"
# Both shipping ramps: r20 is the ship decision (comfort), r10 is what the
# failure-anatomy numbers above were taken on. Keeping both means the comfort
# choice does not have to be re-litigated if the ordering moves.
export DFT_BASE_ARMS="${DFT_BASE_ARMS:-base,b_l24_r20=bank@l24:cons:20,b_l24_r10=bank@l24:cons:10}"
export DFT_FT_ARMS="${DFT_FT_ARMS:-ft}"
RUN=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
export DFT_SELECT_CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110}"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ROOT_DIR="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/reactive_${STAMP}"

for sim in ${DFT_REACT_SIMS:-none idm_brake}; do
  echo "=== sim_agents=$sim ==="
  # Each sibling gets its OWN select dir; a shared one would interleave
  # rows.jsonl from two different simulators into one analysis.
  DFT_SIM_AGENTS="$sim" \
  DFT_SELECT_DIR="$ROOT_DIR/$sim" \
    bash rl/diffusion_ft/slurm/submit_select_eval.sh
  echo
done

echo "both arms under $ROOT_DIR"
echo "read: base tells you the level shift between simulators;"
echo "      (b_l24 - base) inside each tells you whether the bank survives."
