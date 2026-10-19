#!/bin/bash
# Sweep submitter for the DPPO *reward shaping*: fan the per-step penalty scale
# out against the goal bonus as independent sibling jobs, then queue one CPU
# aggregation job behind them all. Does no work of its own -- it only calls the
# tested single-run submitter (submit_train_dppo.sh) once per grid cell and
# `sbatch` once for the aggregator.
#
# ---- why this sweep ---------------------------------------------------------
# Under the per-step (dwell-charged, non-terminating) recipe the reward is, per
# world step:  progress +P if the on-route arclength advanced at all (a bounded
# INDICATOR -- `progression_indicator=route_reward` in train_dppo -- not metres),
# off_route -0.2 if off the route, offroad C_off if off-road, collision C_col if
# in contact, plus a one-shot goal bonus that also ends the episode. An episode
# is 8 replans x 5 = 40 world steps, so at P=0.2 the entire progress budget of a
# perfect episode is +8.0.
#
# That fixes the exchange rate. |C_off| / P is how many steps of the best
# possible progress one off-road step forfeits:
#
#   job 36946 "noescape"   C=-0.2  -> ratio 1.0: an off-road step costs exactly
#     the progress step it buys. The policy drove (eval return 4.7 -> 9.3) and
#     stayed dirty. Never instrumented with the rates, so we cannot see how dirty.
#   job 37016 "penalty_up" C=-1.0/-1.5 -> ratio 5.0/7.5: one 8-step excursion
#     wipes out the whole episode's progress. Eval return -24 -> -2.7 (an order
#     of magnitude) but off-road-ever went UP 0.28 -> 0.50 and the goal was
#     reached in 4/32. Return was bought by shortening excursions and by not
#     driving, which is exactly what a dwell charge pays for.
#
# The balance point is between those two, so the default grid walks the scale
# across them at two goal bonuses -- the goal bonus being the only term that
# rewards finishing rather than merely not-offending, and at 5.0 it is worth 25
# progress steps against a penalty budget that can reach -40.
#
# Grid (override by exporting a space-separated list):
#   DFT_PENALTY_GRID     per-step penalty scale, applied as offroad = -scale
#                        and collision = -scale * DFT_COL_RATIO
#                        (default "0.2 0.5 1.0")
#   DFT_GOAL_BONUS_GRID  one-shot goal bonus       (default "5 25")
#   DFT_COL_RATIO        collision:offroad ratio   (default 1.0; penalty_up
#                        used 1.5 -- set it here to keep that weighting)
# Cell (0.2, 5) at ratio 1.0 reproduces `noescape` exactly, now with the rates
# logged, so the sweep carries its own control.
#
# Any other DFT_* (DFT_INDICES, DFT_ITERS, DFT_SIGMA_FLOOR, DFT_TARGET_KL,
# DFT_CKPT, DFT_PARTITION, DFT_NODELIST, ...) is inherited by every arm.
#
# Eval defaults differ from the single-run submitter on purpose: 3 seeds every
# 10 iterations instead of 1 every 5. Same wall-clock, but a 32-scene rate moves
# 0.031 per scene and a 6-way ranking read off single-seed points is a coin flip
# (the s*/lr* sweep already lost time to reading such spikes as trends).
#
# Usage (from repo root):
#   # default 3x2 = 6 arms on the failure set:
#   bash rl/diffusion_ft/slurm/sweep_reward.sh
#   # keep penalty_up's 1.5x collision weighting:
#   DFT_COL_RATIO=1.5 bash rl/diffusion_ft/slurm/sweep_reward.sh
#   # finer scan of the scale only, one goal bonus:
#   DFT_PENALTY_GRID="0.3 0.4 0.5 0.7" DFT_GOAL_BONUS_GRID=5 \
#     bash rl/diffusion_ft/slurm/sweep_reward.sh
#
# Read the first .out before trusting any arm: `[dppo] route: N/32 scenes have a
# valid sdc_path` MUST be 31/32 or better, or every progression reward is 0.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"

PENALTY_GRID="${DFT_PENALTY_GRID:-0.2 0.5 1.0}"
GOAL_GRID="${DFT_GOAL_BONUS_GRID:-5 25}"
COL_RATIO="${DFT_COL_RATIO:-1.0}"

# Keep the arms off the a5000. gpu28 is the only non-a6000 in `general` (24 GB
# against 48), and it is where arm p1p0_g25 died on 2026-08-25 (job 37156):
# RESOURCE_EXHAUSTED allocating 13.05 GiB in a device->host transfer at eval.
# Every arm that ran to completion on this config was on gpu24 (a6000). Override
# with DFT_EXCLUDE= (empty) to let it back in.
export DFT_EXCLUDE="${DFT_EXCLUDE-}"

BASE_OUT="${DFT_OUT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo_reward}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUT="$BASE_OUT/$STAMP"

# The per-step recipe every arm shares (job 36466 showed why: with offroad
# dwell-charged but collision terminal, a cheap crash escapes the rest of the
# bill and the policy learns to crash early). Both dwell-charged, neither
# terminal, so the goal is the only absorbing state.
export DFT_TFRECORD="${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
export DFT_INDICES="${DFT_INDICES:-$(seq -s, 0 31)}"
export DFT_ROUTE_REWARD="${DFT_ROUTE_REWARD:-1}"
export DFT_OFFROAD_PER_STEP="${DFT_OFFROAD_PER_STEP:-1}"
export DFT_TERMINATE_OFFROAD="${DFT_TERMINATE_OFFROAD:-0}"
export DFT_COLLISION_PER_STEP="${DFT_COLLISION_PER_STEP:-1}"
export DFT_TERMINATE_COLLISION="${DFT_TERMINATE_COLLISION:-0}"
export DFT_REWARD_PROGRESS="${DFT_REWARD_PROGRESS:-0.2}"
export DFT_REWARD_OFF_ROUTE="${DFT_REWARD_OFF_ROUTE:--0.2}"
export DFT_GOAL_SHAPING_COEF="${DFT_GOAL_SHAPING_COEF:-1.0}"
export DFT_GOAL_SHAPING_NORMALIZE="${DFT_GOAL_SHAPING_NORMALIZE:-1}"
export DFT_NUM_PATHS="${DFT_NUM_PATHS:-10}"
export DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-300}"
export DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.03}"
export DFT_TARGET_KL="${DFT_TARGET_KL:-0.556}"
export DFT_ITERS="${DFT_ITERS:-500}"
export DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
export DFT_EVAL_SEEDS="${DFT_EVAL_SEEDS:-3}"
export DFT_WANDB="${DFT_WANDB:-1}"
export DFT_WANDB_PROJECT="${DFT_WANDB_PROJECT:-diffusion_ft_dppo}"
export DFT_WANDB_GROUP="${DFT_WANDB_GROUP:-reward_$STAMP}"

# `0.2` / `1.5` -> `0p2` / `1p5` for use in a job name / path.
tag() { echo "$1" | sed 's/\./p/g; s/-//g; s/+//g'; }

# Every scale/ratio/bonus is a shell string; do the arithmetic in awk rather
# than bash (which has no floats) so `0.5 * 1.5` is -0.75 and not a syntax error.
mul() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%g", a*b}'; }

echo "Reward sweep: penalty=[$PENALTY_GRID] x goal_bonus=[$GOAL_GRID]" \
     "(collision = ratio $COL_RATIO x offroad), out=$SWEEP_OUT"
mkdir -p "$SWEEP_OUT"

JOBIDS=()
n=0
for pen in $PENALTY_GRID; do
  for goal in $GOAL_GRID; do
    off="-$pen"
    col="-$(mul "$pen" "$COL_RATIO")"
    arm="p$(tag "$pen")_g$(tag "$goal")"
    ARM_OUT="$SWEEP_OUT/$arm"
    mkdir -p "$ARM_OUT"
    # Arm metadata for the aggregator: it labels the table from this file rather
    # than parsing the directory name back into floats.
    cat > "$ARM_OUT/arm.json" <<JSON
{"arm": "$arm", "offroad": $off, "collision": $col, "goal_bonus": $goal,
 "progress": $DFT_REWARD_PROGRESS, "off_route": $DFT_REWARD_OFF_ROUTE,
 "col_ratio": $COL_RATIO, "iters": $DFT_ITERS,
 "penalty_over_progress": $(awk -v a="$pen" -v b="$DFT_REWARD_PROGRESS" 'BEGIN{printf "%g", a/b}')}
JSON
    out="$(DFT_REWARD_OFFROAD="$off" \
           DFT_REWARD_COLLISION="$col" \
           DFT_REWARD_GOAL_BONUS="$goal" \
           DFT_JOBNAME="dppo_rw_$arm" \
           DFT_OUT_DIR="$ARM_OUT" \
           DFT_WANDB_NAME="dppo_rw_$arm" \
           bash rl/diffusion_ft/slurm/submit_train_dppo.sh)"
    echo "$out" | tail -2
    jid="$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)"
    if [[ -n "$jid" ]]; then
      JOBIDS+=("$jid")
    else
      echo "  WARNING: could not read a job id for arm $arm; it will not gate the aggregator." >&2
    fi
    n=$((n + 1))
    echo "  [$n] arm $arm  offroad=$off collision=$col goal=$goal  job=${jid:-?}"
  done
done

# One CPU aggregation job behind all of them. `afterany`, not `afterok`: an arm
# that hits the time limit still wrote 400 iterations of eval rows worth reading,
# and `afterok` would strand the table on DependencyNeverSatisfied instead.
if ((${#JOBIDS[@]})); then
  DEP="afterany:$(IFS=:; echo "${JOBIDS[*]}")"
  agg="$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --dependency="$DEP" --kill-on-invalid-dep=no \
      rl/diffusion_ft/slurm/aggregate_reward_sweep.sbatch "$SWEEP_OUT")"
  echo "$agg  (dependency=$DEP)"
fi

echo "Submitted $n arms -> $SWEEP_OUT"
echo "Re-render the table any time without waiting:"
echo "  sbatch rl/diffusion_ft/slurm/aggregate_reward_sweep.sbatch $SWEEP_OUT"
