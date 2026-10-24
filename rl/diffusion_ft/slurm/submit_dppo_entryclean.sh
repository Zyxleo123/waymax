#!/bin/bash
# Launch pure DPPO on the ORDERING-CORRECTED reward (`entry_clean`), N seeds as
# sibling jobs. Does no work of its own -- it only calls the tested single-run
# submitter once per seed.
#
# ---- why this run --------------------------------------------------------
# Every DPPO sweep before this one moved the return without moving the solve
# rate, across a 5x span in penalty and a 5x span in goal bonus. The reason was
# not magnitude but ORDERING: `sbatch rl/diffusion_ft/slurm/reward_ranking.sbatch`
# (job 39053) scored episode archetypes under each shipped recipe and found that
# noescape/penalty_up/perstep all rank a 2-step corner cut (+8.60) AND never
# arriving at all (+8.00) ABOVE a clean prompt arrival (+7.00) -- because
# progression_indicator pays progress per world step alive, so reaching the goal
# terminates the episode and forfeits the rest. R(reach@k) rises monotonically
# 8:+6.60 16:+8.20 24:+9.60 32:+11.20: arriving promptly was punished.
#
# `entry_clean` is the only recipe that passes all four ordering properties:
#   P1 solve-is-argmax   argmax=clean_solve_fast (+10.00), 0 inversions
#   P2 avoidance>dwell   ratio 31.50 (was 1.00-6.00)
#   P3 excursions priced 2nd excursion costs +1.00 (was free under all three)
#   P4 promptness        price of dawdling +0.05 (was +4.60)
#
# So this run is the actual test of the thesis: with the ordering corrected and
# NOTHING else changed, does eval_solve move? Pure DPPO, expert_steps=0 -- the
# BC ratio is a separate lever (sweep_expert_ratio.sh) and mixing it in here
# would confound which of the two did the work.
#
# ---- how to read it ------------------------------------------------------
# The reward scale moves entirely vs every earlier run, so do NOT compare the
# return curve to anything before this. Read `eval_solve_deterministic` and
# `eval_offroad_deterministic`, and expect return and solve to move TOGETHER --
# that co-movement is what every previous sweep lacked. Baseline to beat: the
# old recipe held solve at 0.000-0.031 for 500 iters on these same 32 scenes.
#
# 3 seeds because a 32-scene solve rate moves 0.031 per scene, so a single-seed
# read of a "trend" is a coin flip (the s*/lr* sweep already lost time to that).
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_dppo_entryclean.sh
#   DFT_SEED_GRID="0 1 2 3 4" DFT_ITERS=1000 \
#     bash rl/diffusion_ft/slurm/submit_dppo_entryclean.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"

SEED_GRID="${DFT_SEED_GRID:-0 1 2}"
BASE_OUT="${DFT_OUT_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo_entryclean}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- data: the failure set. ScenarioMax route dims are 10x300; raw WOMD's
# 45x800 would yield zero valid routes and a silently dead progression reward.
export DFT_TFRECORD="${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
export DFT_INDICES="${DFT_INDICES:-$(seq -s, 0 31)}"
export DFT_NUM_PATHS="${DFT_NUM_PATHS:-10}"
export DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-300}"

# ---- the entry_clean recipe (the one that passes reward_ranking) -----------
export DFT_ROUTE_REWARD="${DFT_ROUTE_REWARD:-1}"
export DFT_OFFROAD_PER_STEP="${DFT_OFFROAD_PER_STEP:-1}"
export DFT_TERMINATE_OFFROAD="${DFT_TERMINATE_OFFROAD:-0}"
export DFT_COLLISION_PER_STEP="${DFT_COLLISION_PER_STEP:-1}"
export DFT_TERMINATE_COLLISION="${DFT_TERMINATE_COLLISION:-0}"
export DFT_PROGRESSION_INDICATOR="${DFT_PROGRESSION_INDICATOR:-0}"
export DFT_CLEAN_GOAL_BONUS="${DFT_CLEAN_GOAL_BONUS:-1}"
export DFT_REWARD_PROGRESS="${DFT_REWARD_PROGRESS:-5.0}"
export DFT_REWARD_OFFROAD="${DFT_REWARD_OFFROAD:--0.1}"
export DFT_REWARD_OFFROAD_ENTRY="${DFT_REWARD_OFFROAD_ENTRY:--1.0}"
export DFT_REWARD_COLLISION="${DFT_REWARD_COLLISION:--0.1}"
export DFT_REWARD_COLLISION_ENTRY="${DFT_REWARD_COLLISION_ENTRY:--2.0}"
export DFT_REWARD_GOAL_BONUS="${DFT_REWARD_GOAL_BONUS:-5.0}"
export DFT_REWARD_OFF_ROUTE="${DFT_REWARD_OFF_ROUTE:--0.2}"
export DFT_GOAL_SHAPING_COEF="${DFT_GOAL_SHAPING_COEF:-1.0}"
export DFT_GOAL_SHAPING_NORMALIZE="${DFT_GOAL_SHAPING_NORMALIZE:-1}"

# ---- optimizer: the ONLY pair demonstrably observed to take steps (the
# 20260826 reward sweep applied 100% of updates and drifted kl_gauss 0.12->0.29
# at these values). The submitter defaults, 0.1/0.05, are a 10x tighter KL gate
# -- that coupling is what froze learning in the s*_k* sweep.
export DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.03}"
export DFT_TARGET_KL="${DFT_TARGET_KL:-0.556}"

export DFT_EXPERT_STEPS="${DFT_EXPERT_STEPS:-0}"   # pure DPPO; BC is a separate lever
export DFT_ITERS="${DFT_ITERS:-500}"
export DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
export DFT_EVAL_SEEDS="${DFT_EVAL_SEEDS:-3}"

# Disk: one ckpt is ~606 MB and 1075 of them (~634 GB) exhausted the quota
# mid-sweep on 20260830. Save rarely, keep few. train_log.jsonl is never pruned.
export DFT_SAVE_EVERY="${DFT_SAVE_EVERY:-50}"
export DFT_KEEP_CKPTS="${DFT_KEEP_CKPTS:-3}"

# gpu28 is the 24 GB a5000; every arm that completed this config ran on a 48 GB
# a6000, and gpu28 is where an arm died in a 13.05 GiB eval transfer (job 37156).
export DFT_EXCLUDE="${DFT_EXCLUDE-}"
export DFT_WANDB="${DFT_WANDB:-1}"
export DFT_WANDB_PROJECT="${DFT_WANDB_PROJECT:-diffusion_ft_dppo}"
export DFT_WANDB_GROUP="${DFT_WANDB_GROUP:-entryclean_$STAMP}"

echo "entry_clean DPPO: seeds=[$SEED_GRID] iters=$DFT_ITERS -> $BASE_OUT/$STAMP"
n=0
for s in $SEED_GRID; do
  DFT_SEED="$s" \
  DFT_JOBNAME="dppo_ec_s${s}" \
  DFT_OUT_DIR="$BASE_OUT/$STAMP/s${s}" \
    bash rl/diffusion_ft/slurm/submit_train_dppo.sh
  n=$((n + 1))
  echo "  [$n] submitted seed $s"
done
echo "Submitted $n sibling seeds -> $BASE_OUT/$STAMP"
echo
echo "Check the FIRST .out before trusting the run:"
echo "  * 'route[train]: 31/32' or better, else progression is identically 0"
echo "  * it=0000 reward line shows prog_indicator=0 clean_goal=1 offroad_entry=-1.0"
