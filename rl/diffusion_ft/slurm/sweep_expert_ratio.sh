#!/bin/bash
# Sweep the expert:DPPO STEP RATIO on the full WOMD training corpus.
#
# Each arm is a full fine-tuning run of the 2026-06-16 pretrained diffusion
# checkpoint: fresh scenes drawn uniformly from all 1000 raw-WOMD training shards
# every iteration, a fixed held-out eval set, and `DFT_EXPERT_STEPS` expert
# (behavior-cloning) gradient steps per `DFT_PPO_EPOCHS` DPPO steps. The arms are
# siblings with no dependencies, so they queue concurrently; each has its own job
# name, log, and out_dir, so a bad arm cannot corrupt another.
#
# This does no work of its own -- it only calls the tested single-run submitter
# once per grid cell (which writes one timestamped env file per submission and
# passes its path positionally; never `sbatch --export`).
#
# Grid (override by exporting a space-separated list):
#   DFT_EXPERT_STEPS_GRID   expert steps per iteration   (default "0 1 2 4 8 16")
#   DFT_PPO_EPOCHS          DPPO steps per iteration     (default 4, the denominator)
#
# The two ends are the controls the middle is read against: `0` is pure DPPO on
# the corpus, and a large value is essentially DAgger-style BC with DPPO as a
# rounding error. What we are looking for is the ratio where the eval solve rate
# beats BOTH -- i.e. the reward is actually adding something the demonstrations
# do not, without the exploration noise walking the pretrained policy downhill.
#
# Read the arms by `eval_solve_deterministic` on the held-out set (the objective),
# with `kl_gauss` vs `kl_gauss_expert` in train_log.jsonl saying how much of each
# iteration's policy motion came from DPPO vs the expert -- that split is what
# makes "balance" a measurement rather than a guess.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/sweep_expert_ratio.sh
#   # smoke the whole grid for 2 iterations first:
#   DFT_ITERS=2 DFT_EVAL_EVERY=1 bash rl/diffusion_ft/slurm/sweep_expert_ratio.sh
#   # fewer arms, longer:
#   DFT_EXPERT_STEPS_GRID="0 2 8" DFT_ITERS=1000 \
#     bash rl/diffusion_ft/slurm/sweep_expert_ratio.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

EXPERT_STEPS_GRID="${DFT_EXPERT_STEPS_GRID:-0 1 2 4 8 16}"
BASE_OUT="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo_corpus}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- shared configuration for every arm -------------------------------------
# Data: the whole training corpus, which is also the corpus the base checkpoint
# was pretrained on -- so the expert loss anchors to the same distribution.
# No-colon `-` (not `:-`): an explicitly EMPTY DFT_SHARD_GLOB must survive, because
# empty is what switches train_dppo.sbatch from corpus sampling back to a fixed
# DFT_TFRECORD/DFT_INDICES batch -- i.e. how this sweep is pointed at the failure
# set. With `:-` an empty override silently reverts to the corpus. Same idiom as
# DFT_EXCLUDE in sweep_reward.sh.
export DFT_SHARD_GLOB="${DFT_SHARD_GLOB-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*-of-01000}"
export DFT_SCENES_PER_ITER="${DFT_SCENES_PER_ITER:-32}"
export DFT_EVAL_SCENES="${DFT_EVAL_SCENES:-32}"
export DFT_RESAMPLE_EVERY="${DFT_RESAMPLE_EVERY:-1}"
# Raw WOMD tf_example route dims. ScenarioMax's 10x300 would silently yield zero
# valid routes and a dead progression reward -- watch the `route[train]:` line.
export DFT_NUM_PATHS="${DFT_NUM_PATHS:-45}"
export DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-800}"
# Reward: the "no escape" recipe. Offroad and collision are both charged per
# dwell step and NEITHER terminates, so the goal is the only absorbing state --
# the policy cannot shorten its way out of a penalty, and every episode yields
# its full replan budget of signal.
export DFT_ROUTE_REWARD="${DFT_ROUTE_REWARD:-1}"
export DFT_OFFROAD_PER_STEP="${DFT_OFFROAD_PER_STEP:-1}"
export DFT_TERMINATE_OFFROAD="${DFT_TERMINATE_OFFROAD:-0}"
export DFT_COLLISION_PER_STEP="${DFT_COLLISION_PER_STEP:-1}"
export DFT_TERMINATE_COLLISION="${DFT_TERMINATE_COLLISION:-0}"
export DFT_REWARD_PROGRESS="${DFT_REWARD_PROGRESS:-0.2}"
export DFT_REWARD_OFFROAD="${DFT_REWARD_OFFROAD:--0.2}"
export DFT_REWARD_COLLISION="${DFT_REWARD_COLLISION:--0.2}"
export DFT_REWARD_GOAL_BONUS="${DFT_REWARD_GOAL_BONUS:-5.0}"
export DFT_REWARD_OFF_ROUTE="${DFT_REWARD_OFF_ROUTE:--0.2}"
export DFT_GOAL_SHAPING_COEF="${DFT_GOAL_SHAPING_COEF:-1.0}"
export DFT_ITERS="${DFT_ITERS:-1000}"
export DFT_PPO_EPOCHS="${DFT_PPO_EPOCHS:-4}"
export DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
export DFT_EVAL_SEEDS="${DFT_EVAL_SEEDS:-2}"
# Host RAM: 48G host-OOM'd a fixed-batch run ~58 iters in (job 31985), and the
# corpus stream holds a prefetched batch on top of that.
export DFT_MEM="${DFT_MEM:-96G}"

n=0
echo "Sweep expert_steps=[$EXPERT_STEPS_GRID] over ppo_epochs=$DFT_PPO_EPOCHS"
echo "base_out=$BASE_OUT/$STAMP  iters=$DFT_ITERS  scenes/iter=$DFT_SCENES_PER_ITER"
for es in $EXPERT_STEPS_GRID; do
  arm="x${es}_p${DFT_PPO_EPOCHS}"
  DFT_EXPERT_STEPS="$es" \
  DFT_JOBNAME="dppo_corpus_${arm}" \
  DFT_OUT_DIR="$BASE_OUT/$STAMP/$arm" \
  DFT_WANDB_GROUP="${DFT_WANDB_GROUP:-dppo_corpus_$STAMP}" \
    bash rl/diffusion_ft/slurm/submit_train_dppo.sh
  n=$((n + 1))
  echo "  [$n] submitted arm $arm (expert:dppo = ${es}:${DFT_PPO_EPOCHS})"
done
echo "Submitted $n sibling arms -> $BASE_OUT/$STAMP"
