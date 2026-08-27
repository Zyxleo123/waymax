#!/bin/bash
# Submitter for DPPO fine-tuning. Only calls sbatch.
#
# Writes a timestamped env file with this run's config and passes its path as a
# positional arg to the batch script (no `sbatch --export`).
#
# Usage (from repo root):
#   # short GPU smoke (2 iters) to shake out integration shapes:
#   DFT_ITERS=2 DFT_INDICES=0,1 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#   # overfit test (8 scenes):
#   bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#   # DPPO + expert anchor. Keep the weight small: at 0.1 the anchor was ~1000x
#   # the PG term and the run was behavior cloning with PPO as a rounding error.
#   DFT_EXPERT_WEIGHT=0.01 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#
# ---- V-Max parity reward recipe --------------------------------------------
# The diagnosis (diffusion_ft/diag/*) showed the default goal-distance reward is
# farmable: offroad is charged once and does NOT terminate while progress is
# unbounded metres, so cutting the corner across a median strictly pays (dppo500
# collected +15 progress for -3 offroad, 66% offroad, 0 solved). This maps the
# reward of the 94.3%-reaching V-Max SAC run onto DPPO: dense progress becomes
# on-route arclength (a beeline off-road earns none), offroad TERMINATES, event
# magnitudes shrink to -1, and potential-based goal shaping (policy-invariant)
# carries credit assignment. Run it on the failure set (ScenarioMax => 10x300):
#
#   DFT_TFRECORD=/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord \
#   DFT_INDICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
#   DFT_ROUTE_REWARD=1 DFT_TERMINATE_OFFROAD=1 DFT_REWARD_PROGRESS=0.2 \
#   DFT_REWARD_OFFROAD=-1.0 DFT_REWARD_COLLISION=-1.0 DFT_REWARD_GOAL_BONUS=5.0 \
#   DFT_REWARD_OFF_ROUTE=-0.2 DFT_GOAL_SHAPING_COEF=1.0 \
#   DFT_ITERS=2 DFT_JOBNAME=dppo_vmax_parity_smoke \
#   bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#
# Terminating on offroad turned collection into a one-step bandit (job 34866: 80%
# of iterations collected ONE replan; the sigma sweep of 20260821_192513 found no
# noise level that both keeps the episode and leaves the KL gate meaningful).
# The per-step variant charges dwell instead of ending the episode, so a noisy
# rollout still yields its full 8 replans of signal:
#
#   DFT_ROUTE_REWARD=1 DFT_OFFROAD_PER_STEP=1 DFT_TERMINATE_OFFROAD=0 \
#   DFT_REWARD_PROGRESS=0.2 DFT_REWARD_OFFROAD=-0.2 DFT_REWARD_COLLISION=-1.0 \
#   DFT_REWARD_GOAL_BONUS=5.0 DFT_REWARD_OFF_ROUTE=-0.2 DFT_GOAL_SHAPING_COEF=1.0 \
#   DFT_ITERS=2 DFT_JOBNAME=dppo_perstep_smoke \
#   bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#
# That run (job 36466) kept all 8 replans but bought them with a new farm: a
# cheap TERMINAL collision escapes the remaining per-step offroad bill (-1.0 once
# vs up to 80 x -0.2 = -16), so the policy learned to crash early --
# corr(return, n_alive) = -0.80, episode length 8.0 -> 5.9 replans, train return
# "improving" -1.82 -> -0.78 purely by shortening, eval solve 0.000 in all 60
# evals. Charge collision by dwell too and the only absorbing state is the goal:
#
#   DFT_ROUTE_REWARD=1 DFT_OFFROAD_PER_STEP=1 DFT_TERMINATE_OFFROAD=0 \
#   DFT_COLLISION_PER_STEP=1 DFT_TERMINATE_COLLISION=0 \
#   DFT_REWARD_PROGRESS=0.2 DFT_REWARD_OFFROAD=-0.2 DFT_REWARD_COLLISION=-0.2 \
#   DFT_REWARD_GOAL_BONUS=5.0 DFT_REWARD_OFF_ROUTE=-0.2 DFT_GOAL_SHAPING_COEF=1.0 \
#   DFT_ITERS=2 DFT_JOBNAME=dppo_noescape_smoke \
#   bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#
# Verified by rl/diffusion_ft/tests/test_collision_per_step.py (CPU, seconds).
#
# ---- Full training corpus + expert step ratio --------------------------------
# Every run above trains on ONE fixed batch of scenes for its whole life, which is
# an overfit target. DFT_SHARD_GLOB switches to a fresh batch drawn uniformly over
# the entire 1000-shard raw-WOMD training set each iteration (O(1) random access
# via the byte-offset indexes -- build them once with
# `bash rl/diffusion_ft/slurm/submit_build_womd_offsets.sh`), with a fixed
# held-out eval set so the eval curve measures generalization.
#
# That corpus is also what the base checkpoint was pretrained on
# (metadata.json: tfrecord_path=.../training_tfexample.tfrecord@1000), which is
# what makes the expert term a coherent anchor. Raw WOMD route dims are 45x800,
# NOT the 10x300 of failures.tfrecord.
#
# DFT_EXPERT_STEPS expert-only gradient steps run after each iteration's PPO
# epochs, so DFT_EXPERT_STEPS : DFT_PPO_EPOCHS is a step ratio. Sweep it with
# `bash rl/diffusion_ft/slurm/sweep_expert_ratio.sh`.
#
#   DFT_SHARD_GLOB='/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*-of-01000' \
#   DFT_NUM_PATHS=45 DFT_NUM_POINTS_PER_PATH=800 \
#   DFT_ROUTE_REWARD=1 DFT_OFFROAD_PER_STEP=1 DFT_TERMINATE_OFFROAD=0 \
#   DFT_COLLISION_PER_STEP=1 DFT_TERMINATE_COLLISION=0 \
#   DFT_REWARD_PROGRESS=0.2 DFT_REWARD_OFFROAD=-0.2 DFT_REWARD_COLLISION=-0.2 \
#   DFT_REWARD_GOAL_BONUS=5.0 DFT_REWARD_OFF_ROUTE=-0.2 DFT_GOAL_SHAPING_COEF=1.0 \
#   DFT_EXPERT_STEPS=4 DFT_PPO_EPOCHS=4 \
#   DFT_ITERS=2 DFT_JOBNAME=dppo_corpus_smoke \
#   bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#
# Read the first .out: `[dppo] route: N/32 scenes have a valid sdc_path` MUST be
# 32/32 before trusting the run -- a 0 there means the route dims are wrong for
# the tfrecord and every progression reward is silently 0.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
# Name the env file after the job so two arms submitted close together cannot
# end up sharing one config (27647/27648 both ran expert_weight=0.01 and were
# accidental duplicates rather than a pure-vs-anchor comparison).
ENV_FILE="$ENV_DIR/dppo_${DFT_JOBNAME:-dft_dppo}_${STAMP}.env"

{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7}"
  echo "DFT_ITERS=${DFT_ITERS:-50}"
  echo "DFT_EXPERT_WEIGHT=${DFT_EXPERT_WEIGHT:-0.0}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_ACTOR_LR=${DFT_ACTOR_LR:-3e-5}"
  echo "DFT_PPO_CLIP=${DFT_PPO_CLIP:-0.01}"
  echo "DFT_GAE_LAMBDA=${DFT_GAE_LAMBDA:-1.0}"
  echo "DFT_PPO_EPOCHS=${DFT_PPO_EPOCHS:-4}"
  echo "DFT_VALUE_EPOCHS=${DFT_VALUE_EPOCHS:-10}"
  echo "DFT_WARMUP=${DFT_WARMUP:-0}"
  echo "DFT_EVAL_EVERY=${DFT_EVAL_EVERY:-5}"
  echo "DFT_EVAL_SEEDS=${DFT_EVAL_SEEDS:-1}"
  echo "DFT_CLEAR_CACHES=${DFT_CLEAR_CACHES:-0}"
  echo "DFT_PROFILE=${DFT_PROFILE:-0}"
  echo "DFT_RESUME=${DFT_RESUME:-}"
  echo "DFT_ITER_OFFSET=${DFT_ITER_OFFSET:-0}"
  echo "DFT_PG_SCALE=${DFT_PG_SCALE:-0}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.1}"
  echo "DFT_TARGET_KL=${DFT_TARGET_KL:-0.05}"
  echo "DFT_SEED=${DFT_SEED:-0}"
  echo "DFT_WANDB=${DFT_WANDB:-0}"
  echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_dppo}"
  echo "DFT_WANDB_GROUP=${DFT_WANDB_GROUP:-}"
  echo "DFT_WANDB_NAME=${DFT_WANDB_NAME:-${DFT_JOBNAME:-}}"
  echo "DFT_OUT_DIR=${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo}"
  # Reward design. Defaults reproduce the original goal-distance reward exactly.
  # The V-Max parity recipe (see the header) is turned on by setting these.
  echo "DFT_ROUTE_REWARD=${DFT_ROUTE_REWARD:-0}"
  echo "DFT_TERMINATE_OFFROAD=${DFT_TERMINATE_OFFROAD:-0}"
  echo "DFT_OFFROAD_PER_STEP=${DFT_OFFROAD_PER_STEP:-0}"
  echo "DFT_COLLISION_PER_STEP=${DFT_COLLISION_PER_STEP:-0}"
  echo "DFT_TERMINATE_COLLISION=${DFT_TERMINATE_COLLISION:-1}"
  echo "DFT_REWARD_PROGRESS=${DFT_REWARD_PROGRESS:-1.0}"
  echo "DFT_REWARD_OFFROAD=${DFT_REWARD_OFFROAD:--5.0}"
  echo "DFT_REWARD_COLLISION=${DFT_REWARD_COLLISION:--10.0}"
  echo "DFT_REWARD_GOAL_BONUS=${DFT_REWARD_GOAL_BONUS:-10.0}"
  echo "DFT_REWARD_OFF_ROUTE=${DFT_REWARD_OFF_ROUTE:--0.2}"
  echo "DFT_GOAL_SHAPING_COEF=${DFT_GOAL_SHAPING_COEF:-0.0}"
  echo "DFT_GOAL_SHAPING_NORMALIZE=${DFT_GOAL_SHAPING_NORMALIZE:-1}"
  # sdc_paths dims for the route reward: ScenarioMax (failures.tfrecord) = 10x300;
  # raw WOMD tf_example = 45x800.
  echo "DFT_NUM_PATHS=${DFT_NUM_PATHS:-10}"
  echo "DFT_NUM_POINTS_PER_PATH=${DFT_NUM_POINTS_PER_PATH:-300}"
  # Full-corpus scene sampling (empty DFT_SHARD_GLOB = the historical fixed batch).
  # Quoted: the glob must reach the batch script unexpanded.
  echo "DFT_SHARD_GLOB='${DFT_SHARD_GLOB:-}'"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-32}"
  echo "DFT_RESAMPLE_EVERY=${DFT_RESAMPLE_EVERY:-1}"
  echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-32}"
  echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
  echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
  # Expert:DPPO step ratio (DFT_EXPERT_STEPS : DFT_PPO_EPOCHS).
  echo "DFT_EXPERT_STEPS=${DFT_EXPERT_STEPS:-0}"
  echo "DFT_EXPERT_LR=${DFT_EXPERT_LR:-0}"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# Optional scheduling overrides (all `general` a6000s are often full). These are
# plain sbatch flags -- NOT --export -- so they don't trigger user-env retrieval.
#   DFT_JOBNAME   distinct job name (also names the log via %x)
#   DFT_PARTITION e.g. project (h200 gpu2) or legacy (v100 gpu20/21/23).
#                 MUST be paired with DFT_QOS: every partition mandates its own
#                 QOS and sbatch otherwise dies on "Invalid qos specification"
#                 (measured 2026-08-19, --partition=project with no --qos).
#                   general -> qos_general (default, no flag needed)
#                   legacy  -> qos_legacy      debug -> qos_debug
#                   cpu     -> qos_cpu         preempt -> qos_preempt
#                   project -> qos_radio_freq | qos_project_preempt
#   DFT_QOS       the QOS matching DFT_PARTITION (see above)
#   DFT_NODELIST  pin a node, e.g. gpu2  (or gpu20)
#   DFT_EXCLUDE   e.g. exclude the 11GB 2080 Tis on legacy
#   DFT_MEM       host RAM, e.g. 96G. Overrides the 48G in train_dppo.sbatch;
#                 job 31985 host-OOM'd at 48G ~58 iters in (slow RSS creep).
SB_FLAGS=()
[[ -n "${DFT_JOBNAME:-}" ]]   && SB_FLAGS+=(--job-name="$DFT_JOBNAME")
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
[[ -n "${DFT_MEM:-}" ]]       && SB_FLAGS+=(--mem="$DFT_MEM")

if [[ -n "${DFT_PARTITION:-}" && -z "${DFT_QOS:-}" && "$DFT_PARTITION" != "general" ]]; then
  echo "WARNING: DFT_PARTITION=$DFT_PARTITION set without DFT_QOS; sbatch will"
  echo "WARNING: likely fail with 'Invalid qos specification'." >&2
fi
echo "sbatch flags: ${SB_FLAGS[*]:-<none>}"
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch "${SB_FLAGS[@]}" rl/diffusion_ft/slurm/train_dppo.sbatch "$ENV_FILE"
