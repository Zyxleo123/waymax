#!/bin/bash
#SBATCH --job-name=dft_dppo
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# DPPO fine-tuning of the diffusion policy. `general` hosts only a6000/a5000
# (>=24 GB), pinning the GPU class away from the 11 GB 2080 Tis.
#
# Submit via the companion submitter (writes a timestamped env file, passes its
# path as $1). Never use `sbatch --export` (requeue-hold on this cluster).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

# Config (overridable via the env file / positional VAR=value args).
DFT_INDICES="${DFT_INDICES:-0,1,2,3,4,5,6,7}"
DFT_ITERS="${DFT_ITERS:-50}"
DFT_EXPERT_WEIGHT="${DFT_EXPERT_WEIGHT:-0.0}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_ACTOR_LR="${DFT_ACTOR_LR:-3e-5}"
DFT_PPO_CLIP="${DFT_PPO_CLIP:-0.01}"   # clip acts on the averaged ratio; start tight
DFT_GAE_LAMBDA="${DFT_GAE_LAMBDA:-1.0}"   # 1.0 = Monte-Carlo advantage
DFT_PPO_EPOCHS="${DFT_PPO_EPOCHS:-4}"
DFT_VALUE_EPOCHS="${DFT_VALUE_EPOCHS:-10}"
DFT_WARMUP="${DFT_WARMUP:-0}"   # critic-only iters before the actor is unfrozen
DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-5}"   # fixed-seed eval cadence (higher = faster)
DFT_EVAL_SEEDS="${DFT_EVAL_SEEDS:-1}"   # >1 averages eval over N seeds (reports std)
DFT_CLEAR_CACHES="${DFT_CLEAR_CACHES:-0}"   # 1 = jax.clear_caches() each iter (leak probe/stopgap)
DFT_PROFILE="${DFT_PROFILE:-0}"   # 1 = per-phase timing (collect sample/env, update)
DFT_RESUME="${DFT_RESUME:-}"          # save() ckpt dir to restore actor+critic and continue
DFT_CRITIC_INIT="${DFT_CRITIC_INIT:-}"  # save() ckpt dir to restore ONLY the critic (actor stays base)
# One ckpt is ~606 MB (actor + critic + both Adam optimizer states). At save_every
# 10 a 500-iter run writes 50 = ~30 GB, and a 6-arm sweep ~180 GB; nothing pruned
# them until 1075 had piled up (~634 GB) and every arm of the 20260830 ratio sweep
# died 75-157 iters in on "Disk quota exceeded". Save less often, keep only the
# newest few. train_log.jsonl is never pruned.
DFT_SAVE_EVERY="${DFT_SAVE_EVERY:-50}"
DFT_KEEP_CKPTS="${DFT_KEEP_CKPTS:-3}"   # 0 = keep every checkpoint (the old behavior)
DFT_ITER_OFFSET="${DFT_ITER_OFFSET:-0}"   # add to logged iter so a resumed curve continues
DFT_PG_SCALE="${DFT_PG_SCALE:-0}"   # 0 = auto (channels*prefix_len)
# Equal sample/log-prob sigma floor (the actor requires equality). 0.1 adds ~1sigma
# of ~10 m / 2.5 m/s / 18deg at the final step (outputs scaled by 100/25/pi); sweep
# ~0.0025/0.005/0.01 for usable rollouts.
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.1}"
# Exact Gaussian trajectory-KL early-stop target (nats); see --target_kl help.
DFT_TARGET_KL="${DFT_TARGET_KL:-0.05}"
DFT_SEED="${DFT_SEED:-0}"
DFT_WANDB="${DFT_WANDB:-0}"   # 1 = mirror per-iter rows to Weights & Biases
DFT_WANDB_PROJECT="${DFT_WANDB_PROJECT:-diffusion_ft_dppo}"
DFT_WANDB_GROUP="${DFT_WANDB_GROUP:-}"
DFT_WANDB_NAME="${DFT_WANDB_NAME:-${DFT_JOBNAME:-}}"
DFT_OUT_DIR="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_TFRECORD="${DFT_TFRECORD:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"
# Reward design (defaults reproduce the original goal-distance reward exactly).
DFT_ROUTE_REWARD="${DFT_ROUTE_REWARD:-0}"
DFT_TERMINATE_OFFROAD="${DFT_TERMINATE_OFFROAD:-0}"
# Charge the offroad penalty every off-road world step instead of once. The
# non-terminating alternative to DFT_TERMINATE_OFFROAD=1: keeps the episode (and
# so the gradient signal) intact under collection noise. Cost scales with dwell,
# so pair it with a smaller DFT_REWARD_OFFROAD (~-0.2).
DFT_OFFROAD_PER_STEP="${DFT_OFFROAD_PER_STEP:-0}"
# The collision twin of the two flags above, and the reason both exist: with
# offroad dwell-charged, a cheap TERMINAL collision is an escape from the rest of
# the bill (80 steps x -0.2 = -16 vs -1.0 once, then the world freezes), so the
# policy learns to crash early -- measured in job 36466 as
# corr(return, n_alive) = -0.80 with solve stuck at 0.000. Set
# DFT_COLLISION_PER_STEP=1 with DFT_TERMINATE_COLLISION=0 and the only absorbing
# state left is the goal. Cost scales with dwell: pair with ~-0.2.
DFT_COLLISION_PER_STEP="${DFT_COLLISION_PER_STEP:-0}"
DFT_TERMINATE_COLLISION="${DFT_TERMINATE_COLLISION:-1}"
DFT_REWARD_PROGRESS="${DFT_REWARD_PROGRESS:-1.0}"
DFT_REWARD_OFFROAD="${DFT_REWARD_OFFROAD:--5.0}"
DFT_REWARD_COLLISION="${DFT_REWARD_COLLISION:--10.0}"
DFT_REWARD_GOAL_BONUS="${DFT_REWARD_GOAL_BONUS:-10.0}"
DFT_REWARD_OFF_ROUTE="${DFT_REWARD_OFF_ROUTE:--0.2}"
# ---- reward ORDERING knobs (see rl/diffusion_ft/tests/test_reward_ranking.py) --
# The three flags below are the ones that change which trajectories the reward
# RANKS highest, as opposed to how hard it pushes. Every recipe shipped before
# them scores a 2-step corner cut above a clean prompt arrival, and scores never
# reaching the goal above reaching it quickly -- which is why every sweep so far
# moved the return without moving the solve rate.
#
# Entry charge: levied on each on-road -> off-road transition. A per-step charge
# is paid for making an excursion shorter; the charged-once form makes the second
# excursion free. Only this term is paid for not leaving the road at all. Pair
# with DFT_OFFROAD_PER_STEP=1 and a small DFT_REWARD_OFFROAD (~-0.1).
DFT_REWARD_OFFROAD_ENTRY="${DFT_REWARD_OFFROAD_ENTRY:-0.0}"
# The contact twin; only meaningful with DFT_TERMINATE_COLLISION=0.
DFT_REWARD_COLLISION_ENTRY="${DFT_REWARD_COLLISION_ENTRY:-0.0}"
# Pay the goal bonus only to an episode that arrives having never gone off-road
# or collided -- i.e. only to a scene the solve metric counts. Puts the reward's
# discontinuity where the solve criterion's already is.
DFT_CLEAN_GOAL_BONUS="${DFT_CLEAN_GOAL_BONUS:-0}"
# 1 (default, and what every run so far used) = progress paid per world step in
# which the route fraction advanced: a DURATION reward, total ~progress x episode
# length, so arriving at the goal forfeits the remainder. 0 = progress*(s-prev_s),
# telescoping to progress x (fraction of route covered), independent of duration.
# waymax sdc_progression is normalized to [0,1], so the telescoping form wants a
# coefficient sized against the goal bonus (~5.0), NOT the 0.2 the indicator uses.
DFT_PROGRESSION_INDICATOR="${DFT_PROGRESSION_INDICATOR:-1}"
DFT_REWARD_STEP_PENALTY="${DFT_REWARD_STEP_PENALTY:-0.0}"
DFT_GOAL_SHAPING_COEF="${DFT_GOAL_SHAPING_COEF:-0.0}"
DFT_GOAL_SHAPING_NORMALIZE="${DFT_GOAL_SHAPING_NORMALIZE:-1}"
DFT_NUM_PATHS="${DFT_NUM_PATHS:-10}"
DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-300}"
# ---- full-corpus training ---------------------------------------------------
# Set DFT_SHARD_GLOB to train on the WHOLE WOMD training set (a fresh batch drawn
# uniformly over all 1000 shards every DFT_RESAMPLE_EVERY iterations, with a fixed
# held-out eval set) instead of one memorized batch of DFT_INDICES. Needs the
# byte-offset indexes: bash rl/diffusion_ft/slurm/submit_build_womd_offsets.sh
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-}"
# Restrict corpus sampling to a selected POOL of scene ids (a hard_womd.json from
# rl/diffusion_ft/tools/select_hard_womd.py). Empty = uniform over the glob, which
# on WOMD means the base policy already solves ~95-98% and there is no headroom to
# read a result in. Requires DFT_SHARD_GLOB and the SAME glob it was scanned with.
DFT_SCENE_IDS_JSON="${DFT_SCENE_IDS_JSON:-}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SCENES_PER_ITER="${DFT_SCENES_PER_ITER:-32}"
DFT_RESAMPLE_EVERY="${DFT_RESAMPLE_EVERY:-1}"
DFT_EVAL_SCENES="${DFT_EVAL_SCENES:-32}"
# Additionally eval the FIXED named splits (failure/success held-out +
# failure/success train, each capped) from submit_build_splits.sh's
# eval_splits.json, logged under per-split `<split>/` prefixes. Needs
# DFT_SHARD_GLOB = the glob the scan used.
DFT_EVAL_SPLITS_JSON="${DFT_EVAL_SPLITS_JSON:-}"
DFT_EVAL_SPLIT_CAP="${DFT_EVAL_SPLIT_CAP:-48}"
DFT_DATA_SEED="${DFT_DATA_SEED:-0}"
DFT_PREFETCH="${DFT_PREFETCH:-1}"
# ---- expert (behavior-cloning) steps ----------------------------------------
# The balance knob: DFT_EXPERT_STEPS expert-only gradient steps per iteration
# against DFT_PPO_EPOCHS DPPO steps, i.e. a step RATIO. Distinct from
# DFT_EXPERT_WEIGHT, which mixes the two into one gradient by a coefficient on
# incommensurable loss scales (that is what went wrong at 0.1).
DFT_EXPERT_STEPS="${DFT_EXPERT_STEPS:-0}"
DFT_EXPERT_LR="${DFT_EXPERT_LR:-0}"   # 0 = reuse DFT_ACTOR_LR

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "indices=$DFT_INDICES iters=$DFT_ITERS expert_weight=$DFT_EXPERT_WEIGHT k=$DFT_K_TRAINABLE"
echo "actor_lr=$DFT_ACTOR_LR ppo_clip=$DFT_PPO_CLIP gae_lambda=$DFT_GAE_LAMBDA"
echo "ppo_epochs=$DFT_PPO_EPOCHS value_epochs=$DFT_VALUE_EPOCHS pg_scale=$DFT_PG_SCALE"
echo "sigma_floor=$DFT_SIGMA_FLOOR target_kl=$DFT_TARGET_KL warmup=$DFT_WARMUP"
echo "expert_steps=$DFT_EXPERT_STEPS expert_lr=$DFT_EXPERT_LR" \
     "ratio=${DFT_EXPERT_STEPS}:${DFT_PPO_EPOCHS}"
if [[ -n "$DFT_SHARD_GLOB" ]]; then
  echo "data=full-corpus glob=$DFT_SHARD_GLOB scenes/iter=$DFT_SCENES_PER_ITER" \
       "resample_every=$DFT_RESAMPLE_EVERY eval_scenes=$DFT_EVAL_SCENES" \
       "data_seed=$DFT_DATA_SEED index_dir=$DFT_INDEX_DIR" \
       "pool=${DFT_SCENE_IDS_JSON:-none}"
else
  echo "data=fixed indices=$DFT_INDICES tfrecord=$DFT_TFRECORD"
fi

# The out_dir lives on a quota'd filesystem; job 27279 died mid-run on
# `OSError: [Errno 122] Disk quota exceeded` writing train_log.jsonl. Fail fast
# and loudly here rather than 27 iterations in.
mkdir -p "$DFT_OUT_DIR"
if ! (echo probe > "$DFT_OUT_DIR/.write_probe" && rm -f "$DFT_OUT_DIR/.write_probe"); then
  echo "FATAL: cannot write to DFT_OUT_DIR=$DFT_OUT_DIR (quota?). Not starting." >&2
  exit 1
fi

"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

# Weights & Biases args (only when DFT_WANDB=1). BooleanOptionalAction => the
# flag is --wandb / --no-wandb; project/group/name passed only when non-empty.
CACHE_ARGS=(--no-clear_caches)
if [[ "$DFT_CLEAR_CACHES" != "0" ]]; then
  CACHE_ARGS=(--clear_caches)
fi

PROFILE_ARGS=(--no-profile)
if [[ "$DFT_PROFILE" != "0" ]]; then
  PROFILE_ARGS=(--profile)
fi

RESUME_ARGS=(--iter_offset "$DFT_ITER_OFFSET")
if [[ -n "$DFT_RESUME" ]]; then
  RESUME_ARGS+=(--resume "$DFT_RESUME")
fi
if [[ -n "$DFT_CRITIC_INIT" ]]; then
  RESUME_ARGS+=(--critic_init "$DFT_CRITIC_INIT")
fi

PREFETCH_ARGS=(--prefetch)
[[ "$DFT_PREFETCH" == "0" ]] && PREFETCH_ARGS=(--no-prefetch)

DATA_ARGS=("${PREFETCH_ARGS[@]}" --index_dir "$DFT_INDEX_DIR"
           --scenes_per_iter "$DFT_SCENES_PER_ITER"
           --resample_every "$DFT_RESAMPLE_EVERY"
           --eval_scenes "$DFT_EVAL_SCENES"
           --data_seed "$DFT_DATA_SEED")
[[ -n "$DFT_SHARD_GLOB" ]] && DATA_ARGS+=(--shard_glob "$DFT_SHARD_GLOB")
[[ -n "$DFT_SCENE_IDS_JSON" ]] && DATA_ARGS+=(--scene_ids_json "$DFT_SCENE_IDS_JSON")
[[ -n "$DFT_EVAL_SPLITS_JSON" ]] && DATA_ARGS+=(--eval_splits_json "$DFT_EVAL_SPLITS_JSON"
                                               --eval_split_cap "$DFT_EVAL_SPLIT_CAP")

WANDB_ARGS=(--no-wandb)
if [[ "$DFT_WANDB" != "0" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "$DFT_WANDB_PROJECT")
  [[ -n "$DFT_WANDB_GROUP" ]] && WANDB_ARGS+=(--wandb_group "$DFT_WANDB_GROUP")
  [[ -n "$DFT_WANDB_NAME" ]]  && WANDB_ARGS+=(--wandb_name "$DFT_WANDB_NAME")
fi

# Reward-design flags. BooleanOptionalAction => --route_reward / --no-route_reward.
ROUTE_ARGS=(--no-route_reward)
[[ "$DFT_ROUTE_REWARD" != "0" ]] && ROUTE_ARGS=(--route_reward)
TERMOFF_ARGS=(--no-terminate_on_offroad)
[[ "$DFT_TERMINATE_OFFROAD" != "0" ]] && TERMOFF_ARGS=(--terminate_on_offroad)
PERSTEP_ARGS=(--no-offroad_per_step)
[[ "$DFT_OFFROAD_PER_STEP" != "0" ]] && PERSTEP_ARGS=(--offroad_per_step)
COLSTEP_ARGS=(--no-collision_per_step)
[[ "$DFT_COLLISION_PER_STEP" != "0" ]] && COLSTEP_ARGS=(--collision_per_step)
TERMCOL_ARGS=(--terminate_on_collision)
[[ "$DFT_TERMINATE_COLLISION" == "0" ]] && TERMCOL_ARGS=(--no-terminate_on_collision)
SHAPNORM_ARGS=(--goal_shaping_normalize)
[[ "$DFT_GOAL_SHAPING_NORMALIZE" == "0" ]] && SHAPNORM_ARGS=(--no-goal_shaping_normalize)
CLEANGOAL_ARGS=(--no-clean_goal_bonus)
[[ "$DFT_CLEAN_GOAL_BONUS" != "0" ]] && CLEANGOAL_ARGS=(--clean_goal_bonus)
PROGIND_ARGS=(--progression_indicator)
[[ "$DFT_PROGRESSION_INDICATOR" == "0" ]] && PROGIND_ARGS=(--no-progression_indicator)
REWARD_ARGS=(
  "${ROUTE_ARGS[@]}" "${TERMOFF_ARGS[@]}" "${PERSTEP_ARGS[@]}" "${SHAPNORM_ARGS[@]}"
  "${COLSTEP_ARGS[@]}" "${TERMCOL_ARGS[@]}"
  "${CLEANGOAL_ARGS[@]}" "${PROGIND_ARGS[@]}"
  --reward_progress "$DFT_REWARD_PROGRESS"
  --reward_offroad "$DFT_REWARD_OFFROAD"
  --reward_collision "$DFT_REWARD_COLLISION"
  --reward_goal_bonus "$DFT_REWARD_GOAL_BONUS"
  --reward_off_route "$DFT_REWARD_OFF_ROUTE"
  --reward_offroad_entry "$DFT_REWARD_OFFROAD_ENTRY"
  --reward_collision_entry "$DFT_REWARD_COLLISION_ENTRY"
  --reward_step_penalty "$DFT_REWARD_STEP_PENALTY"
  --goal_shaping_coef "$DFT_GOAL_SHAPING_COEF"
  --num_paths "$DFT_NUM_PATHS"
  --num_points_per_path "$DFT_NUM_POINTS_PER_PATH"
)
echo "reward: route=$DFT_ROUTE_REWARD term_offroad=$DFT_TERMINATE_OFFROAD" \
     "offroad_per_step=$DFT_OFFROAD_PER_STEP" \
     "collision_per_step=$DFT_COLLISION_PER_STEP" \
     "term_collision=$DFT_TERMINATE_COLLISION" \
     "progress=$DFT_REWARD_PROGRESS offroad=$DFT_REWARD_OFFROAD" \
     "collision=$DFT_REWARD_COLLISION goal=$DFT_REWARD_GOAL_BONUS" \
     "off_route=$DFT_REWARD_OFF_ROUTE shaping=$DFT_GOAL_SHAPING_COEF" \
     "offroad_entry=$DFT_REWARD_OFFROAD_ENTRY" \
     "collision_entry=$DFT_REWARD_COLLISION_ENTRY" \
     "clean_goal=$DFT_CLEAN_GOAL_BONUS prog_indicator=$DFT_PROGRESSION_INDICATOR" \
     "step_penalty=$DFT_REWARD_STEP_PENALTY"

"$PYTHON" -m rl.diffusion_ft.train --method dppo \
  --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" \
  --iters "$DFT_ITERS" --expert_weight "$DFT_EXPERT_WEIGHT" \
  --expert_steps "$DFT_EXPERT_STEPS" --expert_lr "$DFT_EXPERT_LR" \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --actor_lr "$DFT_ACTOR_LR" --ppo_clip "$DFT_PPO_CLIP" \
  --gae_lambda "$DFT_GAE_LAMBDA" --ppo_epochs "$DFT_PPO_EPOCHS" \
  --value_epochs "$DFT_VALUE_EPOCHS" --pg_scale "$DFT_PG_SCALE" \
  --warmup_iters "$DFT_WARMUP" --eval_every "$DFT_EVAL_EVERY" \
  --save_every "$DFT_SAVE_EVERY" --keep_last_ckpts "$DFT_KEEP_CKPTS" \
  --eval_seeds "$DFT_EVAL_SEEDS" \
  --sigma_sample_floor "$DFT_SIGMA_FLOOR" --sigma_logprob_floor "$DFT_SIGMA_FLOOR" \
  --target_kl "$DFT_TARGET_KL" \
  "${CACHE_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${REWARD_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  --seed "$DFT_SEED" --out_dir "$DFT_OUT_DIR"
