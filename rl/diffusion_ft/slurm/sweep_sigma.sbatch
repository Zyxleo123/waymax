#!/bin/bash
#SBATCH --job-name=dft_sigsweep
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
# Preemption-safe: a sigma point whose rollouts.jsonl exists is skipped, so a
# requeued job resumes. --open-mode=append keeps the previous attempt's log.
#SBATCH --requeue
#SBATCH --open-mode=append
#
# How large may DPPO's exploration noise be before the BEHAVIOUR policy stops
# being able to drive?
#
# Job 35195 measured that at sigma_floor=0.1 the sampled trajectory is off-road
# on the very first world step 41% of the time -- 96% of stochastic episodes go
# offroad, against 16% for the identical weights run deterministically. Under the
# parity reward offroad terminates, so 98% of worlds absorb in replan 1 and the
# episode collapses to a one-step bandit (job 34866: nstep=1 in 80% of iters,
# solve rate pinned at 0). Sigma cannot simply be lowered either: below ~0.03 the
# candidate-KL trust region stops opening and no step is ever applied. So the
# number worth having is the LARGEST sigma whose stochastic driving still
# resembles the deterministic policy's -- this sweeps for it.
#
# Rolls `diagnose_drive` once per sigma under the parity reward, base arm only,
# into $DFT_SWEEP_OUT/sigma_<tag>/. Each point also runs the deterministic arm,
# which never samples: the aggregator asserts it is identical across sigma, so a
# sigma leak into the deterministic path cannot pass unnoticed.
#
# RESUMABLE: a point whose rollouts.jsonl already exists is skipped, so hitting
# the time limit means resubmitting, not restarting.
#
# GPU job, forward passes only. The table and figure are drawn by the chained CPU
# job from the artifacts, so they re-render without re-running the policy.
#
# Submit via submit_sweep_sigma.sh (writes a timestamped env file and passes its
# path as $1). Never use `sbatch --export` (requeue-hold on this cluster).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
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

export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_TFRECORD="${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_INDICES="${DFT_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
# Base policy only: the question is about the noise, not about any checkpoint.
DFT_ARMS="${DFT_ARMS:-base=}"
DFT_DIAG_SEEDS="${DFT_DIAG_SEEDS:-4}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_REPLAN="${DFT_REPLAN:-10}"
# Brackets the known-bad end (0.1, measured undriveable) and the known-frozen end
# (~0.003, where the candidate-KL gate never opened in the s x kl sweep).
DFT_SIGMA_LIST="${DFT_SIGMA_LIST:-0.003 0.01 0.02 0.03 0.05 0.1}"
DFT_SWEEP_OUT="${DFT_SWEEP_OUT:?set DFT_SWEEP_OUT}"

# Parity reward, verbatim from job 34866 -- the config under which offroad
# terminates and the collapse happens.
DFT_REWARD_PROGRESS="${DFT_REWARD_PROGRESS:-0.2}"
DFT_REWARD_OFFROAD="${DFT_REWARD_OFFROAD:--1.0}"
DFT_REWARD_COLLISION="${DFT_REWARD_COLLISION:--1.0}"
DFT_REWARD_GOAL_BONUS="${DFT_REWARD_GOAL_BONUS:-5.0}"
DFT_REWARD_OFF_ROUTE="${DFT_REWARD_OFF_ROUTE:--0.2}"
DFT_GOAL_SHAPING_COEF="${DFT_GOAL_SHAPING_COEF:-1.0}"
DFT_NUM_PATHS="${DFT_NUM_PATHS:-10}"
DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-300}"
DFT_GAMMA="${DFT_GAMMA:-0.99}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "sigmas=[$DFT_SIGMA_LIST] arms=$DFT_ARMS seeds=$DFT_DIAG_SEEDS out=$DFT_SWEEP_OUT"

mkdir -p "$DFT_SWEEP_OUT"
if ! (echo probe > "$DFT_SWEEP_OUT/.write_probe" && rm -f "$DFT_SWEEP_OUT/.write_probe"); then
  echo "FATAL: cannot write to DFT_SWEEP_OUT=$DFT_SWEEP_OUT (quota?). Not starting." >&2
  exit 1
fi

"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

for SIG in $DFT_SIGMA_LIST; do
  TAG="sigma_${SIG//./p}"
  PT="$DFT_SWEEP_OUT/$TAG"
  if [[ -f "$PT/rollouts.jsonl" ]]; then
    echo
    echo "==== $TAG: already done, skipping (resume) ===="
    continue
  fi
  echo
  echo "======================= sigma_floor = $SIG ======================="
  mkdir -p "$PT"
  # The aggregator prefers this over parsing the directory name.
  printf '{"sigma_floor": %s}\n' "$SIG" > "$PT/sweep_point.json"
  "$PYTHON" -m rl.diffusion_ft.tests.diagnose_drive \
    --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" \
    --arms "$DFT_ARMS" --seeds "$DFT_DIAG_SEEDS" \
    --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
    --sigma_floor "$SIG" --replan_interval_steps "$DFT_REPLAN" \
    --stochastic --no-expert \
    --route_reward --terminate_offroad \
    --reward_progress "$DFT_REWARD_PROGRESS" \
    --reward_offroad "$DFT_REWARD_OFFROAD" \
    --reward_collision "$DFT_REWARD_COLLISION" \
    --reward_goal_bonus "$DFT_REWARD_GOAL_BONUS" \
    --reward_off_route "$DFT_REWARD_OFF_ROUTE" \
    --goal_shaping_coef "$DFT_GOAL_SHAPING_COEF" \
    --gamma "$DFT_GAMMA" \
    --num_paths "$DFT_NUM_PATHS" --num_points_per_path "$DFT_NUM_POINTS_PER_PATH" \
    --out_dir "$PT"
done

echo
echo "[sweep] points -> $DFT_SWEEP_OUT/sigma_*/rollouts.jsonl"
