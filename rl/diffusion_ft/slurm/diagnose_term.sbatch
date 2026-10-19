#!/bin/bash
#SBATCH --job-name=dft_term
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# H3: WHAT absorbs the whole batch in the first replan under the V-Max parity
# reward? Job 34866 collected exactly one outer replan step in 80% of its 500
# iterations -- all 32 worlds absorbed inside 10 world steps, already at
# iteration 0 with the frozen base policy -- so the episode was a one-step
# bandit and the solve rate could not move. Measured per-world absorb rates at
# replan 1 are 29%, which makes all-32 ~1e-17: the terminating condition is the
# suspect, not the driving.
#
# This runs `diagnose_drive` TWICE on the same scenes and arms, once per reward
# config, and prints the termination-attribution table plus the per-world-step
# flag rate over the first prefix:
#
#   legacy/  -- bare RewardConfig() (progress=1/m, offroad -5 non-terminal,
#               collision -10 terminal, goal +10). Doubles as the REGRESSION
#               CHECK: base/det here must reproduce diag/20260819_021146
#               (absorb@1 ~ 0.29, offroad-ep ~ 0.47, collided ~ 0.72). If it does
#               not, the instrumentation changed the measurement and the parity
#               numbers below cannot be trusted either.
#   parity/  -- the exact config of job 34866 (route progress, terminate on
#               offroad, magnitudes -1, goal +5, shaping 1.0).
#   perstep/ -- the CANDIDATE FIX: same parity reward but offroad is charged on
#               every off-road world step and does NOT terminate. The bind it
#               answers: terminating collapsed collection to a one-step bandit,
#               and the sigma sweep (sigma_sweep/20260821_192513) found no noise
#               level that keeps the episode AND leaves the candidate-KL gate
#               meaningful (target_kl ~ 1/sigma^2 -> 1.25 at sigma 0.02). Charging
#               dwell instead of ending the episode keeps all 8 replans of signal
#               while a long excursion still costs more than a clipped corner.
#               PASS/FAIL: `nrep` must recover toward the legacy 4.5 (parity is
#               1.66 at sigma 0.02, 1.02 at 0.1) while `offFr` stays well below
#               the legacy arm's -- if dwell is as high as legacy, the charge is
#               too weak to deter the excursion and -0.2 needs raising.
#
# Which passes run is set by DFT_TERM_PASSES (default all three); the legacy and
# parity passes are the regression anchors, so drop them only when re-reading a
# config that has already been anchored in the same output tree.
#
# Read the tail of this .out: an offroad rate already ~1 at world step 1 of the
# first prefix is a metric/threshold bug; a rate that ramps across the 10 steps
# is the policy genuinely leaving the road. The `expert/det` row is the control
# -- the human log replayed through the same executor. If the expert also
# absorbs in replan 1 under parity, the config is unsatisfiable by construction.
#
# GPU job, forward passes only (no PPO update, no critic). Writes artifacts;
# figures are drawn by the chained CPU viz jobs.
#
# Submit via submit_diagnose_term.sh (writes a timestamped env file and passes
# its path as $1). Never use `sbatch --export` (requeue-hold on this cluster).
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
DFT_ARMS="${DFT_ARMS:?set DFT_ARMS, e.g. 'base=,parity500=/path/ckpt_0500'}"
DFT_DIAG_SEEDS="${DFT_DIAG_SEEDS:-4}"
DFT_DIAG_STOCHASTIC="${DFT_DIAG_STOCHASTIC:-1}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
# Must match the run under investigation: the sigma floor defines the stochastic
# policy, and the collapse happened during *collection*, which is stochastic.
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.1}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_DIAG_OUT="${DFT_DIAG_OUT:?set DFT_DIAG_OUT}"

# Parity reward knobs (job 34866's exact recipe). Only the parity pass reads
# these; the legacy pass hardcodes the script's own defaults on purpose.
DFT_REWARD_PROGRESS="${DFT_REWARD_PROGRESS:-0.2}"
DFT_REWARD_OFFROAD="${DFT_REWARD_OFFROAD:--1.0}"
DFT_REWARD_COLLISION="${DFT_REWARD_COLLISION:--1.0}"
DFT_REWARD_GOAL_BONUS="${DFT_REWARD_GOAL_BONUS:-5.0}"
DFT_REWARD_OFF_ROUTE="${DFT_REWARD_OFF_ROUTE:--0.2}"
DFT_GOAL_SHAPING_COEF="${DFT_GOAL_SHAPING_COEF:-1.0}"
# Per-step offroad charge for the `perstep` pass. Deliberately 5x smaller than
# the terminating variant's -1.0: this one is paid on every off-road step, so a
# 24-step excursion already costs -4.8 against a +5 goal bonus.
DFT_REWARD_OFFROAD_PERSTEP="${DFT_REWARD_OFFROAD_PERSTEP:--0.2}"
DFT_TERM_PASSES="${DFT_TERM_PASSES:-legacy parity perstep}"
DFT_NUM_PATHS="${DFT_NUM_PATHS:-10}"
DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-300}"
DFT_GAMMA="${DFT_GAMMA:-0.99}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "arms=$DFT_ARMS seeds=$DFT_DIAG_SEEDS sigma=$DFT_SIGMA_FLOOR out=$DFT_DIAG_OUT"
echo "passes=$DFT_TERM_PASSES"

want_pass() { [[ " $DFT_TERM_PASSES " == *" $1 "* ]]; }

mkdir -p "$DFT_DIAG_OUT"
for _p in $DFT_TERM_PASSES; do mkdir -p "$DFT_DIAG_OUT/$_p"; done
if ! (echo probe > "$DFT_DIAG_OUT/.write_probe" && rm -f "$DFT_DIAG_OUT/.write_probe"); then
  echo "FATAL: cannot write to DFT_DIAG_OUT=$DFT_DIAG_OUT (quota?). Not starting." >&2
  exit 1
fi

"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

STO_ARGS=()
[[ "$DFT_DIAG_STOCHASTIC" != "0" ]] && STO_ARGS=(--stochastic)

COMMON=(
  --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES"
  --arms "$DFT_ARMS" --seeds "$DFT_DIAG_SEEDS"
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN"
  --sigma_floor "$DFT_SIGMA_FLOOR" --replan_interval_steps "$DFT_REPLAN"
  "${STO_ARGS[@]}"
)

# The parity knobs shared by pass 2 and pass 3 (they differ only in HOW offroad
# is charged), so the comparison cannot drift on an unrelated weight.
PARITY_COMMON=(
  --route_reward
  --reward_progress "$DFT_REWARD_PROGRESS"
  --reward_collision "$DFT_REWARD_COLLISION"
  --reward_goal_bonus "$DFT_REWARD_GOAL_BONUS"
  --reward_off_route "$DFT_REWARD_OFF_ROUTE"
  --goal_shaping_coef "$DFT_GOAL_SHAPING_COEF"
  --gamma "$DFT_GAMMA"
  --num_paths "$DFT_NUM_PATHS" --num_points_per_path "$DFT_NUM_POINTS_PER_PATH"
)

if want_pass legacy; then
  echo
  echo "================ PASS legacy: bare RewardConfig (regression check) ================"
  "$PYTHON" -m rl.diffusion_ft.tests.diagnose_drive \
    "${COMMON[@]}" \
    --no-route_reward --no-terminate_offroad --no-offroad_per_step \
    --out_dir "$DFT_DIAG_OUT/legacy"
fi

if want_pass parity; then
  echo
  echo "================ PASS parity: V-Max parity reward (job 34866) ================"
  "$PYTHON" -m rl.diffusion_ft.tests.diagnose_drive \
    "${COMMON[@]}" "${PARITY_COMMON[@]}" \
    --terminate_offroad --no-offroad_per_step \
    --reward_offroad "$DFT_REWARD_OFFROAD" \
    --out_dir "$DFT_DIAG_OUT/parity"
fi

if want_pass perstep; then
  echo
  echo "================ PASS perstep: offroad charged per step, NON-terminating ================"
  "$PYTHON" -m rl.diffusion_ft.tests.diagnose_drive \
    "${COMMON[@]}" "${PARITY_COMMON[@]}" \
    --no-terminate_offroad --offroad_per_step \
    --reward_offroad "$DFT_REWARD_OFFROAD_PERSTEP" \
    --out_dir "$DFT_DIAG_OUT/perstep"
fi

echo
echo "[term] compare the per-arm summary lines across passes:"
echo "  nrep  (mean replans/episode)   -- perstep must recover toward legacy"
echo "  offFr (fraction of steps off)  -- perstep must stay below legacy"
echo "[term] artifacts -> $DFT_DIAG_OUT/{$(echo $DFT_TERM_PASSES | tr ' ' ',')}/term_attribution.json"
