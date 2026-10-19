#!/bin/bash
#SBATCH --job-name=dft_safeprobe
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Validate the differentiable safety loss against the simulator's own verdict on
# the failure set: does c_edge < 0 coincide exactly with waymax's `offroad`, does
# the disc cover ever miss an `overlap`, does the loss separate violating from
# clean frames, is it flat when clear, and does its gradient buy real clearance.
#
# Forward passes plus one jax.grad over a [B, T] pose array -- no training, no
# critic. It holds a B=32 batched Waymax state and a 30k-point roadgraph, so keep
# it off the 11 GB 2080 Tis (`general` is a6000/a5000).
#
# Writes artifacts only; figures come from the companion CPU job.
#
# Submit via submit_probe_safety_loss.sh (writes a timestamped env file and
# passes its path as $1). Never use `sbatch --export` (requeue-hold on this
# cluster).
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
DFT_PROBE_SEEDS="${DFT_PROBE_SEEDS:-2}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.03}"
# --- loss geometry --- #
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_EDGE_TAU="${DFT_EDGE_TAU:-0.15}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_AGENT_TAU="${DFT_AGENT_TAU:-0.25}"
DFT_EDGE_WEIGHT="${DFT_EDGE_WEIGHT:-1.0}"
DFT_AGENT_WEIGHT="${DFT_AGENT_WEIGHT:-1.0}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_TIME_CHUNK="${DFT_TIME_CHUNK:-8}"
DFT_GRAD_STEP="${DFT_GRAD_STEP:-0.05}"
DFT_PROBE_OUT="${DFT_PROBE_OUT:?set DFT_PROBE_OUT}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "scenes=$DFT_INDICES seeds=$DFT_PROBE_SEEDS out=$DFT_PROBE_OUT"
echo "loss: edge margin=$DFT_EDGE_MARGIN tau=$DFT_EDGE_TAU | "\
"agent margin=$DFT_AGENT_MARGIN tau=$DFT_AGENT_TAU discs=$DFT_N_DISCS"

mkdir -p "$DFT_PROBE_OUT"
if ! (echo probe > "$DFT_PROBE_OUT/.write_probe" && rm -f "$DFT_PROBE_OUT/.write_probe"); then
  echo "FATAL: cannot write to DFT_PROBE_OUT=$DFT_PROBE_OUT (quota?). Not starting." >&2
  exit 1
fi

"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

"$PYTHON" -m rl.diffusion_ft.tests.probe_safety_loss \
  --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" \
  --seeds "$DFT_PROBE_SEEDS" --replan_interval_steps "$DFT_REPLAN" \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --edge_margin_m "$DFT_EDGE_MARGIN" --edge_tau_m "$DFT_EDGE_TAU" \
  --agent_margin_m "$DFT_AGENT_MARGIN" --agent_tau_m "$DFT_AGENT_TAU" \
  --agent_geometry "$DFT_AGENT_GEOMETRY" \
  --edge_weight "$DFT_EDGE_WEIGHT" --agent_weight "$DFT_AGENT_WEIGHT" \
  --n_discs "$DFT_N_DISCS" --time_chunk "$DFT_TIME_CHUNK" \
  --grad_step "$DFT_GRAD_STEP" \
  --out_dir "$DFT_PROBE_OUT"
