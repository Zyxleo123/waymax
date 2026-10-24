#!/bin/bash
#SBATCH --job-name=dft_diag
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
# Roll out DPPO checkpoints and record HOW they drive (reward decomposition,
# separated outcome flags, paths with per-world-step offroad masks). Forward
# passes only -- no PPO update and no critic -- so it is much lighter than
# training, but it still holds a B=32 batched Waymax state, so keep it off the
# 11 GB 2080 Tis (`general` is a6000/a5000; override with DFT_PARTITION/NODELIST).
#
# Writes artifacts only. Figures are drawn by the companion CPU job.
#
# Submit via submit_diagnose_drive.sh (writes a timestamped env file and passes
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
DFT_ARMS="${DFT_ARMS:?set DFT_ARMS, e.g. 'base=,dppo500=/path/ckpt_0250'}"
DFT_DIAG_SEEDS="${DFT_DIAG_SEEDS:-4}"
DFT_DIAG_STOCHASTIC="${DFT_DIAG_STOCHASTIC:-1}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.03}"
DFT_DIAG_OUT="${DFT_DIAG_OUT:?set DFT_DIAG_OUT}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "arms=$DFT_ARMS seeds=$DFT_DIAG_SEEDS out=$DFT_DIAG_OUT"

mkdir -p "$DFT_DIAG_OUT"
if ! (echo probe > "$DFT_DIAG_OUT/.write_probe" && rm -f "$DFT_DIAG_OUT/.write_probe"); then
  echo "FATAL: cannot write to DFT_DIAG_OUT=$DFT_DIAG_OUT (quota?). Not starting." >&2
  exit 1
fi

"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

STO_ARGS=()
[[ "$DFT_DIAG_STOCHASTIC" != "0" ]] && STO_ARGS=(--stochastic)

"$PYTHON" -m rl.diffusion_ft.tests.diagnose_drive \
  --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" \
  --arms "$DFT_ARMS" --seeds "$DFT_DIAG_SEEDS" \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  "${STO_ARGS[@]}" \
  --out_dir "$DFT_DIAG_OUT"
