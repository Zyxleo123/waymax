#!/bin/bash
#SBATCH --job-name=dft_scan
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
# Preemption-safe: the scan skips rows it already wrote, so a requeued task
# resumes. --open-mode=append keeps the previous attempt's log instead of
# truncating it, which is the only way to tell a resume from a fresh start.
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Stage A of hard-WOMD selection: roll the BASE policy (deterministic and
# stochastic) plus the expert log replay over a seeded sample of WOMD scenes and
# write one row per scene. GPU, forward passes only.
#
# `general` hosts a6000/a5000 (>=24 GB); a v100 dies in cuDNN init within
# seconds, so do not move this to `legacy`. Array-parallel over contiguous slices
# of the scan set, and RESUMABLE -- each task skips rows it already wrote, so a
# time-limit kill is fixed by resubmitting the same array.
#
# Submit via submit_hard_womd.sh (writes a timestamped env file, passes its path
# as $1). Never `sbatch --export` (requeue-hold on this cluster).
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

DFT_SHARD_GLOB="${DFT_SHARD_GLOB:?set DFT_SHARD_GLOB}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SCAN_OUT="${DFT_SCAN_OUT:?set DFT_SCAN_OUT}"
DFT_SCAN_SCENES="${DFT_SCAN_SCENES:-4096}"
DFT_SCAN_BATCH="${DFT_SCAN_BATCH:-32}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.03}"
DFT_NUM_PATHS="${DFT_NUM_PATHS:-45}"
DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-800}"
DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SCAN_SEED="${DFT_SCAN_SEED:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset} task=${TASK_ID}/${NTASKS}"
echo "glob=$DFT_SHARD_GLOB out=$DFT_SCAN_OUT scenes=$DFT_SCAN_SCENES batch=$DFT_SCAN_BATCH"

mkdir -p "$DFT_SCAN_OUT"
if ! (echo probe > "$DFT_SCAN_OUT/.write_probe" && rm -f "$DFT_SCAN_OUT/.write_probe"); then
  echo "FATAL: cannot write to DFT_SCAN_OUT=$DFT_SCAN_OUT (quota?). Not starting." >&2
  exit 1
fi

"$PYTHON" -m rl.diffusion_ft.tools.scan_womd_headroom \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --out_dir "$DFT_SCAN_OUT" --ckpt "$DFT_CKPT" \
  --n_scenes "$DFT_SCAN_SCENES" --batch "$DFT_SCAN_BATCH" \
  --sigma_floor "$DFT_SIGMA_FLOOR" --seed "$DFT_SCAN_SEED" \
  --num_paths "$DFT_NUM_PATHS" --num_points_per_path "$DFT_NUM_POINTS_PER_PATH" \
  --task_id "$TASK_ID" --n_tasks "$NTASKS"
