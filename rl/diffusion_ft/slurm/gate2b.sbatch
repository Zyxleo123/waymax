#!/bin/bash
#SBATCH --job-name=dft_gate2b
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gate 2b (RL-actor) unit gates. The model here is a tiny synthetic UNet, so this
# is CPU-only work and belongs on the `cpu` partition -- asking for a GPU would
# just queue it behind real training. It will NOT run on a login node: the XLA
# CPU compile is heavy enough to get the process killed (observed exit 137).
#
# Submit with the companion submitter:
#     bash rl/diffusion_ft/slurm/submit_gate2b.sh
#
# Configuration travels in the positional env-file argument (never --export,
# which triggers user-env retrieval and requeue-hold on this cluster).
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
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES=""
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

echo "host=${SLURMD_NODENAME:-unknown}  root=$ROOT"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

# Equivalence check for the jitted K-batched update path (log-probs, posterior
# means/std, gate KL must match the reference Python loop within fp32 rounding).
"$PYTHON" -m rl.diffusion_ft.tests.verify_jit_equiv

# DDIM frozen-prefix subsampling: strided-step correctness + PPO-invariance.
"$PYTHON" -m rl.diffusion_ft.tests.test_ddim_stride

"$PYTHON" -m rl.diffusion_ft.tests.test_gate2b
