#!/bin/bash
#SBATCH --job-name=dft_selfcond_gates
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
# Gates for the FlowMol3 self-conditioning + late-stage distortion port.
# Tiny synthetic UNet, no checkpoint -> CPU partition; asking for a GPU would
# only queue it behind real training. It will NOT run on a login node (the XLA
# CPU compile gets the process killed, exit 137).
#
# It also re-runs the pre-existing Gate 2 core, because the flags-off path of
# the diffusion model is exactly what that gate pins: if self-conditioning broke
# the default sampler, Gate 2 is where it shows up.
#
#     bash rl/diffusion_ft/slurm/submit_test_self_cond.sh
#
# Configuration arrives in the positional env FILE (never --export).
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

# Run BOTH suites and report both, then fail at the end if either did. Under a
# plain `set -e` chain the first failure aborts the job, so the Gate 2
# regression -- the one that says whether the DEFAULT sampler still works -- goes
# unrun exactly when it is most worth knowing (observed: job 58125).
rc=0

echo; echo "===== self-cond / distortion gates ====="
"$PYTHON" -m rl.diffusion_ft.tests.test_self_cond || rc=1

echo; echo "===== Gate 2 core (regression: default sampler untouched) ====="
"$PYTHON" -m rl.diffusion_ft.tests.test_gate2 || rc=1

echo; echo "===== Gate 2b / DDIM stride (regression: RL + strided prefix) ====="
"$PYTHON" -m rl.diffusion_ft.tests.test_ddim_stride || rc=1

echo
if [[ $rc -eq 0 ]]; then echo "ALL SUITES PASS"; else echo "AT LEAST ONE SUITE FAILED"; fi
exit $rc
