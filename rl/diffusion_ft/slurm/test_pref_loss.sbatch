#!/bin/bash
#SBATCH --job-name=dft_pref_tests
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
# Diffusion-DPO term (train_safety --pref_weight): loss math, pair loading, and
# the regressions it could cause (base_anchor_loss default, DDIM/late-region).
# Tiny synthetic UNet, no checkpoint -> CPU partition.
#
#     sbatch rl/diffusion_ft/slurm/test_pref_loss.sbatch
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

echo "host=${SLURMD_NODENAME:-unknown}  root=$ROOT  job=${SLURM_JOB_ID:-?}"
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

rc=0
echo; echo "===== preference loss + pair loading ====="
"$PYTHON" -m rl.diffusion_ft.tests.test_pref_loss || rc=1
echo; echo "===== regression: BC term routing ====="
"$PYTHON" -m rl.diffusion_ft.tests.test_bc_term || rc=1
echo; echo "===== regression: DDIM stride / late region ====="
"$PYTHON" -m rl.diffusion_ft.tests.test_ddim_stride || rc=1

echo
if [[ $rc -eq 0 ]]; then echo "ALL SUITES PASS"; else echo "AT LEAST ONE SUITE FAILED"; fi
exit $rc
