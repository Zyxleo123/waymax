#!/bin/bash
#SBATCH --job-name=dft_test_adapter
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# CPU tests for the gated tail adapter (rl/diffusion_ft/adapter.py), plus the
# existing suites that touch what it edited: the UNet forward (tail_hook), the
# sampler, and proposal_select (adapter arms, --dump_pairs). Exits non-zero on any FAIL.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
for t in test_adapter test_proposal_select test_lateral_opt test_gate2 test_ddim_stride test_self_cond; do
  echo "=== $t"; "$PY" -m "rl.diffusion_ft.tests.$t"
done
echo "ALL PASS"
