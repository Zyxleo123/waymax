#!/bin/bash
#SBATCH --job-name=dft_test_tb
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# CPU tests for the transition-replay term (--tb_weight) plus the suites that
# touch the code it edited: rl_actor (the trace sampler + clean_prediction_from_xt),
# safety_objective (per-row t0) and the teacher-forced unroll it reuses.
# Exits non-zero on any FAIL.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
for t in test_transition_buffer test_unroll_jit test_gate2 test_gate2b test_safety_loss \
         test_ddim_stride test_bc_term test_resume_state; do
  echo "=== $t"; "$PY" -m "rl.diffusion_ft.tests.$t"
done
echo "ALL TESTS PASSED"
