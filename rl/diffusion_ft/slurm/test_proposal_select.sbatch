#!/bin/bash
#SBATCH --job-name=dft_test_sel
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# CPU tests: proposal selection + preservation logic, and the pre-existing suites
# that touch the trainer code this change edited. Exits non-zero on any FAIL.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
for t in test_proposal_select test_branching test_lateral_opt test_marginal_rate test_resume_state test_bc_term \
         test_eval_chunking test_probe_selection; do
  echo "=== $t"; "$PY" -m "rl.diffusion_ft.tests.$t"
done
