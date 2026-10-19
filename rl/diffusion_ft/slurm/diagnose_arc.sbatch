#!/bin/bash
#SBATCH --job-name=dft_arc
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Speed or chatter? Reads a finished safety_probe.npz -- executed poses only, no
# model and no GPU. Redrawable from saved artefacts, so it belongs on `cpu`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then source "$arg"; elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2
DFT_PROBE_DIR="${DFT_PROBE_DIR:?set DFT_PROBE_DIR to a finished probe output dir}"
echo "host=${SLURMD_NODENAME:-unknown} probe_dir=$DFT_PROBE_DIR"
"$PYTHON" -m rl.diffusion_ft.tests.diagnose_arc --probe_dir "$DFT_PROBE_DIR"
