#!/bin/bash
#SBATCH --job-name=dft_timebase
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Prove or refute the time-base hypothesis for the broken BC target BEFORE
# patching shared preprocessing again. Pure log arrays: no checkpoint, no policy,
# no GPU.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then source "$arg"; elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
export PYTHON="${PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2
DFT_TFRECORD="${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_TB_SCENES="${DFT_TB_SCENES:-8}"
DFT_TB_OUT="${DFT_TB_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/timebase/$(date +%Y%m%d_%H%M%S)}"
echo "host=${SLURMD_NODENAME:-unknown} out=$DFT_TB_OUT"
"$PYTHON" -m rl.diffusion_ft.tests.diagnose_timebase \
  --tfrecord "$DFT_TFRECORD" --n_scenes "$DFT_TB_SCENES" --out_dir "$DFT_TB_OUT"
