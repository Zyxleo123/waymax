#!/bin/bash
#SBATCH --job-name=dft_bcprobe
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Why the bank BC fine-tune moved so little: did it fit the edits (loss) and can
# the late copy express them from the deployed prefix (d_free vs d_tf)?
# rl/diffusion_ft/tools/probe_bc_transfer.py. a6000/a5000 only (v100 dies at cuDNN
# autotune, 2080 Ti OOMs); batch 32 fits an a5000.
#
#   sbatch rl/diffusion_ft/slurm/probe_bc_transfer.sbatch <env file>
#
# Config arrives as a positional env FILE (never --export).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_BC_DATA="${DFT_BC_DATA:?set DFT_BC_DATA (glob of bc_data/<arm>/*.npz)}"
DFT_FT_CKPTS="${DFT_FT_CKPTS:?set DFT_FT_CKPTS (space-separated iter_* dirs)}"
DFT_OUT="${DFT_OUT:?set DFT_OUT (json path)}"
DFT_N_ROWS="${DFT_N_ROWS:-512}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-} " \
     "cpus=${SLURM_CPUS_PER_TASK:-} mem=${SLURM_MEM_PER_NODE:-}MB gres=gpu:1"
nvidia-smi -L || true

# Gate: every checkpoint must exist; say which one is missing and exit 0.
for c in $DFT_FT_CKPTS; do
  [[ -s "$c/tparams.npz" ]] || { echo "PROBE NOT RUN: no $c/tparams.npz"; exit 0; }
done
compgen -G "$DFT_BC_DATA" > /dev/null || { echo "PROBE NOT RUN: no rows match $DFT_BC_DATA"; exit 0; }

# shellcheck disable=SC2086
"$PY" -u -m rl.diffusion_ft.tools.probe_bc_transfer --ckpt "$DFT_CKPT" \
  --bc_data "$DFT_BC_DATA" --ft $DFT_FT_CKPTS --n_rows "$DFT_N_ROWS" --out "$DFT_OUT"
