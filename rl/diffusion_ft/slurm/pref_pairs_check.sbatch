#!/bin/bash
#SBATCH --job-name=dft_pref_pairs
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# What the collected Diffusion-DPO pairs are, and whether they are the bank-BC
# run's edit rows (tools/pref_pairs_check.py). Informational: training does not
# wait on it -- train_safety._init_pref enforces the hard pair invariants itself.
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
DATA="${DFT_PREF_DATA:?set DFT_PREF_DATA (glob of bc_data/<arm>/*.npz)}"
OUT="${DFT_PREF_CHECK_OUT:?set DFT_PREF_CHECK_OUT (json path)}"
echo "host=${SLURMD_NODENAME:-unknown} job=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "mem=${SLURM_MEM_PER_NODE:-?}M"
compgen -G "$DATA" > /dev/null || { echo "CHECK NOT RUN: no rows match $DATA (did collect run?)"; exit 0; }
"$PY" -m rl.diffusion_ft.tools.pref_pairs_check --data "$DATA" \
  ${DFT_PREF_REF_DATA:+--ref "$DFT_PREF_REF_DATA"} --out "$OUT"
