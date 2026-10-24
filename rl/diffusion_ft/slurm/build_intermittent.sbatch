#!/bin/bash
#SBATCH --job-name=dft_interm
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Write the three-stratum splits file (adds intermittent_heldout; the verified
# file is not modified). CPU, seconds. A failure here is a real error (bad
# inputs), so it exits non-zero and holds the dependents -- by design.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
V="${DFT_VERIFIED_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777}"
"$PY" -m rl.diffusion_ft.tools.add_intermittent_split \
  --rows "${DFT_VERIFY_ROWS:-$V/verify_rows.jsonl}" \
  --in_json "${DFT_VERIFIED_JSON:-$V/eval_splits_verified.json}" \
  --source_json "${DFT_SOURCE_JSON:-$V/../eval_splits.json}" \
  --out_json "${DFT_SPLITS3_JSON:-$V/eval_splits_3strata.json}"
