#!/bin/bash
#SBATCH --job-name=dft_buildsplits
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
# Stage B of the eval-split build: read the headroom-scan rows and partition the
# expert-clean scenes into failure/success held-out + train id lists. CPU only,
# reads saved rows -- rerun by hand against any scan dir. Gate is INSIDE the job:
# a missing scan exits 0 with an explanation rather than stranding nothing.
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
DFT_SCAN_OUT="${DFT_SCAN_OUT:?set DFT_SCAN_OUT}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:?set DFT_SHARD_GLOB}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-${DFT_SCAN_OUT}/eval_splits.json}"
DFT_HELDOUT_FAILURE="${DFT_HELDOUT_FAILURE:-64}"
DFT_HELDOUT_SUCCESS="${DFT_HELDOUT_SUCCESS:-64}"
DFT_SPLIT_SEED="${DFT_SPLIT_SEED:-0}"

echo "host=${SLURMD_NODENAME:-unknown} scan=$DFT_SCAN_OUT"

if ! ls "$DFT_SCAN_OUT"/rows_task*.jsonl >/dev/null 2>&1; then
  echo "GATE: no rows_task*.jsonl in $DFT_SCAN_OUT -- the scan stage produced"
  echo "GATE: nothing. Nothing to split; check the dft scan .out."
  exit 0
fi

JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tools.build_eval_splits \
  --scan_dir "$DFT_SCAN_OUT" --shard_glob "$DFT_SHARD_GLOB" \
  --index_dir "$DFT_INDEX_DIR" --out_json "$DFT_SPLITS_JSON" \
  --heldout_failure "$DFT_HELDOUT_FAILURE" \
  --heldout_success "$DFT_HELDOUT_SUCCESS" --seed "$DFT_SPLIT_SEED"
echo "splits -> $DFT_SPLITS_JSON"
