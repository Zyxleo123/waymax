#!/bin/bash
#SBATCH --job-name=dft_esan
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Population-search read-out (rl/diffusion_ft/tools/es_analysis.py): the es_p1 ==
# b_cv plumbing check, every arm paired against b_cv, and the per-replan ES
# diagnostics (feasible right off the bat? did the smoothing keep the repair?).
# Pure re-read of rows.jsonl + feat/*.npz: no GPU, re-runnable any time:
#   sbatch rl/diffusion_ft/slurm/es_analysis.sbatch <select_dir>/es_analysis.env
#
# Gate inside the job: a scoring task with no rows is named and the job exits 0.
# Config arrives as a positional env FILE. Never `sbatch --export`.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
SEL_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR}"
REF="${DFT_ES_REF:-b_cv}"
IDENT="${DFT_ES_IDENTITY-es_p1:b_cv}"
echo "hostname=$(hostname) job=${SLURM_JOB_ID:-} cpus=${SLURM_CPUS_PER_TASK:-} mem=${SLURM_MEM_PER_NODE:-}MB partition=${SLURM_JOB_PARTITION:-}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi -L: no GPU (CPU job)"
echo "select_dir=$SEL_DIR ref=$REF identity=${IDENT:-<skip>}"

missing=0
for d in "$SEL_DIR"/tasks/*/; do
  [[ -s "$d/rows.jsonl" ]] || { echo "ES ANALYSIS NOT RUN: no rows in $d -- check that task's .out"; missing=1; }
done
[[ "$missing" == 0 ]] || exit 0

"$PY" -u -m rl.diffusion_ft.tools.es_analysis --select_dir "$SEL_DIR" --ref "$REF" \
  --identity "$IDENT" --out "$SEL_DIR/es_summary.json"
