#!/bin/bash
#SBATCH --job-name=dft_adp_report
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Results table of the gated tail adapter pilot (tools/adapter_report.py) from rows
# and JSON already on disk; re-runnable from its env file:
#   sbatch rl/diffusion_ft/slurm/adapter_report.sbatch <run>/report.env
# Gate inside: no eval rows -> say so and exit 0.
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
EVAL="${DFT_RPT_EVAL:?set DFT_RPT_EVAL}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} eval=$EVAL"
if ! compgen -G "$EVAL/tasks/*/rows.jsonl" > /dev/null; then
  echo "REPORT NOT RUN: no rows under $EVAL/tasks -- the eval did not run (no adapter"
  echo "  checkpoint? see ${DFT_RPT_TRAIN:-the train dir}/preflight.json)"
  exit 0
fi
"$PY" -m rl.diffusion_ft.tools.adapter_report --eval "$EVAL" \
  ${DFT_RPT_REF:+--ref "$DFT_RPT_REF"} --train "${DFT_RPT_TRAIN:-}" \
  --sweep "${DFT_RPT_SWEEP:-}" --splits_json "${DFT_SPLITS_JSON:?}" --out "${DFT_RPT_OUT:?}"
echo "report -> ${DFT_RPT_OUT}.md / .json"
