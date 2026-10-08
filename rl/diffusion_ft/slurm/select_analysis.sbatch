#!/bin/bash
#SBATCH --job-name=dft_selan
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
# Proposal-selection analysis: rows -> select_summary.json -> select_summary.png
# (the figure is drawn FROM the JSON). CPU only; re-runnable without a GPU:
#   sbatch rl/diffusion_ft/slurm/select_analysis.sbatch <select_dir>/analyze.env
#
# Gate inside the job: if a scoring task left no rows, say which and exit 0.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
SEL_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR}"
SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
REPEATS="${DFT_SELECT_REPEATS:-4}"

if [[ ! -s "$SPLITS_JSON" ]]; then
  echo "ANALYSIS NOT RUN: splits file missing: $SPLITS_JSON (did the build job fail?)"; exit 0
fi
missing=0
for d in "$SEL_DIR"/tasks/*/; do
  [[ -s "$d/rows.jsonl" ]] || { echo "ANALYSIS NOT RUN: no rows in $d -- check that task's .out"; missing=1; }
done
[[ "$missing" == 0 ]] || exit 0

"$PY" -m rl.diffusion_ft.tools.select_analysis --splits_json "$SPLITS_JSON" \
  --rows "$SEL_DIR/tasks/*/rows.jsonl" --repeats "$REPEATS" \
  --out "$SEL_DIR/select_summary.json"
"$PY" -m rl.diffusion_ft.tools.plot_select --in_json "$SEL_DIR/select_summary.json" \
  --out_png "$SEL_DIR/select_summary.png"
# Rows from runners that record the termination-aware flags get a second summary
# scored on them (tools/select_analysis.py --term); older rows are left alone.
first_rows=("$SEL_DIR"/tasks/*/rows.jsonl)
line1="$(head -n 1 "${first_rows[0]}")"
if [[ "$line1" == *'"solved_term"'* ]]; then
  "$PY" -m rl.diffusion_ft.tools.select_analysis --splits_json "$SPLITS_JSON" \
    --rows "$SEL_DIR/tasks/*/rows.jsonl" --repeats "$REPEATS" --term \
    --out "$SEL_DIR/select_summary_term.json"
fi
