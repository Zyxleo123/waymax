#!/bin/bash
#SBATCH --job-name=dft_pacean
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
# Pace experiment readout (tools/pace_analysis.py): rows of every cell ->
# pace_summary.json -> pace_summary.png (the figure is drawn FROM the JSON).
# CPU only; re-runnable without a GPU:
#   sbatch rl/diffusion_ft/slurm/pace_analysis.sbatch <pace_dir>/analyze.env
#
# Gate inside the job: a cell with no rows (e.g. `realised` gated out by its
# verdict) is reported and skipped by the tool; no control rows at all -> say so
# and exit 0.
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
PACE_DIR="${DFT_PACE_DIR:?set DFT_PACE_DIR}"
SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
REPEATS="${DFT_SELECT_REPEATS:-4}"
CELLS="${DFT_PACE_CELLS:-commanded realised}"
CONTROL="${DFT_PACE_CONTROL:-commanded:base}"

ctrl_cell="${CONTROL%%:*}"
if ! compgen -G "$PACE_DIR/$ctrl_cell/tasks/*/rows.jsonl" >/dev/null; then
  echo "ANALYSIS NOT RUN: control cell '$ctrl_cell' has no rows under $PACE_DIR/$ctrl_cell/tasks"
  echo "  -- check that sibling's select .out files"
  exit 0
fi
[[ -f "$PACE_DIR/gate/verdict.txt" ]] && echo "gate verdict: $(head -1 "$PACE_DIR/gate/verdict.txt")"
args=()
for c in $CELLS; do args+=(--cell "$c=$PACE_DIR/$c"); done
"$PY" -m rl.diffusion_ft.tools.pace_analysis --splits_json "$SPLITS_JSON" \
  "${args[@]}" --repeats "$REPEATS" --control "$CONTROL" \
  --out "$PACE_DIR/pace_summary.json"
"$PY" -m rl.diffusion_ft.tools.pace_analysis --plot "$PACE_DIR/pace_summary.json" \
  --out_png "$PACE_DIR/pace_summary.png"
