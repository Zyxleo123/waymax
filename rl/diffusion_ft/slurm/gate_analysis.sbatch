#!/bin/bash
#SBATCH --job-name=dft_gate
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Hardness-gate analysis: read per-scene verify rows (base + fine-tuned
# checkpoints) and the risk table, write gate_curve.json, then redraw the figure
# FROM that JSON. CPU only; re-runnable at will without a GPU.
#
# Gate inside the job: if the scoring array left a rows file missing or empty,
# say which and exit 0, so nothing downstream strands on a non-zero exit.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
GATE_DIR="${DFT_GATE_DIR:?set DFT_GATE_DIR}"
SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
RISK="${DFT_RISK:-$ROOT/runs/hardness/risk.npz}"
FT_TAGS="${DFT_FT_TAGS:?set DFT_FT_TAGS}"

BASE_ROWS="$GATE_DIR/base/rows.jsonl"
FT_ROWS=()
for f in "$BASE_ROWS" $(for t in $FT_TAGS; do echo "$GATE_DIR/$t/rows.jsonl"; done); do
  if [[ ! -s "$f" ]]; then
    echo "GATE NOT RUN: rows file missing or empty: $f"
    echo "  (the scoring array did not finish that policy -- check its .out)"
    exit 0
  fi
done
for t in $FT_TAGS; do FT_ROWS+=("$GATE_DIR/$t/rows.jsonl"); done

"$PY" -m rl.diffusion_ft.tools.gate_analysis --splits_json "$SPLITS_JSON" \
  --base_rows "$BASE_ROWS" --ft_rows "${FT_ROWS[@]}" --risk "$RISK" \
  --out "$GATE_DIR/gate_curve.json"
"$PY" -m rl.diffusion_ft.tools.plot_gate_curve --in_json "$GATE_DIR/gate_curve.json" \
  --out_png "$GATE_DIR/gate_curve.png"
