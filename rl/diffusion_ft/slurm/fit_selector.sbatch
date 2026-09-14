#!/bin/bash
#SBATCH --job-name=dft_fitsel
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Fit the learned selector from counterfactual labels (CPU, numpy). Also
# summarizes the repair search if its rows exist. Re-runnable without a GPU.
#
# Gate: no label rows -> say so and exit 0 WITHOUT writing selector.json, so the
# downstream selection job reports "no selector" instead of crashing opaquely.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
CF_DIR="${DFT_CF_DIR:?set DFT_CF_DIR}"
SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"

if compgen -G "$CF_DIR/repair/*/repairs.jsonl" > /dev/null; then
  "$PY" -m rl.diffusion_ft.tools.repair_summary --rows "$CF_DIR/repair/*/repairs.jsonl" \
    --out "$CF_DIR/repair_summary.json" || echo "repair summary failed (non-fatal)"
fi
if ! compgen -G "$CF_DIR/labels/*/labels.jsonl" > /dev/null; then
  echo "SELECTOR NOT FIT: no labels under $CF_DIR/labels -- check the labels array's .out"
  exit 0
fi
"$PY" -m rl.diffusion_ft.tools.fit_selector --labels "$CF_DIR/labels/*/labels.jsonl" \
  --splits_json "$SPLITS_JSON" --out "$CF_DIR/selector.json"
