#!/bin/bash
#SBATCH --job-name=dft_recov
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Per-stratum headline figure (failure recovery / success non-regression /
# intermittent) from select_summary.json (rl/diffusion_ft/tools/viz_recovery.py).
# Pure re-read of already-scored summaries: no GPU, no re-roll, free to re-run.
#
# Config arrives as a positional env FILE (or VAR=value args). Never sbatch --export.
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
IN_JSON="${DFT_RECOV_IN_JSON:?set DFT_RECOV_IN_JSON (space-separated select_summary.json paths)}"
OUT_PNG="${DFT_RECOV_OUT_PNG:?set DFT_RECOV_OUT_PNG}"
ARMS="${DFT_RECOV_ARMS:-}"
TITLE="${DFT_RECOV_TITLE:-Deploy-time repair: per-stratum solve rate vs base}"

echo "host=${SLURMD_NODENAME:-unknown}  in=$IN_JSON  out=$OUT_PNG  arms=${ARMS:-<all>}"

# Gate inside the job: explain and exit 0 rather than strand anything downstream.
missing=0
for f in $IN_JSON; do
  [[ -f "$f" ]] || { echo "MISSING: $f"; missing=1; }
done
if [[ "$missing" == 1 ]]; then
  echo "RECOVERY FIGURE NOT DRAWN: one or more select_summary.json absent"; exit 0
fi

# shellcheck disable=SC2086
"$PY" -u -m rl.diffusion_ft.tools.viz_recovery \
  --in_json $IN_JSON --out_png "$OUT_PNG" \
  ${ARMS:+--arms "$ARMS"} --title "$TITLE"
echo "out -> $OUT_PNG"
