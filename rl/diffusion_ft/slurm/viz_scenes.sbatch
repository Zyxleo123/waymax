#!/bin/bash
#SBATCH --job-name=dft_scenehtml
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Build the interactive scene page from scenes.json
# (rl/diffusion_ft/tools/build_scene_viz.py). Pure re-read, CPU, free to re-run.
#
# Config arrives as a positional env FILE. Never sbatch --export.
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
DFT_OUT_JSON="${DFT_OUT_JSON:?set DFT_OUT_JSON}"
DFT_OUT_HTML="${DFT_OUT_HTML:?set DFT_OUT_HTML}"
# Arms to DRAW. The dump keeps every arm it scored, so narrowing the page is a
# CPU rebuild off the same scenes.json -- never a reason to re-run the GPU dump.
DFT_SHOW_ARMS="${DFT_SHOW_ARMS:-}"

# Gate inside the job: the GPU step exits 0 without a json if the dump was empty.
if [[ ! -s "$DFT_OUT_JSON" ]]; then
  echo "SCENE PAGE NOT BUILT: $DFT_OUT_JSON is missing or empty (upstream dump wrote nothing)"; exit 0
fi

"$PY" -u -m rl.diffusion_ft.tools.build_scene_viz \
  --in_json "$DFT_OUT_JSON" --out_html "$DFT_OUT_HTML" --arms "$DFT_SHOW_ARMS"
echo "html -> $DFT_OUT_HTML"
