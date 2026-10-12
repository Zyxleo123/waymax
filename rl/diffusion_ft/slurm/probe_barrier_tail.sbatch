#!/bin/bash
#SBATCH --job-name=dft_btail
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Does the safety loss fire on scenes where nothing is wrong? The softplus
# barrier has infinite support and is SUMMED over 4 corners x N agents x T
# steps, so a per-term epsilon is multiplied by a factor that grows with the
# scene's traffic density and length. This splits every scene's loss and its
# gradient into the part inside the margin (what the term is for) and the part
# already clear of it (the misfire), and re-scores the same geometry under the
# compact-support shapes.
#
# Pure geometry over log_trajectory: no env, no diffusion checkpoint, no GPU.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

DFT_TFRECORD="${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_TAIL_SCENES="${DFT_TAIL_SCENES:-64}"
DFT_TAIL_START="${DFT_TAIL_START:-0}"
DFT_TAIL_CHUNK="${DFT_TAIL_CHUNK:-8}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_EDGE_TAU="${DFT_EDGE_TAU:-0.15}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_AGENT_TAU="${DFT_AGENT_TAU:-0.25}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"
DFT_TAIL_OUT="${DFT_TAIL_OUT:?set DFT_TAIL_OUT}"

echo "host=${SLURMD_NODENAME:-unknown}  scenes=$DFT_TAIL_SCENES from $DFT_TFRECORD"
mkdir -p "$DFT_TAIL_OUT"
if ! (echo probe > "$DFT_TAIL_OUT/.w" && rm -f "$DFT_TAIL_OUT/.w"); then
  echo "FATAL: cannot write to DFT_TAIL_OUT=$DFT_TAIL_OUT (quota?)." >&2; exit 1
fi

"$PYTHON" -m rl.diffusion_ft.tests.probe_barrier_tail \
  --tfrecord "$DFT_TFRECORD" --n_scenes "$DFT_TAIL_SCENES" \
  --start "$DFT_TAIL_START" --chunk "$DFT_TAIL_CHUNK" \
  --edge_margin_m "$DFT_EDGE_MARGIN" --edge_tau_m "$DFT_EDGE_TAU" \
  --agent_margin_m "$DFT_AGENT_MARGIN" --agent_tau_m "$DFT_AGENT_TAU" \
  --agent_geometry "$DFT_AGENT_GEOMETRY" --n_discs "$DFT_N_DISCS" \
  --out_dir "$DFT_TAIL_OUT"
