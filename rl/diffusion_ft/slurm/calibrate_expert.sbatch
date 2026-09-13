#!/bin/bash
#SBATCH --job-name=dft_expcal
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# What does the safety loss charge the HUMAN DEMONSTRATION? That decides whether
# the term can coexist with an objective that pulls the policy toward the expert,
# and it is prior to every other question about the loss.
#
# No policy, no diffusion checkpoint, no GPU: the human trajectory is already in
# log_trajectory and the other agents are log-replayed, so this is pure geometry.
# Hence the `cpu` partition -- it must not queue behind GPU work.
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
DFT_CAL_SCENES="${DFT_CAL_SCENES:-64}"
DFT_CAL_START="${DFT_CAL_START:-0}"
DFT_CAL_CHUNK="${DFT_CAL_CHUNK:-8}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_EDGE_TAU="${DFT_EDGE_TAU:-0.15}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_AGENT_TAU="${DFT_AGENT_TAU:-0.25}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_CAL_OUT="${DFT_CAL_OUT:?set DFT_CAL_OUT}"

echo "host=${SLURMD_NODENAME:-unknown}  scenes=$DFT_CAL_SCENES from $DFT_TFRECORD"
mkdir -p "$DFT_CAL_OUT"
if ! (echo probe > "$DFT_CAL_OUT/.w" && rm -f "$DFT_CAL_OUT/.w"); then
  echo "FATAL: cannot write to DFT_CAL_OUT=$DFT_CAL_OUT (quota?)." >&2; exit 1
fi

"$PYTHON" -m rl.diffusion_ft.tests.calibrate_expert \
  --tfrecord "$DFT_TFRECORD" --n_scenes "$DFT_CAL_SCENES" --start "$DFT_CAL_START" \
  --chunk "$DFT_CAL_CHUNK" \
  --edge_margin_m "$DFT_EDGE_MARGIN" --edge_tau_m "$DFT_EDGE_TAU" \
  --agent_margin_m "$DFT_AGENT_MARGIN" --agent_tau_m "$DFT_AGENT_TAU" \
  --n_discs "$DFT_N_DISCS" --agent_geometry "$DFT_AGENT_GEOMETRY" \
  --out_dir "$DFT_CAL_OUT"
