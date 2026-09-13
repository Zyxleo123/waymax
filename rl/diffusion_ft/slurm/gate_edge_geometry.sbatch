#!/bin/bash
#SBATCH --job-name=dft_edgegeo
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Does edge_geometry="segment" remove the zero-crossing cliff WITHOUT moving the
# offroad boundary? Pure geometry -- no policy, no checkpoint, no GPU.
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
echo "host=${SLURMD_NODENAME:-unknown}"

# The synthetic gates run first and need no corpus; --tfrecord repeats 1 and 2
# where the roadgraph has real gaps, curves and junctions.
JAX_PLATFORMS=cpu "$PY" -u -m rl.diffusion_ft.tests.gate_edge_geometry \
  --tfrecord "${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}" \
  --indices "${DFT_GATE_INDICES:-0,1,2,3,4,5}"
