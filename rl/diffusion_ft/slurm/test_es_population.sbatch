#!/bin/bash
#SBATCH --job-name=dft_esgate
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gates for the population search's pure rules (tests/test_es_population.py):
# the projection picks exactly bank_choice's edit for one member, clean members
# are never edited, fitness ranks feasible above infeasible, replication matches
# Diffusion-ES, the survival ratio is calibrated, the config grammar fails loudly.
#
# CPU, pure numpy, seconds. DECISION GATE: always exits 0 and leaves its answer
# in DFT_GATE_VERDICT (first line PASS, FAIL: <gates>, or CRASH), which the
# dependent select job reads (select_eval.sbatch, DFT_REQUIRE_GATE).
#
# Config arrives as a positional env FILE. Never `sbatch --export`.
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
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
DFT_GATE_VERDICT="${DFT_GATE_VERDICT:?set DFT_GATE_VERDICT (where the verdict goes)}"

echo "=== ES population gates: $(date) on $(hostname) ==="
echo "job=${SLURM_JOB_ID:-} cpus=${SLURM_CPUS_PER_TASK:-} mem=${SLURM_MEM_PER_NODE:-}MB partition=${SLURM_JOB_PARTITION:-}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi -L: no GPU (CPU job)"
mkdir -p "$(dirname "$DFT_GATE_VERDICT")"
rm -f "$DFT_GATE_VERDICT"

rc=0
"$PY" -m rl.diffusion_ft.tests.test_es_population --verdict "$DFT_GATE_VERDICT" || rc=$?

# The test writes the verdict itself on PASS and on FAIL; no file means it died
# before reaching the end, which is not the same thing as a failed gate.
if [[ ! -s "$DFT_GATE_VERDICT" ]]; then
  echo "CRASH (exit $rc, see output above)" > "$DFT_GATE_VERDICT"
fi
echo "=== verdict: $(head -1 "$DFT_GATE_VERDICT") (python exit $rc) ==="
exit 0
