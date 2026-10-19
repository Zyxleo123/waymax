#!/bin/bash
#SBATCH --job-name=disc_abl_compare
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/es_discriminator/logs/disc_abl_compare_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/es_discriminator/logs/disc_abl_compare_%j.err

# After an ablation round has trained: GPU latency benchmark of the trained checkpoints, then the
# table on the identical test replans with paired bootstrap vs the baseline. Takes the newest
# safety_progress_v3_<AX>_* run of each ablation. Submit from rhea / lov4:
#
#   sbatch es_discriminator/analyse_discriminator/compare_ablations.sh                 # A1-A4 vs A0
#   sbatch es_discriminator/analyse_discriminator/compare_ablations.sh --baseline A4   # A5-A8 vs A4
#
# Output: es_discriminator/analyse_discriminator/compare (A0) or compare_a4 (A4).
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
OUT=${SCRATCH}/es_discriminator/analyse_discriminator
SP=${SCRATCH}/es_discriminator/safety_progress
BASELINE=A0
[[ "${1:-}" == --baseline ]] && BASELINE="${2:?--baseline needs A0 or A4}"
case "${BASELINE}" in
  A0) NAMES=(A1 A2 A3 A4) RUNS=() BENCH=() SUFFIX="" ;;
  A4) A4_DIR=${SP}/safety_progress_v3_A4_20261017_212234_87831
      NAMES=(A5 A6 A7 A8) RUNS=(--run "A4=${A4_DIR}") BENCH=(--run "A4=${A4_DIR}/best.pt") SUFFIX=_a4 ;;
  *) echo "ERROR: --baseline must be A0 or A4" >&2; exit 1 ;;
esac
for name in "${NAMES[@]}"; do
  dir=$(ls -dt "${SP}"/safety_progress_v3_${name}_* 2>/dev/null | head -1 || true)
  if [[ -n "${dir}" && -f "${dir}/eval_test.json" ]]; then
    RUNS+=(--run "${name}=${dir}")
    BENCH+=(--run "${name}=${dir}/best.pt")
  else
    echo "WARNING: no finished run for ${name}; it is left out" >&2
  fi
done

set --
set +u
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
cd "${ROOT}"
mkdir -p "${OUT}"

python -u -m es_discriminator.analyse_discriminator.benchmark --device cuda --worlds 1,5 "${BENCH[@]}" \
  --out "${OUT}/benchmark_cuda${SUFFIX}.json"
python -u -m es_discriminator.analyse_discriminator.compare --baseline "${BASELINE}" "${RUNS[@]}" \
  --benchmark "${OUT}/benchmark_cuda${SUFFIX}.json" --latency-worlds worlds_5 --paired --out-dir "${OUT}/compare${SUFFIX}"
echo "done: ${OUT}/compare${SUFFIX}/ablations_test.md"
