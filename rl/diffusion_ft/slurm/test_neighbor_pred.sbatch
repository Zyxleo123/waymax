#!/bin/bash
#SBATCH --job-name=dft_nbrpred
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gates for the repair bank's non-oracle neighbour sources
# (tests/test_neighbor_pred.py): `cv` and `none` never read past t0 (scrambling
# every object's future changes nothing they compute, while the `log` oracle
# sees it); the default scorer is untouched; constant velocity is within 0.5 m
# of the truth one step ahead.
#
# CPU, no policy: waymax reset + log replay only. 16G is ample (1.8 GB at 4 scenes).
#
# DECISION GATE, so it always exits 0 and leaves its answer in a verdict file
# (DFT_GATE_VERDICT): first line PASS, FAIL: <gates>, or CRASH. A dependent
# select job reads that file (select_eval.sbatch, DFT_REQUIRE_GATE) and exits 0
# with the reason instead of stranding in DependencyNeverSatisfied.
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
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_GATE_VERDICT="${DFT_GATE_VERDICT:?set DFT_GATE_VERDICT (where the verdict goes)}"

echo "=== neighbour-prediction gates: $(date) on $(hostname) ==="
echo "splits=$DFT_SPLITS_JSON verdict=$DFT_GATE_VERDICT"
mkdir -p "$(dirname "$DFT_GATE_VERDICT")"
rm -f "$DFT_GATE_VERDICT"

rc=0
"$PY" -m rl.diffusion_ft.tests.test_neighbor_pred \
  --splits_json "$DFT_SPLITS_JSON" --shard_glob "$DFT_SHARD_GLOB" \
  --index_dir "$DFT_INDEX_DIR" --verdict "$DFT_GATE_VERDICT" || rc=$?

# The test writes the verdict itself on PASS and on FAIL; no file means it died
# before reaching the end, which is not the same thing as a failed gate.
if [[ ! -s "$DFT_GATE_VERDICT" ]]; then
  echo "CRASH (exit $rc, see $0 output above)" > "$DFT_GATE_VERDICT"
fi
echo "=== verdict: $(head -1 "$DFT_GATE_VERDICT") (python exit $rc) ==="
exit 0
