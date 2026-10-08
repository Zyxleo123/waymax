#!/bin/bash
#SBATCH --job-name=dft_lopt_gate
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gates for the loss-optimized lateral repair (rl/diffusion_ft/lateral_opt.py):
# first the pure algebra (tests/test_lateral_opt.py: the lateral_shift refactor is
# bit-identical, the comfort Hessian, the ramp-20 cubic as the special case), then
# the real-scene plumbing (tests/gate_lateral_opt.py: identity, anchor, descent,
# it repairs, finite, union contains the bank) on failure_heldout scenes with the
# expert window distorted into violation. The real-scene gate also REPORTS fix
# rate / ride quality per arm per distortion (the numbers land in DFT_LOPT_JSON).
#
# CPU, no policy: waymax reset + log replay only. ~15-30 min at 16 scenes.
#
# DECISION GATE, so it always exits 0 and leaves its answer in a verdict file
# (DFT_GATE_VERDICT): first line PASS, FAIL: <gates>, or CRASH. The dependent
# select job reads it (select_eval.sbatch, DFT_REQUIRE_GATE) and exits 0 with the
# reason instead of stranding in DependencyNeverSatisfied.
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
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_GATE_VERDICT="${DFT_GATE_VERDICT:?set DFT_GATE_VERDICT (where the verdict goes)}"
DFT_LOPT_JSON="${DFT_LOPT_JSON:-$(dirname "$DFT_GATE_VERDICT")/gate_lateral_opt.json}"
DFT_LOPT_NEIGHBORS="${DFT_LOPT_NEIGHBORS:-cv}"
DFT_SELECT_OPT_CFG="${DFT_SELECT_OPT_CFG:-}"
DFT_LOPT_GATE_SCENES="${DFT_LOPT_GATE_SCENES:-16}"

echo "=== lateral-opt gates: $(date) on $(hostname) ==="
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?} (CPU job, no GPU)"
echo "splits=$DFT_SPLITS_JSON verdict=$DFT_GATE_VERDICT neighbors=$DFT_LOPT_NEIGHBORS"
echo "opt_cfg=${DFT_SELECT_OPT_CFG:-<defaults>}"
mkdir -p "$(dirname "$DFT_GATE_VERDICT")"
rm -f "$DFT_GATE_VERDICT"

rc=0
"$PY" -m rl.diffusion_ft.tests.test_lateral_opt || rc=$?
if [[ "$rc" != 0 ]]; then
  echo "FAIL: pure tests (tests/test_lateral_opt.py exit $rc)" > "$DFT_GATE_VERDICT"
  echo "=== verdict: $(head -1 "$DFT_GATE_VERDICT") ==="
  exit 0
fi

"$PY" -m rl.diffusion_ft.tests.gate_lateral_opt \
  --splits_json "$DFT_SPLITS_JSON" --shard_glob "$DFT_SHARD_GLOB" \
  --index_dir "$DFT_INDEX_DIR" --n_scenes "$DFT_LOPT_GATE_SCENES" \
  --neighbors "$DFT_LOPT_NEIGHBORS" --opt_cfg "$DFT_SELECT_OPT_CFG" \
  --verdict "$DFT_GATE_VERDICT" --json_out "$DFT_LOPT_JSON" || rc=$?

# The test writes the verdict itself on PASS and on FAIL; no file means it died
# before reaching the end, which is not the same thing as a failed gate.
if [[ ! -s "$DFT_GATE_VERDICT" ]]; then
  echo "CRASH (exit $rc, see $0 output above)" > "$DFT_GATE_VERDICT"
fi
echo "=== verdict: $(head -1 "$DFT_GATE_VERDICT") (python exit $rc) ==="
exit 0
