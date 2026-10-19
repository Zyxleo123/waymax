#!/bin/bash
#SBATCH --job-name=dft_egovel
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gates for `ego_vel_source="realised"` (tests/test_ego_vel_splice.py): the
# default splice is bit-identical to the old one, the estimator reproduces the
# human's logged speed from the human's positions ("the log column still reads
# 1.000"), only the ego's executed velocity moves, and the observation then
# reports the speed the car actually drove.
#
# CPU: the env owns a policy but no denoiser is called -- every trajectory is
# handed to env.step. No `--qos`: the cpu partition's default is qos_cpu.
#
# DECISION GATE, so it always exits 0 and leaves its answer in a verdict file
# (DFT_GATE_VERDICT): first line PASS, FAIL: <gates>, or CRASH. A dependent
# select job reads that file (select_eval.sbatch, DFT_REQUIRE_GATE) and exits 0
# with the reason instead of stranding in DependencyNeverSatisfied.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_GATE_VERDICT="${DFT_GATE_VERDICT:?set DFT_GATE_VERDICT (where the verdict goes)}"
DFT_CALIB_TOL="${DFT_CALIB_TOL:-0.01}"

echo "=== ego-velocity splice gates: $(date) on $(hostname) ==="
echo "splits=$DFT_SPLITS_JSON verdict=$DFT_GATE_VERDICT calib_tol=$DFT_CALIB_TOL"
mkdir -p "$(dirname "$DFT_GATE_VERDICT")"
rm -f "$DFT_GATE_VERDICT"

rc=0
"$PY" -m rl.diffusion_ft.tests.test_ego_vel_splice \
  --splits_json "$DFT_SPLITS_JSON" --shard_glob "$DFT_SHARD_GLOB" \
  --index_dir "$DFT_INDEX_DIR" --ckpt "$DFT_CKPT" \
  --calib_tol "$DFT_CALIB_TOL" --verdict "$DFT_GATE_VERDICT" || rc=$?

# The test writes the verdict itself on PASS and on FAIL; no file means it died
# before reaching the end, which is not the same thing as a failed gate.
if [[ ! -s "$DFT_GATE_VERDICT" ]]; then
  echo "CRASH (exit $rc, see $0 output above)" > "$DFT_GATE_VERDICT"
fi
echo "=== verdict: $(head -1 "$DFT_GATE_VERDICT") (python exit $rc) ==="
exit 0
