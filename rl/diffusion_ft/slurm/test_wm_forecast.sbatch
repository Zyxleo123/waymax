#!/bin/bash
#SBATCH --job-name=dft_wm_gate
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gates for the learned world-model neighbour sources (rl/diffusion_ft/
# world_model_forecast.py, tests/test_wm_forecast.py): vendored-import isolation,
# decode parity with es_WM's own decoder, weights/masks, the external-box path ==
# cv, forecast accuracy vs cv under log replay, and o_am's optimizer per source.
#
# CPU (torch + jax on CPU), no policy: waymax reset + expert log replay only.
#
# DECISION GATE, so it always exits 0 and leaves its answer in a verdict file
# (DFT_GATE_VERDICT): first line PASS, FAIL: <gates>, or CRASH. The select tasks
# read it (select_eval.sbatch, DFT_REQUIRE_GATE).
#
# Config arrives as a positional env FILE. Never `sbatch --export`.
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
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_GATE_VERDICT="${DFT_GATE_VERDICT:?set DFT_GATE_VERDICT (where the verdict goes)}"
DFT_WM_JSON="${DFT_WM_JSON:-$(dirname "$DFT_GATE_VERDICT")/test_wm_forecast.json}"
DFT_WM_GATE_SCENES="${DFT_WM_GATE_SCENES:-8}"
DFT_WM_DET_CKPT="${DFT_WM_DET_CKPT:-}"
DFT_WM_MF_CKPT="${DFT_WM_MF_CKPT:-}"

echo "=== world-model gates: $(date) on $(hostname) ==="
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?} (CPU job, no GPU)"
echo "splits=$DFT_SPLITS_JSON scenes=$DFT_WM_GATE_SCENES verdict=$DFT_GATE_VERDICT"
echo "wm det=${DFT_WM_DET_CKPT:-<default>} mf=${DFT_WM_MF_CKPT:-<default>}"
mkdir -p "$(dirname "$DFT_GATE_VERDICT")"
rm -f "$DFT_GATE_VERDICT"

rc=0
"$PY" -m rl.diffusion_ft.tests.test_wm_forecast \
  --splits_json "$DFT_SPLITS_JSON" --shard_glob "$DFT_SHARD_GLOB" \
  --index_dir "$DFT_INDEX_DIR" --n_scenes "$DFT_WM_GATE_SCENES" \
  --wm_det_ckpt "$DFT_WM_DET_CKPT" --wm_mf_ckpt "$DFT_WM_MF_CKPT" \
  --verdict "$DFT_GATE_VERDICT" --json_out "$DFT_WM_JSON" || rc=$?

# The test writes the verdict itself on PASS and on FAIL; no file means it died
# before reaching the end, which is not the same thing as a failed gate.
if [[ ! -s "$DFT_GATE_VERDICT" ]]; then
  echo "CRASH (exit $rc, see output above)" > "$DFT_GATE_VERDICT"
fi
echo "=== verdict: $(head -1 "$DFT_GATE_VERDICT") (python exit $rc) ==="
exit 0
