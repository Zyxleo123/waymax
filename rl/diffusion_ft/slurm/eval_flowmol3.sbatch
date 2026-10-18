#!/bin/bash
#SBATCH --job-name=dft_fm3_eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#
# Closed-loop eval of one FlowMol3 arm, through the SAME harness
# (`train_safety.py --eval_only`) and the SAME named splits every other arm in
# this project was scored with -- that is the whole reason not to write a new
# eval: a number that cannot be put next to the existing ones is not a result.
#
# GPU CLASS IS NOT NEGOTIABLE: a6000/a5000 on `general`/`preempt`.
#
#     bash rl/diffusion_ft/slurm/submit_eval_flowmol3.sh
#
# Config arrives as a positional env FILE or DIRECTORY (array mode). Never --export.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "FATAL: dir arg but no array" >&2; exit 1; }
    ARM_ENV="$arg/${SLURM_ARRAY_TASK_ID}.env"
    [[ -f "$ARM_ENV" ]] || { echo "FATAL: no env file $ARM_ENV" >&2; exit 1; }
    # shellcheck disable=SC1090
    source "$ARM_ENV"
  elif [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

FM3_EVAL_NAME="${FM3_EVAL_NAME:?set FM3_EVAL_NAME}"
FM3_EVAL_GLOB="${FM3_EVAL_GLOB:?set FM3_EVAL_GLOB (the arm dir glob, or the base checkpoint)}"
FM3_EVAL_OUT="${FM3_EVAL_OUT:?set FM3_EVAL_OUT}"
# Pinned, not defaulted: an eval scored against a different splits.json or a
# different seed is measuring a different thing, and both fail SILENTLY.
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON (the VERIFIED splits; a third of a raw failure_heldout is not a failure)}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:?set DFT_SHARD_GLOB (must be the glob splits.json was built against)}"
DFT_SPLITS="${DFT_SPLITS:-failure_heldout,success_heldout}"
DFT_SPLIT_CAP="${DFT_SPLIT_CAP:-1000}"
DFT_EVAL_CHUNK="${DFT_EVAL_CHUNK:-32}"
DFT_EVAL_SEED="${DFT_EVAL_SEED:-12345}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"

# ---- decision gate, INSIDE the job ------------------------------------------
# A missing checkpoint means the training arm has not produced one yet (or
# failed). Say so and exit 0: a non-zero exit here would strand every dependent
# job in DependencyNeverSatisfied with no explanation, which is strictly worse
# than a log line.
CKPT="$(ls -d $FM3_EVAL_GLOB 2>/dev/null | sort | tail -1 || true)"
if [[ -z "$CKPT" || ! -d "$CKPT" ]]; then
  echo "[gate] no checkpoint matched '$FM3_EVAL_GLOB' for arm '$FM3_EVAL_NAME'."
  echo "[gate] the training arm has not written one. Nothing to score; exiting 0."
  exit 0
fi
if [[ ! -f "$CKPT/metadata.json" ]]; then
  echo "[gate] '$CKPT' has no metadata.json, so its architecture is unknown."
  echo "[gate] exiting 0 rather than loading it with guessed defaults."
  exit 0
fi

echo "host=${SLURMD_NODENAME:-unknown}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "arm=$FM3_EVAL_NAME"
echo "ckpt=$CKPT"
"$PY" -c "import json,sys; m=json.load(open('$CKPT/metadata.json')); print('[eval] self_cond=%s distort_p=%s distort_sigma=%s mode=%s' % (m.get('self_cond'), m.get('distort_p'), m.get('distort_sigma'), m.get('distort_mode')))"

OUT="$FM3_EVAL_OUT/$FM3_EVAL_NAME"
mkdir -p "$OUT"

"$PY" -m rl.diffusion_ft.train_safety \
  --eval_only \
  --ckpt "$CKPT" \
  --out_dir "$OUT" \
  --dump_dir "$OUT/eval" \
  --shard_glob "$DFT_SHARD_GLOB" \
  --index_dir "$DFT_INDEX_DIR" \
  --splits_json "$DFT_SPLITS_JSON" \
  --splits "$DFT_SPLITS" \
  --split_cap "$DFT_SPLIT_CAP" \
  --eval_chunk "$DFT_EVAL_CHUNK" \
  --eval_seed "$DFT_EVAL_SEED"

echo "[$FM3_EVAL_NAME] eval rows under $OUT/eval"
