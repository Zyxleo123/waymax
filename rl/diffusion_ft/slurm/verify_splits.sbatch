#!/bin/bash
#SBATCH --job-name=dft_verify
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Stage A of the split rebuild: re-score the FROZEN base checkpoint per scene on
# the candidate splits, through the SAME code path the training curve evaluates
# with (`train_safety.rollout(train=False)`), at a seed disjoint from
# --eval_seed. Writes one JSONL row per scene.
#
# WHY. The current splits were labelled by scan_womd_headroom's DETERMINISTIC
# rollout at sigma_floor=0.03. Under the eval protocol the base solves 0.328 of
# `failure_heldout` -- a split defined as scenes it fails. A third of the failure
# set is therefore not a failure of the policy being fine-tuned.
#
# Resumable: rows already written are skipped, so a time-limit kill is fixed by
# resubmitting. --open-mode=append keeps the previous attempt's log.
#
# GPU CLASS: `general` a6000/a5000 only. This env's jaxlib dies at cuDNN autotune
# on v100 and OOMs on the 11 GB 2080 Tis -- do not widen the partition.
#
# Config arrives as a positional env FILE (never `sbatch --export`).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

# A DIRECTORY argument is array mode (<dir>/<task_id>.env), same contract as
# train_safety.sbatch, so several policies can be scored under one %N throttle.
for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "FATAL: '$arg' is a dir but not an array job" >&2; exit 1; }
    ARM_ENV="$arg/${SLURM_ARRAY_TASK_ID}.env"
    [[ -f "$ARM_ENV" ]] || { echo "FATAL: no $ARM_ENV" >&2; exit 1; }
    echo "array task ${SLURM_ARRAY_TASK_ID} -> $ARM_ENV"
    # shellcheck disable=SC1090
    source "$ARM_ENV"
  elif [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON (the splits file to verify)}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
# Which splits to re-measure, and how many scenes of each. All 628 failures are
# verified; success_train is ~23k and only needs enough verified scenes to fill
# the held-out success set, so it is capped.
DFT_VERIFY_SPLITS="${DFT_VERIFY_SPLITS:-failure_heldout,failure_train,success_heldout,success_train}"
# Passed as --verify_cap=... (one token). argparse reads a separate value
# starting with "-" as an OPTION unless it matches its negative-number pattern
# ^-\d+$ -- a bare "-1" survives that, "-1,-1,-1,2500" does not, and the job
# dies in argument parsing (job 43942).
DFT_VERIFY_CAP="${DFT_VERIFY_CAP:--1,-1,-1,2500}"
DFT_VERIFY_REPEATS="${DFT_VERIFY_REPEATS:-4}"
# NOT 12345 (--eval_seed): selecting scenes on the draws they are later scored on
# would bake this sample's noise into the split. The trainer refuses if equal.
DFT_VERIFY_SEED="${DFT_VERIFY_SEED:-777}"
DFT_VERIFY_OUT="${DFT_VERIFY_OUT:-${DFT_OUT_DIR}/verify_rows.jsonl}"
DFT_EVAL_CHUNK="${DFT_EVAL_CHUNK:-32}"
# Rollout shape MUST match the training curve's eval, or the verification
# measures a different policy than the one whose failures are being selected.
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_SEED="${DFT_SEED:-0}"
# Empty = score the frozen base. A path (a run's `latest` or an `iter_*` snapshot)
# = score those fine-tuned weights through the identical eval path.
DFT_LOAD_CKPT="${DFT_LOAD_CKPT:-}"
LOAD_ARGS=(); [[ -n "$DFT_LOAD_CKPT" ]] && LOAD_ARGS=(--load_ckpt "$DFT_LOAD_CKPT")

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "verify splits=$DFT_VERIFY_SPLITS caps=$DFT_VERIFY_CAP"
echo "repeats=$DFT_VERIFY_REPEATS seed=$DFT_VERIFY_SEED chunk=$DFT_EVAL_CHUNK"
echo "in=$DFT_SPLITS_JSON  out=$DFT_VERIFY_OUT"
echo "policy=${DFT_LOAD_CKPT:-<frozen base>}"

mkdir -p "$DFT_OUT_DIR"

srun "$PY" -u -m rl.diffusion_ft.train_safety \
  --verify_only --ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --splits_json "$DFT_SPLITS_JSON" --splits "$DFT_VERIFY_SPLITS" \
  --verify_cap="$DFT_VERIFY_CAP" --verify_repeats "$DFT_VERIFY_REPEATS" \
  --verify_seed "$DFT_VERIFY_SEED" --verify_out "$DFT_VERIFY_OUT" \
  --eval_chunk "$DFT_EVAL_CHUNK" --eval_scenes 0 --prefetch 0 \
  --probe_per_split 0 \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --replan_interval_steps "$DFT_REPLAN" --max_replans "$DFT_MAX_REPLANS" \
  --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN" \
  --seed "$DFT_SEED" --no-wandb ${LOAD_ARGS[@]+"${LOAD_ARGS[@]}"}
echo "verify rows -> $DFT_VERIFY_OUT"
