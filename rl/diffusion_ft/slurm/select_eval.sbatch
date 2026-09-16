#!/bin/bash
#SBATCH --job-name=dft_select
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
# Training-free proposal selection (rl/diffusion_ft/proposal_select.py): every arm
# closed-loop on the same scenes and sampler seeds, one JSONL row per
# (scene, arm, repeat) plus per-replan selector features in feat/*.npz.
#
# Resumable: finished chunks are skipped, so a time-limit kill is a resubmit.
#
# GPU CLASS: `general` a6000 (a5000 excluded by the submitter). This env's jaxlib
# dies at cuDNN autotune on v100 and OOMs on 2080 Tis; selection arms also hold
# two candidates per replan, so do not widen the partition.
#
# Config arrives as a positional env FILE, or a DIRECTORY in array mode
# (<dir>/<task_id>.env). Never `sbatch --export`.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "FATAL: '$arg' is a dir but not an array job" >&2; exit 1; }
    TASK_ENV="$arg/${SLURM_ARRAY_TASK_ID}.env"
    [[ -f "$TASK_ENV" ]] || { echo "FATAL: no $TASK_ENV" >&2; exit 1; }
    echo "array task ${SLURM_ARRAY_TASK_ID} -> $TASK_ENV"
    # shellcheck disable=SC1090
    source "$TASK_ENV"
  elif [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON (three-stratum splits file)}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SELECT_SPLITS="${DFT_SELECT_SPLITS:-failure_heldout,success_heldout,intermittent_heldout}"
DFT_FT_CKPT="${DFT_FT_CKPT:-}"
DFT_ARMS="${DFT_ARMS:?set DFT_ARMS}"
DFT_SELECT_SEED="${DFT_SELECT_SEED:-4242}"
DFT_SELECT_REPEATS="${DFT_SELECT_REPEATS:-4}"
DFT_SELECT_TOL_M="${DFT_SELECT_TOL_M:-1.0}"
DFT_SELECT_MARGIN_M="${DFT_SELECT_MARGIN_M:-0.3}"
DFT_SELECT_DELTA_M="${DFT_SELECT_DELTA_M:-0.2}"
DFT_SHARD_INDEX="${DFT_SHARD_INDEX:-0}"
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-1}"
DFT_EVAL_CHUNK="${DFT_EVAL_CHUNK:-32}"
DFT_SELECT_CAP="${DFT_SELECT_CAP:--1}"
DFT_SELECTOR_JSON="${DFT_SELECTOR_JSON:-}"
DFT_LOOK_REPLANS="${DFT_LOOK_REPLANS:-2}"
DFT_SELECT_BANK="${DFT_SELECT_BANK:-tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5}"
DFT_SELECT_RAMP="${DFT_SELECT_RAMP:-10}"
DFT_SELECT_BANKS="${DFT_SELECT_BANKS:-}"
# Rollout shape MUST match the eval the fine-tunes were selected under.
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_SEED="${DFT_SEED:-0}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "arms=$DFT_ARMS"
echo "ft_ckpt=${DFT_FT_CKPT:-<none>} shard=$DFT_SHARD_INDEX/$DFT_NUM_SHARDS chunk=$DFT_EVAL_CHUNK"
echo "splits=$DFT_SELECT_SPLITS seed=$DFT_SELECT_SEED repeats=$DFT_SELECT_REPEATS"
if [[ "$DFT_ARMS" == *"bank"* ]]; then
  echo "bank=$DFT_SELECT_BANK ramp=$DFT_SELECT_RAMP"
  if [[ -n "$DFT_SELECT_BANKS" ]]; then echo "banks=$DFT_SELECT_BANKS"; fi
fi
mkdir -p "$DFT_OUT_DIR"
# Gate: a 'learned' arm with no selector means the fit job gated out (no labels).
# Say so and exit 0 so nothing downstream strands on DependencyNeverSatisfied.
if [[ "$DFT_ARMS" == *":learned"* && ! -s "$DFT_SELECTOR_JSON" ]]; then
  echo "SELECTION NOT RUN: arms use 'learned' but no selector at '${DFT_SELECTOR_JSON}'"
  echo "  (see the fit_selector job's .out: it exits 0 without a file when labels are missing)"
  exit 0
fi

srun "$PY" -u -m rl.diffusion_ft.proposal_select \
  --ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --splits_json "$DFT_SPLITS_JSON" --splits "$DFT_SELECT_SPLITS" \
  --ft_ckpt "$DFT_FT_CKPT" --arms "$DFT_ARMS" \
  --select_seed "$DFT_SELECT_SEED" --select_repeats "$DFT_SELECT_REPEATS" \
  --select_tol_m "$DFT_SELECT_TOL_M" --select_margin_m "$DFT_SELECT_MARGIN_M" \
  --select_delta_m "$DFT_SELECT_DELTA_M" \
  --shard_index "$DFT_SHARD_INDEX" --num_shards "$DFT_NUM_SHARDS" \
  --select_cap="$DFT_SELECT_CAP" \
  --selector_json "$DFT_SELECTOR_JSON" --look_replans "$DFT_LOOK_REPLANS" \
  --select_bank "$DFT_SELECT_BANK" --select_ramp_steps "$DFT_SELECT_RAMP" \
  --select_banks "$DFT_SELECT_BANKS" \
  --eval_chunk "$DFT_EVAL_CHUNK" --eval_scenes 0 --prefetch 0 --probe_per_split 0 \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --replan_interval_steps "$DFT_REPLAN" --max_replans "$DFT_MAX_REPLANS" \
  --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN" \
  --seed "$DFT_SEED" --no-wandb
echo "rows -> $DFT_OUT_DIR/rows.jsonl"
