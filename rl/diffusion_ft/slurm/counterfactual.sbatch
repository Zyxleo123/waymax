#!/bin/bash
#SBATCH --job-name=dft_cf
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Counterfactual rollouts (rl/diffusion_ft/counterfactual.py): --mode repair
# (rewind + search + verified repairs) or --mode labels (one-switch labels for a
# learned selector). Resumable per chunk via <out>/done/ markers.
#
# GPU CLASS: `general` a6000 (a5000 excluded by the submitter): repair data holds
# per-candidate conditions on the host and the env on the card.
#
# Config: positional env FILE, or DIRECTORY in array mode. Never `sbatch --export`.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
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

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_CF_MODE="${DFT_CF_MODE:?set DFT_CF_MODE (repair|labels)}"
DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_CF_SPLITS="${DFT_CF_SPLITS:-}"
DFT_CF_CAPS="${DFT_CF_CAPS:--1}"
DFT_CF_SEED="${DFT_CF_SEED:-2718}"
DFT_CANDS="${DFT_CANDS:-id,tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5,tw0.6h2,tw1.4h2,rn20s0,rn20s1,rn50s0,rn50s1}"
DFT_REWINDS="${DFT_REWINDS:-3}"
DFT_RAMP_STEPS="${DFT_RAMP_STEPS:-10}"
DFT_FT_CKPT="${DFT_FT_CKPT:-}"
DFT_LABEL_ALT="${DFT_LABEL_ALT:-ft}"
DFT_SHARD_INDEX="${DFT_SHARD_INDEX:-0}"
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-1}"
DFT_EVAL_CHUNK="${DFT_EVAL_CHUNK:-32}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_SEED="${DFT_SEED:-0}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "mode=$DFT_CF_MODE splits=${DFT_CF_SPLITS:-<default>} caps=$DFT_CF_CAPS shard=$DFT_SHARD_INDEX/$DFT_NUM_SHARDS"
echo "cands=$DFT_CANDS rewinds=$DFT_REWINDS ft_ckpt=${DFT_FT_CKPT:-<none>} alt=$DFT_LABEL_ALT"
mkdir -p "$DFT_OUT_DIR"

srun "$PY" -u -m rl.diffusion_ft.counterfactual --mode "$DFT_CF_MODE" \
  --ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --splits_json "$DFT_SPLITS_JSON" --cf_splits "$DFT_CF_SPLITS" --cf_caps="$DFT_CF_CAPS" \
  --cf_seed "$DFT_CF_SEED" --cands "$DFT_CANDS" --rewinds "$DFT_REWINDS" \
  --ramp_steps "$DFT_RAMP_STEPS" --ft_ckpt "$DFT_FT_CKPT" --label_alt "$DFT_LABEL_ALT" \
  --shard_index "$DFT_SHARD_INDEX" --num_shards "$DFT_NUM_SHARDS" \
  --eval_chunk "$DFT_EVAL_CHUNK" --eval_scenes 0 --prefetch 0 --probe_per_split 0 \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --replan_interval_steps "$DFT_REPLAN" --max_replans "$DFT_MAX_REPLANS" \
  --seed "$DFT_SEED" --no-wandb
echo "out -> $DFT_OUT_DIR"
