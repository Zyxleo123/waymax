#!/bin/bash
#SBATCH --job-name=dft_evalreach
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Post-hoc reach/solved scoring of a finished safety_train checkpoint, on the
# SAME held-out eval scenes the run used (same shard_glob + eval_scenes +
# data_seed => same batch). Scores base AND fine-tuned and dumps the driving
# artifacts the CPU viz job (viz_reach.sbatch) renders. No training.
#
# GPU CLASS IS NOT NEGOTIABLE: this env's jaxlib only runs on a6000/a5000
# (general/preempt gpu24-31, gpu27/30/31+gpu1). A v100 dies at cuDNN autotune;
# the 11 GB 2080 Tis OOM with no traceback. See reference_gpu_arch_a6000_only.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR (the finished safety_train run dir)}"
DFT_LOAD_CKPT="${DFT_LOAD_CKPT:-${DFT_OUT_DIR}/latest}"
DFT_DUMP_DIR="${DFT_DUMP_DIR:-${DFT_OUT_DIR}/eval}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
# data: must match the run so eval_batch() returns the identical held-out scenes
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_EVAL_SCENES="${DFT_EVAL_SCENES:-32}"
# In-distribution check: also score a train-pool batch (disjoint from held-out).
DFT_EVAL_TRAIN="${DFT_EVAL_TRAIN:-1}"          # 1 = also score train scenes
DFT_EVAL_TRAIN_SCENES="${DFT_EVAL_TRAIN_SCENES:-100}"
# Named splits: score failure/success held-out sets by explicit scene id.
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-}"         # build_eval_splits.py output
DFT_SPLITS="${DFT_SPLITS:-failure_heldout,success_heldout}"
DFT_SPLIT_CAP="${DFT_SPLIT_CAP:-64}"
DFT_DATA_SEED="${DFT_DATA_SEED:-0}"
# model: must match so the checkpoint's leaf count matches the built actor
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"
# schedule of the eval rollout (matches the trainer's held-out eval)
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_SEED="${DFT_SEED:-0}"

EVAL_TRAIN_FLAG=""
[[ "$DFT_EVAL_TRAIN" == "1" ]] && EVAL_TRAIN_FLAG="--eval_train"
SPLITS_FLAG=()
[[ -n "$DFT_SPLITS_JSON" ]] && SPLITS_FLAG=(--splits_json "$DFT_SPLITS_JSON" \
  --splits "$DFT_SPLITS" --split_cap "$DFT_SPLIT_CAP")

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "out_dir=$DFT_OUT_DIR  ckpt=$DFT_LOAD_CKPT"
echo "dump=$DFT_DUMP_DIR  heldout=$DFT_EVAL_SCENES train=${DFT_EVAL_TRAIN_SCENES}(on=$DFT_EVAL_TRAIN) data_seed=$DFT_DATA_SEED k=$DFT_K_TRAINABLE"
mkdir -p "$DFT_DUMP_DIR"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

# scenes_per_iter sizes the train-pool batch scored by --eval_train.
"$PYTHON" -m rl.diffusion_ft.train_safety --eval_only $EVAL_TRAIN_FLAG \
  "${SPLITS_FLAG[@]}" \
  --out_dir "$DFT_OUT_DIR" --load_ckpt "$DFT_LOAD_CKPT" --dump_dir "$DFT_DUMP_DIR" \
  --ckpt "$DFT_CKPT" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --eval_scenes "$DFT_EVAL_SCENES" --scenes_per_iter "$DFT_EVAL_TRAIN_SCENES" \
  --data_seed "$DFT_DATA_SEED" --prefetch 0 \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN" \
  --agent_geometry "$DFT_AGENT_GEOMETRY" --n_discs "$DFT_N_DISCS" \
  --replan_interval_steps "$DFT_REPLAN" --max_replans "$DFT_MAX_REPLANS" \
  --seed "$DFT_SEED" --resume 0 --no-wandb
