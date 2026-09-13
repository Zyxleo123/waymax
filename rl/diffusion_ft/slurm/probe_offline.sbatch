#!/bin/bash
#SBATCH --job-name=dft_probeoff
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Post-hoc probe re-selection for the 8-panel slider. Given a FINISHED run's
# out_dir, classify each split scene by its outcome at the last checkpoint, pick
# K per (base-outcome x now-outcome x train/heldout) bucket -- so the `regressed`
# panel is populated, which the in-run probe cannot do -- then re-roll that
# pinned set over the frozen base and every iter_* checkpoint.
#
# GPU CLASS IS NOT NEGOTIABLE: `general`/`preempt` a6000/a5000 only. On a v100
# this env's jaxlib dies at cuDNN autotune; the 11 GB 2080 Tis OOM. `general`
# holds only a6000/a5000, so no --constraint is needed.
#
# Config arrives as a positional env FILE (never `sbatch --export`, which
# triggers SLURM_GET_USER_ENV and requeue-holds the job).
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
# 0.95 + chunk 32 is the validated a5000 envelope (see train_safety.sbatch).
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

DFT_OFFLINE_RUN_DIR="${DFT_OFFLINE_RUN_DIR:?set DFT_OFFLINE_RUN_DIR to a finished run's out_dir}"
DFT_OFFLINE_DIR="${DFT_OFFLINE_DIR:-$DFT_OFFLINE_RUN_DIR/probe_offline}"
DFT_OFFLINE_K="${DFT_OFFLINE_K:-3}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON (the run's splits file)}"
DFT_SPLIT_CAP="${DFT_SPLIT_CAP:-1000}"
DFT_EVAL_CHUNK="${DFT_EVAL_CHUNK:-32}"
DFT_EVAL_SEED="${DFT_EVAL_SEED:-12345}"
DFT_PROBE_SEED="${DFT_PROBE_SEED:-0}"
# Model/rollout geometry -- MUST match the run being probed (defaults are the
# base_anchor sweep's values).
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "run_dir=$DFT_OFFLINE_RUN_DIR"
echo "out=$DFT_OFFLINE_DIR k=$DFT_OFFLINE_K cap=$DFT_SPLIT_CAP chunk=$DFT_EVAL_CHUNK"
echo "splits_json=$DFT_SPLITS_JSON"

# Gate INSIDE the job: no checkpoints means nothing to probe. Say so, exit 0.
if ! ls -d "$DFT_OFFLINE_RUN_DIR"/iter_* >/dev/null 2>&1; then
  echo "GATE: no iter_* checkpoints in $DFT_OFFLINE_RUN_DIR -- was keep_every set?"
  exit 0
fi

srun "$PY" -u -m rl.diffusion_ft.train_safety \
  --probe_offline --probe_offline_k "$DFT_OFFLINE_K" \
  --probe_offline_dir "$DFT_OFFLINE_DIR" \
  --ckpt "$DFT_CKPT" --out_dir "$DFT_OFFLINE_RUN_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --scenes_per_iter 32 --eval_scenes 32 --data_seed 0 --prefetch 1 \
  --splits_json "$DFT_SPLITS_JSON" --split_cap "$DFT_SPLIT_CAP" \
  --eval_chunk "$DFT_EVAL_CHUNK" --eval_seed "$DFT_EVAL_SEED" \
  --probe_seed "$DFT_PROBE_SEED" \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --max_replans "$DFT_MAX_REPLANS" --replan_interval_steps "$DFT_REPLAN" \
  --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN" \
  --agent_geometry "$DFT_AGENT_GEOMETRY" --n_discs "$DFT_N_DISCS" \
  --no-wandb
