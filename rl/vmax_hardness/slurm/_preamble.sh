#!/bin/bash
# Shared configuration for the V-Max SAC hardness-balanced fine-tune pipeline.
#
# Sourced by every stage's job BODY (not via `#SBATCH`), after the body has set
# `set -euo pipefail` itself -- a failed source must not leave the job running on
# the system python and exiting 0, releasing afterok dependents that did no work.
#
# Config travels as positional ARGUMENTS, never `sbatch --export` (which requeue-
# holds on this cluster): the submitter writes one timestamped env file and passes
# its path; any path arg is sourced, any VAR=value arg is exported. Self-sufficient
# under `env -i` (all defaults live here).

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1

# --- Paths --------------------------------------------------------------- #
# Pretrained V-Max SAC on raw WOMD (lq encoder). model_best.pkl is the warm start.
VMH_RUN_DIR="${VMH_RUN_DIR:-/zfsauton/scratch/yixiz/waymax_rs/vmax_womd_raw/womd_raw_sac_lq}"
VMH_BEST_CKPT="${VMH_BEST_CKPT:-$VMH_RUN_DIR/model/model_best.pkl}"

VMH_SHARD_GLOB="${VMH_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*-of-01000}"
VMH_INDEX_DIR="${VMH_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"

VMH_OUT="${VMH_OUT:-/zfsauton/scratch/yixiz/waymax_rs/vmax_hardness}"
VMH_RESCORE_DIR="${VMH_RESCORE_DIR:-$VMH_OUT/rescore_labels}"   # Stage A per-scene JSONs
VMH_RISK_OUT="${VMH_RISK_OUT:-$VMH_OUT/risk}"                   # Stage B labels.npz + risk.npz

# Expert log-replay labels (policy-independent -> reused) and the a-priori
# descriptors already computed for all 1000 shards. Both from the diffusion run.
VMH_EXPERT_DIR="${VMH_EXPERT_DIR:-/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_log-replay_training_20260716_025213}"
VMH_DESC_DIR="${VMH_DESC_DIR:-/zfsauton2/home/yixiz/waymax_rs/runs/hardness/desc_all}"

# --- Stage A: re-score slice --------------------------------------------- #
VMH_N_SHARDS="${VMH_N_SHARDS:-250}"     # shards to evaluate
VMH_PER_SHARD="${VMH_PER_SHARD:-40}"    # records per shard -> ~10k labelled scenes
VMH_MAX_NUM_OBJECTS="${VMH_MAX_NUM_OBJECTS:-64}"
VMH_NUM_PATHS="${VMH_NUM_PATHS:-45}"
VMH_NUM_POINTS_PER_PATH="${VMH_NUM_POINTS_PER_PATH:-800}"

# --- Stage B: risk refit -------------------------------------------------- #
VMH_FIT_SLICE="${VMH_FIT_SLICE:-expert_clean}"   # fit only on scenes the expert drives clean

# --- Stage C: hardness-balanced fine-tune -------------------------------- #
VMH_ALPHA="${VMH_ALPHA:-1.0}"
VMH_WEIGHT_CLIP="${VMH_WEIGHT_CLIP:-10.0}"
VMH_EXCLUDE_DIRTY="${VMH_EXCLUDE_DIRTY:-true}"
VMH_TOTAL_TIMESTEPS="${VMH_TOTAL_TIMESTEPS:-5000000}"   # fine-tune budget (warm start resets the step counter)
VMH_SAVE_ROOT="${VMH_SAVE_ROOT:-/zfsauton/scratch/yixiz/waymax_rs/vmax_hardness_ft}"
VMH_NAME_RUN="${VMH_NAME_RUN:-womd_raw_sac_lq_hardness_a${VMH_ALPHA}}"
VMH_WANDB="${VMH_WANDB:-true}"
VMH_WANDB_PROJECT="${VMH_WANDB_PROJECT:-vmax-womd-hardness}"

export VMH_RUN_DIR VMH_BEST_CKPT VMH_SHARD_GLOB VMH_INDEX_DIR VMH_OUT \
  VMH_RESCORE_DIR VMH_RISK_OUT VMH_EXPERT_DIR VMH_DESC_DIR \
  VMH_N_SHARDS VMH_PER_SHARD VMH_MAX_NUM_OBJECTS VMH_NUM_PATHS VMH_NUM_POINTS_PER_PATH \
  VMH_FIT_SLICE VMH_ALPHA VMH_WEIGHT_CLIP VMH_EXCLUDE_DIRTY VMH_TOTAL_TIMESTEPS \
  VMH_SAVE_ROOT VMH_NAME_RUN VMH_WANDB VMH_WANDB_PROJECT
