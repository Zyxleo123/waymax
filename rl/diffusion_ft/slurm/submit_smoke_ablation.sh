#!/bin/bash
# Smoke test for the code paths sweep_safety_ablation.sh adds, at a size that
# finishes in minutes. Only calls sbatch. RUN THIS FIRST: the sweep is ~30 GPU-h,
# and a typo in the new plumbing would otherwise surface after hours of queue.
#
# WHAT IS NEW AND THEREFORE UNPROVEN (everything else the sweep uses has run):
#   * --edge_tau / --agent_tau      reach SafetyLossConfig at all
#   * --safety_window               the scored window decoupled from the EXECUTED
#                                   one; `neighbor_window` slices the log at
#                                   t0+1..t0+n_steps, so a window running off the
#                                   end of a scenario is a shape mismatch, and
#                                   `n_steps` is a STATIC jit arg so each distinct
#                                   value is its own compilation
#   * --anchor_window               the [T] time weight through _expert_loss,
#                                   base_anchor_loss and both loss paths
#                                   (the jitted _free_core and the closure)
#   * train_safety.sbatch's         a DIRECTORY argument -> <task_id>.env
#     array-mode env selection
#
# TWO ARMS, as a %2 array so it never holds more than 2 GPUs:
#   0  smoke_all   every new knob off its default AT ONCE, and all three anchor
#                  terms on together, so one job traces every new branch:
#                  safety_window=-1 (full 5 s candidate), anchor_window=0
#                  (executed only), non-default tau, base_anchor + bc + anchor.
#                  This is NOT a configuration to train with -- it is maximum
#                  code coverage in one job.
#   1  smoke_base  the sweep's actual centre, 2 iterations, to prove the centre
#                  itself still starts after the edits.
#
# WHAT TO READ IN THE .out (the trainer prints the resolved windows at startup):
#   [safety] safety window: 50 world steps (5.0 s of the candidate's 5.0 s);
#            env executes 10 (1.0 s)  -- SCORING PAST THE REPLAN: ...
#   [safety] anchor window: 5/25 model steps (1.0 s) -- the anchors are charged...
# and for smoke_base:
#   [safety] safety window: 10 world steps (1.0 s ...)  / anchor window: 25/25
# Then check the first `iter` line has FINITE safety/anchor/bc/base_anchor terms.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_smoke_ablation.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE="${DFT_SMOKE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/smoke_ablation/${STAMP}}"
ENV_DIR="$BASE/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

SPLITS_JSON="${DFT_SPLITS_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_verified.json}"
CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
[[ -s "$SPLITS_JSON" ]] || { echo "FATAL: no splits file $SPLITS_JSON" >&2; exit 1; }

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

# name|overrides -- same format as the sweep.
ARMS=(
  "smoke_all|DFT_SAFETY_WINDOW=-1,DFT_ANCHOR_WINDOW=0,DFT_EDGE_TAU=0.075,DFT_AGENT_TAU=0.5,DFT_BC_WEIGHT=100,DFT_ANCHOR_WEIGHT=1.0,DFT_K_TRAINABLE=3"
  "smoke_base|"
)

i=0
for spec in "${ARMS[@]}"; do
  TAG="${spec%%|*}"; OVR="${spec#*|}"; [[ "$OVR" == "$spec" ]] && OVR=""
  mkdir -p "$BASE/$TAG"
  {
    echo "DFT_OUT_DIR=$BASE/$TAG"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SHARD_GLOB=$GLOB"
    echo "DFT_INDEX_DIR=$IDX"
    # Tiny everything: this proves the code runs, not that it learns.
    echo "DFT_SCENES_PER_ITER=8"
    echo "DFT_EVAL_SCENES=8"
    echo "DFT_MAX_REPLANS=2"
    echo "DFT_ITERS=2"
    echo "DFT_EVAL_EVERY=1"
    echo "DFT_SAVE_EVERY=100"
    echo "DFT_KEEP_EVERY=0"
    # The eval path still runs, at 16 scenes/split in 8-scene chunks, because the
    # split eval is where an OOM or a shape error would otherwise first appear.
    echo "DFT_SPLITS_JSON=$SPLITS_JSON"
    echo "DFT_SPLITS=failure_heldout,success_heldout"
    echo "DFT_SPLIT_CAP=16"
    echo "DFT_EVAL_CHUNK=8"
    echo "DFT_EVAL_REPEATS=1"
    echo "DFT_EVAL_BASE=1"
    echo "DFT_TRAIN_SPLITS=failure_train,success_train"
    echo "DFT_TRAIN_SPLIT_CAP=-1"
    echo "DFT_TRAIN_SPLIT_RATIO=1,1"
    echo "DFT_PROBE_PER_SPLIT=0"
    echo "DFT_PROBE_EVERY=0"
    # the sweep's centre
    echo "DFT_SAFETY_WEIGHT=1.0"
    echo "DFT_ANCHOR_WEIGHT=0.0"
    echo "DFT_BC_WEIGHT=0.0"
    echo "DFT_BASE_ANCHOR_WEIGHT=100"
    echo "DFT_BASE_ANCHOR_SUCCESS_ONLY=1"
    echo "DFT_EDGE_MARGIN=0.3"
    echo "DFT_AGENT_MARGIN=0.75"
    echo "DFT_EDGE_TAU=0.15"
    echo "DFT_AGENT_TAU=0.25"
    echo "DFT_SAFETY_WINDOW=0"
    echo "DFT_ANCHOR_WINDOW=-1"
    echo "DFT_K_TRAINABLE=10"
    echo "DFT_RESUME=0"
    echo "DFT_WANDB=0"
    if [[ -n "$OVR" ]]; then
      echo "# --- arm override: $TAG ---"
      IFS=',' read -ra KV <<< "$OVR"
      for kv in "${KV[@]}"; do echo "$kv"; done
    fi
  } > "$ENV_DIR/${i}.env"
  printf '%2d  %-11s %s\n' "$i" "$TAG" "${OVR:-<centre>}"
  i=$((i + 1))
done
echo

AID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" --array="0-$((i - 1))%2" \
    --job-name="dft_smokeabl" --time=1:00:00 \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_DIR")
echo "smoke array -> $AID  ($BASE)"
echo
echo "check:"
echo "  grep -E 'safety window|anchor window|array task' $ROOT/logs/dft/slurm-dft_smokeabl-*.out"
echo "  grep -E '^iter ' $ROOT/logs/dft/slurm-dft_smokeabl-*.out"
