#!/bin/bash
# Submitter for the FlowMol3 2x2 ablation (self-conditioning x late-stage
# distortion). Only calls sbatch.
#
#   bash rl/diffusion_ft/slurm/submit_train_flowmol3.sh
#
# Pick a subset of arms with FM3_ARRAY (sbatch --array syntax); the mapping is
# 0=base 1=sc 2=dist 3=sc_dist:
#   FM3_ARRAY=0,3 bash rl/diffusion_ft/slurm/submit_train_flowmol3.sh
#
# Set the distortion scale from tools/probe_sampler_drift.py rather than the
# transplanted default:
#   FM3_DISTORT_SIGMA=0.45 bash rl/diffusion_ft/slurm/submit_train_flowmol3.sh
#
# %2 throttles the array to two concurrent GPUs; independent sibling jobs have
# no such knob, which is the only reason this is an array and not four jobs.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR_BASE="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR_BASE"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_DIR="$ENV_DIR_BASE/flowmol3_${STAMP}"; mkdir -p "$ENV_DIR"

SAVE_DIR="${FM3_SAVE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/flowmol3/${STAMP}}"
ARMS=(base sc dist sc_dist)

for i in "${!ARMS[@]}"; do
  {
    echo "FM3_ARM='${ARMS[$i]}'"
    echo "FM3_SAVE_DIR='${SAVE_DIR}'"
    for v in FM3_INIT_FROM FM3_TFRECORD FM3_EPOCHS FM3_STEPS_PER_EPOCH \
             FM3_SAVE_EVERY FM3_BATCH_SIZE FM3_LR FM3_WARMUP FM3_SEED \
             FM3_DISTORT_P FM3_DISTORT_SIGMA FM3_DISTORT_MODE \
             FM3_DISTORT_T_MAX FM3_DISTORT_T_MIN FM3_DISTORT_CHANNELS \
             FM3_SC_TRAIN_PROB FM3_WANDB_MODE FM3_WANDB_PROJECT DFT_PYTHON; do
      [[ -n "${!v:-}" ]] && echo "${v}='${!v}'"
    done
    true
  } > "$ENV_DIR/${i}.env"
done

# Verify the round trip before sbatch: a quoting slip becomes a run that trains
# a different arm than its name says.
for i in "${!ARMS[@]}"; do
  ( set -euo pipefail; set -a; . "$ENV_DIR/${i}.env"; set +a
    [[ "$FM3_ARM" == "${ARMS[$i]}" && "$FM3_SAVE_DIR" == "$SAVE_DIR" ]] \
      || { echo "FATAL: env round trip broken for task $i" >&2; exit 1; } )
done

echo "Env dir: $ENV_DIR"
for i in "${!ARMS[@]}"; do echo "  [$i] ${ARMS[$i]}"; done
echo "Save dir: $SAVE_DIR"

env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --array="${FM3_ARRAY:-0-3}%2" rl/diffusion_ft/slurm/train_flowmol3.sbatch "$ENV_DIR"
