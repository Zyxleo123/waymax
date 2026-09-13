#!/bin/bash
# Submitter for the DDIM frozen-prefix A/B. Only calls sbatch.
#
# WHAT IT ANSWERS. The free-running proposal costs 100 denoiser evals and each
# replan pays it twice (collection rollout + gradient pass), while only the last
# DFT_K_TRAINABLE steps carry a gradient. --collect_steps S strides the frozen
# prefix to S - k steps. This measures what that buys (train_iter_s) against
# what it costs (solved/offroad/drift of the base policy at that step count), so
# the value is picked from a table instead of guessed.
#
# Then: DFT_COLLECT_STEPS=<S> bash rl/diffusion_ft/slurm/submit_verified_safety.sh
# and leave DFT_EVAL_STEPS=0 so the curve keeps measuring the deployable policy.
#
# NOTE ON THE OLDER DPPO BENCH. tests/bench_ddim_ab.py could not work before the
# 2026-09-09 fix to DiffusionRLActor._sampler_for: the jitted sampler was cached
# on rl_mode alone while num_sample_steps was a closure constant, so its sweep
# (`actor.num_sample_steps = S`) never changed the sampler and every row was the
# same full chain. Do not carry its conclusions over.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea, so a
# compute node fails with "Access/permission denied"):
#
#   bash rl/diffusion_ft/slurm/submit_bench_safety_ddim.sh
#   DFT_AB_STEPS=0,40,25,15,12 bash rl/diffusion_ft/slurm/submit_bench_safety_ddim.sh
#
# Node: `general` advertises only a6000/a5000 (gpu24-31). v100 dies at cuDNN
# autotune in this env and the 2080 Ti OOMs -- do NOT widen the partition.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
# $$ as well as the second: two submissions in the same second would otherwise
# share an env filename and an out_dir.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="$ENV_DIR/ddimab_${STAMP}.env"
OUT_DIR="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bench_ddim/${STAMP}}"

{
  echo "DFT_OUT_DIR=$OUT_DIR"
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_SHARD_GLOB=${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
  echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  # Same batch and replan budget as the training arm, or the timing is not the
  # timing that arm pays.
  echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-32}"
  echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
  echo "DFT_REPLAN=${DFT_REPLAN:-10}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
  echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
  echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
  # 0 (full chain) MUST stay first -- it is the reference row every delta is
  # taken against. Values must satisfy k_trainable <= S <= timesteps.
  echo "DFT_AB_STEPS=${DFT_AB_STEPS:-0,50,30,20,15}"
  echo "DFT_AB_ROUNDS=${DFT_AB_ROUNDS:-5}"
} > "$ENV_FILE"
echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"; echo

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

# Scrub SLURM_* so a submission from inside an allocation does not inherit it.
JOB_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/bench_safety_ddim.sbatch "$ENV_FILE")
echo "bench -> job $JOB_ID  ($OUT_DIR)"
echo
echo "watch:  tail -f $ROOT/logs/dft/slurm-dft_ddimab-${JOB_ID}.out"
echo "table:  grep -A20 'DDIM frozen-prefix A/B' $ROOT/logs/dft/slurm-dft_ddimab-${JOB_ID}.out"
