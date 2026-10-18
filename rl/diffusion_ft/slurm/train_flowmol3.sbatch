#!/bin/bash
#SBATCH --job-name=dft_fm3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%A_%a.out
#
# FlowMol3 (arXiv 2508.12629) inference-OOD fine-tune of the 6/16 base diffusion
# checkpoint: self-conditioning and late-stage geometry distortion, as a 2x2
# ablation over the SAME denoising objective the base was pretrained on.
#
#   task 0  base      neither    -- the control. It is not the base checkpoint:
#                                   it is the base checkpoint given the same
#                                   extra fine-tune steps on the same data with
#                                   neither feature on. Without it, any change in
#                                   the eval is confounded with "trained longer".
#   task 1  sc        self-cond only
#   task 2  dist      distortion only
#   task 3  sc_dist   both        -- FlowMol3's shipped combination
#
# WHY A FINE-TUNE AND NOT A RE-PRETRAIN. Self-conditioning changes the parameter
# tree (the stem's fan-in doubles), which normally forces training from scratch.
# `--init_from` instead ZERO-EXPANDS the stem, so the expanded model computes the
# base's function exactly at step 0 (gated in tests/test_self_cond.py) and the
# run starts from the base policy's behaviour rather than from noise. Distortion
# needs no surgery at all.
#
# GPU CLASS IS NOT NEGOTIABLE: a6000/a5000 on `general`/`preempt`. On a v100 this
# env's jaxlib dies at cuDNN autotune with no Python traceback; the 11 GB
# 2080 Tis OOM and an XLA allocation failure there kills the process silently.
# Usable: gpu24-31 (general), gpu27/30/31 + gpu1 (preempt).
#
#     bash rl/diffusion_ft/slurm/submit_train_flowmol3.sh
#
# Config arrives as a positional env FILE or an env DIRECTORY (array mode:
# <dir>/<task_id>.env). NEVER --export, which triggers SLURM_GET_USER_ENV and
# requeue-holds the job.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "FATAL: '$arg' is a directory but this is not an array job." >&2
      exit 1
    fi
    ARM_ENV="$arg/${SLURM_ARRAY_TASK_ID}.env"
    [[ -f "$ARM_ENV" ]] || { echo "FATAL: no env file $ARM_ENV" >&2; exit 1; }
    echo "array task ${SLURM_ARRAY_TASK_ID} -> $ARM_ENV"
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

FM3_ARM="${FM3_ARM:?set FM3_ARM}"
FM3_SAVE_DIR="${FM3_SAVE_DIR:?set FM3_SAVE_DIR}"
FM3_INIT_FROM="${FM3_INIT_FROM:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
FM3_TFRECORD="${FM3_TFRECORD:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord@1000}"

# ---- fine-tune budget -------------------------------------------------------
# A FINE-TUNE, not a pretrain: the base ran 500 epochs x 1000 steps. The lr is an
# order of magnitude below the pretrain's 1e-4 and the warmup is short, because
# `--init_from` deliberately starts at the base's behaviour and the question is
# whether the two features move it, not whether the model can be retrained.
FM3_EPOCHS="${FM3_EPOCHS:-20}"
FM3_STEPS_PER_EPOCH="${FM3_STEPS_PER_EPOCH:-1000}"
FM3_SAVE_EVERY="${FM3_SAVE_EVERY:-5}"
FM3_BATCH_SIZE="${FM3_BATCH_SIZE:-64}"
FM3_LR="${FM3_LR:-1e-5}"
FM3_WARMUP="${FM3_WARMUP:-500}"
FM3_SEED="${FM3_SEED:-0}"

# ---- FlowMol3 knobs ---------------------------------------------------------
# p_distort: FlowMol3's 0.2, i.e. one waypoint in five is kicked.
FM3_DISTORT_P="${FM3_DISTORT_P:-0.2}"
# sigma in `relative` mode is a MULTIPLE of the forward noise sqrt(1-abar_t)
# already present at that timestep. 0.8 is FlowMol3's sigma_distort expressed
# against their own late-stage noise level; replace it with the value
# tools/probe_sampler_drift.py measures on THIS model before committing to a
# long run -- that probe exists precisely so this number is not a transplant
# from a molecule.
FM3_DISTORT_SIGMA="${FM3_DISTORT_SIGMA:-0.8}"
FM3_DISTORT_MODE="${FM3_DISTORT_MODE:-relative}"
# -1 == timesteps//2 - 1 == 49, FlowMol3's t_distort=0.5 mapped onto this DDPM
# index (their t runs noise->data, so their "late" half is this chain's LOW t).
FM3_DISTORT_T_MAX="${FM3_DISTORT_T_MAX:--1}"
FM3_DISTORT_T_MIN="${FM3_DISTORT_T_MIN:-0}"
# Positions only (channels x, y), as in FlowMol3 -- it is a GEOMETRY distortion.
FM3_DISTORT_CHANNELS="${FM3_DISTORT_CHANNELS:-2}"
FM3_SC_TRAIN_PROB="${FM3_SC_TRAIN_PROB:-0.5}"

FM3_WANDB_MODE="${FM3_WANDB_MODE:-offline}"
FM3_WANDB_PROJECT="${FM3_WANDB_PROJECT:-flowmol3_diffusion}"

# Per-arm feature flags. Set here, in the job, rather than in the submitter, so
# the arm definition and the arm name can never disagree.
FLAGS=()
case "$FM3_ARM" in
  base)    ;;
  sc)      FLAGS+=(--self_cond) ;;
  dist)    FLAGS+=(--distort_p "$FM3_DISTORT_P" --distort_sigma "$FM3_DISTORT_SIGMA") ;;
  sc_dist) FLAGS+=(--self_cond --distort_p "$FM3_DISTORT_P" --distort_sigma "$FM3_DISTORT_SIGMA") ;;
  *) echo "FATAL: unknown FM3_ARM='$FM3_ARM' (base|sc|dist|sc_dist)" >&2; exit 1 ;;
esac

ARM_DIR="$FM3_SAVE_DIR/$FM3_ARM"
mkdir -p "$ARM_DIR"

echo "host=${SLURMD_NODENAME:-unknown}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "arm=$FM3_ARM  flags=${FLAGS[*]:-<none>}"
echo "init_from=$FM3_INIT_FROM"
echo "save_dir=$ARM_DIR"
echo "budget=${FM3_EPOCHS}x${FM3_STEPS_PER_EPOCH} steps  lr=$FM3_LR  batch=$FM3_BATCH_SIZE"

"$PY" -m train_diffusion.pretrain_diffusion_policy \
  --init_from "$FM3_INIT_FROM" \
  --tfrecord_path "$FM3_TFRECORD" \
  --save_dir "$ARM_DIR" \
  --epochs "$FM3_EPOCHS" \
  --steps_per_epoch "$FM3_STEPS_PER_EPOCH" \
  --save_every "$FM3_SAVE_EVERY" \
  --batch_size "$FM3_BATCH_SIZE" \
  --lr "$FM3_LR" \
  --warmup_steps "$FM3_WARMUP" \
  --seed "$FM3_SEED" \
  --distort_mode "$FM3_DISTORT_MODE" \
  --distort_t_max "$FM3_DISTORT_T_MAX" \
  --distort_t_min "$FM3_DISTORT_T_MIN" \
  --distort_channels "$FM3_DISTORT_CHANNELS" \
  --sc_train_prob "$FM3_SC_TRAIN_PROB" \
  --log_jsonl_path "$ARM_DIR/train_logs.jsonl" \
  --wandb_project "$FM3_WANDB_PROJECT" \
  --wandb_name "flowmol3_$FM3_ARM" \
  --wandb_mode "$FM3_WANDB_MODE" \
  "${FLAGS[@]}"

echo "[$FM3_ARM] done. checkpoints under $ARM_DIR/flowmol3_${FM3_ARM}_*/latest"
