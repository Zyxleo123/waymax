#!/bin/bash
# The alpha sweep: five independent training arms, submitted as siblings so they
# queue concurrently rather than chaining.
#
#   alpha=0  keep dirty     the exact control -- reproduces the current run
#   alpha=0  exclude dirty  "clean the data, change nothing else"
#   alpha=0.3 / 0.5 / 1.0   tempered reweighting, dirty excluded
#
# The second arm matters on its own: 22.6% of policy failures sit on scenes the
# logged expert also fails. Without it, a win from alpha>0 cannot be separated
# from simply having dropped that contamination.
#
# Run submit_phase2.sh first and read runs/hardness/risk.json.
#
# Usage (from the repo root, on rhea):
#     HRD_CKPT=/path/to/pretrained/latest bash train_diffusion/hardness/slurm/submit_alpha_sweep.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

HRD_OUT="${HRD_OUT:-$ROOT/runs/hardness}"
if [[ ! -f "$HRD_OUT/risk.npz" ]]; then
  echo "ERROR: $HRD_OUT/risk.npz not found. Run submit_phase2.sh first." >&2
  exit 1
fi
: "${HRD_CKPT:?set HRD_CKPT to the pretrained diffusion checkpoint}"

# Pre-flight the checkpoint HERE, not on the compute node. Five arms are
# submitted as siblings, so an unusable path costs five queued jobs that each
# fail seconds in -- and a copy-pasted placeholder looks exactly like a real
# argument until then.
if [[ ! -d "$HRD_CKPT" ]]; then
  echo "ERROR: HRD_CKPT is not a directory: $HRD_CKPT" >&2
  echo "       Point it at a checkpoint dir, e.g. .../pretrain_diffusion_without_subgoal_<stamp>/latest" >&2
  exit 1
fi
if [[ ! -f "$HRD_CKPT/metadata.json" ]]; then
  echo "ERROR: no metadata.json under $HRD_CKPT" >&2
  echo "       train_diffusion.utils.infer reads the model contract from it; this is not a checkpoint dir." >&2
  exit 1
fi
echo "checkpoint OK: $HRD_CKPT"

ENV_DIR="$ROOT/logs/hardness/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

emit() { printf '%s=%q\n' "$1" "$2"; }
SUBMIT=(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch)

submit_arm() {
  local alpha="$1" dirty="$2" label="$3"
  local env_file="$ENV_DIR/sweep_${STAMP}_${label}.env"
  {
    emit HRD_OUT "$HRD_OUT"
    emit HRD_ALPHA "$alpha"
    emit HRD_EXCLUDE_DIRTY "$dirty"
    emit HRD_WEIGHT_CLIP "${HRD_WEIGHT_CLIP:-10.0}"
    emit HRD_CKPT "$HRD_CKPT"
    emit HRD_CACHE_DIR "${HRD_CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/}"
    emit HRD_SAVE_DIR "${HRD_SAVE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/hardness_sweep}/$label"
    emit HRD_EPOCHS "${HRD_EPOCHS:-20}"
    emit HRD_STEPS "${HRD_STEPS:-1000}"
    # Same seed across arms: the sweep must vary alpha and nothing else.
    emit HRD_SEED "${HRD_SEED:-0}"
    emit HRD_SHUFFLE_SEED "${HRD_SHUFFLE_SEED:-0}"
    # Closed-loop safe-goal-reach eval. sbatch has matching defaults, but the
    # submitter scrubs the environment (env -i), so anything you want to override
    # from the submit command must be forwarded through the env file explicitly.
    emit HRD_EVAL_EVERY "${HRD_EVAL_EVERY:-5}"
    emit HRD_EVAL_TFRECORD_DIR "${HRD_EVAL_TFRECORD_DIR:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures}"
    emit HRD_EVAL_NUM "${HRD_EVAL_NUM:-218}"
    emit HRD_EVAL2_TFRECORD_DIR "${HRD_EVAL2_TFRECORD_DIR:-/zfsauton/scratch/eshau/womd/tf_example/training}"
    emit HRD_EVAL2_NUM "${HRD_EVAL2_NUM:-200}"
    emit HRD_EVAL_LANE_GRAPH_DIR "${HRD_EVAL_LANE_GRAPH_DIR:-/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"
  } > "$env_file"
  local job
  job="$("${SUBMIT[@]}" --parsable --job-name="hrd_$label" \
    train_diffusion/hardness/slurm/train_alpha.sbatch "$env_file")"
  printf '  %-22s alpha=%-4s exclude_dirty=%s  job=%s\n' "$label" "$alpha" "$dirty" "$job"
}

echo "Submitting the alpha sweep (ckpt: $HRD_CKPT):"
submit_arm 0.0 0 control
submit_arm 0.0 1 cleanonly
submit_arm 0.3 1 alpha03
submit_arm 0.5 1 alpha05
submit_arm 1.0 1 alpha10

echo
echo "Watch: squeue -u \$USER -o '%.10i %.9P %.14j %.2t %.10M %R'"
