#!/bin/bash
#SBATCH --job-name=dft_reactunit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Mechanics gates for `sim_agents="idm"` and `candidate_safety(neighbor_state=)`.
# Run this BEFORE the collision probe: the probe's numbers are only meaningful
# once these say the flag moves the NPCs, leaves the ego alone, and addresses the
# right rows and timesteps.
#
# CPU: the env owns a policy but this never calls the denoiser -- every rollout
# is `expert_replay_episode`, which feeds logged poses as actions. The module
# pins JAX_PLATFORMS=cpu.
#
# A SLURM job rather than a login-node run because it loads the checkpoint and
# rolls two full episodes of a batch, well past the login node's ~60 s reaper.
#
# No `--qos` line on purpose: the `cpu` partition applies qos_cpu as its default,
# and naming a QOS this account has no association for fails submission with a
# bare "Invalid qos specification".
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

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
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export CUDA_VISIBLE_DEVICES=""

DFT_TFRECORD="${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
# 8 scenes, not 32: these are mechanics gates (does the flag move the right
# arrays?), not a measurement, and gate 3 does an O(T^2) nearest-point search per
# object. The rates are measured by probe_logreplay_collision.
DFT_INDICES="${DFT_INDICES:-0-7}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
# Which reactive model to gate: `idm_brake` (the fix) or `idm` (waymax stock,
# kept so job 58075's failure stays reproducible).
DFT_REACTIVE="${DFT_REACTIVE:-idm_brake}"

echo "=== reactive-NPC mechanics gates: $(date) on $(hostname) ==="
echo "tfrecord=$DFT_TFRECORD indices=$DFT_INDICES reactive=$DFT_REACTIVE"
echo

"$PYTHON" -m rl.diffusion_ft.tests.test_reactive_npc \
  --tfrecord "$DFT_TFRECORD" \
  --indices "$DFT_INDICES" \
  --ckpt "$DFT_CKPT" \
  --reactive "$DFT_REACTIVE"

echo "=== exit $? ==="
