#!/bin/bash
#SBATCH --job-name=dft_lrprobe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Is the DPPO collision rate the policy's fault, or the simulator's?
#
# Replays the human demonstration under a controlled deviation from the log and
# reports collision vs deviation size. Every non-ego agent is on rails
# (controlled_object=SDC) and the metric is symmetric `overlap` with no fault
# attribution, so an SDC that leaves the recorded path can be driven into and
# charged for it. See rl/diffusion_ft/tests/probe_logreplay_collision.py.
#
# CPU-only: expert replay does no model forward, so this must not sit on a GPU
# partition queueing behind the training runs it exists to interpret.
#
# No `--qos` line on purpose: the `cpu` partition applies qos_cpu as its default,
# and naming a QOS this account has no association for fails submission with a
# bare "Invalid qos specification".
#
# 4h and 48G (was 2h/32G): DFT_SIM_AGENTS now defaults to running every
# perturbation level under BOTH agent models, so this is ~24 full-episode replays
# rather than ~12, and the IDM arm additionally builds an
# (B, N, N, lookahead, 5) pairwise box tensor at every world step to find each
# agent's leader. Set DFT_SIM_AGENTS=none to get the old single-arm cost back.
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

DFT_TFRECORD="${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
DFT_INDICES="${DFT_INDICES:-0-31}"
DFT_LATERAL="${DFT_LATERAL:-0,0.25,0.5,1.0,1.5,2.0,3.0}"
DFT_DELAY="${DFT_DELAY:-0,2,5,10,20,40}"
# Which non-ego agent models to run: `none` = log replay (every result before
# 2026-09-19), `idm_brake` = braking-only IDM, `idm` = waymax stock, which races
# every unobstructed car toward 108 km/h (see rl/diffusion_ft/sim_agents.py and
# job 58076). Two arms by default, so each case is paired.
DFT_SIM_AGENTS="${DFT_SIM_AGENTS:-none,idm_brake}"
DFT_PROBE_OUT="${DFT_PROBE_OUT:-runs/dft_logreplay_probe/collision_vs_deviation.json}"

echo "=== log-replay collision probe: $(date) on $(hostname) ==="
echo "tfrecord=$DFT_TFRECORD indices=$DFT_INDICES"
echo "lateral=[$DFT_LATERAL] delay=[$DFT_DELAY] sim_agents=[$DFT_SIM_AGENTS]"
echo

"$PYTHON" -m rl.diffusion_ft.tests.probe_logreplay_collision \
  --tfrecord "$DFT_TFRECORD" \
  --indices "$DFT_INDICES" \
  --lateral "$DFT_LATERAL" \
  --delay "$DFT_DELAY" \
  --sim_agents "$DFT_SIM_AGENTS" \
  --out "$DFT_PROBE_OUT"

echo "=== exit $? ==="
