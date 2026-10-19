#!/bin/bash
# The exploration-noise sweep, pointed at WOMD, including sigma = 0.
#
# Does no work of its own and defines no new pipeline: it sets WOMD data + route
# dims and calls the tested `submit_sweep_sigma.sh`, which chains the GPU sweep
# (one diagnose_drive per sigma) to the CPU aggregate (table + figure).
#
# ---- why this is the blocker ---------------------------------------------
# Measured on WOMD at sigma_floor=0.03 (jobs 40469/40470), the base policy:
#
#   deterministic (deployment) : solve 0.979  collided 0.000  offroad 0.021
#   stochastic (what DPPO collects with) : solve 0.052  collided 0.750  offroad 0.750
#
# Every DPPO advantage is computed on the stochastic rollouts. So the gradient is
# fitted to a policy that solves 5% of the time in order to update one that solves
# 98% -- and it moves solve DOWN (0.979 -> 0.948 in a single update). Until that
# gap is closed, no reward-design result on this stack is readable, which is why
# this sweep runs before any further fine-tune.
#
# 0.03 is in NORMALIZED units and ego positions are normalized by ego_range =
# 100 m, so the floor is ~3 m of per-step positional jitter against ~0.87 m of
# per-step motion. `tests/diagnose_arc.py` (job 40319) already separated jitter
# from speed: the proposal arcs 27.79 m over 10 world steps while its mean
# deviation from the log is 2.87 m, and the executed deterministic rollout is
# smooth (path efficiency 1.000, 0.91x expert speed). Zero-mean jitter inflates
# arc length without moving the mean.
#
# ---- what this measures, and the known trap ------------------------------
# The largest sigma whose STOCHASTIC driving still resembles the deterministic
# policy's. sigma=0.0 is included because it is the pending unconfirmed test --
# the run that would have confirmed the arc collapses to ~8-9 m was blocked by
# the 09-01 slurmdbd outage.
#
# The trap: sigma cannot simply be lowered. The 08-11 sigma x KL sweep froze
# learning outright, because below ~0.03 the candidate-KL trust region stops
# opening and no step is ever applied. So a small sigma that drives beautifully
# and never takes a step is NOT a solution, and this sweep cannot tell you that
# it happened -- it measures driving, not learning. If the answer is "every
# drivable sigma is untrainable", the fix is structural (a group baseline over N
# rollouts per scene, which removes the value net and much of the noise
# sensitivity from the critical path), not another sweep.
#
# Reward flags do not affect the measurement's core (the stochastic/deterministic
# driving gap is reward-independent), but termination does, so the parity defaults
# inherited from submit_sweep_sigma.sh are left alone deliberately.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_sweep_sigma_womd.sh
#   DFT_SIGMA_LIST="0.0 0.005 0.01" bash rl/diffusion_ft/slurm/submit_sweep_sigma_womd.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"

# WOMD, and its route dims. 10x300 (ScenarioMax) would give zero valid paths.
export DFT_TFRECORD="${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"
export DFT_INDICES="${DFT_INDICES:-$(seq -s, 0 31)}"
export DFT_NUM_PATHS="${DFT_NUM_PATHS:-45}"
export DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-800}"

# 0.0 first: it is the control (no exploration noise at all) and the cheapest
# point to read. 0.03 is where the entry_clean runs collect, so it anchors the
# sweep to a measured point rather than an extrapolation.
export DFT_SIGMA_LIST="${DFT_SIGMA_LIST:-0.0 0.003 0.006 0.01 0.02 0.03 0.05 0.1}"
export DFT_DIAG_SEEDS="${DFT_DIAG_SEEDS:-4}"
export DFT_SWEEP_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/sigma_sweep_womd/$(date +%Y%m%d_%H%M%S)}"

# Do not pin gpu28 (the 24 GB a5000) the way submit_sweep_sigma.sh defaults to:
# these are 32-scene WOMD batches at 45x800 routes, and gpu28 is where an
# equivalently sized eval transfer died (job 37156).
export DFT_NODELIST="${DFT_NODELIST:-}"
export DFT_EXCLUDE="${DFT_EXCLUDE-}"

echo "sigma sweep on WOMD: [$DFT_SIGMA_LIST]"
echo "  data=$DFT_TFRECORD  routes=${DFT_NUM_PATHS}x${DFT_NUM_POINTS_PER_PATH}"
echo "  out=$DFT_SWEEP_OUT"
exec bash rl/diffusion_ft/slurm/submit_sweep_sigma.sh
