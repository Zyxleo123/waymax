#!/bin/bash
# Submitter for the differentiable safety-loss validation.
# Only calls sbatch. Chains: CPU-cheap unit gates (GPU node, seconds) ->
# GPU probe on the failure set -> CPU figures, each afterok the last.
#
# The unit gates run first on purpose: they check that our road-edge distance is
# bit-identical to waymax's. If that fails, the probe's headline gate would be
# comparing our arithmetic against itself and would pass on a broken loss, so
# there is no point spending the GPU rollout.
#
# Writes a timestamped env file and passes its path as a positional arg to each
# batch script (no `sbatch --export`, which requeue-holds on this cluster).
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_probe_safety_loss.sh
#
# Useful knobs:
#   DFT_EDGE_MARGIN / DFT_AGENT_MARGIN   where the barrier starts to bite (m)
#   DFT_EDGE_TAU / DFT_AGENT_TAU         width of the knee (m)
#   DFT_INDICES                          which failure scenes (default 0..31)
#   DFT_PROBE_SEEDS                      rollout seeds per arm (default 2)
#   DFT_PARTITION / DFT_QOS / DFT_NODELIST   placement (see below)
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/safeprobe_${STAMP}.env"

DFT_PROBE_OUT="${DFT_PROBE_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_probe/${STAMP}}"

{
  echo "DFT_CKPT=${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
  echo "DFT_PROBE_SEEDS=${DFT_PROBE_SEEDS:-2}"
  echo "DFT_REPLAN=${DFT_REPLAN:-10}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.03}"
  # Margins are in metres of real clearance, and are set from what the HUMAN
  # keeps, not from what sounds safe: over 16,880 expert-clean steps of this
  # failure set the human's own road-edge clearance is p01 +0.31 / p05 +0.42 /
  # p50 +1.72 m. The old 0.5 m default sat on the human p10 and charged 35.9%
  # of expert steps, which fights whatever pulls the policy toward the expert.
  # 0.3 m is that measured p01. Re-run submit_calibrate_expert.sh if the scene
  # set changes -- these numbers are corpus-specific.
  echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
  echo "DFT_EDGE_TAU=${DFT_EDGE_TAU:-0.15}"
  # Expert p05 of the EXACT box gap (job 39985): the human's own clearance runs
  # p1 +0.17 / p5 +0.91 / p50 +4.96 m. The old 0.3 was p05 under the disc cover,
  # which for a pair of cars is ~1.3 m of real gap -- so this is a measured
  # envelope replacing an inherited approximation, not a loosening.
  echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
  echo "DFT_AGENT_TAU=${DFT_AGENT_TAU:-0.25}"
  echo "DFT_EDGE_WEIGHT=${DFT_EDGE_WEIGHT:-1.0}"
  echo "DFT_AGENT_WEIGHT=${DFT_AGENT_WEIGHT:-1.0}"
  echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
  echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
  echo "DFT_TIME_CHUNK=${DFT_TIME_CHUNK:-8}"
  echo "DFT_GRAD_STEP=${DFT_GRAD_STEP:-0.05}"
  echo "DFT_VIZ_SCENES=${DFT_VIZ_SCENES:-12}"
  echo "DFT_PROBE_OUT=$DFT_PROBE_OUT"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# Every partition on this cluster mandates its own QOS, and sbatch rejects the
# job with a bare "Invalid qos specification" if the two disagree:
#   general -> qos_general (the default; no flag needed)
#   legacy  -> qos_legacy      debug -> qos_debug      cpu -> qos_cpu
#   project -> qos_radio_freq | qos_project_preempt   (h200 gpu2; no association
#              for this account -- do not use)        preempt -> qos_preempt
SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
if [[ -n "${DFT_PARTITION:-}" && -z "${DFT_QOS:-}" && "$DFT_PARTITION" != "general" ]]; then
  echo "WARNING: DFT_PARTITION=$DFT_PARTITION set without DFT_QOS; sbatch will" >&2
  echo "WARNING: likely fail with 'Invalid qos specification'." >&2
fi
echo "sbatch flags: ${SB_FLAGS[*]:-<none>}"

# Scrub SLURM_* so submitting from inside an allocation cannot leak this job's
# environment into the new one.
#
# The unit gates are pure CPU (the module pins JAX_PLATFORMS=cpu), so they go to
# the `cpu` partition on their own and deliberately do NOT take SB_FLAGS -- those
# place the GPU work, and forwarding a GPU nodelist here would just park a
# synthetic seconds-long test on an a6000.
UID_=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable \
  rl/diffusion_ft/slurm/test_safety_loss.sbatch "$ENV_FILE")
echo "unit-gate job: $UID_  (cpu partition)"

PID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency=afterok:"$UID_" "${SB_FLAGS[@]}" \
  rl/diffusion_ft/slurm/probe_safety_loss.sbatch "$ENV_FILE")
echo "probe job:     $PID  (afterok:$UID_)"

VID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency=afterok:"$PID" \
  rl/diffusion_ft/slurm/viz_safety_loss.sbatch "$DFT_PROBE_OUT" "$ENV_FILE")
echo "viz job:       $VID  (afterok:$PID)"
echo
echo "artifacts -> $DFT_PROBE_OUT/safety_probe.{npz,jsonl}, safety_probe_summary.json"
echo "figures   -> $DFT_PROBE_OUT/safety_{calibration,transfer,timelines,gradient}.png"
echo "digest    -> $DFT_PROBE_OUT/safety_probe_summary.md"
