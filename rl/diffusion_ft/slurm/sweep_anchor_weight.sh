#!/bin/bash
# Sweep the safety:anchor weighting before committing to a training run.
# Only calls sbatch -- one independent gate job per weight, submitted as siblings
# so they queue concurrently.
#
# WHY THIS EXISTS. The two terms are on incommensurable scales: measured at
# anchor_weight=1 the safety loss is 39.87 and the anchor 0.2653, so the anchor is
# 0.66% of the objective and 0.05% of the gradient norm -- inert. Turned up too
# far it goes the other way and the safety term stops mattering, which is exactly
# how the DPPO expert anchor failed at 0.1 (~1000:1 the other direction). The
# right value is empirical and cheap to find: each arm is minutes, and gate 4b
# reports whether the anchor actually moves the policy toward the demonstration.
#
# Read the results as: the LOWEST weight whose gate 4b passes (the anchor is
# materially constraining) while gate 3 still shows the safety loss descending.
# A weight where 4b passes but safety no longer descends has over-corrected.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/sweep_anchor_weight.sh
#   DFT_ANCHOR_WEIGHTS="1 30 100 300 1000 3000" bash rl/.../sweep_anchor_weight.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${DFT_SWEEP_OUT:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/anchor_sweep/${STAMP}}"
WEIGHTS="${DFT_ANCHOR_WEIGHTS:-1 30 100 300 1000 3000}"

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

echo "sweep -> $BASE_OUT"
for W in $WEIGHTS; do
  ENV_FILE="$ENV_DIR/anchorsweep_${STAMP}_w${W}.env"
  {
    echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord}"
    echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7}"
    echo "DFT_GRAD_STEPS=${DFT_GRAD_STEPS:-30}"
    echo "DFT_SAFETY_LR=${DFT_SAFETY_LR:-1e-5}"
    echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
    echo "DFT_ANCHOR_WEIGHT=$W"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-0.03}"
    echo "DFT_GATE_OUT=$BASE_OUT/w${W}"
  } > "$ENV_FILE"
  JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable "${SB_FLAGS[@]}" --job-name="dft_anchw${W}" \
    rl/diffusion_ft/slurm/gate_safety_grad.sbatch "$ENV_FILE")
  echo "  anchor_weight=$W -> job $JID  ($BASE_OUT/w${W})"
done
echo
echo "compare with:"
echo "  for d in $BASE_OUT/w*/gate_safety_grad.json; do"
echo "    python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); r=d['results']['anchored'];"
echo "    print(f\\\"{d['config']['anchor_weight']:>6} anchor {r['anchor_grad_share']*100:6.2f}% of grad\\\","
echo "          f\\\"safety {r['safety_first']:.2f}->{r['safety_last']:.2f}\\\","
echo "          f\\\"drift {r['drift']:.3f}\\\", [g['passed'] for g in d['gates']])\" \$d; done"
