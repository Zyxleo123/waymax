#!/bin/bash
#SBATCH --job-name=vmh_refit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#
# Stage B (CPU): build labels.npz from the Stage-A SAC failures + the EXISTING
# expert log-replay labels (policy-independent, reused), then refit the descriptor
# risk model over all 1000 shards -> risk.npz. Reads saved artefacts only, so it
# must not queue behind GPU work for a card it never uses.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/vmax_hardness"
# shellcheck source=/dev/null
source "$ROOT/rl/vmax_hardness/slurm/_preamble.sh" "$@"

# Gate: Stage A must have produced per-scene JSONs; distinguish "gate failed" from
# "upstream never ran" and exit 0 so dependents are not stranded.
if ! compgen -G "$VMH_RESCORE_DIR/*.scenario_*.json" > /dev/null; then
  echo "GATE: no re-score JSONs under $VMH_RESCORE_DIR -- Stage A produced nothing." ; exit 0
fi
if ! compgen -G "$VMH_DESC_DIR/*.desc.npz" > /dev/null; then
  echo "GATE: no descriptors under $VMH_DESC_DIR -- run the diffusion descriptors_all job first." ; exit 0
fi

mkdir -p "$VMH_RISK_OUT"
echo "host=${SLURMD_NODENAME:-unknown} policy=$VMH_RESCORE_DIR expert=$VMH_EXPERT_DIR"
echo "labelled scenes: $(ls "$VMH_RESCORE_DIR"/*.scenario_*.json | wc -l)"

# 1) labels.npz: pol_success from the SAC re-score, exp_* from the reused expert dir.
"$PYTHON" -m train_diffusion.hardness.labels \
  --policy_dir "$VMH_RESCORE_DIR" \
  --expert_dir "$VMH_EXPERT_DIR" \
  --out_dir "$VMH_RISK_OUT"

# 2) risk.npz: fit on expert-clean labelled scenes, score every shard's descriptors.
"$PYTHON" -m train_diffusion.hardness.risk_model \
  --labels "$VMH_RISK_OUT/labels.npz" \
  --fit_desc_dir "$VMH_DESC_DIR" \
  --score_desc_dir "$VMH_DESC_DIR" \
  --fit_slice "$VMH_FIT_SLICE" \
  --out_dir "$VMH_RISK_OUT"

echo "refit done: $VMH_RISK_OUT/risk.npz"
