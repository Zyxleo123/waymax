#!/bin/bash
# Draw the per-stratum headline figure (failure recovery / success non-regression
# / intermittent) for the deploy-time repair line. Only calls sbatch; the work is
# a pure re-read of select_summary.json on the CPU partition.
#
# Writes one timestamped env file and passes its path as a POSITIONAL argument to
# the batch script (never sbatch --export). Merges any number of select runs into
# one figure: the 3-seed b_l24 run gives the shipping arm, the ramp/blend sweep
# gives the r10/r20/r20q/r30q comparison.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_viz_recovery.sh
#   DFT_RECOV_ARMS="base,ft,b_l24,c_r20" bash rl/diffusion_ft/slurm/submit_viz_recovery.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"

SEL=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select
# 3-seed ladder run (b_l24, ft, bf_w10/w20) + ramp/blend sweep (c_r10..c_r30q).
IN_JSON="${DFT_RECOV_IN_JSON:-$SEL/20260918_110955_3370740/select_summary.json $SEL/20260918_110957_3371380/select_summary.json}"
ARMS="${DFT_RECOV_ARMS:-ft,bf_w20,bf_w10,b_l24,c_r30q,c_r20q,c_r20,c_r10}"
OUT_DIR="${DFT_RECOV_OUT_DIR:-$ROOT/logs/dft/figs}"; mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_PNG="${DFT_RECOV_OUT_PNG:-$OUT_DIR/recovery_${STAMP}.png}"

# Scrub SLURM_* so a submission from inside an allocation does not trip user-env retrieval.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

ENVF="$OUT_DIR/recovery_${STAMP}.env"
{
  echo "DFT_RECOV_IN_JSON='$IN_JSON'"
  echo "DFT_RECOV_OUT_PNG='$OUT_PNG'"
  echo "DFT_RECOV_ARMS='$ARMS'"
} > "$ENVF"

# Verify the env file round-trips under a clean environment before sbatch.
env -i bash -c "source '$ENVF' && [[ -n \"\$DFT_RECOV_IN_JSON\" && -n \"\$DFT_RECOV_OUT_PNG\" ]]" \
  || { echo "env file did not round-trip: $ENVF"; exit 1; }

echo "env  -> $ENVF"
echo "fig  -> $OUT_PNG"
sbatch rl/diffusion_ft/slurm/viz_recovery.sbatch "$ENVF"
