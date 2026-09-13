#!/bin/bash
# Would a HARDNESS GATE make the safety fine-tune a corpus-level win? Only calls sbatch.
#
# WHY. Ungated, the fine-tune is a small net loss (3-seed stopping sweep: best
# mean marginal -0.0093 at iter 110) and no training knob moves it beyond the ~0.03
# seed spread. The budget is structural: failures are 1.5% of the corpus, so a
# +0.28 failure gain pays for only ~0.004 of success. A gate changes the budget
# rather than the policy: run the fine-tune only where `runs/hardness/risk.npz`
# (AUC 0.847 on these verified labels) says the scene is hard, the base elsewhere.
# Back-of-envelope with UNIFORM per-scene effects: about +0.001 at a 2-10% gate,
# against a hard oracle ceiling of +0.0041. Whether the effects really are uniform
# -- whether the fine-tune's damage to successes is concentrated in the same
# high-risk scenes it helps -- is exactly what this measures.
#
# CHAIN
#   1. score (GPU array, %3): `train_safety --verify_only` per scene on
#      failure_heldout + success_heldout, for the FROZEN BASE and for the three
#      stopping-sweep `base` seeds at iter DFT_GATE_ITER (default 110, the best
#      3-seed mean). All at ONE fresh verify seed (4242): not 777, which selected
#      the splits (scoring on the selection draws is biased), not 12345, which
#      chose iter 110. Same seed for every policy -> each scene's (ft, base) pair
#      shares its sampler draws, so the difference is paired.
#   2. analyze (CPU, afterok): tools/gate_analysis.py -> gate_curve.json, then
#      plot_gate_curve.py redraws gate_curve.png FROM that JSON.
#
# COST. 1173 scenes x 4 repeats per policy ~= 30 min + startup; 4 policies at 3-wide
# ~= 1-1.5 h. The analysis is minutes.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_gate_analysis.sh
#   DFT_GATE_ITER=130 bash rl/diffusion_ft/slurm/submit_gate_analysis.sh
#   # re-analyze only (rows already written):
#   sbatch rl/diffusion_ft/slurm/gate_analysis.sbatch <gate_dir>/analyze.env
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
GATE_DIR="${DFT_GATE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/gate/${STAMP}}"
RUN="${DFT_GATE_RUN:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947}"
ITER="${DFT_GATE_ITER:-110}"
SEEDS="${DFT_GATE_SEEDS:-base_s0 base_s1 base_s2}"
SPLITS_JSON="${DFT_SPLITS_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_verified.json}"
VSEED="${DFT_VERIFY_SEED:-4242}"
REPEATS="${DFT_VERIFY_REPEATS:-4}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"
ENV_DIR="$GATE_DIR/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

[[ -s "$SPLITS_JSON" ]] || { echo "FATAL: no splits $SPLITS_JSON" >&2; exit 1; }
[[ -s "$ROOT/runs/hardness/risk.npz" ]] || { echo "FATAL: no runs/hardness/risk.npz" >&2; exit 1; }
[[ "$VSEED" != "777" && "$VSEED" != "12345" ]] || {
  echo "FATAL: verify seed $VSEED reuses selection draws (777 = splits, 12345 = iter choice)" >&2; exit 1; }
snap="$(printf 'iter_%06d' "$ITER")"
FT_TAGS=""
for s in $SEEDS; do
  [[ -s "$RUN/$s/$snap/tparams.npz" ]] || { echo "FATAL: missing $RUN/$s/$snap/tparams.npz" >&2; exit 1; }
  FT_TAGS="$FT_TAGS ft_${s}_${ITER}"
done

i=0
write_env() {  # tag ckpt
  mkdir -p "$GATE_DIR/$1"
  {
    echo "DFT_OUT_DIR=$GATE_DIR/$1"
    echo "DFT_VERIFY_OUT=$GATE_DIR/$1/rows.jsonl"
    echo "DFT_SPLITS_JSON=$SPLITS_JSON"
    echo "DFT_VERIFY_SPLITS=failure_heldout,success_heldout"
    echo "DFT_VERIFY_CAP=-1,-1"
    echo "DFT_VERIFY_REPEATS=$REPEATS"
    echo "DFT_VERIFY_SEED=$VSEED"
    echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-32}"
    # rollout shape must match the stopping sweep's eval exactly
    echo "DFT_K_TRAINABLE=10"
    echo "DFT_PREFIX_LEN=5"
    echo "DFT_SIGMA_FLOOR=1e-4"
    echo "DFT_REPLAN=10"
    echo "DFT_MAX_REPLANS=8"
    echo "DFT_LOAD_CKPT=$2"
  } > "$ENV_DIR/$i.env"
  printf '%2d  %-18s %s\n' "$i" "$1" "${2:-<frozen base>}"
  i=$((i + 1))
}
write_env base ""
for s in $SEEDS; do write_env "ft_${s}_${ITER}" "$RUN/$s/$snap"; done

AENV="$GATE_DIR/analyze.env"
{ echo "DFT_GATE_DIR=$GATE_DIR"; echo "DFT_SPLITS_JSON=$SPLITS_JSON"
  echo "DFT_RISK=$ROOT/runs/hardness/risk.npz"; echo "DFT_FT_TAGS=\"${FT_TAGS# }\""; } > "$AENV"

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
SB_FLAGS=()
[[ -n "${DFT_EXCLUDE:-}" ]] && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
AID=$(sb --parsable ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} --array="0-$((i - 1))%${MAX_CONCURRENT}" \
  --job-name=dft_gatescore rl/diffusion_ft/slurm/verify_splits.sbatch "$ENV_DIR")
GID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/gate_analysis.sbatch "$AENV")
echo
echo "score   -> array $AID ($i policies, %$MAX_CONCURRENT, GPU)"
echo "analyze -> job $GID (CPU, afterok:$AID)"
echo "out     -> $GATE_DIR/gate_curve.{json,png}"
