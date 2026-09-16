#!/bin/bash
# Experiment 1: training-free proposal selection. Only calls sbatch.
#
# QUESTION. Does keeping the base as the default and letting a second candidate
# earn each intervention beat the base at the corpus level -- and if not, is the
# selector or the proposal the bottleneck? See rl/diffusion_ft/proposal_select.py.
#
# CHAIN
#   0. build (CPU, only if the three-stratum splits file is missing): adds
#      intermittent_heldout (base solves 1-3 of 4) to a NEW splits file.
#   1. score (GPU array, %MAX_CONCURRENT): checkpoints x shards. The FIRST
#      checkpoint's tasks also run the checkpoint-independent arms (base, second
#      base sample, independent base sample); every task runs the fine-tune arms.
#      One sampler seed (4242, as the hardness gate) and FIXED chunking across all
#      tasks, so every arm of every checkpoint is paired draw for draw per scene.
#   2. analyze (CPU, afterok): select_summary.{json,png}.
#
# ARMS (defaults)
#   ft-dependent  : ft | bf_cons=ft:cons:20 | bf_cons10=ft:cons:10 | bf_margin=ft:margin:20
#   base-only (1x): base | bb_cons=base2:cons:20 (matched compute) | bfull_cons=basefull:cons:20
#
# THE REPAIR BANK (DFT_SELECT_BANK, alt 'bank'). Alternatives are time warps and
# lateral shifts of the BASE's own proposal -- the two families that account for
# 82 of the 85 repairs `counterfactual --mode repair` verified, offered at
# decision time instead of distilled into weights (which did not transfer). They
# cost NO denoiser evaluations, so a bank arm is cheaper than bb_cons and needs
# no checkpoint at all; it is placed in DFT_BASE_ARMS for that reason.
#
# COST (a6000, 1450 scenes x 4 repeats; one plain-eval arm ~37 min): first
# checkpoint ~7.5 arm-units, the others ~4.3; at 2 shards that is ~2.3 h / ~1.3 h
# per task, 6 tasks at 3-wide ~4-5 h wall.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_select_eval.sh
#   DFT_SELECT_CKPTS="/path/a/iter_000110 /path/b/iter_000110" bash ...   # other policies
#   # theta sweep for a learned selector (theta is the 4th arm field):
#   DFT_SELECTOR_JSON=.../selector.json DFT_BASE_ARMS=base \
#     DFT_FT_ARMS="bf_cons10=ft:cons:10,l02=ft:learned:20:0.2,l05=ft:learned:20:0.5" \
#     DFT_SELECT_CKPTS=".../base_s0/iter_000110" bash ...
#   DFT_SELECT_SMOKE=1 bash ...    # 4 scenes/split, 1 repeat, 1 checkpoint (~20 min)
#   # attribution: lateral-only vs timing-only vs both, PAIRED in one job. Named
#   # banks (DFT_SELECT_BANKS, 'name=spec;name=spec') are referenced as 'bank@name'.
#   DFT_SELECT_CKPTS=".../base_s0/iter_000110" DFT_FT_ARMS="bf_cons10=ft:cons:10" \
#     DFT_BASE_ARMS="base,bank_all=bank:cons:20,bank_lat=bank@lat:cons:20,bank_tw=bank@tw:cons:20" \
#     DFT_SELECT_BANKS="lat=lat0.25,lat-0.25,lat0.5,lat-0.5;tw=tw0.85,tw1.15,tw0.7,tw1.3" \
#     bash rl/diffusion_ft/slurm/submit_select_eval.sh
#   # repair bank only, no fine-tune needed at all:
#   DFT_FT_ARMS=ft DFT_BASE_ARMS="base,bank_cons10=bank:cons:10" \
#     DFT_SELECT_BANK="tw0.85,tw0.7,lat0.25,lat-0.25" bash ...
#   # re-analyze only:
#   sbatch rl/diffusion_ft/slurm/select_analysis.sbatch <select_dir>/analyze.env
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SEL_DIR="${DFT_SELECT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/${STAMP}}"
VDIR=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777
SPLITS3="${DFT_SPLITS3_JSON:-$VDIR/eval_splits_3strata.json}"
RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110 $RUN/base_s1/iter_000110 $RUN/base_s2/iter_000110}"
FT_ARMS="${DFT_FT_ARMS:-ft,bf_cons=ft:cons:20,bf_cons10=ft:cons:10,bf_margin=ft:margin:20}"
BASE_ARMS="${DFT_BASE_ARMS:-base,bb_cons=base2:cons:20,bfull_cons=basefull:cons:20,bank_cons10=bank:cons:10,bank_cons=bank:cons:20}"
SEED="${DFT_SELECT_SEED:-4242}"
REPEATS="${DFT_SELECT_REPEATS:-4}"
SHARDS="${DFT_NUM_SHARDS:-2}"
CHUNK="${DFT_EVAL_CHUNK:-32}"
CAP="${DFT_SELECT_CAP:--1}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"
EXCLUDE="${DFT_EXCLUDE-gpu28}"        # a5000: two candidates per replan; keep to a6000
ENV_DIR="$SEL_DIR/env"; mkdir -p "$ENV_DIR" "$SEL_DIR/tasks" "$ROOT/logs/dft"

[[ "$SEED" != "777" && "$SEED" != "12345" ]] || {
  echo "FATAL: seed $SEED reuses selection draws (777 = splits, 12345 = training eval)" >&2; exit 1; }
for c in $CKPTS; do
  [[ -s "$c/tparams.npz" ]] || { echo "FATAL: missing $c/tparams.npz" >&2; exit 1; }
done
if [[ -n "${DFT_SELECT_SMOKE:-}" ]]; then
  REPEATS=1; SHARDS=1; CKPTS="${CKPTS%% *}"; CAP="${DFT_SELECT_CAP:-4}"; CHUNK=4
  echo "SMOKE: 1 checkpoint, 1 repeat, 1 shard, $CAP scenes per split"
fi

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
SB_FLAGS=()
[[ -n "$EXCLUDE" ]] && SB_FLAGS+=(--exclude="$EXCLUDE")

DEP=()
if [[ ! -s "$SPLITS3" ]]; then
  BID=$(sb --parsable rl/diffusion_ft/slurm/build_intermittent.sbatch "DFT_SPLITS3_JSON=$SPLITS3")
  DEP=(--dependency="afterok:${BID}")
  echo "build   -> job $BID (CPU) -> $SPLITS3"
fi

i=0; first=1
for c in $CKPTS; do
  tag="$(basename "$(dirname "$c")")_$(basename "$c")"
  arms="$FT_ARMS"; [[ "$first" == 1 ]] && arms="$BASE_ARMS,$FT_ARMS"
  for ((s = 0; s < SHARDS; s++)); do
    d="$SEL_DIR/tasks/${tag}_shard${s}"; mkdir -p "$d"
    {
      echo "DFT_OUT_DIR=$d"
      echo "DFT_SPLITS_JSON=$SPLITS3"
      echo "DFT_SELECT_SPLITS=${DFT_SELECT_SPLITS:-failure_heldout,success_heldout,intermittent_heldout}"
      echo "DFT_FT_CKPT=$c"
      echo "DFT_ARMS=$arms"
      echo "DFT_SELECT_SEED=$SEED"
      echo "DFT_SELECT_REPEATS=$REPEATS"
      echo "DFT_SHARD_INDEX=$s"
      echo "DFT_NUM_SHARDS=$SHARDS"
      echo "DFT_EVAL_CHUNK=$CHUNK"
      echo "DFT_SELECT_CAP=$CAP"
      echo "DFT_SELECTOR_JSON=${DFT_SELECTOR_JSON:-}"
      echo "DFT_LOOK_REPLANS=${DFT_LOOK_REPLANS:-2}"
      echo "DFT_SELECT_BANK=${DFT_SELECT_BANK:-tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5}"
      echo "DFT_SELECT_RAMP=${DFT_SELECT_RAMP:-10}"
      echo "DFT_SELECT_BANKS=${DFT_SELECT_BANKS:-}"
      echo "DFT_SELECT_TOL_M=${DFT_SELECT_TOL_M:-1.0}"
      echo "DFT_SELECT_MARGIN_M=${DFT_SELECT_MARGIN_M:-0.3}"
      echo "DFT_SELECT_DELTA_M=${DFT_SELECT_DELTA_M:-0.2}"
      # rollout shape = the stopping sweep's eval exactly
      echo "DFT_K_TRAINABLE=10"; echo "DFT_PREFIX_LEN=5"; echo "DFT_SIGMA_FLOOR=1e-4"
      echo "DFT_REPLAN=10"; echo "DFT_MAX_REPLANS=8"
    } > "$ENV_DIR/$i.env"
    printf '%2d  %-28s shard %d/%d  %s\n' "$i" "$tag" "$s" "$SHARDS" "$arms"
    i=$((i + 1))
  done
  first=0
done

AID=$(sb --parsable ${DEP[@]+"${DEP[@]}"} ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} \
  --array="0-$((i - 1))%${MAX_CONCURRENT}" --job-name=dft_select \
  rl/diffusion_ft/slurm/select_eval.sbatch "$ENV_DIR")
AENV="$SEL_DIR/analyze.env"
{ echo "DFT_SELECT_DIR=$SEL_DIR"; echo "DFT_SPLITS_JSON=$SPLITS3"
  echo "DFT_SELECT_REPEATS=$REPEATS"; } > "$AENV"
GID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/select_analysis.sbatch "$AENV")
echo
echo "score   -> array $AID ($i tasks, %$MAX_CONCURRENT, GPU, exclude=${EXCLUDE:-none})"
echo "analyze -> job $GID (CPU, afterok:$AID)"
echo "out     -> $SEL_DIR/select_summary.{json,png}"
