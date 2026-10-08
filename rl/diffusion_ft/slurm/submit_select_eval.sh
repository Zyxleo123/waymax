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
#   # REACTIVE NPCs: use submit_reactive_select.sh, which runs the rails and the
#   # idm_brake arms as two sibling jobs. `overlap` is a DIFFERENT metric under
#   # DFT_SIM_AGENTS=idm_brake, so nothing there is comparable to a run above and
#   # `base` has to be inside the same job.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SEL_DIR="${DFT_SELECT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/${STAMP}}"
VDIR=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777
SPLITS3="${DFT_SPLITS3_JSON:-$VDIR/eval_splits_3strata.json}"
RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110 $RUN/base_s1/iter_000110 $RUN/base_s2/iter_000110}"
# `-`, not `:-`: an explicitly EMPTY DFT_FT_ARMS means "no fine-tune arms" (a
# base/bank-only job still needs one checkpoint dir, but never loads it).
FT_ARMS="${DFT_FT_ARMS-ft,bf_cons=ft:cons:20,bf_cons10=ft:cons:10,bf_margin=ft:margin:20}"
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
# With DFT_SELECT_DEP set the checkpoints may not exist YET (a training array
# upstream writes them); the batch job re-checks at start and exits 0 with the
# reason if one is still missing.
if [[ -z "${DFT_SELECT_DEP:-}" ]]; then
  for c in $CKPTS; do
    [[ -s "$c/tparams.npz" ]] || { echo "FATAL: missing $c/tparams.npz" >&2; exit 1; }
  done
fi
if [[ -n "${DFT_SELECT_SMOKE:-}" ]]; then
  REPEATS=1; SHARDS=1; CKPTS="${CKPTS%% *}"; CAP="${DFT_SELECT_CAP:-4}"; CHUNK=4
  echo "SMOKE: 1 checkpoint, 1 repeat, 1 shard, $CAP scenes per split"
fi

sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }

# Env files are SOURCED by the batch script, so a bare `KEY=value` line is a shell
# assignment: an unquoted `;` in the value ends it and the rest becomes a separate
# variable. That is not hypothetical -- `DFT_SELECT_BANKS="lat=...;tw=..."` was
# silently truncated to the `lat` half (job 55580/55642, 2026-09-16), which the
# runner's own arm/bank guard caught at startup. Single-quote every value, with
# embedded single quotes escaped, so any value round-trips verbatim.
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
SB_FLAGS=()
[[ -n "$EXCLUDE" ]] && SB_FLAGS+=(--exclude="$EXCLUDE")

DEPS=()
if [[ ! -s "$SPLITS3" ]]; then
  BID=$(sb --parsable rl/diffusion_ft/slurm/build_intermittent.sbatch "DFT_SPLITS3_JSON=$SPLITS3")
  DEPS+=("afterok:${BID}")
  echo "build   -> job $BID (CPU) -> $SPLITS3"
fi
# A caller's own upstream (e.g. a gate job: DFT_SELECT_DEP=afterok:<id>). Pair it
# with DFT_REQUIRE_GATE so the tasks read the gate's ANSWER, not just its exit.
[[ -n "${DFT_SELECT_DEP:-}" ]] && DEPS+=("$DFT_SELECT_DEP")
DEP=()
if ((${#DEPS[@]})); then DEP=(--dependency="$(IFS=,; echo "${DEPS[*]}")"); fi

i=0; first=1
for c in $CKPTS; do
  tag="$(basename "$(dirname "$c")")_$(basename "$c")"
  arms="$FT_ARMS"; [[ "$first" == 1 ]] && arms="$BASE_ARMS${FT_ARMS:+,$FT_ARMS}"
  if [[ -z "$arms" ]]; then
    echo "skip    $c: no arms for it (DFT_FT_ARMS is empty)"; first=0; continue
  fi
  for ((s = 0; s < SHARDS; s++)); do
    d="$SEL_DIR/tasks/${tag}_shard${s}"; mkdir -p "$d"
    {
      kv DFT_OUT_DIR "$d"
      kv DFT_SPLITS_JSON "$SPLITS3"
      kv DFT_SELECT_SPLITS "${DFT_SELECT_SPLITS:-failure_heldout,success_heldout,intermittent_heldout}"
      kv DFT_FT_CKPT "$c"
      kv DFT_ARMS "$arms"
      kv DFT_SELECT_SEED "$SEED"
      kv DFT_SELECT_REPEATS "$REPEATS"
      kv DFT_SHARD_INDEX "$s"
      kv DFT_NUM_SHARDS "$SHARDS"
      kv DFT_EVAL_CHUNK "$CHUNK"
      kv DFT_SELECT_CAP "$CAP"
      kv DFT_SELECTOR_JSON "${DFT_SELECTOR_JSON:-}"
      kv DFT_LOOK_REPLANS "${DFT_LOOK_REPLANS:-2}"
      kv DFT_SELECT_BANK "${DFT_SELECT_BANK:-tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5}"
      kv DFT_SELECT_RAMP "${DFT_SELECT_RAMP:-10}"
      kv DFT_SELECT_BANKS "${DFT_SELECT_BANKS:-}"
      kv DFT_SELECT_BANK_RAMPS "${DFT_SELECT_BANK_RAMPS:-}"
      kv DFT_SELECT_BANK_BLENDS "${DFT_SELECT_BANK_BLENDS:-}"
      kv DFT_SELECT_BANK_NEIGHBORS "${DFT_SELECT_BANK_NEIGHBORS:-}"
      kv DFT_SELECT_BANK_OPTS "${DFT_SELECT_BANK_OPTS:-}"
      kv DFT_SELECT_OPT_CFG "${DFT_SELECT_OPT_CFG:-}"
      kv DFT_SELECT_BANK_OPT_CFGS "${DFT_SELECT_BANK_OPT_CFGS:-}"
      kv DFT_SELECT_BANK_PREFER "${DFT_SELECT_BANK_PREFER:-}"
      kv DFT_SELECT_ES "${DFT_SELECT_ES:-}"
      kv DFT_DUMP_TRAJ "${DFT_DUMP_TRAJ:-0}"
      kv DFT_SELECT_TOL_M "${DFT_SELECT_TOL_M:-1.0}"
      kv DFT_SELECT_MARGIN_M "${DFT_SELECT_MARGIN_M:-0.3}"
      kv DFT_SELECT_DELTA_M "${DFT_SELECT_DELTA_M:-0.2}"
      # rollout shape = the stopping sweep's eval exactly
      kv DFT_K_TRAINABLE 10; kv DFT_PREFIX_LEN 5; kv DFT_SIGMA_FLOOR 1e-4
      kv DFT_REPLAN 10; kv DFT_MAX_REPLANS 8
      # `idm` makes the other agents reactive, which changes what `overlap`
      # means -- so it changes every number in the run, not just the arms'
      # relative order. Keep BASE_ARMS in the same job when setting it.
      kv DFT_SIM_AGENTS "${DFT_SIM_AGENTS:-none}"
      # Which velocity the policy is shown for its own executed steps. Like
      # sim_agents it changes every arm, so `base` belongs in the same job.
      kv DFT_EGO_VEL_SOURCE "${DFT_EGO_VEL_SOURCE:-commanded}"
      # The plan's world time grid. `median` (default since 2026-09-24, validated
      # by jobs 62629-62634) is the scene's own clock; `first` reproduces every
      # earlier result. Written explicitly, so a requeue can never switch methods.
      kv DFT_WORLD_DT_METHOD "${DFT_WORLD_DT_METHOD:-median}"
      kv DFT_REQUIRE_GATE "${DFT_REQUIRE_GATE:-}"
    } > "$ENV_DIR/$i.env"
    # Cheap invariant: source what we just wrote in a CLEAN shell and require
    # every key to come back byte-identical. Catches any value that needs quoting
    # before a GPU is spent on it, not after.
    while IFS= read -r k; do
      want="$(sed -n "s/^${k}='\(.*\)'$/\1/p" "$ENV_DIR/$i.env")"
      got="$(env -i bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2}"' _ "$ENV_DIR/$i.env" "$k")"
      [[ "$want" == "$got" ]] || {
        echo "FATAL: $ENV_DIR/$i.env does not round-trip through 'source':" >&2
        echo "  $k  wrote [$want]  read back [$got]" >&2; exit 1; }
    done < <(sed -n 's/^\(DFT_[A-Z_]*\)=.*/\1/p' "$ENV_DIR/$i.env")
    printf '%2d  %-28s shard %d/%d  %s\n' "$i" "$tag" "$s" "$SHARDS" "$arms"
    i=$((i + 1))
  done
  first=0
done

(( i > 0 )) || { echo "FATAL: no tasks to submit (no arms for any checkpoint)" >&2; exit 1; }
AID=$(sb --parsable ${DEP[@]+"${DEP[@]}"} ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} \
  --array="0-$((i - 1))%${MAX_CONCURRENT}" --job-name=dft_select \
  rl/diffusion_ft/slurm/select_eval.sbatch "$ENV_DIR")
AENV="$SEL_DIR/analyze.env"
{ echo "DFT_SELECT_DIR=$SEL_DIR"; echo "DFT_SPLITS_JSON=$SPLITS3"
  echo "DFT_SELECT_REPEATS=$REPEATS"; } > "$AENV"
GID=$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/select_analysis.sbatch "$AENV")
echo
# Record the array id so a caller can chain onto it without guessing from squeue.
echo "$AID" > "$SEL_DIR/jobid.txt"
echo "score   -> array $AID ($i tasks, %$MAX_CONCURRENT, GPU, exclude=${EXCLUDE:-none})"
echo "analyze -> job $GID (CPU, afterok:$AID)"
echo "out     -> $SEL_DIR/select_summary.{json,png}"
