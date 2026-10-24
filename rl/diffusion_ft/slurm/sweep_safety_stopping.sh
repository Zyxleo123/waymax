#!/bin/bash
# WHERE DO YOU STOP, and is the marginal actually >= 0 there? Only calls sbatch.
#
# THE QUESTION. Two geometry sweeps (ablation array 46733, combo array 47510)
# agree that the objective's knobs are inert: every one moves
# `marginal/delta_vs_base` by <= 0.01, inside the run-to-run spread. What moves
# it by 0.03 is WHEN YOU STOP. The identical config:
#
#     ablation `base` @100   fail 0.295  succ 0.990  margD -0.0027
#     ablation `base` @200   fail 0.335  succ 0.985  margD -0.0071
#     ablation `base` @299   fail 0.335  succ 0.954  margD -0.0376
#
# and ranking EVERY eval point of both sweeps, the top of the list is iteration
# 100 of whatever arm, not any particular arm:
#
#     abl   swin_25       @100  fail 0.347  succ 0.996  margD +0.0039  (positive)
#     combo combo_noAwin  @100  fail 0.324  succ 0.991  margD -0.0013
#     abl   margin_lo     @100  fail 0.301  succ 0.990  margD -0.0027
#     abl   tau_lo/k25/base @100 fail 0.295 succ 0.990  margD -0.0027
#
# So "the fine-tune is a corpus-level net loss" may be an artifact of having
# scored it at 200-300 iterations. At ~100 it looks indistinguishable from
# neutral WHILE taking failure_heldout from 0.017 to ~0.30-0.35. This run exists
# to find out whether that is real.
#
# WHY IT IS NOT ALREADY ANSWERED. Two reasons, both about resolution:
#   1. The eval grid was 50-100 iterations wide, and the curve is NOT monotonic
#      -- every combo arm dipped to success 0.87-0.92 at iter 50 and recovered by
#      100. A coarse grid aliases that, so the iter-100 "peak" could be where the
#      grid happened to land rather than a real optimum. Here the grid is 10.
#   2. Nothing has ever been replicated. A single +0.0039 with no measure of
#      run-to-run spread is not a result. Every config here runs at THREE seeds.
#
# THE DESIGN IS 2 CONFIGS x 3 SEEDS, NOT 6 CONFIGS. Ranking more configurations
# is exactly the mistake the last two sweeps made -- their arms differed by less
# than the noise. Spend the GPU on precision instead: `base` is the reference and
# `swin25` is the single nominal best point. If the three seeds of a config
# disagree by more than the two configs differ, the answer is "the knob is noise"
# and that is a real finding too.
#
# SEEDS vary DFT_SEED *and* DFT_DATA_SEED together, so replicates differ in both
# the rollout noise and the order scenes arrive in. DFT_EVAL_SEED is PINNED at
# 12345 for every arm: the eval must be the same measurement across arms or the
# curves are not comparable ([[feedback_pin_eval_splits_and_seed]]).
#
# THE COMPARISON IS ALREADY PAIRED. `base_reference` scores the FROZEN base on
# the same fixed splits with the same eval seed and caches it, so
# `marginal/delta_vs_base` is a same-scene, same-sampler-noise difference -- the
# unpaired binomial se (0.006 on 1000 success scenes) OVERSTATES the error on it.
# For an exact paired test on a chosen checkpoint afterwards, `--verify_only`
# writes per-scene rows for the fine-tuned and base arms; McNemar on the
# discordant scenes is the honest significance test and needs no new code.
#
# COST. DFT_SPLITS drops the two `_train` panels: the marginal is computed from
# failure_heldout + success_heldout only (weights 0.0150 / 0.9850, covering 0.965
# of the corpus), so an eval point is 1173 scenes (~6.5 min) not 2346 (~13 min).
# 16 eval points + 150 iters x ~12 s = ~2.3 h/arm; 6 arms at 3-wide = ~4.6 h.
# keep_every=10 puts a checkpoint at EVERY eval point -- the point of the run is
# to be able to take the weights from the optimum, not just to see it on a plot.
# ~16 snapshots x 121 MB x 6 arms = ~12 GB (8.9 T free, fine).
#
# WHAT WOULD COUNT AS AN ANSWER:
#   * a margD peak that appears at the SAME iteration in all three seeds, with
#     the seed spread smaller than the peak's height  -> a real stopping point;
#     take that checkpoint and run --verify_only on it for the paired test.
#   * seeds that peak at different iterations, or a spread that swamps it -> the
#     iter-100 result was grid luck; the honest conclusion is that the fine-tune
#     is neutral-to-negative and the knobs cannot fix it.
#
# THE bc300 ARMS, and why the old bc300 curve could not be read. The 09-07
# `bc_sweep` (diffusion_ft/bc_sweep/20260907_201620_354034) is NOT comparable to
# anything measured since, for two independent reasons:
#
#   1. ITS SPLITS FILE WAS A DIFFERENT FILE. Its env points at
#      `verified_seed777/eval_splits_verified.json`, but that path was REWRITTEN
#      on 09-08 10:09, 14 h after the sweep started. The proof is in the runs'
#      own base reference: bc_sweep saw base failure 0.040 / success 0.998, the
#      ablation and combo saw 0.017 / 0.997. Different base = different scenes.
#      This is exactly the silent failure [[feedback_pin_eval_splits_and_seed]]
#      warns about -- the path stayed valid while its contents changed.
#   2. `marginal/delta_vs_base` is identically +0.0000 in every bc_sweep row --
#      the marginal was not computed yet. So whatever looks better in those
#      curves, it is not the decision metric.
#
# Within its OWN sweep, at matched iterations, bc300 was worse than bc0 on BOTH
# axes (it25 0.225/0.765 vs 0.329/0.873; it50 0.289/0.932 vs 0.327/0.940; it75
# 0.295/0.962 vs 0.303/0.969), and on the current splits the ablation's
# `anchor_bc` (bc=100) was worse than the base anchor at every matched iteration
# with drift climbing 0.70 -> 1.58 m. So the prior is that bc300 loses. But it
# has never actually been scored on the current splits with the marginal, its
# late-run points (fail 0.39 @250, 0.36 @299 at success ~0.96) are the best-
# looking failure/success pairs in that old sweep, and settling it costs ~7 GPU-h.
# That is what these three arms are for. Read them the same way as the others:
# a peak that repeats across all three seeds, or nothing.
#
# Node: `general` = a6000/a5000 (gpu24-31) only -- v100 dies at cuDNN autotune,
# 2080 Ti OOMs. Concurrency is the array's `%3`.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/sweep_safety_stopping.sh
#   DFT_ARMS="base_s0 swin25_s0" bash rl/diffusion_ft/slurm/sweep_safety_stopping.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/${STAMP}}"
ENV_DIR="$BASE_OUT/env"
mkdir -p "$ENV_DIR" "$ROOT/logs/dft"

SPLITS_JSON="${DFT_SPLITS_JSON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_verified.json}"
CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
IDX="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
[[ -s "$SPLITS_JSON" ]] || { echo "FATAL: no splits file $SPLITS_JSON" >&2; exit 1; }

ITERS="${DFT_ITERS:-150}"
EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
KEEP_EVERY="${DFT_KEEP_EVERY:-10}"
MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-3}"

# The centre is the ablation's `base` -- plain, no geometry changes. The sweeps
# established the knobs are inert, so the reference is the unmodified config.
ALL_ARMS=(
  "base_s0|DFT_SAFETY_WINDOW=0,DFT_SEED=0,DFT_DATA_SEED=0"
  "base_s1|DFT_SAFETY_WINDOW=0,DFT_SEED=1,DFT_DATA_SEED=1"
  "base_s2|DFT_SAFETY_WINDOW=0,DFT_SEED=2,DFT_DATA_SEED=2"
  "swin25_s0|DFT_SAFETY_WINDOW=25,DFT_SEED=0,DFT_DATA_SEED=0"
  "swin25_s1|DFT_SAFETY_WINDOW=25,DFT_SEED=1,DFT_DATA_SEED=1"
  "swin25_s2|DFT_SAFETY_WINDOW=25,DFT_SEED=2,DFT_DATA_SEED=2"
  # bc300: the log-target anchor at weight 300, base anchor OFF -- like-for-like
  # with the 09-07 bc_sweep and with the ablation's `anchor_bc` (which was 100).
  # Here it is finally on the SAME splits and WITH the marginal computed, which
  # is the whole reason to re-run it; see the bc300 note in the header.
  "bc300_s0|DFT_SAFETY_WINDOW=0,DFT_BC_WEIGHT=300,DFT_BASE_ANCHOR_WEIGHT=0.0,DFT_SEED=0,DFT_DATA_SEED=0"
  "bc300_s1|DFT_SAFETY_WINDOW=0,DFT_BC_WEIGHT=300,DFT_BASE_ANCHOR_WEIGHT=0.0,DFT_SEED=1,DFT_DATA_SEED=1"
  "bc300_s2|DFT_SAFETY_WINDOW=0,DFT_BC_WEIGHT=300,DFT_BASE_ANCHOR_WEIGHT=0.0,DFT_SEED=2,DFT_DATA_SEED=2"
)
# A caller's own arm table (space-separated 'tag|K=V,K=V'), for a recipe that is
# this one plus shared overrides -- those go in DFT_EXTRA_ENV_FILE, appended to
# every arm's env file before the arm's own lines, so values may hold commas.
if [[ -n "${DFT_ARM_SPECS:-}" ]]; then read -ra ALL_ARMS <<< "$DFT_ARM_SPECS"; fi
if [[ -n "${DFT_EXTRA_ENV_FILE:-}" && ! -s "$DFT_EXTRA_ENV_FILE" ]]; then
  echo "FATAL: DFT_EXTRA_ENV_FILE=$DFT_EXTRA_ENV_FILE is missing or empty" >&2; exit 1
fi

WANT="${DFT_ARMS:-}"
ARMS=()
if [[ -z "$WANT" ]]; then ARMS=("${ALL_ARMS[@]}"); else
  for w in $WANT; do
    hit=""
    for a in "${ALL_ARMS[@]}"; do
      [[ "${a%%|*}" == "$w" ]] && { ARMS+=("$a"); hit=1; break; }
    done
    [[ -n "$hit" ]] || { echo "FATAL: unknown arm '$w'" >&2; exit 1; }
  done
fi
N=${#ARMS[@]}
GROUP="${DFT_WANDB_GROUP:-stopping_${STAMP}}"

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")
[[ -n "${DFT_TIME:-}" ]]      && SB_FLAGS+=(--time="$DFT_TIME")
# A caller's upstream, e.g. the calibration that writes DFT_EDGE_MARGIN_FILE.
[[ -n "${DFT_DEPENDENCY:-}" ]] && SB_FLAGS+=(--dependency="$DFT_DEPENDENCY")

echo "sweep     -> $BASE_OUT"
echo "arms      -> $N (2 configs x 3 seeds), at most $MAX_CONCURRENT at once"
echo "schedule  -> $ITERS iters, eval+keep every $EVAL_EVERY  (16 eval points)"
echo "splits    -> failure_heldout,success_heldout only (marginal needs no more)"
echo "sim       -> sim_agents=${DFT_SIM_AGENTS:-none} neighbor_source=${DFT_NEIGHBOR_SOURCE:-log} world_dt=${DFT_WORLD_DT_METHOD:-median}"
echo "edge      -> geometry=${DFT_EDGE_GEOMETRY:-point} margin=${DFT_EDGE_MARGIN_FILE:-${DFT_EDGE_MARGIN:-0.3}}"
echo

MANIFEST="$BASE_OUT/arms.txt"; : > "$MANIFEST"
IDX_N=0
for spec in "${ARMS[@]}"; do
  TAG="${spec%%|*}"; OVR="${spec#*|}"; [[ "$OVR" == "$spec" ]] && OVR=""
  OUT_DIR="$BASE_OUT/$TAG"; mkdir -p "$OUT_DIR"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=$CKPT"
    echo "DFT_SHARD_GLOB=$GLOB"
    echo "DFT_INDEX_DIR=$IDX"
    echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-32}"
    echo "DFT_EVAL_SCENES=${DFT_EVAL_SCENES:-16}"
    echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
    echo "DFT_SPLITS_JSON=$SPLITS_JSON"
    # Only the two held-out panels: the marginal is their weighted sum, and the
    # _train panels double the eval bill for a question this run is not asking.
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_heldout,success_heldout}"
    echo "DFT_SPLIT_CAP=${DFT_SPLIT_CAP:-1000}"
    echo "DFT_EVAL_CHUNK=${DFT_EVAL_CHUNK:-32}"
    echo "DFT_TRAIN_SPLITS=${DFT_TRAIN_SPLITS:-failure_train,success_train}"
    echo "DFT_TRAIN_SPLIT_CAP=${DFT_TRAIN_SPLIT_CAP:--1}"
    echo "DFT_TRAIN_SPLIT_RATIO=${DFT_TRAIN_SPLIT_RATIO:-1,1}"
    echo "DFT_PROBE_PER_SPLIT=0"
    echo "DFT_PROBE_EVERY=0"
    # objective: the unmodified centre (knobs measured inert in both sweeps)
    echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
    echo "DFT_ANCHOR_WEIGHT=0.0"
    echo "DFT_BC_WEIGHT=${DFT_BC_WEIGHT:-0.0}"
    echo "DFT_BASE_ANCHOR_WEIGHT=${DFT_BASE_ANCHOR_WEIGHT:-100}"
    echo "DFT_BASE_ANCHOR_SUCCESS_ONLY=1"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_EDGE_TAU=${DFT_EDGE_TAU:-0.15}"
    echo "DFT_AGENT_TAU=${DFT_AGENT_TAU:-0.25}"
    echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    # 2026-09-25. `point` reproduces every earlier sweep. A non-empty margin file
    # overrides DFT_EDGE_MARGIN inside the job (see train_safety.sbatch).
    echo "DFT_EDGE_GEOMETRY=${DFT_EDGE_GEOMETRY:-point}"
    echo "DFT_EDGE_MARGIN_FILE=${DFT_EDGE_MARGIN_FILE:-}"
    echo "DFT_ANCHOR_WINDOW=-1"
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    echo "DFT_COLLECT_STEPS=0"
    echo "DFT_EVAL_STEPS=0"
    echo "DFT_UNROLL_STEPS=0"
    echo "DFT_ANCHOR_STATES=policy"
    echo "DFT_ITERS=$ITERS"
    echo "DFT_EPOCHS=1"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_GRAD_ACCUM=0"
    echo "DFT_EVAL_EVERY=$EVAL_EVERY"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_KEEP_EVERY=$KEEP_EVERY"
    # PINNED across every arm: the eval has to be the same measurement, or the
    # seed replicates measure the eval instead of the run.
    echo "DFT_EVAL_SEED=12345"
    # The clock (2026-09-24): `median` unless told otherwise. The 09-11 sweep ran
    # `first`; DFT_WORLD_DT_METHOD=first reproduces it.
    echo "DFT_WORLD_DT_METHOD=${DFT_WORLD_DT_METHOD:-median}"
    # The simulator (2026-09-25): forwarded so a sweep can run the same recipe
    # against reactive NPCs. Defaults reproduce every earlier sweep (rails).
    echo "DFT_SIM_AGENTS=${DFT_SIM_AGENTS:-none}"
    echo "DFT_NEIGHBOR_SOURCE=${DFT_NEIGHBOR_SOURCE:-log}"
    echo "DFT_EVAL_REPEATS=1"
    echo "DFT_EVAL_BASE=1"
    echo "DFT_RESUME=1"
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
    if [[ -n "${DFT_EXTRA_ENV_FILE:-}" ]]; then
      echo "# --- shared overrides: $DFT_EXTRA_ENV_FILE ---"; cat "$DFT_EXTRA_ENV_FILE"
    fi
    echo "# --- arm: $TAG ---"
    IFS=',' read -ra KV <<< "$OVR"; for kv in "${KV[@]}"; do echo "$kv"; done
  } > "$ENV_DIR/${IDX_N}.env"
  printf '%2d  %-11s %s\n' "$IDX_N" "$TAG" "$OVR" | tee -a "$MANIFEST"
  IDX_N=$((IDX_N + 1))
done
echo
# Write the env files and stop, so a caller can check them before anything queues.
if [[ -n "${DFT_DRY_RUN:-}" ]]; then
  echo "DRY RUN: env files in $ENV_DIR, nothing submitted"; exit 0
fi

ARRAY_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable "${SB_FLAGS[@]}" \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" --job-name="dft_stop" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_DIR")
echo "submitted array $ARRAY_ID  (${N} tasks, %${MAX_CONCURRENT})"
# Record the array id so a caller can chain onto it without guessing from squeue.
echo "$ARRAY_ID" > "$BASE_OUT/jobid.txt"
echo
cat <<EOF
read it (per-seed curves; is there a peak at the SAME iter in all three?):
  python3 - <<'PY'
import json,glob,collections
C=collections.defaultdict(dict)
for p in sorted(glob.glob('$BASE_OUT/*/train_log.jsonl')):
    arm=p.split('/')[-2]; cfg,seed=arm.rsplit('_',1)
    for l in open(p):
        r=json.loads(l)
        if 'marginal/delta_vs_base' in r: C[cfg].setdefault(r['iter'],{})[seed]=r['marginal/delta_vs_base']
for cfg,by in C.items():
    print('==',cfg)
    for it in sorted(by):
        v=list(by[it].values())
        print('  it%-4d %s   mean %+.4f  spread %.4f'%(it,' '.join('%+.4f'%x for x in v),
              sum(v)/len(v), max(v)-min(v) if len(v)>1 else 0))
PY
EOF
