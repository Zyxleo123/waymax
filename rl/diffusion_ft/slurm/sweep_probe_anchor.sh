#!/bin/bash
# The "does the loss actually fix it?" experiment: two training arms, no-anchor
# and with-anchor, both TRAINED ON the agent's own failures, both dumping the
# rollout of a fixed 2x2 of scenes at every probe point; then one CPU job that
# renders the slider. Only calls sbatch.
#
# WHY THIS EXISTS. Every safety run before 2026-09-04 drew its training batch
# uniformly from ~482k WOMD scenes. A 48-scene split named `failure_train` is
# therefore drawn about 0.3 times in a 200-iteration run -- so `failure_train`
# and `failure_heldout` were BOTH held out, their curves agree for that reason
# and no other, and no run had ever taken a gradient step on a scene it then
# scored. Nothing published from this package yet shows a single off-road
# trajectory being repaired; it shows aggregate rates moving, on scenes the
# gradient never touched. DFT_TRAIN_SPLITS is what closes that gap.
#
#   arms      no-anchor  safety_weight=1 anchor_weight=0   (the current default)
#             anchor     safety_weight=1 anchor_weight=$DFT_ANCHOR_WEIGHT
#   pool      DFT_TRAIN_SPLITS, capped per split so the mix is deliberate
#   probe     DFT_PROBE_PER_SPLIT scenes from each of failure_train /
#             success_train / failure_heldout / success_heldout, rolled out under
#             ONE fixed eval seed every DFT_PROBE_EVERY iters
#   figures   probe_slider.html (slider), probe_filmstrip.png, probe_curves.png
#
# Node: `general` advertises only a6000/a5000 (gpu24-31). This env's jaxlib dies
# at cuDNN autotune on v100 and OOMs on the 2080 Ti -- do NOT widen the partition.
#
# Usage (from repo root, ON RHEA -- every partition sets AllocNodes=rhea,lov4):
#   bash rl/diffusion_ft/slurm/sweep_probe_anchor.sh
#   DFT_ITERS=500 DFT_ANCHOR_WEIGHT=300 bash rl/diffusion_ft/slurm/sweep_probe_anchor.sh
#   # failure-only pool (no do-no-harm pressure in the objective at all):
#   DFT_TRAIN_SPLITS=failure_train DFT_TRAIN_SPLIT_CAP=-1 \
#     bash rl/diffusion_ft/slurm/sweep_probe_anchor.sh
#   # re-render the figures alone, against a pair that already ran:
#   DFT_PROBE_ARMS="no-anchor=$OUT/noanchor anchor=$OUT/anchor" \
#     sbatch rl/diffusion_ft/slurm/viz_probe_slider.sbatch
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"; mkdir -p "$ENV_DIR" "$ROOT/logs/dft"
# $$ as well as the second: two sibling submissions in the same second would
# otherwise share an env filename and an out_dir and interleave one log.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_OUT="${DFT_SWEEP_OUT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/probe_anchor/${STAMP}}"
GROUP="${DFT_WANDB_GROUP:-probe_anchor_${STAMP}}"

# ---- the pool the GRADIENT is taken on --------------------------------------
# Both panels of the "trained on" column have to be real, so the default pool is
# failures AND successes at equal caps. success_train is ~23k scenes against
# failure_train's 564, so an uncapped union would be 98% successes and the
# failure half would never be seen. A tighter pool is also what makes "trained
# on" mean something inside a few hundred iterations: 16 scenes/iter x 300 iters
# over 512 pool scenes is ~9 visits each.
DFT_TRAIN_SPLITS="${DFT_TRAIN_SPLITS:-failure_train,success_train}"
DFT_TRAIN_SPLIT_CAP="${DFT_TRAIN_SPLIT_CAP:-256,256}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/eval_splits.json}"
# A DIFFERENT splits file is silently wrong: the ids are positions in the scan's
# shard_glob order, so arms scored against another file are not comparable to
# anything already measured, and nothing errors.
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"

DFT_ITERS="${DFT_ITERS:-300}"
DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
DFT_PROBE_EVERY="${DFT_PROBE_EVERY:-10}"
# 6 per split = a 24-scene probe batch. More than the four panels need, on
# purpose: ~15% of the failure pool is `never_reached`, a failure with no
# off-road and no contact in it, and the viz leads each column with a scene
# the base actually drove off the road rather than whichever came first.
DFT_PROBE_PER_SPLIT="${DFT_PROBE_PER_SPLIT:-6}"
# Held-out eval cost is linear in cap x repeats x 4 splits and is charged at
# every eval point; 32x2 keeps a 300-iteration arm inside ~3 h on an a6000.
DFT_SPLIT_CAP="${DFT_SPLIT_CAP:-32}"
DFT_EVAL_REPEATS="${DFT_EVAL_REPEATS:-2}"
DFT_EVAL_SCENES="${DFT_EVAL_SCENES:-16}"
# The anchor arm's weight. At aw=1 the anchor measured 0.66% of the objective, so
# "with anchor" at 1 is "without anchor" with extra compute. The gate's
# objective-share ladder (job 39648-39653, pre-rng-fix) read 30 -> 17%,
# 100 -> 40%, 300 -> 67%. Read `anchor_share` in train_log.jsonl at iter 0 and
# move this if the post-fix anchor lands somewhere else.
DFT_ANCHOR_WEIGHT="${DFT_ANCHOR_WEIGHT:-100}"
# 0 (default): the two arms are independent, so they queue as siblings and run
# concurrently. 1 chains them so only one GPU is ever occupied.
SERIAL="${DFT_SERIAL:-0}"

SB_FLAGS=()
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_QOS:-}" ]]       && SB_FLAGS+=(--qos="$DFT_QOS")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

echo "sweep     -> $BASE_OUT"
echo "pool      -> $DFT_TRAIN_SPLITS  caps $DFT_TRAIN_SPLIT_CAP"
echo "splits    -> $DFT_SPLITS_JSON"
echo "probe     -> $DFT_PROBE_PER_SPLIT scenes/split every $DFT_PROBE_EVERY iters"
echo "arms      -> no-anchor(aw=0)  anchor(aw=$DFT_ANCHOR_WEIGHT)   iters=$DFT_ITERS"
echo

TRAIN_IDS=(); PROBE_ARMS=(); PREV=""
for ARM in noanchor:0 anchor:"$DFT_ANCHOR_WEIGHT"; do
  TAG="${ARM%%:*}"; AW="${ARM##*:}"
  OUT_DIR="$BASE_OUT/$TAG"
  ENV_FILE="$ENV_DIR/probeanchor_${STAMP}_${TAG}.env"
  {
    echo "DFT_OUT_DIR=$OUT_DIR"
    echo "DFT_CKPT=${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
    echo "DFT_SHARD_GLOB=$DFT_SHARD_GLOB"
    echo "DFT_INDEX_DIR=${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
    echo "DFT_SCENES_PER_ITER=${DFT_SCENES_PER_ITER:-16}"
    echo "DFT_EVAL_SCENES=$DFT_EVAL_SCENES"
    echo "DFT_DATA_SEED=${DFT_DATA_SEED:-0}"
    echo "DFT_PREFETCH=${DFT_PREFETCH:-1}"
    # named splits: scored every eval point AND (TRAIN_SPLITS) the gradient's pool
    echo "DFT_SPLITS_JSON=$DFT_SPLITS_JSON"
    echo "DFT_SPLITS=${DFT_SPLITS:-failure_train,success_train,failure_heldout,success_heldout}"
    echo "DFT_SPLIT_CAP=$DFT_SPLIT_CAP"
    echo "DFT_TRAIN_SPLITS=$DFT_TRAIN_SPLITS"
    echo "DFT_TRAIN_SPLIT_CAP=$DFT_TRAIN_SPLIT_CAP"
    # the probe: identical scenes and seed in BOTH arms, or the panels are not
    # comparable and the viz refuses to render them side by side
    echo "DFT_PROBE_PER_SPLIT=$DFT_PROBE_PER_SPLIT"
    echo "DFT_PROBE_SPLITS=${DFT_PROBE_SPLITS:-failure_train,success_train,failure_heldout,success_heldout}"
    echo "DFT_PROBE_EVERY=$DFT_PROBE_EVERY"
    echo "DFT_PROBE_SEED=${DFT_PROBE_SEED:-0}"
    # objective -- the swept axis
    echo "DFT_SAFETY_WEIGHT=${DFT_SAFETY_WEIGHT:-1.0}"
    echo "DFT_ANCHOR_WEIGHT=$AW"
    echo "DFT_EDGE_MARGIN=${DFT_EDGE_MARGIN:-0.3}"
    echo "DFT_AGENT_MARGIN=${DFT_AGENT_MARGIN:-0.75}"
    echo "DFT_AGENT_GEOMETRY=${DFT_AGENT_GEOMETRY:-exact}"
    echo "DFT_N_DISCS=${DFT_N_DISCS:-5}"
    echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
    echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
    echo "DFT_SIGMA_FLOOR=${DFT_SIGMA_FLOOR:-1e-4}"
    # proposal: free-running by default -- the loss is applied to the trajectory
    # the policy itself emits from its own drifted states, which is the only
    # place the barrier has anything to bite on. The human log scores ~0.
    echo "DFT_UNROLL_STEPS=${DFT_UNROLL_STEPS:-0}"
    echo "DFT_UNROLL_GRAD_STEPS=${DFT_UNROLL_GRAD_STEPS:-1}"
    echo "DFT_UNROLL_T_START=${DFT_UNROLL_T_START:--1}"
    echo "DFT_UNROLL_T_END=${DFT_UNROLL_T_END:-0}"
    echo "DFT_UNROLL_ETA=${DFT_UNROLL_ETA:-0.0}"
    echo "DFT_ANCHOR_STATES=${DFT_ANCHOR_STATES:-policy}"
    echo "DFT_ITERS=$DFT_ITERS"
    echo "DFT_EPOCHS=${DFT_EPOCHS:-1}"
    echo "DFT_MAX_REPLANS=${DFT_MAX_REPLANS:-8}"
    echo "DFT_REPLAN=${DFT_REPLAN:-10}"
    echo "DFT_LR=${DFT_LR:-1e-5}"
    echo "DFT_EVAL_EVERY=$DFT_EVAL_EVERY"
    echo "DFT_SAVE_EVERY=${DFT_SAVE_EVERY:-25}"
    echo "DFT_GRAD_ACCUM=${DFT_GRAD_ACCUM:-0}"
    echo "DFT_EVAL_SEED=${DFT_EVAL_SEED:-12345}"
    echo "DFT_EVAL_REPEATS=$DFT_EVAL_REPEATS"
    echo "DFT_EVAL_BASE=${DFT_EVAL_BASE:-1}"
    echo "DFT_RESUME=${DFT_RESUME:-1}"
    echo "DFT_SEED=${DFT_SEED:-0}"
    echo "DFT_WANDB=${DFT_WANDB:-1}"
    echo "DFT_WANDB_PROJECT=${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
    echo "DFT_WANDB_GROUP=$GROUP"
    echo "DFT_WANDB_NAME=$TAG"
  } > "$ENV_FILE"

  DEP=()
  [[ "$SERIAL" == "1" && -n "$PREV" ]] && DEP=(--dependency="afterany:${PREV}")
  # Scrub SLURM_* so a submission from inside an allocation does not inherit it.
  ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
    sbatch --parsable "${SB_FLAGS[@]}" "${DEP[@]}" --job-name="dft_pa_${TAG}" \
    rl/diffusion_ft/slurm/train_safety.sbatch "$ENV_FILE")
  echo "  $TAG (aw=$AW)  train -> job $ID   ($OUT_DIR)"
  TRAIN_IDS+=("$ID"); PREV="$ID"
  PROBE_ARMS+=("$([[ $TAG == noanchor ]] && echo no-anchor || echo anchor)=$OUT_DIR")
done

VIZ_ENV="$ENV_DIR/probeanchor_${STAMP}_viz.env"
{
  # QUOTE every value: the sbatch `source`s this file, so an unquoted value with
  # spaces runs its second word as a command (jobs 41878/41881 died on
  # `the: command not found` in the title line and rendered no figures).
  echo "DFT_PROBE_ARMS='${PROBE_ARMS[*]}'"
  echo "DFT_PROBE_FIG_DIR='$BASE_OUT/figures'"
  echo "DFT_PROBE_TITLE='${DFT_PROBE_TITLE:-Does the safety loss fix an off-road trajectory?}'"
} > "$VIZ_ENV"
# afterany, not afterok: an arm that hits the wall clock still has probe dumps
# worth rendering, and afterok would strand the figures in
# DependencyNeverSatisfied with nothing said about why.
DEP_LIST=$(IFS=:; echo "${TRAIN_IDS[*]}")
VIZ_ID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch --parsable --dependency="afterany:${DEP_LIST}" \
  rl/diffusion_ft/slurm/viz_probe_slider.sbatch "$VIZ_ENV")
echo "  slider    -> job $VIZ_ID  (CPU, afterany:${DEP_LIST})  $BASE_OUT/figures"
echo
echo "watch:"
for i in "${!TRAIN_IDS[@]}"; do
  echo "  tail -f $ROOT/logs/dft/slurm-dft_pa_*-${TRAIN_IDS[$i]}.out"
done
echo
echo "render the slider early, without waiting for the run to finish:"
echo "  sbatch rl/diffusion_ft/slurm/viz_probe_slider.sbatch $VIZ_ENV"
