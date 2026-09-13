#!/bin/bash
# Re-run of the `entry_clean` DPPO experiment on **raw WOMD**, in two data
# regimes x N seeds as sibling jobs. Does no work of its own -- it only calls the
# tested single-run submitter once per arm.
#
# ---- why this run exists -------------------------------------------------
# Every DPPO result before 2026-09-01 that was taken on the ScenarioMax failure
# set (`failures.tfrecord`) is unreadable, and `submit_dppo_entryclean.sh` is one
# of them. Every ScenarioMax-converted record carries a constant timestamp
# column, so `ego_source_time_seconds` (data/preprocess.py) had a degenerate
# `src_t`, `searchsorted` returned a meaningless bracket, and the ego frame
# (origin_xy / anchor_yaw / anchor_world_state) was pinned to the LOG'S STEP 0
# instead of the current step: the policy was conditioned on where it had been
# ~1 s earlier, and the BC target round-tripped 60.9 m away from the log it is.
# Fixed 09-01 (BC round-trip 60.9 m -> 0.002 m, base drift 10.29 m -> 3.05 m).
#
# The fix is a NO-OP on WOMD, whose timestamps are real -- and WOMD is the corpus
# the base checkpoint was pretrained on (metadata.json:
# tfrecord_path=.../training_tfexample.tfrecord@1000), which is what makes the
# expert/BC term a coherent anchor rather than a pull toward another
# distribution. So this re-run moves to WOMD rather than re-running on SMX.
#
# Nothing about the recipe changes: this is still the reward-ORDERING test that
# `reward_ranking.sbatch` (job 39053) selected `entry_clean` for -- P1 solve is
# argmax, P2 avoidance > dwell (31.5x), P3 second excursion priced, P4 arriving
# promptly pays. The question is unchanged: with the ordering corrected and
# nothing else, does `eval_solve_deterministic` move?
#
# ---- what changes vs submit_dppo_entryclean.sh ---------------------------
#   * data: raw WOMD, not failures.tfrecord
#   * route dims 45x800, not 10x300. ScenarioMax's 10x300 against WOMD yields
#     ZERO valid sdc_paths and a silently dead progression reward -- the whole
#     recipe is route-based, so this is the one line that must not be wrong.
#   * two regimes (below), because they answer different questions.
#
# ---- the two regimes ------------------------------------------------------
#   fixed   32 scenes of WOMD shard 0, loaded once. The like-for-like analogue of
#           the invalidated run: an OVERFIT target, eval is on the training
#           scenes. Answers "does the corrected ordering move solve at all".
#   corpus  a fresh 32 scenes drawn uniformly over all 1000 shards each iter,
#           with a fixed held-out 32-scene eval set. Answers "does it
#           generalize". Needs the byte-offset indexes (already built:
#           1000/1000 in DFT_INDEX_DIR); rebuild with submit_build_womd_offsets.sh.
# They queue concurrently; neither is a prerequisite for the other.
#
# ---- how to read it -------------------------------------------------------
# Do NOT compare the return curve to any pre-09-01 run: both the corpus and the
# ego frame changed. Read `eval_solve_deterministic` / `eval_offroad_deterministic`
# from train_log.jsonl (not the console line -- it rounds). Expect return and
# solve to move TOGETHER; that co-movement is what every earlier sweep lacked.
# WOMD scenes are much easier than the failure set, so the `fixed` arm may start
# near ceiling -- read the headroom off it=0 before reading a trend.
#
# Usage (from repo root, on rhea):
#   bash rl/diffusion_ft/slurm/submit_dppo_entryclean_womd.sh
#   # smoke the whole grid for 2 iterations first (checks route[train]: 32/32):
#   DFT_ITERS=2 DFT_EVAL_EVERY=1 DFT_SEED_GRID=0 \
#     bash rl/diffusion_ft/slurm/submit_dppo_entryclean_womd.sh
#   # one regime only:
#   DFT_REGIMES=corpus DFT_SEED_GRID="0 1 2 3 4" \
#     bash rl/diffusion_ft/slurm/submit_dppo_entryclean_womd.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

SEED_GRID="${DFT_SEED_GRID:-0 1 2}"
# A selected scene pool (rl/diffusion_ft/tools/select_hard_womd.py) only means
# anything for corpus sampling -- the `fixed` regime loads its scenes by index
# from one shard and never consults the pool -- so a pool switches the default
# regime set to `corpus` alone rather than silently training half the arms on
# uniform scenes.
export DFT_SCENE_IDS_JSON="${DFT_SCENE_IDS_JSON:-}"
if [[ -n "$DFT_SCENE_IDS_JSON" ]]; then
  REGIMES="${DFT_REGIMES:-corpus}"
  [[ -f "$DFT_SCENE_IDS_JSON" ]] || {
    echo "DFT_SCENE_IDS_JSON=$DFT_SCENE_IDS_JSON does not exist" >&2; exit 1; }
  if [[ " $REGIMES " == *" fixed "* ]]; then
    echo "regime 'fixed' ignores DFT_SCENE_IDS_JSON -- drop one of them" >&2
    exit 1
  fi
  echo "pool: $DFT_SCENE_IDS_JSON ($(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['scene_ids']))" "$DFT_SCENE_IDS_JSON") scenes)"
else
  REGIMES="${DFT_REGIMES:-fixed corpus}"
fi
BASE_OUT="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo_entryclean_womd}"
STAMP="$(date +%Y%m%d_%H%M%S)"

WOMD_DIR="/zfsauton/scratch/eshau/womd/tf_example/training"
WOMD_GLOB="$WOMD_DIR/training_tfexample.tfrecord-*-of-01000"
export DFT_TFRECORD="${DFT_TFRECORD:-$WOMD_DIR/training_tfexample.tfrecord-00000-of-01000}"
export DFT_INDICES="${DFT_INDICES:-$(seq -s, 0 31)}"
# Raw WOMD tf_example route dims. 10x300 (ScenarioMax) here = zero valid routes.
export DFT_NUM_PATHS="${DFT_NUM_PATHS:-45}"
export DFT_NUM_POINTS_PER_PATH="${DFT_NUM_POINTS_PER_PATH:-800}"
export DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
export DFT_SCENES_PER_ITER="${DFT_SCENES_PER_ITER:-32}"
export DFT_EVAL_SCENES="${DFT_EVAL_SCENES:-32}"
export DFT_RESAMPLE_EVERY="${DFT_RESAMPLE_EVERY:-1}"
export DFT_DATA_SEED="${DFT_DATA_SEED:-0}"

# ---- the entry_clean recipe (unchanged from submit_dppo_entryclean.sh) -----
export DFT_ROUTE_REWARD="${DFT_ROUTE_REWARD:-1}"
export DFT_OFFROAD_PER_STEP="${DFT_OFFROAD_PER_STEP:-1}"
export DFT_TERMINATE_OFFROAD="${DFT_TERMINATE_OFFROAD:-0}"
export DFT_COLLISION_PER_STEP="${DFT_COLLISION_PER_STEP:-1}"
export DFT_TERMINATE_COLLISION="${DFT_TERMINATE_COLLISION:-0}"
export DFT_PROGRESSION_INDICATOR="${DFT_PROGRESSION_INDICATOR:-0}"
export DFT_CLEAN_GOAL_BONUS="${DFT_CLEAN_GOAL_BONUS:-1}"
export DFT_REWARD_PROGRESS="${DFT_REWARD_PROGRESS:-5.0}"
export DFT_REWARD_OFFROAD="${DFT_REWARD_OFFROAD:--0.1}"
export DFT_REWARD_OFFROAD_ENTRY="${DFT_REWARD_OFFROAD_ENTRY:--1.0}"
export DFT_REWARD_COLLISION="${DFT_REWARD_COLLISION:--0.1}"
export DFT_REWARD_COLLISION_ENTRY="${DFT_REWARD_COLLISION_ENTRY:--2.0}"
export DFT_REWARD_GOAL_BONUS="${DFT_REWARD_GOAL_BONUS:-5.0}"
export DFT_REWARD_OFF_ROUTE="${DFT_REWARD_OFF_ROUTE:--0.2}"
export DFT_GOAL_SHAPING_COEF="${DFT_GOAL_SHAPING_COEF:-1.0}"
export DFT_GOAL_SHAPING_NORMALIZE="${DFT_GOAL_SHAPING_NORMALIZE:-1}"

# ---- optimizer: the only (sigma, target_kl) pair observed to take steps. The
# submitter defaults 0.1/0.05 are a 10x tighter KL gate, and that coupling is
# what froze the s*_k* sweep outright. Caveat worth knowing before reading the
# result: 0.03 is in NORMALIZED units against ego_range=100 m, i.e. ~3 m of
# per-step positional jitter -- see the sigma section of the README.
export DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-0.03}"
export DFT_TARGET_KL="${DFT_TARGET_KL:-0.556}"

export DFT_EXPERT_STEPS="${DFT_EXPERT_STEPS:-0}"   # pure DPPO; BC is a separate lever
export DFT_ITERS="${DFT_ITERS:-500}"
export DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
export DFT_EVAL_SEEDS="${DFT_EVAL_SEEDS:-3}"

# Disk: one ckpt is ~606 MB; 1075 of them (~634 GB) exhausted the quota mid-sweep
# on 20260830. Save rarely, keep few. train_log.jsonl is never pruned.
export DFT_SAVE_EVERY="${DFT_SAVE_EVERY:-50}"
export DFT_KEEP_CKPTS="${DFT_KEEP_CKPTS:-3}"

# gpu28 is the 24 GB a5000; an arm died there in a 13.05 GiB eval transfer (job
# 37156). Every arm that completed this config ran on a 48 GB a6000 (`general`).
export DFT_EXCLUDE="${DFT_EXCLUDE-gpu28}"
# Host RAM: 48G host-OOM'd a fixed-batch run ~58 iters in (job 31985), and the
# corpus stream holds a prefetched batch on top of that.
export DFT_MEM="${DFT_MEM:-96G}"
export DFT_WANDB="${DFT_WANDB:-1}"
export DFT_WANDB_PROJECT="${DFT_WANDB_PROJECT:-diffusion_ft_dppo}"
export DFT_WANDB_GROUP="${DFT_WANDB_GROUP:-entryclean_womd_$STAMP}"

echo "entry_clean DPPO on WOMD: regimes=[$REGIMES] seeds=[$SEED_GRID]" \
     "iters=$DFT_ITERS routes=${DFT_NUM_PATHS}x${DFT_NUM_POINTS_PER_PATH}"
echo "base_out=$BASE_OUT/$STAMP"
n=0
for regime in $REGIMES; do
  case "$regime" in
    # Empty glob is what switches train_dppo.sbatch back to the fixed
    # DFT_TFRECORD/DFT_INDICES batch.
    fixed)  arm_glob="" ;;
    corpus) arm_glob="$WOMD_GLOB" ;;
    *) echo "unknown regime '$regime' (want: fixed | corpus)" >&2; exit 1 ;;
  esac
  for s in $SEED_GRID; do
    DFT_SHARD_GLOB="$arm_glob" \
    DFT_SEED="$s" \
    DFT_JOBNAME="dppo_ecw_${regime}_s${s}" \
    DFT_OUT_DIR="$BASE_OUT/$STAMP/${regime}/s${s}" \
      bash rl/diffusion_ft/slurm/submit_train_dppo.sh
    n=$((n + 1))
    echo "  [$n] submitted $regime seed $s"
  done
done
echo "Submitted $n sibling arms -> $BASE_OUT/$STAMP"
echo
echo "Check the FIRST .out of each regime before trusting the run:"
echo "  * 'route[train]: 32/32' (and 'route[eval]: 32/32' on corpus). A 0 or a"
echo "    low count means the 45x800 dims did not take and progression is dead."
echo "  * the it=0000 reward line shows prog_indicator=0 clean_goal=1"
echo "    offroad_entry=-1.0 collision_entry=-2.0"
echo "  * host= is an a6000; a v100 dies in cuDNN init within seconds."
