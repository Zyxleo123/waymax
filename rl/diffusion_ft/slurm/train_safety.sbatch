#!/bin/bash
#SBATCH --job-name=dft_safetrain
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
# A requeued or chained job APPENDS to its predecessor's log instead of
# truncating it -- the only way to tell a resume from a fresh start after the
# fact, and a multi-day run is a chain of these.
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Analytic safety fine-tune of the 6/16 base diffusion checkpoint.
#
# GPU CLASS IS NOT NEGOTIABLE HERE -- it is `general`/`preempt` a6000/a5000 or
# nothing. Measured 2026-09-02 (jobs 40492 gpu20, 40494 gpu21): on a v100 this
# env's jaxlib dies at iteration 0 with every cuDNN autotune config failing --
# `Autotuning failed for HLO: %cudnn-conv-bias-activation`, 205x
# `<unknown cudnn status: 5003>` / CUDNN_STATUS_EXECUTION_FAILED -- on a
# f32[16,256,14] conv, i.e. 57 KB. That is a library/architecture mismatch, not
# memory; raising --mem or shrinking the batch does nothing. The 11 GB 2080 Tis
# (`legacy`, `debug` gpu12/13) are out for the separate reason that they OOM, and
# an XLA allocation failure there kills the process with no Python traceback.
# Usable nodes: gpu24-31 (general), gpu27/30/31 + gpu1 (preempt), gpu1 (debug).
#
# Config arrives as a positional env FILE (never `sbatch --export`, which
# triggers SLURM_GET_USER_ENV and requeue-holds the job).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

# A DIRECTORY argument is array mode: this task sources <dir>/<task_id>.env, so
# one job array can run N differently-configured arms under a single `%N`
# concurrency throttle (that throttle is the only way to bound how many GPUs a
# sweep holds at once -- independent sibling jobs have no such knob). Everything
# else is unchanged: a FILE argument is sourced, a VAR=value argument exported.
for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "FATAL: '$arg' is a directory but this is not an array job." >&2
      echo "       Submit with --array, or pass a single .env FILE." >&2
      exit 1
    fi
    ARM_ENV="$arg/${SLURM_ARRAY_TASK_ID}.env"
    if [[ ! -f "$ARM_ENV" ]]; then
      echo "FATAL: array task ${SLURM_ARRAY_TASK_ID} has no env file $ARM_ENV" >&2
      exit 1
    fi
    echo "array task ${SLURM_ARRAY_TASK_ID} -> $ARM_ENV"
    # shellcheck disable=SC1090
    source "$ARM_ENV"
  elif [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"

# JAX preallocates 75% of the card by default, and a transient larger than what
# is left then fails even though the GPU has room. Measured 2026-09-07 (jobs
# 44862/44864 on gpu28, an a5000): both died with
# `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.84GiB` at their
# first eval point after warm-up, while the identical arm on gpu27 (a6000, 48 GB)
# ran fine -- so this is the 24 GB cards, and the peak is the chunked eval
# rollout, not training. 0.95 buys ~4.8 GB more on an a5000, well past the 1.84
# that failed. Pair it with a smaller DFT_EVAL_CHUNK rather than relying on it
# alone.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"

# ---- data: full WOMD corpus by default (1000 indexed shards) -----------------
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_TFRECORD="${DFT_TFRECORD:-}"
DFT_INDICES="${DFT_INDICES:-0,1,2,3,4,5,6,7}"
# 32, not 16: at 16 the denoiser leaves the a6000 idle. Measured on the same
# denoiser+env under DPPO (see the collect-bottleneck note): B=32 is 4x the data
# for +11% wall time, and B=64 OOMs in the actor gradient. 48 needs an a6000 --
# the 24 GB a5000s already OOM on the eval rollout at DFT_EVAL_CHUNK=64.
DFT_SCENES_PER_ITER="${DFT_SCENES_PER_ITER:-32}"
DFT_EVAL_SCENES="${DFT_EVAL_SCENES:-32}"
DFT_DATA_SEED="${DFT_DATA_SEED:-0}"
DFT_PREFETCH="${DFT_PREFETCH:-1}"
# Named eval splits (build_eval_splits.py output). When set, each split in
# DFT_SPLITS is additionally evaluated every eval_every and logged under a
# `<split>/` prefix. Needs DFT_SHARD_GLOB = the glob the scan used.
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-}"
DFT_SPLITS="${DFT_SPLITS:-failure_heldout,success_heldout,failure_train,success_train}"
# Scenes scored per split at every eval point. Was 48 -- a binomial floor of
# ~0.07 on a rate near 0.8, i.e. the same size as the effect being measured.
# 1000 puts that floor at ~0.015. Cost is linear in cap x DFT_EVAL_REPEATS x
# |DFT_SPLITS| and is charged at every DFT_EVAL_EVERY, so raising this without
# raising DFT_EVAL_EVERY turns a 2 h run into a 20 h one.
#
# That is exactly what happened. Measured 2026-09-09 on the 700-iteration run:
# training is 11 s/iter (`iter_s`, stamped BEFORE the eval block, so eval is on
# top of it), i.e. 2.1 h for the whole run -- while 70 eval points at
# 2346 scenes x 4 repeats cost 25-40 h. Training was 5-10% of the wall clock and
# the safety loss was never the constraint. DFT_EVAL_EVERY and DFT_EVAL_REPEATS
# below are set from that measurement; raise the cap only together with them.
DFT_SPLIT_CAP="${DFT_SPLIT_CAP:-1000}"
# Splits longer than this are rolled in chunks and aggregated scene-weighted. A
# single batched state of 1000 scenes is ~2 GB and its rollout does not fit on a
# 24 GB a5000; four of them resident do not fit at all.
# 32, not 64: the eval rollout is the peak allocation of the whole job (8
# replans of the 100-step sampler over a whole chunk at once), and 64 does not
# fit on a 24 GB a5000 -- see the OOM note above. The old cap-48 runs worked, so
# 32 sits a third below the largest size known to run.
DFT_EVAL_CHUNK="${DFT_EVAL_CHUNK:-32}"
# Which scenes the GRADIENT is taken on. Empty = uniform over the ~482k-scene
# corpus, which is what every run before 2026-09-04 did -- and under it a split
# named `failure_train` is drawn ~0.3 times in 200 iterations, so `failure_train`
# and `failure_heldout` are BOTH held out and their curves agree for that reason
# and no other. Set this to train on the agent's own failures for real.
DFT_TRAIN_SPLITS="${DFT_TRAIN_SPLITS:-}"
# Per-split cap, one value or a comma list aligned to DFT_TRAIN_SPLITS; -1 keeps
# the split whole. success_train is ~23k against failure_train's 564, so a mixed
# pool needs equal caps or it is 98% successes.
DFT_TRAIN_SPLIT_CAP="${DFT_TRAIN_SPLIT_CAP:--1}"
# Per-batch quota per split, e.g. "1,1" for a 50/50 failure/success batch. This
# is the alternative to capping: the pool stays ALL of the data and the mix is
# fixed per batch, instead of throwing away 22k successes to balance a few
# hundred failures. Empty = uniform over the union (98% successes).
DFT_TRAIN_SPLIT_RATIO="${DFT_TRAIN_SPLIT_RATIO:-}"

# ---- per-iteration rollout probe (the slider's data) -------------------------
# A fixed handful of named scenes rolled out every DFT_PROBE_EVERY iterations and
# dumped to <out_dir>/probe/ as npz + jsonl. Costs one extra eval pass per probe
# point and makes the training curve re-renderable as PICTURES on a CPU node.
# 0 scenes disables it.
DFT_PROBE_PER_SPLIT="${DFT_PROBE_PER_SPLIT:-4}"
DFT_PROBE_SPLITS="${DFT_PROBE_SPLITS:-failure_train,success_train,failure_heldout,success_heldout}"
DFT_PROBE_EVERY="${DFT_PROBE_EVERY:-0}"
DFT_PROBE_SEED="${DFT_PROBE_SEED:-0}"

# ---- objective --------------------------------------------------------------
# anchor_weight 0 = safety only. The denoising anchor measured 0.06% of the
# gradient and flat under a 20x drift increase (job 40324), so it is off by
# default; the eval block below is what catches the resulting degeneracy.
DFT_ANCHOR_WEIGHT="${DFT_ANCHOR_WEIGHT:-0.0}"
# The ORIGINAL pretraining denoising loss, on the HUMAN's states, from a second
# sampling-free pass over the same batch. NOT the anchor: the anchor conditions
# on policy-visited states and relabels the human future from a drifted pose
# (DAgger), which is why it damaged the policy while buying no behaviour. This
# is the term the pretraining pipeline actually optimized. 0 = off. Size it from
# the MEDIAN `bc_share` in train_log.jsonl, not its iter-0 value: the term is
# minimized within a few iterations, so iter 0 overstates it ~13x. Measured:
# weight 100 -> 0.30 of the objective, 300 -> 0.57.
DFT_BC_WEIGHT="${DFT_BC_WEIGHT:-0.0}"
# The FROZEN-BASE-POLICY anchor (beta): Gaussian KL between the trainable and
# the pretrained denoiser's one-step posterior means, on the policy's own
# visited states. NOT --bc_weight/--anchor_weight, whose target is the human log
# (which the base already departs from while succeeding, so it perturbs a working
# policy). This term's minimizer IS the base, so it is zero where the policy
# still matches base and pulls only against drift -- the do-no-harm leash on the
# successes. Size it from the MEDIAN `base_anchor_share` in train_log.jsonl, not
# iter 0 (the KL collapses within a few steps). 0 = off.
DFT_BASE_ANCHOR_WEIGHT="${DFT_BASE_ANCHOR_WEIGHT:-0.0}"
# Restrict the base anchor to the SUCCESS group of a balanced batch (1 = on,
# the default). On a failure scene the anchor pulls toward base -- i.e. directly
# against the safety term trying to fix that scene. The 2026-09-08 beta30/beta100
# arms ran with this OFF and reached failure_heldout ~0.17 where the unanchored
# arm reached ~0.33, so the leash was costing about half the failure gain. No-op
# unless DFT_TRAIN_SPLITS defines groups.
DFT_BASE_ANCHOR_SUCCESS_ONLY="${DFT_BASE_ANCHOR_SUCCESS_ONLY:-1}"
DFT_SAFETY_WEIGHT="${DFT_SAFETY_WEIGHT:-1.0}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
# tau = the WIDTH OF THE KNEE of `tau*softplus((margin-c)/tau)`, a separate knob
# from the margin: the margin says WHERE cost starts, tau says HOW SHARPLY. It
# sets the loss at the margin (tau*ln2) and the ~2.7*tau band past it over which
# gradient still exists, so small tau is a near-hinge that only pushes frames
# already inside the margin and large tau is a soft field that also pulls frames
# that are already clear. Defaults are the calibrated ones (tests/calibrate_expert).
DFT_EDGE_TAU="${DFT_EDGE_TAU:-0.15}"
DFT_AGENT_TAU="${DFT_AGENT_TAU:-0.25}"
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"

# ---- which part of the candidate each term is charged on ---------------------
# Two different time bases, so two different units (see `_init_windows`):
#  * SAFETY_WINDOW is in WORLD steps (0.1 s). The candidate resamples to ~50 of
#    them (5 s) and the env drives only DFT_REPLAN (10, 1 s) before replanning.
#    0 = the executed window, what every run before 2026-09-09 scored; -1 = the
#    whole candidate; N = N world steps. Widening prices trajectory that gets
#    replanned before it is driven -- but a 1 s window cannot see a violation the
#    candidate commits at 2.5 s, so the policy is free to aim at one.
#  * ANCHOR_WINDOW is in MODEL steps (0.2 s), the denoiser's own horizon axis
#    (25 steps / 5 s). -1 = the whole horizon, the pre-2026-09-09 behaviour;
#    0 = auto, the executed window in model steps (5); N = N model steps. It
#    reweights the loss only -- the denoiser still predicts the full horizon.
DFT_SAFETY_WINDOW="${DFT_SAFETY_WINDOW:-0}"
# Matched-sample preservation + where the safety term applies (2026-09-13).
# Defaults are the pre-09-13 behaviour: term off, safety on every live row.
DFT_SAFETY_ROWS="${DFT_SAFETY_ROWS:-all}"
DFT_PRESERVE_WEIGHT="${DFT_PRESERVE_WEIGHT:-0.0}"
DFT_PRESERVE_SCENES="${DFT_PRESERVE_SCENES:-16}"
DFT_PRESERVE_WINDOW="${DFT_PRESERVE_WINDOW:-0}"
DFT_PRESERVE_RISK="${DFT_PRESERVE_RISK:-}"
DFT_PRESERVE_HARD_FRAC="${DFT_PRESERVE_HARD_FRAC:-0.5}"
DFT_PRESERVE_HARD_QUANTILE="${DFT_PRESERVE_HARD_QUANTILE:-0.8}"
DFT_PRESERVE_REPLAY="${DFT_PRESERVE_REPLAY:-256}"
DFT_PRESERVE_REPLAY_PER_ITER="${DFT_PRESERVE_REPLAY_PER_ITER:-8}"
DFT_PRESERVE_REPLAY_USES="${DFT_PRESERVE_REPLAY_USES:-10}"
DFT_REPAIR_DATA="${DFT_REPAIR_DATA:-}"
DFT_REPAIR_WEIGHT="${DFT_REPAIR_WEIGHT:-0.0}"
DFT_REPAIR_BATCH="${DFT_REPAIR_BATCH:-16}"
# ---- transition replay (2026-09-14) ------------------------------------------
# The collection rollout's OWN reverse-diffusion transitions, kept one per datum
# and replayed later. The safety loss is charged on the clean prediction reached
# from a stored (x_t, t): TB_UNROLL_STEPS=1 scores x0_hat directly (one network
# eval, no unroll -- a posterior MEAN, so it carries the Jensen gap the unrolled
# path exists to shrink), K>1 unrolls K-1 DDIM steps to TB_UNROLL_T_END first.
# Unlike --unroll_steps, the state is NOT q_sample(x0_expert, t): it is one the
# deployed sampler passed through, so there is no diffusion-state mismatch.
# TB_GRAD_STEPS=1 means the update differentiates exactly one transition.
# 0 weight = off, nothing stored, the run is bit-for-bit what it was.
DFT_TB_WEIGHT="${DFT_TB_WEIGHT:-0.0}"
DFT_TB_CAPACITY="${DFT_TB_CAPACITY:-8192}"
DFT_TB_BATCH="${DFT_TB_BATCH:-32}"
DFT_TB_STORE_PER_REPLAN="${DFT_TB_STORE_PER_REPLAN:-1}"
DFT_TB_MAX_USES="${DFT_TB_MAX_USES:-0}"
DFT_TB_MIN_SIZE="${DFT_TB_MIN_SIZE:-0}"
DFT_TB_T_MIN="${DFT_TB_T_MIN:-0}"
DFT_TB_T_MAX="${DFT_TB_T_MAX:--1}"
DFT_TB_UNROLL_STEPS="${DFT_TB_UNROLL_STEPS:-1}"
DFT_TB_UNROLL_T_END="${DFT_TB_UNROLL_T_END:-0}"
DFT_TB_UNROLL_ETA="${DFT_TB_UNROLL_ETA:-0.0}"
DFT_TB_GRAD_STEPS="${DFT_TB_GRAD_STEPS:-1}"
DFT_TB_WINDOW="${DFT_TB_WINDOW:-0}"
DFT_ANCHOR_WINDOW="${DFT_ANCHOR_WINDOW:--1}"

# ---- where the gradient enters ----------------------------------------------
# k_trainable is the low-noise window: the gradient reaches the denoiser only at
# reverse steps t=k-1..0. On the 100-step cosine schedule the residual forward
# noise on xy (ego_range=100 m) is t=0:2.5m  2:5.8m  9:16.7m  49:71m. So 10 is
# already the bottom 10% of the chain; 3 restricts to the precise-turning end.
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"

# ---- DDIM striding of the frozen prefix (the loss-speed knob) ----------------
# The free-running proposal costs 100 denoiser evals, and each replan pays it
# TWICE: once in the collection rollout, once inside the gradient pass. Only the
# last DFT_K_TRAINABLE steps carry a gradient, so the other 90 are frozen forward
# compute. S here replaces that prefix with S - k strided DDIM steps: S=20 at
# k=10 is 20 evals instead of 100 on both passes, ~5x off the training portion.
#
# It is an APPROXIMATION -- a strided prefix hands the trainable chain a slightly
# different x_{k-1}, so it is a different policy. 0 (full chain) until
# submit_bench_safety_ddim.sh says which S holds solved/offroad flat.
DFT_COLLECT_STEPS="${DFT_COLLECT_STEPS:-0}"
# Eval deliberately stays on the full chain even when collection is strided: the
# curve has to measure the policy you would deploy, or a movement in it cannot be
# attributed between the weights and the sampler.
DFT_EVAL_STEPS="${DFT_EVAL_STEPS:-0}"

# ---- how the scored trajectory is proposed ----------------------------------
# 0 = free-running: denoise from pure noise through all 100 steps (the original
# path). K >= 1 = teacher-forced unroll: corrupt the expert future to a random
# timestep in [t_end+1, t_start] and unroll K reverse steps back down, scoring
# that endpoint -- K nn evals per gradient step instead of 100. K=1 is the
# one-step posterior mean, the Jensen-gap control. Gate it with
# submit_gate_bond_unroll.sh before spending a run on it.
DFT_UNROLL_STEPS="${DFT_UNROLL_STEPS:-0}"
DFT_UNROLL_GRAD_STEPS="${DFT_UNROLL_GRAD_STEPS:-1}"
# -1 = k_trainable - 1. Must stay inside the trainable window: the sampler runs
# the frozen denoiser above it, so training there only reaches deployment
# through weight sharing. The trainer rejects a larger value.
DFT_UNROLL_T_START="${DFT_UNROLL_T_START:--1}"
DFT_UNROLL_T_END="${DFT_UNROLL_T_END:-0}"
DFT_UNROLL_ETA="${DFT_UNROLL_ETA:-0.0}"
# policy = collect contexts by rolling the policy out closed-loop (its own
# states). log = drive the collection rollout with the human's log window, which
# removes every 100-step sample from the iteration but only ever supervises
# states a human visited.
DFT_ANCHOR_STATES="${DFT_ANCHOR_STATES:-policy}"
# Collection is free-running and needs no exploration noise. 0.03 normalized is
# 3 m of jitter and would drown the 0.3/0.75 m margins (jobs 40093 vs 40324).
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"

# ---- optimization -----------------------------------------------------------
DFT_ITERS="${DFT_ITERS:-500}"
DFT_EPOCHS="${DFT_EPOCHS:-1}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_LR="${DFT_LR:-1e-5}"
# 100, not 10. At 10 the run spent 90-95% of its wall clock evaluating (see the
# DFT_SPLIT_CAP note): 70 eval points against 2.1 h of training. 100 still gives
# ~30 curve points over a 3000-iteration run, which is more resolution than the
# 700-iteration runs it replaces ever had.
DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-100}"
# 1 = the gradient replays only the k_trainable late steps of the EXECUTED
# sample from its stored frozen prefix (tests/bench_reuse_prefix.py checks it
# matches the full chain on the same key). 0 = old behaviour: a fresh full-chain
# sample per gradient, ~90 extra denoiser evals x 8 contexts per iteration.
DFT_REUSE_PREFIX="${DFT_REUSE_PREFIX:-1}"
DFT_SAVE_EVERY="${DFT_SAVE_EVERY:-25}"
# Un-overwritten snapshots (weights only, ~121 MB). `latest` is overwritten, so
# without these a long run cannot answer "the success split was best 2000
# iterations ago, give me that model" -- which is the question a multi-day run
# exists to ask, since more Adam steps means more displacement and displacement
# rank-orders damage. 0 = off.
DFT_KEEP_EVERY="${DFT_KEEP_EVERY:-0}"
# Replan contexts averaged into ONE Adam step; 0 = the whole iteration. Adam's
# step is ~lr per parameter whatever the gradient norm, so the step COUNT sets
# how far the weights travel: the old 1-step-per-context schedule spent 1600
# steps on a 200-iteration run and moved the random-walk distance. 1 restores it.
DFT_GRAD_ACCUM="${DFT_GRAD_ACCUM:-0}"
# Evals draw from their own seed, not the training stream, and average this many
# of them. This was 4, and 4 was calibrated against DFT_SPLIT_CAP=48, where a
# single pass has a binomial floor of ~0.07 on a rate near 0.8 -- the same size
# as the effect being measured. The cap then went 48 -> 1000, which drops that
# floor to ~0.013 in ONE pass, and nobody took the repeats back out: the run was
# paying 4x to re-roll the same fixed 1000 scenes for precision it already had.
# 1 unless you drop the cap back down; the two knobs buy the same thing.
DFT_EVAL_SEED="${DFT_EVAL_SEED:-12345}"
DFT_EVAL_REPEATS="${DFT_EVAL_REPEATS:-1}"
# Score the frozen base once on the same seeds and carry it in every eval row.
DFT_EVAL_BASE="${DFT_EVAL_BASE:-1}"
DFT_RESUME="${DFT_RESUME:-1}"
DFT_SEED="${DFT_SEED:-0}"

# ---- Weights & Biases --------------------------------------------------------
# ON by default: a run that is not in the dashboard is a run nobody looks at, and
# 500 iterations of this cost hours of a6000 time. train_log.jsonl is still
# written either way, so `DFT_WANDB=0` loses nothing but the mirror. The Python
# flag stays default-off (see train_safety.py) -- that path is for interactive
# debugging, where a stray run is noise. A resumed job reuses the id in
# <out_dir>/wandb_run_id.txt, so a 12 h time limit does not split one training
# curve across two charts.
DFT_WANDB="${DFT_WANDB:-1}"
DFT_WANDB_PROJECT="${DFT_WANDB_PROJECT:-diffusion_ft_safety}"
DFT_WANDB_GROUP="${DFT_WANDB_GROUP:-}"
DFT_WANDB_NAME="${DFT_WANDB_NAME:-}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}" \
     "mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "out_dir=$DFT_OUT_DIR iters=$DFT_ITERS lr=$DFT_LR k_trainable=$DFT_K_TRAINABLE"
echo "anchor_weight=$DFT_ANCHOR_WEIGHT safety_weight=$DFT_SAFETY_WEIGHT" \
     "bc_weight=$DFT_BC_WEIGHT base_anchor_weight=$DFT_BASE_ANCHOR_WEIGHT" \
     "base_anchor_success_only=$DFT_BASE_ANCHOR_SUCCESS_ONLY"
echo "margins edge=$DFT_EDGE_MARGIN agent=$DFT_AGENT_MARGIN geom=$DFT_AGENT_GEOMETRY" \
     "tau edge=$DFT_EDGE_TAU agent=$DFT_AGENT_TAU"
echo "windows safety=$DFT_SAFETY_WINDOW world-steps (0=executed, -1=full)" \
     "anchor=$DFT_ANCHOR_WINDOW model-steps (-1=full horizon, 0=executed)"
echo "safety_rows=$DFT_SAFETY_ROWS preserve_weight=$DFT_PRESERVE_WEIGHT" \
     "preserve_scenes=$DFT_PRESERVE_SCENES risk=${DFT_PRESERVE_RISK:-none}"
echo "sigma_floor=$DFT_SIGMA_FLOOR replan=$DFT_REPLAN max_replans=$DFT_MAX_REPLANS"
echo "collect_steps=$DFT_COLLECT_STEPS eval_steps=$DFT_EVAL_STEPS (0=full chain)"
echo "grad_accum=$DFT_GRAD_ACCUM eval_seed=$DFT_EVAL_SEED" \
     "eval_repeats=$DFT_EVAL_REPEATS eval_base=$DFT_EVAL_BASE"
echo "eval_splits cap=$DFT_SPLIT_CAP chunk=$DFT_EVAL_CHUNK every=$DFT_EVAL_EVERY"
echo "reuse_prefix=$DFT_REUSE_PREFIX"
echo "tb_weight=$DFT_TB_WEIGHT capacity=$DFT_TB_CAPACITY batch=$DFT_TB_BATCH" \
     "store/replan=$DFT_TB_STORE_PER_REPLAN max_uses=$DFT_TB_MAX_USES" \
     "t=[$DFT_TB_T_MIN,$DFT_TB_T_MAX] K=$DFT_TB_UNROLL_STEPS" \
     "grad_steps=$DFT_TB_GRAD_STEPS"
echo "iters=$DFT_ITERS save_every=$DFT_SAVE_EVERY keep_every=$DFT_KEEP_EVERY" \
     "resume=$DFT_RESUME"
echo "train_pool=${DFT_TRAIN_SPLITS:-uniform-corpus} cap=$DFT_TRAIN_SPLIT_CAP" \
     "ratio=${DFT_TRAIN_SPLIT_RATIO:-uniform}" \
     "probe=${DFT_PROBE_PER_SPLIT}/split every=${DFT_PROBE_EVERY:-eval_every}"
echo "wandb=$DFT_WANDB project=$DFT_WANDB_PROJECT group=${DFT_WANDB_GROUP:-none}"
if [[ "$DFT_UNROLL_STEPS" -gt 0 ]]; then
  echo "proposal=teacher-forced unroll K=$DFT_UNROLL_STEPS grad=$DFT_UNROLL_GRAD_STEPS" \
       "eta=$DFT_UNROLL_ETA t_start=$DFT_UNROLL_T_START t_end=$DFT_UNROLL_T_END" \
       "anchor_states=$DFT_ANCHOR_STATES"
else
  echo "proposal=free-running (full reverse chain)"
fi
if [[ -n "$DFT_TFRECORD" ]]; then
  echo "data=fixed tfrecord=$DFT_TFRECORD indices=$DFT_INDICES"
else
  echo "data=corpus glob=$DFT_SHARD_GLOB scenes/iter=$DFT_SCENES_PER_ITER" \
       "eval_scenes=$DFT_EVAL_SCENES data_seed=$DFT_DATA_SEED"
fi

# The out_dir lives on a quota'd filesystem; job 27279 died 27 iterations in on
# `OSError: [Errno 122] Disk quota exceeded`. Fail in seconds instead.
mkdir -p "$DFT_OUT_DIR"
# Gate: a repair arm whose search produced no verified repairs has nothing to learn.
if [[ "$DFT_REPAIR_WEIGHT" != "0" && "$DFT_REPAIR_WEIGHT" != "0.0" ]] \
   && ! compgen -G "$DFT_REPAIR_DATA" > /dev/null; then
  echo "TRAINING NOT RUN: repair_weight=$DFT_REPAIR_WEIGHT but no repair data matches '$DFT_REPAIR_DATA'"
  echo "  (either the repair search found no verified repair, or it did not run -- see its .out)"
  exit 0
fi
if ! (dd if=/dev/zero of="$DFT_OUT_DIR/.probe" bs=1M count=8 status=none && \
      rm -f "$DFT_OUT_DIR/.probe"); then
  echo "FATAL: cannot write 8 MB to $DFT_OUT_DIR (quota?)" >&2; exit 1
fi

ARGS=(--ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR"
      --iters "$DFT_ITERS" --epochs "$DFT_EPOCHS"
      --max_replans "$DFT_MAX_REPLANS" --replan_interval_steps "$DFT_REPLAN"
      --lr "$DFT_LR" --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN"
      --sigma_floor "$DFT_SIGMA_FLOOR"
      --collect_steps "$DFT_COLLECT_STEPS" --eval_steps "$DFT_EVAL_STEPS"
      --unroll_steps "$DFT_UNROLL_STEPS" --unroll_grad_steps "$DFT_UNROLL_GRAD_STEPS"
      --unroll_t_start "$DFT_UNROLL_T_START" --unroll_t_end "$DFT_UNROLL_T_END"
      --unroll_eta "$DFT_UNROLL_ETA" --anchor_states "$DFT_ANCHOR_STATES"
      --safety_weight "$DFT_SAFETY_WEIGHT" --anchor_weight "$DFT_ANCHOR_WEIGHT"
      --bc_weight "$DFT_BC_WEIGHT" --base_anchor_weight "$DFT_BASE_ANCHOR_WEIGHT"
      --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN"
      --edge_tau "$DFT_EDGE_TAU" --agent_tau "$DFT_AGENT_TAU"
      # `=` form: both take -1, and a bare `--flag -1` is what argparse misread
      # as an option string in job 43942.
      --safety_window="$DFT_SAFETY_WINDOW" --anchor_window="$DFT_ANCHOR_WINDOW"
      --safety_rows "$DFT_SAFETY_ROWS" --preserve_weight "$DFT_PRESERVE_WEIGHT"
      --preserve_scenes "$DFT_PRESERVE_SCENES" --preserve_window="$DFT_PRESERVE_WINDOW"
      --preserve_risk "$DFT_PRESERVE_RISK"
      --preserve_hard_frac "$DFT_PRESERVE_HARD_FRAC"
      --preserve_hard_quantile "$DFT_PRESERVE_HARD_QUANTILE"
      --preserve_replay "$DFT_PRESERVE_REPLAY"
      --preserve_replay_per_iter "$DFT_PRESERVE_REPLAY_PER_ITER"
      --preserve_replay_uses "$DFT_PRESERVE_REPLAY_USES"
      --repair_data "$DFT_REPAIR_DATA" --repair_weight "$DFT_REPAIR_WEIGHT"
      --repair_batch "$DFT_REPAIR_BATCH"
      --tb_weight "$DFT_TB_WEIGHT" --tb_capacity "$DFT_TB_CAPACITY"
      --tb_batch "$DFT_TB_BATCH" --tb_store_per_replan "$DFT_TB_STORE_PER_REPLAN"
      --tb_max_uses "$DFT_TB_MAX_USES" --tb_min_size "$DFT_TB_MIN_SIZE"
      --tb_t_min "$DFT_TB_T_MIN" --tb_t_max="$DFT_TB_T_MAX"
      --tb_unroll_steps "$DFT_TB_UNROLL_STEPS"
      --tb_unroll_t_end "$DFT_TB_UNROLL_T_END" --tb_unroll_eta "$DFT_TB_UNROLL_ETA"
      --tb_grad_steps "$DFT_TB_GRAD_STEPS" --tb_window "$DFT_TB_WINDOW"
      --agent_geometry "$DFT_AGENT_GEOMETRY" --n_discs "$DFT_N_DISCS"
      --eval_every "$DFT_EVAL_EVERY" --save_every "$DFT_SAVE_EVERY"
      --keep_every "$DFT_KEEP_EVERY"
      --grad_accum "$DFT_GRAD_ACCUM"
      --eval_seed "$DFT_EVAL_SEED" --eval_repeats "$DFT_EVAL_REPEATS"
      --resume "$DFT_RESUME" --seed "$DFT_SEED")
[[ "$DFT_EVAL_BASE" == "0" ]] && ARGS+=(--no-eval_base) || ARGS+=(--eval_base)
[[ "$DFT_REUSE_PREFIX" == "0" ]] && ARGS+=(--no-reuse_prefix) || ARGS+=(--reuse_prefix)
[[ "$DFT_BASE_ANCHOR_SUCCESS_ONLY" == "0" ]] \
  && ARGS+=(--no-base_anchor_success_only) || ARGS+=(--base_anchor_success_only)
if [[ -n "$DFT_TFRECORD" ]]; then
  ARGS+=(--tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" --eval_scenes 0)
else
  ARGS+=(--shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR"
         --scenes_per_iter "$DFT_SCENES_PER_ITER" --eval_scenes "$DFT_EVAL_SCENES"
         --data_seed "$DFT_DATA_SEED" --prefetch "$DFT_PREFETCH")
  # Named eval splits (corpus mode only; the ids are glob-relative positions).
  if [[ -n "$DFT_SPLITS_JSON" ]]; then
    ARGS+=(--splits_json "$DFT_SPLITS_JSON"
           --splits "$DFT_SPLITS" --split_cap "$DFT_SPLIT_CAP"
           --eval_chunk "$DFT_EVAL_CHUNK"
           --probe_splits "$DFT_PROBE_SPLITS"
           --probe_per_split "$DFT_PROBE_PER_SPLIT"
           --probe_every "$DFT_PROBE_EVERY" --probe_seed "$DFT_PROBE_SEED")
    [[ -n "$DFT_TRAIN_SPLITS" ]] && ARGS+=(--train_splits "$DFT_TRAIN_SPLITS"
           --train_split_cap="$DFT_TRAIN_SPLIT_CAP")
    [[ -n "$DFT_TRAIN_SPLIT_RATIO" ]] && ARGS+=(--train_split_ratio "$DFT_TRAIN_SPLIT_RATIO")
  elif [[ -n "$DFT_TRAIN_SPLITS" ]]; then
    # Fail here rather than train uniformly while the log claims a pool.
    echo "FATAL: DFT_TRAIN_SPLITS='$DFT_TRAIN_SPLITS' needs DFT_SPLITS_JSON" >&2
    exit 1
  fi
fi

# BooleanOptionalAction => the flag is --wandb / --no-wandb; project/group/name
# are passed only when non-empty.
WANDB_ARGS=(--no-wandb)
if [[ "$DFT_WANDB" != "0" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "$DFT_WANDB_PROJECT")
  [[ -n "$DFT_WANDB_GROUP" ]] && WANDB_ARGS+=(--wandb_group "$DFT_WANDB_GROUP")
  [[ -n "$DFT_WANDB_NAME" ]]  && WANDB_ARGS+=(--wandb_name "$DFT_WANDB_NAME")
fi

srun "$PY" -m rl.diffusion_ft.train_safety "${ARGS[@]}" "${WANDB_ARGS[@]}"
