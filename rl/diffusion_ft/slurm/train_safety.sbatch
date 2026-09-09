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

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
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
DFT_SCENES_PER_ITER="${DFT_SCENES_PER_ITER:-16}"
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
DFT_AGENT_GEOMETRY="${DFT_AGENT_GEOMETRY:-exact}"
DFT_N_DISCS="${DFT_N_DISCS:-5}"

# ---- where the gradient enters ----------------------------------------------
# k_trainable is the low-noise window: the gradient reaches the denoiser only at
# reverse steps t=k-1..0. On the 100-step cosine schedule the residual forward
# noise on xy (ego_range=100 m) is t=0:2.5m  2:5.8m  9:16.7m  49:71m. So 10 is
# already the bottom 10% of the chain; 3 restricts to the precise-turning end.
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"

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
DFT_EVAL_EVERY="${DFT_EVAL_EVERY:-10}"
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
# of them. 4 is the point where a 48-scene split's noise stops dominating the
# arm-vs-arm difference; it costs 4x the eval time, which is why eval_every=10.
DFT_EVAL_SEED="${DFT_EVAL_SEED:-12345}"
DFT_EVAL_REPEATS="${DFT_EVAL_REPEATS:-4}"
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
echo "margins edge=$DFT_EDGE_MARGIN agent=$DFT_AGENT_MARGIN geom=$DFT_AGENT_GEOMETRY"
echo "sigma_floor=$DFT_SIGMA_FLOOR replan=$DFT_REPLAN max_replans=$DFT_MAX_REPLANS"
echo "grad_accum=$DFT_GRAD_ACCUM eval_seed=$DFT_EVAL_SEED" \
     "eval_repeats=$DFT_EVAL_REPEATS eval_base=$DFT_EVAL_BASE"
echo "eval_splits cap=$DFT_SPLIT_CAP chunk=$DFT_EVAL_CHUNK every=$DFT_EVAL_EVERY"
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
if ! (dd if=/dev/zero of="$DFT_OUT_DIR/.probe" bs=1M count=8 status=none && \
      rm -f "$DFT_OUT_DIR/.probe"); then
  echo "FATAL: cannot write 8 MB to $DFT_OUT_DIR (quota?)" >&2; exit 1
fi

ARGS=(--ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR"
      --iters "$DFT_ITERS" --epochs "$DFT_EPOCHS"
      --max_replans "$DFT_MAX_REPLANS" --replan_interval_steps "$DFT_REPLAN"
      --lr "$DFT_LR" --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN"
      --sigma_floor "$DFT_SIGMA_FLOOR"
      --unroll_steps "$DFT_UNROLL_STEPS" --unroll_grad_steps "$DFT_UNROLL_GRAD_STEPS"
      --unroll_t_start "$DFT_UNROLL_T_START" --unroll_t_end "$DFT_UNROLL_T_END"
      --unroll_eta "$DFT_UNROLL_ETA" --anchor_states "$DFT_ANCHOR_STATES"
      --safety_weight "$DFT_SAFETY_WEIGHT" --anchor_weight "$DFT_ANCHOR_WEIGHT"
      --bc_weight "$DFT_BC_WEIGHT" --base_anchor_weight "$DFT_BASE_ANCHOR_WEIGHT"
      --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN"
      --agent_geometry "$DFT_AGENT_GEOMETRY" --n_discs "$DFT_N_DISCS"
      --eval_every "$DFT_EVAL_EVERY" --save_every "$DFT_SAVE_EVERY"
      --keep_every "$DFT_KEEP_EVERY"
      --grad_accum "$DFT_GRAD_ACCUM"
      --eval_seed "$DFT_EVAL_SEED" --eval_repeats "$DFT_EVAL_REPEATS"
      --resume "$DFT_RESUME" --seed "$DFT_SEED")
[[ "$DFT_EVAL_BASE" == "0" ]] && ARGS+=(--no-eval_base) || ARGS+=(--eval_base)
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
