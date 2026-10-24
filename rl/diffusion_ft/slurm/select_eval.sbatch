#!/bin/bash
#SBATCH --job-name=dft_select
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Training-free proposal selection (rl/diffusion_ft/proposal_select.py): every arm
# closed-loop on the same scenes and sampler seeds, one JSONL row per
# (scene, arm, repeat) plus per-replan selector features in feat/*.npz.
#
# Resumable: finished chunks are skipped, so a time-limit kill is a resubmit.
#
# GPU CLASS: `general` a6000 (a5000 excluded by the submitter). This env's jaxlib
# dies at cuDNN autotune on v100 and OOMs on 2080 Tis; selection arms also hold
# two candidates per replan, so do not widen the partition.
#
# Config arrives as a positional env FILE, or a DIRECTORY in array mode
# (<dir>/<task_id>.env). Never `sbatch --export`.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "FATAL: '$arg' is a dir but not an array job" >&2; exit 1; }
    TASK_ENV="$arg/${SLURM_ARRAY_TASK_ID}.env"
    [[ -f "$TASK_ENV" ]] || { echo "FATAL: no $TASK_ENV" >&2; exit 1; }
    echo "array task ${SLURM_ARRAY_TASK_ID} -> $TASK_ENV"
    # shellcheck disable=SC1090
    source "$TASK_ENV"
  elif [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
# A wm_* neighbour source runs a PyTorch world model on the same GPU: leave it
# room, or torch's first allocation fails behind JAX's 95% preallocation.
if [[ "${DFT_SELECT_BANK_NEIGHBORS:-}" == *"wm_"* ]]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
else
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
fi
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON (three-stratum splits file)}"
# /data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/ vanished ~2026-10-02; the same 1000 shards (offset index
# verified 10-03) live under /data/user_data/yixiz/autonlab/zfsauton/datasets/womd.
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SELECT_SPLITS="${DFT_SELECT_SPLITS:-failure_heldout,success_heldout,intermittent_heldout}"
DFT_FT_CKPT="${DFT_FT_CKPT:-}"
DFT_ARMS="${DFT_ARMS:?set DFT_ARMS}"
DFT_SELECT_SEED="${DFT_SELECT_SEED:-4242}"
DFT_SELECT_REPEATS="${DFT_SELECT_REPEATS:-4}"
DFT_SELECT_TOL_M="${DFT_SELECT_TOL_M:-1.0}"
DFT_SELECT_MARGIN_M="${DFT_SELECT_MARGIN_M:-0.3}"
DFT_SELECT_DELTA_M="${DFT_SELECT_DELTA_M:-0.2}"
DFT_SHARD_INDEX="${DFT_SHARD_INDEX:-0}"
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-1}"
DFT_EVAL_CHUNK="${DFT_EVAL_CHUNK:-32}"
DFT_SELECT_CAP="${DFT_SELECT_CAP:--1}"
DFT_SELECTOR_JSON="${DFT_SELECTOR_JSON:-}"
DFT_LOOK_REPLANS="${DFT_LOOK_REPLANS:-2}"
DFT_SELECT_BANK="${DFT_SELECT_BANK:-tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5}"
DFT_SELECT_RAMP="${DFT_SELECT_RAMP:-10}"
DFT_SELECT_BANKS="${DFT_SELECT_BANKS:-}"
# Per-bank ride-comfort knobs (peak lateral accel ~ |A|/T^2, jerk ~ |A|/T^3,
# T = ramp * world_dt). Per BANK so a comfort A/B is paired arms in one job.
DFT_SELECT_BANK_RAMPS="${DFT_SELECT_BANK_RAMPS:-}"
DFT_SELECT_BANK_BLENDS="${DFT_SELECT_BANK_BLENDS:-}"
# Per-bank source of the OTHER cars' next steps, 'name=log|cv|none'. Unnamed banks
# use `log`: the recorded future, which under log replay is an oracle.
DFT_SELECT_BANK_NEIGHBORS="${DFT_SELECT_BANK_NEIGHBORS:-}"
# Per-bank LOSS-OPTIMIZED lateral candidates (rl/diffusion_ft/lateral_opt.py),
# 'name=inits;name=inits' with inits from 0 | best | lat<m>, and the optimizer's
# config 'key=value,key=value'. Empty = no optimized candidates (every run before).
DFT_SELECT_BANK_OPTS="${DFT_SELECT_BANK_OPTS:-}"
DFT_SELECT_OPT_CFG="${DFT_SELECT_OPT_CFG:-}"
# Per-bank optimizer overrides on top of it ('name=k=v,k=v;name=...'), and a
# per-bank margin preference among clean candidates ('name=edge_m,agent_m').
DFT_SELECT_BANK_OPT_CFGS="${DFT_SELECT_BANK_OPT_CFGS:-}"
DFT_SELECT_BANK_PREFER="${DFT_SELECT_BANK_PREFER:-}"
# Dump executed states for the ride-quality figures (~51 kB/arm-repeat chunk).
DFT_DUMP_TRAJ="${DFT_DUMP_TRAJ:-0}"
# Dump (condition, executed plan) per alive replan to <out>/bc_data/<arm>/*.npz,
# the train_safety --repair_data format (~33 kB/row). *_train splits only.
DFT_DUMP_BC="${DFT_DUMP_BC:-0}"
# Dump REPLAYABLE base draws (prefix + per-row late noise) next to the base and
# executed plans, to <out>/pair_data/<arm>/*.npz -- train_adapter.py's input.
# *_train splits only.
DFT_DUMP_PAIRS="${DFT_DUMP_PAIRS:-0}"
# Gated tail adapter checkpoint (train_adapter.py) for 'name=adapter:<gate>[:s]' arms.
DFT_ADAPTER_CKPT="${DFT_ADAPTER_CKPT:-}"
# Named population-search configs for 'es@name' arms (rl/diffusion_ft/es_population.py),
# 'name=pop:16,iters:2,bank:<bank>,t:1,scale:0,nb:cv;...'. Empty = no ES arms.
DFT_SELECT_ES="${DFT_SELECT_ES:-}"
# Rollout shape MUST match the eval the fine-tunes were selected under.
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_REPLAN="${DFT_REPLAN:-10}"
DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_SEED="${DFT_SEED:-0}"
# `none` log-replays the other agents (every select result to date); `idm` makes
# them reactive. The `cons` certificate is a ONE-STEP-AHEAD prediction either
# way, but under log replay it is exact (the neighbours cannot deviate from what
# it read) and under `idm` it is genuinely a prediction. Re-scoring a shipping
# arm here is how the load-bearing assumption gets tested; the result is not
# comparable to a `none` run and needs its own base arm in the same job.
DFT_SIM_AGENTS="${DFT_SIM_AGENTS:-none}"
# `commanded` shows the policy the velocity it WROTE for its own executed steps
# (every select result before 2026-09-23); `realised` shows the executed
# positions' backward difference. It changes what the policy is conditioned on,
# so compare across it only with `base` run under each.
DFT_EGO_VEL_SOURCE="${DFT_EGO_VEL_SOURCE:-commanded}"
# How the plan's world grid is read off the log timestamps. `first` (every select
# result before 2026-09-24) runs 44/173 failure_heldout scenes at ~0.75x pace;
# `median` is the fix. See data.preprocess.compute_world_dt_seconds. The
# submitter writes `median` explicitly since 2026-09-24; this fallback stays
# `first` only so an OLDER env file without the key, requeued, finishes on the
# method it started with.
DFT_WORLD_DT_METHOD="${DFT_WORLD_DT_METHOD:-first}"
# Optional upstream gate: a verdict file whose first line must read PASS
# (test_ego_vel.sbatch writes one). Empty = no gate.
DFT_REQUIRE_GATE="${DFT_REQUIRE_GATE:-}"
# World-model checkpoints / device for wm_* bank neighbours (empty = the defaults
# in rl/diffusion_ft/world_model_forecast.py).
DFT_WM_DET_CKPT="${DFT_WM_DET_CKPT:-}"
DFT_WM_MF_CKPT="${DFT_WM_MF_CKPT:-}"
DFT_WM_DEVICE="${DFT_WM_DEVICE:-}"

echo "host=${SLURMD_NODENAME:-unknown} ($(hostname)) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: unavailable"
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "arms=$DFT_ARMS"
echo "ft_ckpt=${DFT_FT_CKPT:-<none>} shard=$DFT_SHARD_INDEX/$DFT_NUM_SHARDS chunk=$DFT_EVAL_CHUNK"
echo "splits=$DFT_SELECT_SPLITS seed=$DFT_SELECT_SEED repeats=$DFT_SELECT_REPEATS"
echo "sim_agents=$DFT_SIM_AGENTS (overlap under idm is NOT the same metric as under none)"
echo "ego_vel_source=$DFT_EGO_VEL_SOURCE world_dt_method=$DFT_WORLD_DT_METHOD"
if [[ "$DFT_ARMS" == *"bank"* ]]; then
  echo "bank=$DFT_SELECT_BANK ramp=$DFT_SELECT_RAMP"
  if [[ -n "$DFT_SELECT_BANKS" ]]; then echo "banks=$DFT_SELECT_BANKS"; fi
  if [[ -n "$DFT_SELECT_BANK_RAMPS" ]]; then echo "bank_ramps=$DFT_SELECT_BANK_RAMPS"; fi
  if [[ -n "$DFT_SELECT_BANK_BLENDS" ]]; then echo "bank_blends=$DFT_SELECT_BANK_BLENDS"; fi
  if [[ -n "$DFT_SELECT_BANK_NEIGHBORS" ]]; then echo "bank_neighbors=$DFT_SELECT_BANK_NEIGHBORS"; fi
  if [[ "$DFT_SELECT_BANK_NEIGHBORS" == *"wm_"* ]]; then
    echo "wm det=${DFT_WM_DET_CKPT:-<default>} mf=${DFT_WM_MF_CKPT:-<default>} device=${DFT_WM_DEVICE:-auto} xla_mem=$XLA_PYTHON_CLIENT_MEM_FRACTION"
  fi
  if [[ -n "$DFT_SELECT_BANK_OPTS" ]]; then
    echo "bank_opts=$DFT_SELECT_BANK_OPTS opt_cfg=${DFT_SELECT_OPT_CFG:-<defaults>}"
    echo "bank_opt_cfgs=${DFT_SELECT_BANK_OPT_CFGS:-<none>}"
  fi
  if [[ -n "$DFT_SELECT_BANK_PREFER" ]]; then echo "bank_prefer=$DFT_SELECT_BANK_PREFER"
  fi
fi
if [[ -n "$DFT_SELECT_ES" ]]; then echo "es=$DFT_SELECT_ES"; fi
if [[ -n "$DFT_ADAPTER_CKPT" ]]; then echo "adapter_ckpt=$DFT_ADAPTER_CKPT"; fi
if [[ "$DFT_DUMP_PAIRS" == "1" ]]; then echo "dump_pairs=1"; fi
echo "shard_glob=$DFT_SHARD_GLOB"
mkdir -p "$DFT_OUT_DIR"
# Gate: the splits file may be written by an upstream job (DFT_SPLITS_UPSTREAM).
if [[ ! -s "$DFT_SPLITS_JSON" ]]; then
  echo "SELECTION NOT RUN: splits file '$DFT_SPLITS_JSON' does not exist -- the job"
  echo "  that writes it never ran or failed; check its .out"
  exit 0
fi
# Gate: a 'learned' arm with no selector means the fit job gated out (no labels).
# Say so and exit 0 so nothing downstream strands on DependencyNeverSatisfied.
# (adapter arms' 'adapter:learned' is the adapter's gate mode, not this selector.)
ARMS_NO_ADAPTER="$(printf ',%s,' "$DFT_ARMS" | sed -E 's/,[^,=]*=adapter:[^,]*/,/g')"
if [[ "$ARMS_NO_ADAPTER" == *":learned"* && ! -s "$DFT_SELECTOR_JSON" ]]; then
  echo "SELECTION NOT RUN: arms use 'learned' but no selector at '${DFT_SELECTOR_JSON}'"
  echo "  (see the fit_selector job's .out: it exits 0 without a file when labels are missing)"
  exit 0
fi

# Gate: an arm needs a fine-tuned checkpoint that an upstream training job was
# supposed to write. Missing = that job did not get there; say so, exit 0.
if [[ -n "$DFT_FT_CKPT" && ",$DFT_ARMS," =~ (,ft,|=ft:) && ! -s "$DFT_FT_CKPT/tparams.npz" ]]; then
  echo "SELECTION NOT RUN: no checkpoint at $DFT_FT_CKPT/tparams.npz -- the training"
  echo "  run that should have written it stopped short; check its .out"
  exit 0
fi
# Gate: an adapter arm needs the checkpoint train_adapter.py was supposed to write.
if [[ "$DFT_ARMS" == *"=adapter:"* && ! -s "${DFT_ADAPTER_CKPT}/adapter.npz" ]]; then
  echo "SELECTION NOT RUN: no adapter checkpoint at '${DFT_ADAPTER_CKPT}/adapter.npz' -- the"
  echo "  training job stopped short (its preflight.json / .out says why)"
  exit 0
fi
# Gate: an upstream decision job must have said PASS. Distinguish "it said no"
# from "it never ran", and exit 0 either way so nothing strands.
if [[ -n "$DFT_REQUIRE_GATE" ]]; then
  if [[ ! -s "$DFT_REQUIRE_GATE" ]]; then
    echo "SELECTION NOT RUN: gate verdict '$DFT_REQUIRE_GATE' does not exist -- the gate"
    echo "  job never ran or was killed before writing it; check its .out"
    exit 0
  fi
  verdict="$(head -1 "$DFT_REQUIRE_GATE")"
  if [[ "$verdict" != "PASS" ]]; then
    echo "SELECTION NOT RUN: gate '$DFT_REQUIRE_GATE' says: $verdict"
    exit 0
  fi
  echo "gate $DFT_REQUIRE_GATE: PASS"
fi

srun "$PY" -u -m rl.diffusion_ft.proposal_select \
  --ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --splits_json "$DFT_SPLITS_JSON" --splits "$DFT_SELECT_SPLITS" \
  --ft_ckpt "$DFT_FT_CKPT" --arms "$DFT_ARMS" \
  --select_seed "$DFT_SELECT_SEED" --select_repeats "$DFT_SELECT_REPEATS" \
  --select_tol_m "$DFT_SELECT_TOL_M" --select_margin_m "$DFT_SELECT_MARGIN_M" \
  --select_delta_m "$DFT_SELECT_DELTA_M" \
  --shard_index "$DFT_SHARD_INDEX" --num_shards "$DFT_NUM_SHARDS" \
  --select_cap="$DFT_SELECT_CAP" \
  --selector_json "$DFT_SELECTOR_JSON" --look_replans "$DFT_LOOK_REPLANS" \
  --select_bank "$DFT_SELECT_BANK" --select_ramp_steps "$DFT_SELECT_RAMP" \
  --select_banks "$DFT_SELECT_BANKS" \
  --select_bank_ramps "$DFT_SELECT_BANK_RAMPS" \
  --select_bank_blends "$DFT_SELECT_BANK_BLENDS" \
  --select_bank_neighbors "$DFT_SELECT_BANK_NEIGHBORS" \
  --select_bank_opts "$DFT_SELECT_BANK_OPTS" --select_opt_cfg "$DFT_SELECT_OPT_CFG" \
  --select_bank_opt_cfgs "$DFT_SELECT_BANK_OPT_CFGS" --select_bank_prefer "$DFT_SELECT_BANK_PREFER" \
  --select_es "$DFT_SELECT_ES" \
  --wm_det_ckpt "$DFT_WM_DET_CKPT" --wm_mf_ckpt "$DFT_WM_MF_CKPT" --wm_device "$DFT_WM_DEVICE" \
  ${DFT_DUMP_TRAJ:+$([[ "$DFT_DUMP_TRAJ" == "1" ]] && echo --dump_traj)} \
  ${DFT_DUMP_BC:+$([[ "$DFT_DUMP_BC" == "1" ]] && echo --dump_bc)} \
  ${DFT_DUMP_PAIRS:+$([[ "$DFT_DUMP_PAIRS" == "1" ]] && echo --dump_pairs)} \
  --adapter_ckpt "$DFT_ADAPTER_CKPT" \
  --eval_chunk "$DFT_EVAL_CHUNK" --eval_scenes 0 --prefetch 0 --probe_per_split 0 \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --sigma_floor "$DFT_SIGMA_FLOOR" \
  --replan_interval_steps "$DFT_REPLAN" --max_replans "$DFT_MAX_REPLANS" \
  --sim_agents "$DFT_SIM_AGENTS" \
  --ego_vel_source "$DFT_EGO_VEL_SOURCE" --world_dt_method "$DFT_WORLD_DT_METHOD" \
  --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN" \
  --seed "$DFT_SEED" --no-wandb
echo "rows -> $DFT_OUT_DIR/rows.jsonl"
