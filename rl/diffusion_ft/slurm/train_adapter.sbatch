#!/bin/bash
#SBATCH --job-name=dft_adapter
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gated tail adapter (rl/diffusion_ft/train_adapter.py): preflight (identity +
# gradient), stages A / B / relabel / C, strength-0 re-check, strength sweep,
# latency bench. Writes <DFT_ADAPTER_OUT>/{preflight,verify,bench,summary}.json,
# train_log.jsonl and checkpoints adapter_{A,B,C*,final}/.
#
# GPU CLASS: `general` a6000/a5000 -- this env's jaxlib dies on v100, OOMs on 2080 Ti.
# Gate inside: a failed preflight or missing pairs prints why and exits 0 with no
# checkpoint, so the eval tasks behind it report "not run" instead of stranding.
#
# Config arrives as a positional env FILE (never `sbatch --export`).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
OUT="${DFT_ADAPTER_OUT:?set DFT_ADAPTER_OUT}"
PAIRS="${DFT_ADAPTER_PAIRS:?set DFT_ADAPTER_PAIRS (glob of pair_data npz)}"
DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
# Shape of the deployed sampler and the scorer -- must equal the collection's.
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_SIGMA_FLOOR="${DFT_SIGMA_FLOOR:-1e-4}"
DFT_EDGE_MARGIN="${DFT_EDGE_MARGIN:-0.3}"
DFT_AGENT_MARGIN="${DFT_AGENT_MARGIN:-0.75}"
DFT_WORLD_DT_METHOD="${DFT_WORLD_DT_METHOD:-median}"
# Adapter and pilot schedule (train_adapter.py --help for each).
DFT_ADAPTER_K="${DFT_ADAPTER_K:-5}"
DFT_ADAPTER_WIDTH="${DFT_ADAPTER_WIDTH:-32}"
DFT_TEACHER_ARM="${DFT_TEACHER_ARM:-o_lpA3}"
DFT_STAGE_A="${DFT_STAGE_A:-500}"
DFT_STAGE_B="${DFT_STAGE_B:-200}"
DFT_STAGE_C="${DFT_STAGE_C:-500}"
DFT_ADAPTER_LR="${DFT_ADAPTER_LR:-1e-4}"
DFT_KEEP_WEIGHT="${DFT_KEEP_WEIGHT:-1.0}"
DFT_REPAIR_WEIGHT="${DFT_REPAIR_WEIGHT:-1.0}"
DFT_GATE_WEIGHT="${DFT_GATE_WEIGHT:-0.1}"
DFT_VAL_EVERY="${DFT_VAL_EVERY:-100}"
DFT_ADAPTER_SEED="${DFT_ADAPTER_SEED:-0}"
DFT_ADAPTER_EXTRA="${DFT_ADAPTER_EXTRA:-}"

echo "host=${SLURMD_NODENAME:-unknown} ($(hostname)) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: unavailable"
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?}"
echo "pairs=$PAIRS"
echo "out=$OUT  K=$DFT_ADAPTER_K width=$DFT_ADAPTER_WIDTH teacher=$DFT_TEACHER_ARM"
echo "stages A/B/C=$DFT_STAGE_A/$DFT_STAGE_B/$DFT_STAGE_C lr=$DFT_ADAPTER_LR keep=$DFT_KEEP_WEIGHT repair=$DFT_REPAIR_WEIGHT gate=$DFT_GATE_WEIGHT"
mkdir -p "$OUT"
n_pairs=$(compgen -G "$PAIRS" | wc -l || true)
if [[ "$n_pairs" == 0 ]]; then
  echo "TRAINING NOT RUN: no pair files match '$PAIRS' -- the collection did not write any"
  echo "  (check the collect array's .out)"
  exit 0
fi
echo "pair files: $n_pairs"

# shellcheck disable=SC2086
srun "$PY" -u -m rl.diffusion_ft.train_adapter \
  --ckpt "$DFT_CKPT" --out_dir "$OUT" --pairs "$PAIRS" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" --splits_json "$DFT_SPLITS_JSON" \
  --eval_scenes 0 --prefetch 0 --probe_per_split 0 --no-wandb \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" --sigma_floor "$DFT_SIGMA_FLOOR" \
  --edge_margin "$DFT_EDGE_MARGIN" --agent_margin "$DFT_AGENT_MARGIN" \
  --sim_agents none --world_dt_method "$DFT_WORLD_DT_METHOD" \
  --teacher_arm "$DFT_TEACHER_ARM" --adapter_K "$DFT_ADAPTER_K" --adapter_width "$DFT_ADAPTER_WIDTH" \
  --stage_a_steps "$DFT_STAGE_A" --stage_b_steps "$DFT_STAGE_B" --stage_c_steps "$DFT_STAGE_C" \
  --adapter_lr "$DFT_ADAPTER_LR" --keep_weight "$DFT_KEEP_WEIGHT" \
  --adapter_repair_weight "$DFT_REPAIR_WEIGHT" --gate_weight "$DFT_GATE_WEIGHT" \
  --val_every "$DFT_VAL_EVERY" --seed "$DFT_ADAPTER_SEED" $DFT_ADAPTER_EXTRA
if [[ -s "$OUT/adapter_final/adapter.npz" ]]; then
  echo "checkpoint -> $OUT/adapter_final"
else
  echo "no final checkpoint; see $OUT/preflight.json"
fi
