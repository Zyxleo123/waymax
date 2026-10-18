#!/bin/bash
#SBATCH --job-name=dft_explabel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Expert log replay on every scene of a splits file, then a copy of the file
# restricted to expert-solvable scenes (tools/expert_label_splits.py). The env is
# built from the SAME flags select_eval.sbatch passes, so the label and the
# policy's `solved` are the same flags in the same env. Resumable.
#
# GPU CLASS: a6000 (the submitter excludes the a5000); this env dies at cuDNN
# autotune on v100 and OOMs on 2080 Tis.
#
# Config: a positional env file. Never `sbatch --export`.
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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_OUT_DIR="${DFT_OUT_DIR:?set DFT_OUT_DIR}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
DFT_LABEL_SPLITS="${DFT_LABEL_SPLITS:?set DFT_LABEL_SPLITS}"
DFT_CLEAN_SPLITS_OUT="${DFT_CLEAN_SPLITS_OUT:?set DFT_CLEAN_SPLITS_OUT}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:?set DFT_SHARD_GLOB}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
# Rollout shape = select_eval.sbatch's.
DFT_REPLAN="${DFT_REPLAN:-10}"; DFT_MAX_REPLANS="${DFT_MAX_REPLANS:-8}"
DFT_WORLD_DT_METHOD="${DFT_WORLD_DT_METHOD:-median}"
DFT_SIM_AGENTS="${DFT_SIM_AGENTS:-none}"

echo "host=${SLURMD_NODENAME:-unknown} ($(hostname)) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: unavailable"
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?}"
echo "splits=$DFT_SPLITS_JSON label=$DFT_LABEL_SPLITS -> $DFT_CLEAN_SPLITS_OUT"
echo "sim_agents=$DFT_SIM_AGENTS world_dt_method=$DFT_WORLD_DT_METHOD"
if [[ ! -s "$DFT_SPLITS_JSON" ]]; then
  echo "LABELLING NOT RUN: splits file '$DFT_SPLITS_JSON' does not exist"; exit 0
fi
mkdir -p "$DFT_OUT_DIR"

srun "$PY" -u -m rl.diffusion_ft.tools.expert_label_splits \
  --ckpt "$DFT_CKPT" --out_dir "$DFT_OUT_DIR" \
  --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --splits_json "$DFT_SPLITS_JSON" --label_splits "$DFT_LABEL_SPLITS" \
  --expert_rows "$DFT_OUT_DIR/expert_rows.jsonl" --clean_splits_out "$DFT_CLEAN_SPLITS_OUT" \
  --eval_scenes 0 --prefetch 0 --probe_per_split 0 \
  --k_trainable 10 --prefix_len 5 --sigma_floor 1e-4 \
  --replan_interval_steps "$DFT_REPLAN" --max_replans "$DFT_MAX_REPLANS" \
  --sim_agents "$DFT_SIM_AGENTS" --world_dt_method "$DFT_WORLD_DT_METHOD" \
  --edge_margin 0.3 --agent_margin 0.75 --seed 0 --no-wandb
