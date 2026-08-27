#!/bin/bash
#SBATCH --job-name=vmh_rescore
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/vmax_hardness/slurm-%x-%j.out
#
# Stage A: re-score a slice of the WOMD training shards with the PRETRAINED V-Max
# SAC (model_best), emitting per-scene JSONs with (shard, scenario_idx) provenance
# so the risk model can be refit on THIS policy's failures.
#
# a6000 (48G) matches the pretrained run's footprint; grows on demand so the
# waymax rollout coexists with the loaded model. Resumable: existing JSONs skip.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/vmax_hardness"
# shellcheck source=/dev/null
source "$ROOT/rl/vmax_hardness/slurm/_preamble.sh" "$@"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

# Gate: the warm-start checkpoint and the byte-offset indexes must exist.
if [[ ! -f "$VMH_BEST_CKPT" ]]; then
  echo "GATE: model_best checkpoint missing: $VMH_BEST_CKPT -- nothing to score." ; exit 0
fi
if ! compgen -G "$VMH_INDEX_DIR/*.offsets.npy" > /dev/null; then
  echo "GATE: no WOMD byte-offset indexes under $VMH_INDEX_DIR -- build them first." ; exit 0
fi

# Stage model_best as model_final so vmax get_model_path() selects it (it prefers
# model_final.pkl; the original run dir also holds the 25M-step final we do NOT want).
STAGE_DIR="$VMH_OUT/staged_best"
mkdir -p "$STAGE_DIR/model"
ln -sfn "$VMH_RUN_DIR/.hydra" "$STAGE_DIR/.hydra"
cp -f "$VMH_BEST_CKPT" "$STAGE_DIR/model/model_final.pkl"

echo "host=${SLURMD_NODENAME:-unknown} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "rescore: n_shards=$VMH_N_SHARDS per_shard=$VMH_PER_SHARD -> $VMH_RESCORE_DIR"

"$PYTHON" -m rl.vmax_hardness.rescore \
  --run-dir "$STAGE_DIR" \
  --shard-glob "$VMH_SHARD_GLOB" \
  --index-dir "$VMH_INDEX_DIR" \
  --out-dir "$VMH_RESCORE_DIR" \
  --n-shards "$VMH_N_SHARDS" \
  --per-shard "$VMH_PER_SHARD" \
  --max-num-objects "$VMH_MAX_NUM_OBJECTS" \
  --num-paths "$VMH_NUM_PATHS" \
  --num-points-per-path "$VMH_NUM_POINTS_PER_PATH" \
  --seed 0
