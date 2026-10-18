#!/bin/bash
#SBATCH --job-name=dft_compact
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Index-preserving compact copy of a scene list (tools/build_compact_shards.py): same shard
# basenames and record indices, non-target records replaced by empty frames. Then an
# independent TensorFlow re-read checks every shard. CPU only; resumable (done shards skipped).
#
# Default = the new goal-cond policy's 1568 WOMD validation failures (mineuih, 2026-10-18).
# Config: positional env-file or VAR=value arguments (DFT_COMPACT_TARGETS, DFT_COMPACT_SRC,
# DFT_COMPACT_NAME_FMT, DFT_COMPACT_OUT, DFT_COMPACT_SPLIT). Never `sbatch --export`.
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
TARGETS="${DFT_COMPACT_TARGETS:-/zfsauton/scratch/mineuih/waymax_rs/exp/validation_0_150_failure_scenarios.json}"
SRC="${DFT_COMPACT_SRC:-/zfsauton/datasets/womd/tf_example/validation}"
# (a '}' inside a ${VAR:-default} would end the expansion early, so no inline default here)
NAME_FMT="${DFT_COMPACT_NAME_FMT:-}"; [[ -n "$NAME_FMT" ]] || NAME_FMT='validation_tfexample.tfrecord-{:05d}-of-00150'
OUT="${DFT_COMPACT_OUT:-/zfsauton/scratch/yixiz/waymax_rs/womd_compact/valfail1568}"
SPLIT="${DFT_COMPACT_SPLIT:-validation}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
nvidia-smi -L 2>/dev/null || echo "gpu: none"
echo "repo=$ROOT @ $(git rev-parse --short HEAD) python=$PY"
echo "targets=$TARGETS src=$SRC name_fmt=$NAME_FMT out=$OUT split=$SPLIT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=

"$PY" -m rl.diffusion_ft.tools.build_compact_shards build --targets "$TARGETS" --src_dir "$SRC" \
  --name_fmt "$NAME_FMT" --out_root "$OUT" --split "$SPLIT" --workers "${SLURM_CPUS_PER_TASK:-8}"
"$PY" -m rl.diffusion_ft.tools.build_compact_shards verify --out_root "$OUT"
