#!/bin/bash
#SBATCH --job-name=verify_womd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/verify_womd-%j.out
#SBATCH --error=logs/verify_womd-%j.out
#
# Is this WOMD copy byte-compatible with the cached offset indexes (and so with every
# split JSON's scene ids)? Checks training (expect 1000 shards) and validation shards
# 0-9 against womd_offsets/ and env/womd_sizes.tsv. CPU only, reads ~3 headers per shard.
#
# No partition/qos line: those are per cluster. From the repo root, after `mkdir -p logs`:
#   sbatch -p <cpu partition> env/verify_womd.sbatch NEWROOT=/abs/new/root
# Config: positional VAR=value or env-file arguments (NEWROOT, PY, WOMD_DIR, INDEX_DIR).
# NEWROOT empty = the autonlab paths themselves.
set -euo pipefail
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
NEWROOT="${NEWROOT:-}"
ROOT="${SLURM_SUBMIT_DIR:-$NEWROOT/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
PY="${PY:-$NEWROOT/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
WOMD_DIR="${WOMD_DIR:-$NEWROOT/zfsauton/datasets/womd/tf_example}"
INDEX_DIR="${INDEX_DIR:-$NEWROOT/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
nvidia-smi -L 2>/dev/null || echo "gpu: none"
echo "repo=$ROOT @ $(git rev-parse --short HEAD) python=$PY"
echo "womd=$WOMD_DIR index=$INDEX_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1

rc=0
"$PY" -m rl.diffusion_ft.tools.verify_womd_offsets --glob "$WOMD_DIR/training/training_tfexample.tfrecord-*" \
  --index_dir "$INDEX_DIR" --sizes env/womd_sizes.tsv --expect 1000 || rc=1
"$PY" -m rl.diffusion_ft.tools.verify_womd_offsets \
  --glob "$WOMD_DIR/validation/validation_tfexample.tfrecord-0000[0-9]-of-00150" \
  --index_dir "$INDEX_DIR" --sizes env/womd_sizes.tsv --expect 10 || rc=1
echo "verify_womd: $([[ $rc == 0 ]] && echo PASS || echo FAIL)"
exit "$rc"
