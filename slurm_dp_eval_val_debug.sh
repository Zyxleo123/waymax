#!/bin/bash
#SBATCH --job-name=dp_eval_dbg
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/world_model_sim/logs/dp_eval_dbg_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model_sim/logs/dp_eval_dbg_%j.err

# slurm_dp_eval_val.sh on the debug partition (2 jobs, 2 GPUs, 32 CPUs per user, 2 h): several validation shards
# in parallel on one GPU, one process per shard (~25 GB RAM, ~2 cores, 40-70 min each); pass --mem to fit them.
#
#   CKPT_NAME=... sbatch --mem=180G --nodelist=gpu22 slurm_dp_eval_val_debug.sh "0 1 2 3 4 5 6" <run_dir> [log|idm] [tag]
set -euo pipefail

SHARDS="${1:?usage: sbatch slurm_dp_eval_val_debug.sh \"<shard ids>\" <run_dir> [log|idm] [tag]}"
shift
LOG_DIR="/zfsauton/scratch/ishanj/world_model_sim/logs"

read -r -a shards <<< "${SHARDS}"
pids=()
for s in "${shards[@]}"; do
  SLURM_ARRAY_TASK_ID="${s}" bash /zfsauton/scratch/ishanj/waymax_rs/slurm_dp_eval_val.sh "$@" \
    > "${LOG_DIR}/dp_eval_dbg_${SLURM_JOB_ID}_${s}.out" 2> "${LOG_DIR}/dp_eval_dbg_${SLURM_JOB_ID}_${s}.err" &
  pids+=($!)
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "shard ${shards[$i]} ok"; else echo "shard ${shards[$i]} FAILED"; status=1; fi
done
exit "${status}"
