#!/bin/bash
#SBATCH --job-name=awm_ceil
#SBATCH --partition=debug
#SBATCH --qos=qos_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/rule_ceiling_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/rule_ceiling_%j.err
#
# Safety ranking of the scorer's rule features vs the learned scorer on bank val groups
# (anchor_wm/rule_ceiling.py).
#
#   CKPT=.../best.pt sbatch anchor_wm/slurm/rule_ceiling.sh
set -euo pipefail
source /zfsauton/scratch/ishanj/waymax_rs/anchor_wm/slurm/common.sh

SET="${SET:-fail1500_plus_womd_train_s02_11}"
V3_SET="/zfsauton/scratch/sbellad/es_discriminator/training_sets/${SET}"
python -m anchor_wm.rule_ceiling --ckpt "${CKPT}" --bank_dir "${V3_SET}/es_candidates" \
  --cache_dir "${BANK_CACHE:-${OUT_ROOT}/bank_cache/${SET}}" --split_json "${V3_SET}/scenario_split_seed0.json" \
  --invalid_json "${V3_SET}/invalid_at_start.json" --groups "${N_GROUPS:-400}" ${EXTRA_ARGS:-}
