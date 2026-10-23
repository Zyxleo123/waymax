#!/bin/bash
#SBATCH --job-name=awm_ft
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/mineuih/ishan/logs/awm_ft_%j.out
#SBATCH --error=/home/mineuih/ishan/logs/awm_ft_%j.err
#
# Scorer fine-tuning (anchor_wm/finetune.py) on babel, on an ES bank collected with --label_in_loop
# (slurm_collect_anchorwm_bank.sh): BANK/es_candidates + BANK/anchor_cache + BANK/scenario_split_seed0.json.
# A v2 checkpoint needs --rule_features --input_heading (finetune.py takes those two from the flags; time
# remaining and near-agent modes come from the checkpoint).
#
#   sbatch anchor_wm/slurm/babel/finetune.sh
#   INIT=.../final.pt BANK=... RUN_NAME=ft_x EXTRA_ARGS="--unfreeze_decoder" sbatch anchor_wm/slurm/babel/finetune.sh
#   SMOKE=1 bash anchor_wm/slurm/babel/finetune.sh        (60 steps, inside an allocation)
#   BANKS="<bank root> <bank root> ..." sbatch ...         (aggregate several banks, each with the layout above)
set -euo pipefail
HERE="${ROOT:-/home/mineuih/ishan/waymax_rs}/anchor_wm/slurm/babel"
source "${HERE}/env.sh"
source "${ROOT}/anchor_wm/slurm/common.sh"

INIT="${INIT:-${OUT_ROOT}/runs/v2_scratch_joint_simple_fix55k_20261021_012534/final.pt}"
BANK="${BANK:-/data/user_data/mineuih/ishan/anchorwm_bank/es_v2_scratch_joint_simple_fix55k_final_seed0_train00002_00003_00004_00005}"
read -r -a BANKS <<< "${BANKS:-${BANK}}"
SCORER_LOSS="${SCORER_LOSS:-simple}"
RUN_NAME="${RUN_NAME:-ft_v2simple_${SCORER_LOSS}}"
ARGS=(--max_steps "${MAX_STEPS:-30000}" --max_hours "${MAX_HOURS:-7}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="smoke_${RUN_NAME}"; ARGS=(--max_steps 60 --val_every 30 --val_groups 40 --log_every 10)
  WANDB_MODE=disabled
fi
BANK_ARGS=(--bank_dir) CACHE_ARGS=(--cache_dir) SPLIT_ARGS=(--split_json)
for b in "${BANKS[@]}"; do
  BANK_ARGS+=("${b}/es_candidates") CACHE_ARGS+=("${b}/anchor_cache") SPLIT_ARGS+=("${b}/scenario_split_seed0.json")
done
# A bank without a scenario split gets one (70/15/15 by scenario-id hash, seed 0, as the v1 fine-tune's banks).
for b in "${BANKS[@]}"; do
  [[ -e "${b}/scenario_split_seed0.json" || ! -d "${b}/es_candidates" ]] && continue
  python - "${b}" <<'EOF'
import collections, sys
import numpy as np
from anchor_wm.bank_data import _batch_files
from es_discriminator.real_es.dataset import load_or_create_split
bank = sys.argv[1]
ids = [s for f in _batch_files(bank + "/es_candidates") for s in np.load(f)["scen_id"].astype(str)]
split = load_or_create_split(bank + "/scenario_split_seed0.json", ids, seed=0)
print(f"created {bank}/scenario_split_seed0.json: {dict(collections.Counter(split.values()))}")
EOF
done
for p in "${INIT}" "${CACHE_DIR}" "${BANK_ARGS[@]:1}" "${CACHE_ARGS[@]:1}" "${SPLIT_ARGS[@]:1}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done
python -m anchor_wm.finetune --init_ckpt "${INIT}" "${BANK_ARGS[@]}" "${CACHE_ARGS[@]}" "${SPLIT_ARGS[@]}" \
  --self_cache_dir "${CACHE_DIR}" \
  --save_dir "${OUT_ROOT}/finetune" --run_name "${RUN_NAME}" --num_workers "${NUM_WORKERS:-12}" \
  --rule_features --input_heading --scorer_loss "${SCORER_LOSS}" \
  --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}" --wandb_mode "${WANDB_MODE}" \
  ${RESUME:+--resume "${RESUME}"} "${ARGS[@]}" ${EXTRA_ARGS:-}
echo "results in $(ls -td "${OUT_ROOT}/finetune/${RUN_NAME}"_2*/ | head -1)"
