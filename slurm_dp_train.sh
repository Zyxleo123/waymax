#!/bin/bash
#SBATCH --job-name=dp_train
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton/scratch/ishanj/Diffusion-Planner
#SBATCH --output=/zfsauton/scratch/ishanj/diffusion_planner_womd/logs/train_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/diffusion_planner_womd/logs/train_%j.err

# Diffusion Planner training on WOMD, single A6000. Their recipe (train_predictor.py defaults: EMA 0.999,
# state-perturbation augmentation, 5 warm-up epochs + cosine decay, 32 agents / 70 lanes / 25 route lanes)
# except:
#   - batch 256 (their per-GPU batch; 2048 does not fit one A6000) and lr 1.8e-4 = 5e-4 * sqrt(256 / 2048);
#   - 66 epochs over ~974k samples = ~64M samples, the training budget of the waymax_rs diffusion policy
#     (1000 x 1000 steps x batch 64), ~38 h at the measured 465 samples/s;
#   - future_len=50 (5 s, the waymax_rs policy horizon).
# Runs through diffusion_planner_womd/train.py (upstream-bug and environment workarounds, see its docstring).
#
# Merges the per-shard data lists on first start. Each job resumes from the newest run of RUN_NAME, so a
# second job chained with --dependency=afterany covers the 2-day limit; it exits at once if training finished.
#
# GOAL_CONDITIONED=1: the goal-conditioned variant (diffusion_planner_womd/goal.py, run name ..._goal), which
# needs the goals from slurm_dp_goals.sh.
set -euo pipefail

DP_REPO="/zfsauton/scratch/ishanj/Diffusion-Planner"  # github.com/ZhengYinan-AIR/Diffusion-Planner @ a3a621f
ROOT="/zfsauton/scratch/ishanj/waymax_rs"
DATA_ROOT="${DATA_ROOT:-/zfsauton/scratch/ishanj/diffusion_planner_womd/data}"
SAVE_DIR="${SAVE_DIR:-/zfsauton/scratch/ishanj/diffusion_planner_womd/runs}"
GOAL_CONDITIONED="${GOAL_CONDITIONED:-0}"
if [[ "${GOAL_CONDITIONED}" == 1 ]]; then
  RUN_NAME="${RUN_NAME:-dp_womd_h50_1gpu_goal}"
else
  RUN_NAME="${RUN_NAME:-dp_womd_h50_1gpu}"
fi
NUM_SHARDS=1000

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/ishanj/miniconda3/bin/activate /zfsauton/scratch/ishanj/miniconda3/envs/waymax_rs
set -u
export PYTHONPATH="${DP_REPO}:${ROOT}/diffusion_planner_womd/nuplan_shim" WANDB_MODE=offline

[[ "$(git -C "${DP_REPO}" rev-parse HEAD)" == a3a621f* ]] || { echo "ERROR: ${DP_REPO} is not at a3a621f"; exit 1; }
if [[ ! -f "${DATA_ROOT}/train_list.json" ]]; then
  n="$(ls "${DATA_ROOT}/lists" 2>/dev/null | wc -l)"
  [[ "${n}" -eq "${NUM_SHARDS}" ]] || { echo "ERROR: ${n}/${NUM_SHARDS} shards converted"; exit 1; }
  python "${ROOT}/diffusion_planner_womd/merge_lists.py" "${DATA_ROOT}"
fi

GOAL_ARGS=()
if [[ "${GOAL_CONDITIONED}" == 1 ]]; then
  n="$(ls "${DATA_ROOT}/goals" 2>/dev/null | wc -l)"
  [[ "${n}" -eq "${NUM_SHARDS}" ]] || { echo "ERROR: goals for ${n}/${NUM_SHARDS} shards"; exit 1; }
  GOAL_ARGS=(--goal_conditioned --goal_dir "${DATA_ROOT}/goals")
fi

RESUME=()
LATEST="$(ls -1dt "${SAVE_DIR}/training_log/${RUN_NAME}"/*/latest.pth 2>/dev/null | head -1 || true)"
if [[ -n "${LATEST}" ]]; then
  RESUME=(--resume_model_path "$(dirname "${LATEST}")")
  echo "resuming from $(dirname "${LATEST}")"
fi

torchrun --standalone --nnodes 1 --nproc-per-node 1 "${ROOT}/diffusion_planner_womd/train.py" \
  --name "${RUN_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --train_set "${DATA_ROOT}" \
  --train_set_list "${DATA_ROOT}/train_list.json" \
  --time_len 21 \
  --future_len 50 \
  --batch_size 256 \
  --learning_rate 1.8e-4 \
  --train_epochs 66 \
  --save_utd 5 \
  --num_workers 14 \
  ${GOAL_ARGS[@]+"${GOAL_ARGS[@]}"} \
  ${RESUME[@]+"${RESUME[@]}"}
