#!/bin/bash
#SBATCH --job-name=mf_gauss_wm
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs/multiFuture_GaussianWorldModel
#SBATCH --output=/zfsauton/scratch/ishanj/world_model/multifuture_gaussian_world_model_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/world_model/multifuture_gaussian_world_model_%j.err

set -euo pipefail

ROOT="${MULTIFUTURE_WM_ROOT:-/zfsauton/scratch/ishanj/waymax_rs/multiFuture_GaussianWorldModel}"
CACHE_DIR="${CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
SCRATCH_WM="${SCRATCH_WM:-/zfsauton/scratch/ishanj/world_model}"
SAVE_DIR="${SAVE_DIR:-${SCRATCH_WM}/ckpts_multifuture_gaussian}"
DEVICE="${WORLD_MODEL_TRAIN_DEVICE:-cuda}"

CONDA_ROOT="${CONDA_ROOT:-/zfsauton/scratch/ishanj/miniconda3}"
WAYMAX_ENV="${WAYMAX_ENV:-/zfsauton/scratch/ishanj/miniconda3/envs/waymax_rs}"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi

# Decoder A/B: DECODER_TYPE=direct|per_step, AGENT_FRAME_DECODING=1|0.
# Each run saves to SAVE_DIR/<WANDB_NAME>_<timestamp>/; use distinct WANDB_NAMEs per variant.
# MODE_EMBED_SCALE: init norm of the mode embeddings (unset = sqrt(hidden_dim), 1.0 = legacy).
DECODER_TYPE="${DECODER_TYPE:-direct}"
if [[ "${AGENT_FRAME_DECODING:-1}" == "1" ]]; then
  AGENT_FRAME_FLAG="--agent_frame_decoding"
else
  AGENT_FRAME_FLAG="--no-agent_frame_decoding"
fi

mkdir -p "${SCRATCH_WM}" "${SAVE_DIR}"

cd "${ROOT}"
# ROOT first so its data/ and model/ shadow the repo root's; repo root provides scores/.
export PYTHONPATH="${ROOT}:$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

echo "host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-}  python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python train_world_model.py \
  --cache_dir "${CACHE_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS:-500}" \
  --steps_per_epoch "${STEPS_PER_EPOCH:-2000}" \
  --val_steps "${VAL_STEPS:-100}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --anchor_steps "${ANCHOR_STEPS:-20,30,40}" \
  --history_steps "${HISTORY_STEPS:-10}" \
  --history_stride "${HISTORY_STRIDE:-2}" \
  --future_steps "${FUTURE_STEPS:-25}" \
  --future_stride "${FUTURE_STRIDE:-2}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --max_range "${MAX_RANGE:-50}" \
  --num_modes "${NUM_MODES:-6}" \
  --yaw_loss_weight "${YAW_LOSS_WEIGHT:-0.5}" \
  --mode_cls_weight "${MODE_CLS_WEIGHT:-1.0}" \
  --sigma_eps "${SIGMA_EPS:-1e-3}" \
  --decoder_type "${DECODER_TYPE}" \
  ${MODE_EMBED_SCALE:+--mode_embed_scale "${MODE_EMBED_SCALE}"} \
  "${AGENT_FRAME_FLAG}" \
  --planning_mode "${PLANNING_MODE:-top1}" \
  --wandb_project "${WANDB_PROJECT:-world Model}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_name "${WANDB_NAME:-multiFuture_GaussianWorldModel}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --amp
