#!/bin/bash
#SBATCH --job-name=mf_wm_per_anchor
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs/multiFuture_GaussianWorldModel
#SBATCH --output=/zfsauton/scratch/sbellad/world_model/eval_per_anchor/mf_wm_per_anchor_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/world_model/eval_per_anchor/mf_wm_per_anchor_%j.err

# Per-anchor (t=10..80) ADE/FDE of the multifuture WM vs constant velocity.
#   sbatch evaluation/slurm_evaluate_per_anchor.sh
#   MAX_SCENES=512 sbatch evaluation/slurm_evaluate_per_anchor.sh   # quicker
set -euo pipefail

ROOT="/zfsauton2/home/sbellad/baseline/waymax_rs"
WM_ROOT="${ROOT}/multiFuture_GaussianWorldModel"
OUT_DIR="/zfsauton/scratch/sbellad/world_model/eval_per_anchor"
CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt}"
mkdir -p "${OUT_DIR}"

set +u
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
cd "${WM_ROOT}"
export PYTHONPATH="${WM_ROOT}:${ROOT}"

python evaluation/evaluate_per_anchor.py \
  --cache_dir /zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz \
  --world_model_ckpt "${CKPT}" \
  --anchor_steps "${ANCHOR_STEPS:-10,20,30,40,50,60,70,80}" \
  --max_scenes "${MAX_SCENES:-1024}" \
  --out_json "${OUT_DIR}/mf_wm_$(basename "${CKPT}" .pt)_per_anchor_$(date +%Y%m%d_%H%M%S).json"
