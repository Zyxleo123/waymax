#!/bin/bash
#SBATCH --job-name=cmp_closed_loop
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/closed_loop/logs/compare_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/closed_loop/logs/compare_%j.err
# Pair the learned-reward closed-loop runs with the matching handcrafted Diffusion-ES baselines.
#   bash slurm/train/compare_closed_loop.sh [--disc_name N] [--allow_partial]   (runs on CPU; also fine on lov1)
set -eo pipefail
SCRATCH=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad
ROOT=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
DISC_NAME=safety_progress_full_gaussian_wm
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --disc_name) DISC_NAME="$2"; shift 2 ;;
    --allow_partial) EXTRA+=(--allow_partial); shift ;;
    *) echo "unknown flag $1" >&2; exit 1 ;;
  esac
done
CL=${SCRATCH}/es_discriminator/closed_loop
DES=${SCRATCH}/Diffusion_ES_sim
WMS=${SCRATCH}/world_model_sim
MF=es_diffusion_wm_mf_epoch0450_expected_independent

latest() { ls -d "$1"/$2 2>/dev/null | sort | tail -1; }
disc() {   # run name, split prefix(es) -> comma-separated simulation dirs
  local name=$1; shift; local out=() d
  for p in "$@"; do
    d=$(latest "${CL}/${name}_disc_${DISC_NAME}" "simulation_es_${p}_*")
    [[ -n "$d" ]] || { echo "missing ${p} run for ${name}_disc_${DISC_NAME}" >&2; exit 1; }
    out+=("$d")
  done
  (IFS=,; echo "${out[*]}")
}

PAIRS=(
  "fail1500/gt=$(disc es_diffusion_gt_fail1500 training)::${DES}/es_diffusion_gt_fail1500/simulation_es_training_es_diffusion_gt_fail1500_20260909_153617"
  "fail1500/cv=$(disc es_diffusion_cv_fail1500 training)::${DES}/es_diffusion_cv_fail1500/simulation_es_training_es_diffusion_cv_fail1500_20260909_152755"
  "fail1500/wm=$(disc ${MF}_fail1500 training)::${WMS}/${MF}_fail1500/simulation_es_training_${MF}_fail1500_20260919_170306"
  "trainval/gt=$(disc es_diffusion_gt training validation)::${DES}/es_diffusion_gt/simulation_es_training_es_diffusion_gt_TRAIN_tfrecord00_01_20260907_102456,${DES}/es_diffusion_gt/simulation_es_validation_es_diffusion_gt_VAL_tfrecord00_01_20260907_203053"
  "trainval/cv=$(disc es_diffusion_cv training validation)::${DES}/es_diffusion_cv/simulation_es_training_es_diffusion_cv_TRAIN_tfrecord00_01_20260907_102432,${DES}/es_diffusion_cv/simulation_es_validation_es_diffusion_cv_VAL_tfrecord00_01_20260907_171131"
  "trainval/wm=$(disc ${MF}_r10 training validation)::${WMS}/${MF}_r10/simulation_es_training_${MF}_r10_TRAIN_tfrecord00_01_20260920_104149,${WMS}/${MF}_r10/simulation_es_validation_${MF}_r10_VAL_tfrecord00_01_20260920_173605"
)

set +u; source ${SCRATCH}/miniconda3/bin/activate ${SCRATCH}/conda_envs/waymax_rs
export PYTHONPATH=${ROOT} CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3
cd "${ROOT}"
args=()
for p in "${PAIRS[@]}"; do args+=(--pair "$p"); done
python -m es_discriminator.real_es.compare_closed_loop "${args[@]}" "${EXTRA[@]}" \
  --out_dir "${CL}/comparison_${DISC_NAME}_$(date +%Y%m%d_%H%M%S)"
