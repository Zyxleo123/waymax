#!/bin/bash
#SBATCH --job-name=wm_disc
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --chdir=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/logs/wm_disc_%j.out
#SBATCH --error=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/logs/wm_disc_%j.err

# Safety + progress discriminator with the full Gaussian world model (real_es/wm_model.py) on
# the recorded fail1500 Diffusion-ES candidates: Waymax safety + progress pairs 80/20,
# stratified candidates, fp32, 64 candidates per replan per step, model selection on the
# validation split only, then offline val/test evaluation and the deploy-scorer consistency
# check (online WM forward vs cached inputs). Needs slurm/train/cache_wm_predictions.sh first.
# --method / --variant are optional and only accept safety_progress / full_gaussian.
# Run name: safety_progress_full_gaussian_wm. Submit from rhea or lov4:
#
#   sbatch slurm/train/train_wm_discriminator.sh --method safety_progress --variant full_gaussian
#   sbatch slurm/train/train_wm_discriminator.sh --smoke
#   sbatch slurm/train/train_wm_discriminator.sh -- --batch-groups 8   # trainer args
#
# --data NAME trains on a combined set built by es_discriminator.build_training_set
# (${SCRATCH}/es_discriminator/training_sets/NAME, with its own split and invalid list), e.g.
# fail1500 + the WOMD training shards from slurm/sim/collect_es_candidates_womd_train.sh:
#
#   python -m es_discriminator.build_training_set --name fail1500_plus_womd_train_s02_11   # on lov1
#   sbatch slurm/train/train_wm_discriminator.sh --data fail1500_plus_womd_train_s02_11
#
# Output: ${SCRATCH}/es_discriminator/safety_progress/<run_name>_<timestamp>_<jobid>/
set -euo pipefail

ROOT=/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad
RUN=fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed0
DATA=${SCRATCH}/es_discriminator/es_candidate_data/${RUN}/es_candidates
TFRECORDS=${SCRATCH}/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords/training
WM_DIR=${WM_DIR:-${SCRATCH}/es_discriminator/wm_predictions/${RUN}}
EDGE_DIR=${EDGE_DIR:-${SCRATCH}/es_discriminator/road_edges/${RUN}}
METHOD=safety_progress
VARIANT=full_gaussian
RUN_NAME=""
DATA_NAME=""
SMOKE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) [[ "$2" == safety_progress ]] || { echo "ERROR: only --method safety_progress is supported" >&2; exit 1; }; shift 2 ;;
    --variant) [[ "$2" == full_gaussian ]] || { echo "ERROR: only --variant full_gaussian is supported" >&2; exit 1; }; shift 2 ;;
    --data) DATA_NAME="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "ERROR: unknown argument '$1' (trainer args go after --)" >&2; exit 1 ;;
  esac
done
RUN_NAME=${RUN_NAME:-${METHOD}_${VARIANT}_wm${DATA_NAME:+_${DATA_NAME}}}
SET_ARGS=()
if [[ -n "${DATA_NAME}" ]]; then
  SET=${SCRATCH}/es_discriminator/training_sets/${DATA_NAME}
  [[ -d "${SET}/es_candidates" ]] || { echo "ERROR: ${SET} missing; run es_discriminator.build_training_set" >&2; exit 1; }
  DATA=${SET}/es_candidates WM_DIR=${SET}/wm_predictions EDGE_DIR=${SET}/road_edges
  TFRECORDS=/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training
  SET_ARGS=(--split "${SET}/scenario_split_seed0.json" --invalid-at-start "${SET}/invalid_at_start.json")
fi
ARGS=(--data-dir "${DATA}" --wm-dir "${WM_DIR}" --edge-dir "${EDGE_DIR}" "${SET_ARGS[@]}" --train-cands 64 --eval-chunk 64)
if [[ ${SMOKE} == 1 ]]; then
  RUN_NAME=${RUN_NAME}_smoke
  ARGS+=(--max-files 3 --epochs 1 --max-steps-per-epoch 10 --n-boot 50 --wandb-mode disabled)
else
  n_batch=$(find -L "${DATA}" -name '*.labels_waymax.npz' | wc -l)
  n_wm=$(find -L "${WM_DIR}" -name '*.wm.npz' | wc -l)
  n_edge=$(find -L "${EDGE_DIR}" -name '*.edges.npz' | wc -l)
  [[ ${n_batch} -gt 0 && ${n_wm} -eq ${n_batch} && ${n_edge} -eq ${n_batch} ]] || {
    echo "ERROR: ${n_batch} labelled batches but ${n_wm} WM / ${n_edge} edge caches" >&2; exit 1; }
fi
ARGS+=(--wandb-name "${RUN_NAME}")
OUT=${SCRATCH}/es_discriminator/${METHOD}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}
set --  # conda's activate script reads "$@"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/train.log") 2>&1

set +u  # conda activate.d hooks reference unset variables
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} method=${METHOD} variant=${VARIANT} out=${OUT}"

python -u -m es_discriminator.safety_progress.train "${ARGS[@]}" --output-dir "${OUT}" "${EXTRA[@]}"

echo "[$(date)] deploy-scorer consistency check (online WM forward vs cached inputs)"
JAX_PLATFORMS=cpu python -u -m es_discriminator.real_es.deploy --checkpoint "${OUT}/best.pt" \
  --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" --wm_dir "${WM_DIR}" --edge_dir "${EDGE_DIR}" \
  --max_files $([[ ${SMOKE} == 1 ]] && echo 1 || echo 2) | tee "${OUT}/deploy_check.json"
echo "[$(date)] done: ${OUT}  (eval_test.md, eval_val.md, best.pt)"
