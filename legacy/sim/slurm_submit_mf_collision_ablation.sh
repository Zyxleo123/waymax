#!/bin/bash
# Submit the MF collision-method ablation (multifuture WM epoch 450, fail1500).
#
#   mean_uniform              SAT on the 6 mode means, w = 1/K
#   mean_learned              SAT on the 6 mode means, w = p_k   (== existing expected_independent)
#   gaussian_uniform_M{4,8,16} SAT on M samples per mode,  w = 1/K
#   gaussian_learned_M{4,8,16} SAT on M samples per mode,  w = p_k
#
# mean_learned is skipped by default: it is the existing expected_independent run
# (Diffusion-ES 50.47%, FM-ES it4/rs20 48.20%). Set INCLUDE_MEAN_LEARNED=1 to re-run it.
#
# Usage (from rhea or lov4, NO --export):
#   bash slurm_submit_mf_collision_ablation.sh es             # Diffusion-ES, full 1500 scenes
#   bash slurm_submit_mf_collision_ablation.sh es_fm          # FM-ES (it4 / rs20 / pop 64 / elite 8)
#   bash slurm_submit_mf_collision_ablation.sh es es_fm       # both planners (same as: ... all)
#   SMOKE=1 bash slurm_submit_mf_collision_ablation.sh es     # 10-scene test runs (separate _smoke_n10 folder)
#   SEED=1 bash slurm_submit_mf_collision_ablation.sh es      # another seed -> seed_1/
#   VARIANTS="gaussian_learned_M08" bash slurm_submit_mf_collision_ablation.sh es   # pick variants
#
# Outputs:
#   es    -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation/<variant>/seed_<SEED>/
#   es_fm -> /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/mf_multiSampling_collision_ablation/<variant>/seed_<SEED>/
set -euo pipefail

PLANNERS=("$@")
[[ ${#PLANNERS[@]} -eq 0 ]] && PLANNERS=(es)
[[ "${PLANNERS[*]}" == "all" ]] && PLANNERS=(es es_fm)
for p in "${PLANNERS[@]}"; do
  if [[ "${p}" != "es" && "${p}" != "es_fm" ]]; then
    echo "Usage: bash slurm_submit_mf_collision_ablation.sh {es|es_fm|all} [es_fm]"
    exit 1
  fi
done

ROOT="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
SCRIPT="${ROOT}/legacy/sim/slurm_es_failure_subset_1500.sh"

export SEED="${SEED:-0}"
export POPULATION_SIZE=64 ELITE_SIZE=8 NUM_ES_ITERATIONS=4 RESAMPLE_NUM_STEPS=20
unset RUN_SUFFIX
if [[ "${SMOKE:-0}" == "1" ]]; then
  export NUM_SCENARIOS="${NUM_SCENARIOS:-10}"
  export MF_ABLATION_SUBDIR="_smoke_n${NUM_SCENARIOS}"
else
  unset NUM_SCENARIOS MF_ABLATION_SUBDIR
fi

DEFAULT_VARIANTS="mean_uniform gaussian_uniform_M04 gaussian_uniform_M08 gaussian_uniform_M16 gaussian_learned_M04 gaussian_learned_M08 gaussian_learned_M16"
[[ "${INCLUDE_MEAN_LEARNED:-0}" == "1" ]] && DEFAULT_VARIANTS="mean_learned ${DEFAULT_VARIANTS}"
VARIANTS="${VARIANTS:-${DEFAULT_VARIANTS}}"

for PLANNER in "${PLANNERS[@]}"; do
  if [[ "${PLANNER}" == "es_fm" ]]; then
    ABL_ROOT="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/mf_multiSampling_collision_ablation"
    PTAG="fm"
  else
    ABL_ROOT="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/mf_multiSampling_collision_ablation"
    PTAG="diff"
  fi
  [[ -n "${MF_ABLATION_SUBDIR:-}" ]] && ABL_ROOT="${ABL_ROOT}/${MF_ABLATION_SUBDIR}"
  SLURM_LOG_DIR="${ABL_ROOT}/slurm_logs"
  mkdir -p "${SLURM_LOG_DIR}"

  echo "MF collision ablation: planner=${PLANNER} seed=${SEED} smoke=${SMOKE:-0}"
  echo "  root: ${ABL_ROOT}"
  for v in ${VARIANTS}; do
    if [[ "${v}" =~ ^(gaussian_(uniform|learned))_M([0-9]+)$ ]]; then
      method="${BASH_REMATCH[1]}"
      m=$((10#${BASH_REMATCH[3]}))
    else
      method="${v}"
      m=8
    fi
    jid=$(MF_COLLISION_METHOD="${method}" MF_GAUSSIAN_NUM_SAMPLES="${m}" \
      sbatch --parsable --job-name="mfabl_${PTAG}_${v}" \
      --output="${SLURM_LOG_DIR}/${v}_seed${SEED}_%j.out" --error="${SLURM_LOG_DIR}/${v}_seed${SEED}_%j.err" \
      "${SCRIPT}" "${PLANNER}" wm_mf expected_independent)
    echo "  ${v} (method=${method}, M=${m}) -> job ${jid}"
  done
  echo
done

echo
squeue -u "${USER}" -o '%.10i %.36j %.2t %.10M %.20R'
