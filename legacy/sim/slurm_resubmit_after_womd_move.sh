#!/bin/bash
# Resubmit every fail1500 run that broke on Oct 2 16:54 when
# /zfsauton/scratch/eshau/womd disappeared, plus a Diffusion-ES GT sanity rerun.
#
# The subset tfrecords are symlinks; they now point to
#   /zfsauton/datasets/womd/tf_example/training
# Old partial outputs were renamed to INVALID_womd_missing_simulation_*.
#
# Usage (from rhea or lov4):
#   bash slurm_resubmit_after_womd_move.sh            # submit all 6 jobs
#   EXCLUDE_NODES=gpuXX,gpuYY bash slurm_resubmit_after_womd_move.sh retry
#                                                     # only FM det/mf WM + GT recheck
#   bash slurm_resubmit_after_womd_move.sh compare    # after the GT job finishes
#
# Jobs:
#   3  Diffusion-ES MF collision ablation  -> immediate_exp/mf_multiSampling_collision_ablation/
#      (mean_uniform, gaussian_uniform_M08, gaussian_learned_M08)
#   2  FM-ES it6/rs40 det WM + mf WM       -> Flow_matching_sim/es_fm_wm_*_it6_rs40_p64_e8/
#   1  Diffusion-ES GT recheck             -> immediate_exp/_womd_path_recheck/es_diffusion_gt_fail1500/
#      compared scene-by-scene against immediate_exp/es_diffusion_gt_fail1500/ (Sep 9, 52.27%)
set -euo pipefail

ROOT="${WAYMAX_RS_ROOT:-/zfsauton2/home/sbellad/baseline/waymax_rs}"
SUBSET_DIR="/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0"
NEW_WOMD="/zfsauton/datasets/womd/tf_example/training"
RECHECK_ROOT="/zfsauton/scratch/sbellad/immediate_exp/_womd_path_recheck"
GT_REF_DIR="/zfsauton/scratch/sbellad/immediate_exp/es_diffusion_gt_fail1500"
GT_NEW_DIR="${RECHECK_ROOT}/es_diffusion_gt_fail1500"
MODE="${1:-submit}"

if [[ "${MODE}" == "compare" ]]; then
  source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
  python - "${GT_REF_DIR}" "${GT_NEW_DIR}" <<'EOF'
import glob, json, os, sys
ref_dir = sorted(glob.glob(f"{sys.argv[1]}/simulation_*"))[-1]
new_dirs = sorted(glob.glob(f"{sys.argv[2]}/simulation_*"))
if not new_dirs:
    sys.exit(f"no recheck output under {sys.argv[2]} yet")
new_dir = new_dirs[-1]
load = lambda d: {os.path.basename(f): json.load(open(f)) for f in glob.glob(f"{d}/*.scenario_*.json")}
ref, new = load(ref_dir), load(new_dir)
common = sorted(set(ref) & set(new))
keys = ["success", "overlap", "offroad", "goal_reached"]
print(f"reference: {ref_dir} ({len(ref)} scenes)")
print(f"recheck:   {new_dir} ({len(new)} scenes)")
print(f"common scenes: {len(common)}")
for k in keys:
    same = sum(ref[s][k] == new[s][k] for s in common)
    r = sum(bool(ref[s][k]) for s in common); n = sum(bool(new[s][k]) for s in common)
    print(f"  {k:13s} identical {same}/{len(common)}   ref={r}  recheck={n}")
goal_same = sum(ref[s]["goal_xy"] == new[s]["goal_xy"] and ref[s]["ego_idx"] == new[s]["ego_idx"] for s in common)
print(f"  goal_xy+ego_idx identical {goal_same}/{len(common)}  (data identity)")
flips = [s for s in common if ref[s]["success"] != new[s]["success"]]
print(f"success flips: {len(flips)}  e.g. {flips[:5]}")
EOF
  exit 0
fi

if [[ "${MODE}" != "submit" && "${MODE}" != "retry" ]]; then
  echo "Usage: bash slurm_resubmit_after_womd_move.sh [submit|retry|compare]"
  exit 1
fi
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
  export SBATCH_EXCLUDE="${EXCLUDE_NODES}"
  echo "Excluding nodes: ${SBATCH_EXCLUDE}"
fi

for l in "${SUBSET_DIR}"/tfrecords/training/*; do
  if [[ "$(readlink "${l}")" != "${NEW_WOMD}/$(basename "${l}")" || ! -s "${l}" ]]; then
    echo "ERROR: ${l} does not resolve into ${NEW_WOMD}; aborting"
    exit 1
  fi
done
echo "Subset symlinks OK -> ${NEW_WOMD}"
echo

if [[ "${MODE}" == "submit" ]]; then
  echo "### MF collision ablation (Diffusion-ES)"
  VARIANTS="mean_uniform gaussian_uniform_M08 gaussian_learned_M08" \
    bash "${ROOT}/legacy/sim/slurm_submit_mf_collision_ablation.sh" es
  echo
fi

echo "### FM-ES it6/rs40 det WM + mf WM"
bash "${ROOT}/legacy/sim/slurm_submit_es_fm_fail1500_cv_wm_it6_rs40.sh"
echo

echo "### Diffusion-ES GT recheck"
mkdir -p "${RECHECK_ROOT}/logs"
unset RUN_SUFFIX SEED NUM_SCENARIOS POPULATION_SIZE ELITE_SIZE NUM_ES_ITERATIONS RESAMPLE_NUM_STEPS MF_COLLISION_METHOD MF_ABLATION_SUBDIR
jid=$(CV_GT_SIM_ROOT="${RECHECK_ROOT}" sbatch --parsable --job-name=recheck_es_diff_gt \
  --output="${RECHECK_ROOT}/logs/es_diffusion_gt_fail1500_%j.out" --error="${RECHECK_ROOT}/logs/es_diffusion_gt_fail1500_%j.err" \
  "${ROOT}/legacy/sim/slurm_es_failure_subset_1500.sh" es gt)
echo "  GT recheck -> job ${jid}  (${GT_NEW_DIR})"
echo "  when done: bash slurm_resubmit_after_womd_move.sh compare"
echo
squeue -u "${USER}" -o '%.10i %.36j %.2t %.10M %.20R'
