#!/bin/bash
#SBATCH --job-name=label_es_waymax
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/es_discriminator/logs/label_es_waymax_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/es_discriminator/logs/label_es_waymax_%j.err

# Prepares the recorded fail1500 Diffusion-ES candidates for discriminator training:
#   1. Waymax-exact overlap/offroad labels for all candidates and the logged expert
#      -> <batch>.labels_waymax.npz (resumable; already-labelled batches are skipped)
#   2. time/frame/context alignment check on 12 batches (fails the job on mismatch)
#   3. the scenario split used by safety_progress, and a check that
#      the ES-unsafe scene rule reproduces the decision report's 285 scenes
# Submit from rhea or lov4:  sbatch slurm/train/label_es_waymax.sh
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
DATA=${SCRATCH}/es_discriminator/es_candidate_data/fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed0/es_candidates
TFRECORDS=${SCRATCH}/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords/training
OUT=${SCRATCH}/es_discriminator/real_es_labels
SPLIT=${SCRATCH}/es_discriminator/splits/fail1500_scenario_split_seed0.json
set --  # conda's activate script reads "$@"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/job_${SLURM_JOB_ID:-local}.log") 2>&1

set +u  # conda activate.d hooks reference unset variables
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} out=${OUT}"

echo "[$(date)] 1/3 Waymax labels"
python -u -m es_discriminator.real_es.waymax_labels --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
  --summary_out "${OUT}/waymax_labels_summary.json"

echo "[$(date)] 2/3 alignment check"
python -u -m es_discriminator.real_es.check_alignment --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" \
  --max_files 12 --out "${OUT}/alignment.json"

echo "[$(date)] 3/3 scenario split + ES-unsafe scene check"
python -u - "${DATA}" "${SPLIT}" <<'EOF'
import json, sys
import numpy as np
from es_discriminator.real_es.dataset import (all_scenario_ids, es_unsafe_scenes, invalid_scene_keys, load_bank,
                                              load_or_create_split)
data, split_path = sys.argv[1:3]
split = load_or_create_split(split_path, all_scenario_ids(data))
bank = load_bank(data, verbose=False)
mask, _ = es_unsafe_scenes(bank, invalid_scene_keys())
counts = {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}
in_test = int(sum(split[s] == "test" for s in bank.scenes["scen_id"][mask]))
print(json.dumps({"batch_files": len(bank.files), "scenes_per_split": counts,
                  "es_unsafe_scenes": int(mask.sum()), "es_unsafe_scenes_in_test": in_test}))
if len(bank.files) != 300 or int(mask.sum()) != 285:
    raise SystemExit("expected 300 labelled batch files and 285 ES-unsafe scenes")
EOF
echo "[$(date)] done: ${OUT}"
