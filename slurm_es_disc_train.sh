#!/bin/bash
#SBATCH --job-name=es_disc_train
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --chdir=/zfsauton/scratch/ishanj/waymax_rs
#SBATCH --output=/zfsauton/scratch/ishanj/es_discriminator/logs/train_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/es_discriminator/logs/train_%j.err

# Trains the es_discriminator safety + progress discriminator (real_es/wm_model.py, full
# Gaussian WM branch) on the chunks written by slurm_es_disc_collect.sh: links every chunk
# into one training set (batches + labels, WM / road-edge caches, invalid-at-start scenes),
# trains with sbellad's settings (64 candidates per replan per step, model selection on the
# val split), writes the offline val / test reports, then checks that the deploy scorer
# (online WM forward) matches the cached inputs. The checkpoint records the WM it was
# trained with; run_simulation.py --discriminator_ckpt uses that WM by default.
#
#   sbatch slurm_es_disc_train.sh                         # after the collection finished
#   sbatch slurm_es_disc_train.sh --smoke                 # on the smoke collection
#   sbatch slurm_es_disc_train.sh -- --epochs 10          # trainer args after --
#
# Output: ${RUN_DIR}/train_set/ (links) and
#         /zfsauton/scratch/ishanj/es_discriminator/safety_progress/<run>_<timestamp>_<jobid>/
set -euo pipefail

WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt}"
OFFROAD_MODE="${OFFROAD_MODE:-binary}"
SEED=0
SMOKE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "ERROR: unknown argument '$1' (trainer args go after --)" >&2; exit 1 ;;
  esac
done

ROOT="/zfsauton/scratch/ishanj/waymax_rs"
DISC_ROOT="/zfsauton/scratch/ishanj/es_discriminator"
TFRECORDS="/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords/training"
WM_RUN="$(basename "$(dirname "${WORLD_MODEL_CKPT}")")"
WM_TAG="${WM_RUN#multiFuture_GaussianWorldModel_}_$(basename "${WORLD_MODEL_CKPT}" .pt)"
RUN_NAME="${RUN_NAME:-fail1500_es_diffusion_wm_mf_${WM_TAG}_expected_independent_off${OFFROAD_MODE}_seed${SEED}}"
[[ ${SMOKE} == 1 ]] && RUN_NAME="smoke_${RUN_NAME}"
RUN_DIR="${DISC_ROOT}/es_candidate_data/${RUN_NAME}"
SET="${RUN_DIR}/train_set"
OUT="${DISC_ROOT}/safety_progress/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}" "${DISC_ROOT}/logs"
exec > >(tee -a "${OUT}/train.log") 2>&1
ls -d "${RUN_DIR}"/chunk_* > /dev/null 2>&1 || { echo "ERROR: no chunks under ${RUN_DIR}; run slurm_es_disc_collect.sh" >&2; exit 1; }

set --  # conda's activate script reads "$@"
set +u  # conda activate.d hooks reference unset variables
# shellcheck source=/dev/null
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
log() { echo "[$(date)] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-local} run=${RUN_NAME} smoke=${SMOKE} commit=$(git -C "${ROOT}" rev-parse --short HEAD) out=${OUT}"

log "training set: link chunks into ${SET}"
python -u - "${RUN_DIR}" "${SET}" "${SMOKE}" <<'EOF'
import glob, json, os, shutil, sys
import numpy as np
from es_discriminator.analyze_es_decisions import batch_files
from es_discriminator.build_training_set import LABEL_SUFFIXES, invalid_from_scenes
from es_discriminator.real_es.dataset import SPLITS

run_dir, out, smoke = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
dirs = {k: os.path.join(out, k) for k in ("es_candidates", "wm_predictions", "road_edges")}
for d in dirs.values():
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
link = lambda src, d: os.symlink(os.path.abspath(src), os.path.join(d, os.path.basename(src)))  # noqa: E731
invalid, report, manifest, ids = [], [], None, []
for chunk in sorted(glob.glob(os.path.join(run_dir, "chunk_*"))):
    data, wm_dir, edge_dir = (os.path.join(chunk, k) for k in ("es_candidates", "wm_predictions", "road_edges"))
    n = n_incomplete = 0
    for f in batch_files(data):
        stem, base = f[:-4], os.path.basename(f)[:-4]
        wm, edges = os.path.join(wm_dir, base + ".wm.npz"), os.path.join(edge_dir, base + ".edges.npz")
        if not all(os.path.exists(stem + s) for s in LABEL_SUFFIXES) or not (os.path.exists(wm) and os.path.exists(edges)):
            n_incomplete += 1
            continue
        for p in (f, *(stem + s for s in LABEL_SUFFIXES)):
            link(p, dirs["es_candidates"])
        link(wm, dirs["wm_predictions"])
        link(edges, dirs["road_edges"])
        with np.load(f) as z:
            ids += z["scen_id"].astype(str).tolist()
        n += 1
    if n:
        if not os.path.exists(os.path.join(dirs["es_candidates"], "run_config.json")):
            link(os.path.join(data, "run_config.json"), dirs["es_candidates"])
        invalid += invalid_from_scenes(os.path.join(chunk, "analysis", "scenes.npz"))
        m = json.load(open(os.path.join(wm_dir, "manifest.json")))
        if manifest is None:
            manifest = m
            json.dump(m, open(os.path.join(dirs["wm_predictions"], "manifest.json"), "w"), indent=1)
        elif m["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
            raise ValueError(f"{chunk}: WM cache from a different checkpoint")
    report.append({"chunk": os.path.basename(chunk), "linked": n, "incomplete": n_incomplete})
if not ids:
    raise SystemExit("no complete batches in any chunk")
json.dump(invalid, open(os.path.join(out, "invalid_at_start.json"), "w"), indent=1)
split_path = os.path.join(out, "scenario_split_seed0.json")
if smoke:
    # A handful of scenes: round-robin so train / val / test are all non-empty.
    assignment = {sid: SPLITS[i % 3] for i, sid in enumerate(sorted(set(ids)))}
    json.dump({"seed": 0, "unit": "scenario_id", "assignment": assignment}, open(split_path, "w"))
elif os.path.exists(split_path):
    os.remove(split_path)  # recreated by the trainer (hash of the scenario id, same as sbellad's)
print(json.dumps({"chunks": report, "batch_files": sum(r["linked"] for r in report), "scenes": len(set(ids)),
                  "invalid_at_start": len(invalid), "wm_checkpoint": manifest["checkpoint"]}, indent=1))
EOF

DATA="${SET}/es_candidates"
ARGS=(--data-dir "${DATA}" --wm-dir "${SET}/wm_predictions" --edge-dir "${SET}/road_edges"
      --split "${SET}/scenario_split_seed0.json" --invalid-at-start "${SET}/invalid_at_start.json"
      --train-cands 64 --eval-chunk 64 --wandb-mode offline --wandb-name "${RUN_NAME}")
if [[ ${SMOKE} == 1 ]]; then
  ARGS+=(--epochs 1 --max-steps-per-epoch 10 --n-boot 50)
fi

log "train: es_discriminator.safety_progress.train"
python -u -m es_discriminator.safety_progress.train "${ARGS[@]}" --output-dir "${OUT}" ${EXTRA[@]+"${EXTRA[@]}"}

log "deploy-scorer consistency check (online WM forward vs cached inputs)"
JAX_PLATFORMS=cpu python -u -m es_discriminator.real_es.deploy --checkpoint "${OUT}/best.pt" \
  --data_dir "${DATA}" --tfrecord_dir "${TFRECORDS}" --wm_dir "${SET}/wm_predictions" --edge_dir "${SET}/road_edges" \
  --max_files $([[ ${SMOKE} == 1 ]] && echo 1 || echo 2) | tee "${OUT}/deploy_check.json"
log "done: ${OUT}  (best.pt, eval_val.md, eval_test.md)"
