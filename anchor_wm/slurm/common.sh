# Shared paths / environment for the AnchorWM slurm scripts (sourced, not submitted).
ROOT="${ROOT:-/zfsauton/scratch/ishanj/waymax_rs}"
OUT_ROOT="${OUT_ROOT:-/zfsauton/scratch/ishanj/anchor_wm}"
CACHE_DIR="${CACHE_DIR:-/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz}"
# B0 = the WM + discriminator pair of the current pipeline (this WM also warm-starts the encoder).
INIT_CKPT="${INIT_CKPT:-/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt}"
B0_DIR="${B0_DIR:-/zfsauton/scratch/ishanj/es_discriminator/safety_progress/fail1500_es_diffusion_wm_mf_20261005_092950_epoch_0450_expected_independent_offbinary_seed0_20261017_065318_86702}"
BANK_DIR="${BANK_DIR:-/zfsauton/scratch/ishanj/es_discriminator/es_candidate_data/fail1500_es_diffusion_wm_mf_20261005_092950_epoch_0450_expected_independent_offbinary_seed0/train_set/es_candidates}"
TFRECORDS="${TFRECORDS:-/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0/tfrecords/training}"
ANCHORS="${ANCHORS:-${OUT_ROOT}/anchors/anchors_k64.npz}"
WANDB_PROJECT="${WANDB_PROJECT:-anchor_wm}"
WANDB_ENTITY="${WANDB_ENTITY:-scbellad2025-carnegie-mellon-university}"
WANDB_MODE="${WANDB_MODE:-online}"

CONDA_ROOT="${CONDA_ROOT:-/zfsauton/scratch/ishanj/miniconda3}"
WAYMAX_ENV="${WAYMAX_ENV:-/zfsauton/scratch/ishanj/miniconda3/envs/waymax_rs}"
if [[ "${CONDA_PREFIX:-}" != "${WAYMAX_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/bin/activate" "${WAYMAX_ENV}"
  set -u
fi
# Repo root only: anchor_wm and es_discriminator are imported as packages; the MF package's
# top-level data/ and model/ stay off the path.
export PYTHONPATH="${ROOT}"
# JAX only reads tfrecords / rebuilds replan states / computes reference Waymax labels here.
# On the GPU it holds on to the memory it grows to and starves PyTorch, so keep it on the CPU.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${OUT_ROOT}/logs"
cd "${ROOT}"
echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} python=$(command -v python) job=${SLURM_JOB_ID:-local}"
