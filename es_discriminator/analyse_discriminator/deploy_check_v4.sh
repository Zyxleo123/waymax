#!/bin/bash
#SBATCH --job-name=disc_v4_deploy_check
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/zfsauton2/home/sbellad/baseline/waymax_rs
#SBATCH --output=/zfsauton/scratch/sbellad/es_discriminator/logs/disc_v4_deploy_check_%j.out
#SBATCH --error=/zfsauton/scratch/sbellad/es_discriminator/logs/disc_v4_deploy_check_%j.err

# CPU only, no simulation: deployment-path config check of A0 and A4 (disc_v4) and the online vs
# offline score consistency with top-1 / top-8 elite agreement (deploy_check_v4.py).
#
#   sbatch es_discriminator/analyse_discriminator/deploy_check_v4.sh
set -euo pipefail

ROOT=/zfsauton2/home/sbellad/baseline/waymax_rs
SCRATCH=/zfsauton/scratch/sbellad
OUT=${SCRATCH}/es_discriminator/analyse_discriminator

set --
set +u
source "${SCRATCH}/miniconda3/bin/activate" "${SCRATCH}/conda_envs/waymax_rs"
set -u
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3
cd "${ROOT}"
mkdir -p "${OUT}"

python -u -m es_discriminator.analyse_discriminator.deploy_check_v4 --max-files 2 \
  --out "${OUT}/disc_v4_deploy_check.json"
echo "done: ${OUT}/disc_v4_deploy_check.json"
