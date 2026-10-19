#!/bin/bash
# FM-ES hyper-parameter probe on a fixed 250-scene subset of fail1500 (GT forecast).
#
#   bash slurm/sim/probe250.sh build                                   # once: build the subset
#   bash slurm/sim/probe250.sh --planner diffusion --forecast gt       # Diffusion-ES reference
#   bash slurm/sim/probe250.sh --planner fm --forecast gt --t_start 0.2,0.5,0.8
#   bash slurm/sim/probe250.sh --planner fm --forecast gt --t_start 0.20 --it 6 --rs 40 --tag_suffix it6_rs40
#   python scripts/summarize_fm_hparam_probe.py                         # summary table
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "build" ]]; then
  shift
  PROBE_N=250 PROBE_SEED=0 PROBE_SOURCE=fail1500
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --probe_n) PROBE_N="$2"; shift 2 ;;
      --probe_seed) PROBE_SEED="$2"; shift 2 ;;
      --source) PROBE_SOURCE="$2"; shift 2 ;;
      *) echo "ERROR: unknown build argument '$1'" >&2; exit 1 ;;
    esac
  done
  ROOT="${WAYMAX_RS_ROOT:-/data/user_data/yixiz/autonlab/zfsauton2/home/sbellad/baseline/waymax_rs}"
  OUT_DIR="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/Flow_matching_sim/fm_hparam_probe_n${PROBE_N}_seed${PROBE_SEED}"
  set --
  set +u
  # shellcheck source=/dev/null
  source /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/miniconda3/bin/activate /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs
  set -u
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  python "${ROOT}/scripts/build_fm_hparam_probe_subset.py" \
    --source "${PROBE_SOURCE}" --n "${PROBE_N}" --seed "${PROBE_SEED}" --num_worlds 1 \
    --tf_root /data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training \
    --out_dir "${OUT_DIR}"
  exit 0
fi

exec bash "${HERE}/_submit.sh" --subset probe250 "$@"
