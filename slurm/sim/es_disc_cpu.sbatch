#!/bin/bash
#SBATCH --job-name=es_nm_cpu
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/es_newmodel/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/es_newmodel/slurm-%x-%j.out
#
# CPU side of slurm/sim/submit_es_disc.sh:
#   ES_CPU_MODE=compare    paired arm comparison per evaluator view (scripts/compare_es_disc.py);
#                          reads saved per-scene JSONs only, re-runnable any time
#   ES_CPU_MODE=test_closed_loop  gate: es_WM closed-loop IDM moves NPCs and shows them to the
#                          planner (scripts/test_idm_closed_loop.py; --mem=64G --cpus-per-task=8)
#   ES_CPU_MODE=test_wm_isolation  gate: each WM package keeps its own data modules (scripts/test_wm_isolation.py)
#   ES_CPU_MODE=test_offroad_modes gate: --offroad_mode binary unchanged by the margin_tiebreak port
# Config arrives as a positional env FILE. Never `sbatch --export`.
set -euo pipefail

ES_COMMON="${SLURM_SUBMIT_DIR:-/nonexistent}/slurm/sim/es_disc_common.sh"
[[ -r "$ES_COMMON" ]] || ES_COMMON=/home/yixiz/waymax_rs_es_newmodel/slurm/sim/es_disc_common.sh
[[ -r "$ES_COMMON" ]] || { echo "FATAL: shared preamble $ES_COMMON missing" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ES_COMMON"
es_load_args "$@"
cd "$ES_ROOT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M mode=${ES_CPU_MODE:?set ES_CPU_MODE} (CPU job, no GPU)"
case "$ES_CPU_MODE" in
  compare)
    "$ES_PYTHON" scripts/compare_es_disc.py --root "${ES_OUT:?}" --arms "${ES_ARMS:?}" \
      --views "${ES_VIEWS:-log,idm}" --pairs "${ES_PAIRS:-}" --expect "${ES_EXPECT:-0}" \
      --split_json "${ES_SPLIT_JSON:-}" --out "${ES_OUT}/${ES_COMPARE_NAME:-compare}" ;;
  test_closed_loop)
    # gate for es_WM's closed-loop IDM evaluator (needs no policy, so it runs on CPU;
    # submit with --mem=64G --cpus-per-task=8)
    export PYTHONPATH="$ES_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$ES_PYTHON" scripts/test_idm_closed_loop.py --n "${ES_TEST_N:-5}" ;;
  test_wm_isolation)
    # gate: the multi-future WM keeps its own map features next to new_model's repo preprocess
    export PYTHONPATH="$ES_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$ES_PYTHON" scripts/test_wm_isolation.py ;;
  test_offroad_modes)
    # gate: the margin_tiebreak port leaves --offroad_mode binary unchanged
    export PYTHONPATH="$ES_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$ES_PYTHON" scripts/test_offroad_modes.py ;;
  *) echo "FATAL: ES_CPU_MODE=$ES_CPU_MODE not compare|test_closed_loop|test_wm_isolation|test_offroad_modes" >&2; exit 1 ;;
esac
