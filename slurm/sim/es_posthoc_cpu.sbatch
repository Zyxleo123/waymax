#!/bin/bash
#SBATCH --job-name=es_posthoc_cpu
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/es_posthoc/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/es_posthoc/slurm-%x-%j.out
#
# CPU side of the ES post-hoc experiment (slurm/sim/submit_es_posthoc.sh):
#   ES_CPU_MODE=prep     rebuild fail1500 from a finished run into shard groups
#                        (scripts/build_fail1500_from_run.py)
#   ES_CPU_MODE=prep_val one group per shard matching ES_VAL_GLOB (symlinks only)
#   ES_CPU_MODE=compare  paired arm comparison (scripts/compare_es_posthoc.py) --
#                        reads saved per-scene JSONs only; re-runnable any time.
# Config arrives as a positional env FILE. Never `sbatch --export`.
set -euo pipefail
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
ES_ROOT="${ES_ROOT:-/zfsauton2/home/yixiz/waymax_rs_es_posthoc}"
cd "$ES_ROOT"
PY="${ES_PYTHON:-/zfsauton/scratch/sbellad/conda_envs/waymax_rs/bin/python}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} mode=${ES_CPU_MODE:?set ES_CPU_MODE} (CPU job, no GPU)"
case "$ES_CPU_MODE" in
  prep)
    "$PY" scripts/build_fail1500_from_run.py --run_dir "${ES_SOURCE_RUN:?}" \
      --out "${ES_SUBSET_DIR:?}" --groups "${ES_GROUPS:-6}" ;;
  prep_val)
    # ES_VAL_PER_GROUP shards per group: fewer, longer (resumable) tasks keep the array
    # under the per-QOS queued-job cap without costing throughput (the CPU cap binds first).
    out="${ES_SUBSET_DIR:?}"; per="${ES_VAL_PER_GROUP:-1}"; i=0
    for f in $(compgen -G "${ES_VAL_GLOB:?}" | sort); do
      g=$((i / per)); d="$out/group_$g/tfrecords/validation"; mkdir -p "$d"; ln -sfn "$f" "$d/$(basename "$f")"
      echo "group_$g <- $(basename "$f")"; i=$((i + 1))
    done
    (( i > 0 )) || { echo "FATAL: no shards match $ES_VAL_GLOB" >&2; exit 1; }
    printf '{"glob": "%s", "n_shards": %d, "per_group": %d, "n_groups": %d}\n' "$ES_VAL_GLOB" "$i" "$per" "$(( (i + per - 1) / per ))" > "$out/manifest.json" ;;
  test_idm)
    JAX_PLATFORMS=cpu "$PY" scripts/test_idm_eval.py --n "${ES_TEST_N:-8}" ;;
  compare)
    "$PY" scripts/compare_es_posthoc.py --root "${ES_OUT:?}" --arms "${ES_ARMS:?}" \
      --pairs "${ES_PAIRS:-}" --expect "${ES_EXPECT:-0}" --out "${ES_OUT}/${ES_COMPARE_NAME:-compare}.json" ;;
  *) echo "FATAL: ES_CPU_MODE=$ES_CPU_MODE not prep|compare" >&2; exit 1 ;;
esac
