#!/bin/bash
#SBATCH --job-name=es_posthoc
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/es_posthoc/slurm-%x-%A_%a.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/es_posthoc/slurm-%x-%A_%a.out
#
# One Diffusion-ES arm on one shard group of fail1500 (planner/diffusion_es_planner.py),
# optionally with every candidate repaired by the o_am post-hoc optimizer before the
# elites are chosen (planner/posthoc_repair.py). Submitted as an array by
# slurm/sim/submit_es_posthoc.sh; each task's config is <env dir>/<task id>.env.
#
# RESUMABLE: results go to $ES_OUT/<arm>/group_<g>/ via --start_from, and finished
# scenes are skipped, so a time limit or requeue is a resubmit, not a restart.
#
# GPU: general (a6000 / a5000). The WM arm shares the card between torch and JAX.
# Config arrives as a positional env FILE or array DIR. Never `sbatch --export`.
set -euo pipefail

for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "FATAL: '$arg' is a dir but not an array job" >&2; exit 1; }
    TASK_ENV="$arg/${SLURM_ARRAY_TASK_ID}.env"
    [[ -f "$TASK_ENV" ]] || { echo "FATAL: no $TASK_ENV" >&2; exit 1; }
    echo "array task ${SLURM_ARRAY_TASK_ID} -> $TASK_ENV"
    # shellcheck disable=SC1090
    source "$TASK_ENV"
  elif [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

ES_ROOT="${ES_ROOT:-/zfsauton2/home/yixiz/waymax_rs_es_posthoc}"
cd "$ES_ROOT"
PY="${ES_PYTHON:-/zfsauton/scratch/sbellad/conda_envs/waymax_rs/bin/python}"
ES_OUT="${ES_OUT:?set ES_OUT}"
ES_ARM="${ES_ARM:?set ES_ARM}"
ES_GROUP_DIR="${ES_GROUP_DIR:?set ES_GROUP_DIR (a build_fail1500_from_run.py group)}"
ES_FORECAST="${ES_FORECAST:?set ES_FORECAST (cv|gt|wm_mf)}"
# es = Diffusion-ES; diffusion = the policy alone (one sample per replan, no search).
ES_PLANNER="${ES_PLANNER:-es}"
# Also score every scene with these non-ego agent models (same executed trajectory;
# planning always sees logged agents): written as <agents>_overlap/_offroad/_success.
ES_ALSO_EVAL="${ES_ALSO_EVAL:-idm}"
# Per-scene world step: "first" = es_WM's original first-pair dt (default, so an older
# env file without the key reproduces its run); "median" = the fix. The submitter
# writes it explicitly.
ES_WORLD_DT_METHOD="${ES_WORLD_DT_METHOD:-first}"
# 1 = also write each scene's executed ego trajectory (<scene>.trajectory.npz, ~2 kB), so a
# later evaluator change can be re-scored without replanning. Default 0 (older env files).
ES_SAVE_TRAJ="${ES_SAVE_TRAJ:-0}"
# training = fail1500 groups (with a target_scenarios.json); validation = whole shards.
ES_SPLIT="${ES_SPLIT:-training}"
ES_POSTHOC="${ES_POSTHOC:-0}"
ES_DIFFUSION_CKPT="${ES_DIFFUSION_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
ES_MF_WM_CKPT="${ES_MF_WM_CKPT:-/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/latest.pt}"
ES_LANE_GRAPH_DIR="${ES_LANE_GRAPH_DIR:-/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"
ES_NUM_SCENARIOS="${ES_NUM_SCENARIOS:-100000}"
ES_NUM_WORLDS="${ES_NUM_WORLDS:-5}"
ES_SEED="${ES_SEED:-0}"
ES_POP="${ES_POP:-64}"
ES_ELITE="${ES_ELITE:-8}"
ES_IT="${ES_IT:-4}"
ES_RESAMPLE_TIMESTEPS="${ES_RESAMPLE_TIMESTEPS:-5}"
ES_POSTHOC_WINDOW="${ES_POSTHOC_WINDOW:-20}"
ES_POSTHOC_CFG="${ES_POSTHOC_CFG:-}"

echo "host=${SLURMD_NODENAME:-unknown} ($(hostname)) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: unavailable"
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M job=${SLURM_JOB_ID:-?} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "repo=$ES_ROOT @ $(git -C "$ES_ROOT" rev-parse --short HEAD) dirty=$(git -C "$ES_ROOT" status --porcelain | wc -l) python=$PY"
echo "planner=$ES_PLANNER also_eval=${ES_ALSO_EVAL:-<none>} world_dt=$ES_WORLD_DT_METHOD save_traj=$ES_SAVE_TRAJ"
SAVE=(); [[ "$ES_SAVE_TRAJ" == 1 ]] && SAVE=(--save_trajectory)
echo "arm=$ES_ARM forecast=$ES_FORECAST posthoc=$ES_POSTHOC window=$ES_POSTHOC_WINDOW cfg=${ES_POSTHOC_CFG:-<o_am>}"
echo "group=$ES_GROUP_DIR scenes<=$ES_NUM_SCENARIOS worlds=$ES_NUM_WORLDS seed=$ES_SEED pop=$ES_POP elite=$ES_ELITE it=$ES_IT rt=$ES_RESAMPLE_TIMESTEPS"
echo "policy=$ES_DIFFUSION_CKPT"

# Gate: every shard of this group must be readable on this node (a node without the
# /zfsauton/datasets mount would otherwise finish with 0 scenes and exit 0).
n_bad=0
for f in "$ES_GROUP_DIR"/tfrecords/"$ES_SPLIT"/*; do [[ -r "$f" ]] || { echo "unreadable: $f"; n_bad=$((n_bad+1)); }; done
if (( n_bad > 0 )); then echo "FATAL: $n_bad shards unreadable on $(hostname)" >&2; exit 1; fi

case "$ES_FORECAST" in
  cv) FC=(--other_agent_forecast cv) ;;
  gt) FC=(--other_agent_forecast gt) ;;
  wm_mf)
    echo "wm_mf=$ES_MF_WM_CKPT (expected_independent)"
    FC=(--other_agent_forecast wm --world_model_ckpt "$ES_MF_WM_CKPT" --world_model_device auto
        --wm_backend multifuture --wm_refresh_steps 10 --num_collision_worlds 1
        --collision_reduce any --collision_sample_seed 0 --wm_planning_mode expected_independent) ;;
  *) echo "FATAL: ES_FORECAST=$ES_FORECAST not cv|gt|wm_mf" >&2; exit 1 ;;
esac

# Without a target list the runner scores every scene of every shard in the group.
TARGET=()
[[ -s "$ES_GROUP_DIR/target_scenarios.json" ]] && TARGET=(--target_scenario_path "$ES_GROUP_DIR/target_scenarios.json")
echo "split=$ES_SPLIT target=${TARGET[1]:-<all scenes>}"
OUT_DIR="$ES_OUT/$ES_ARM/$(basename "$ES_GROUP_DIR")"
mkdir -p "$OUT_DIR"
echo "out=$OUT_DIR (done so far: $(find "$OUT_DIR" -name '*.scenario_*.json' | wc -l))"
export PYTHONPATH="$ES_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
"$PY" -u simulation/run_simulation.py \
  --planner "$ES_PLANNER" --also_eval_sim_agents "$ES_ALSO_EVAL" --world_dt_method "$ES_WORLD_DT_METHOD" --diffusion_ckpt_path "$ES_DIFFUSION_CKPT" --resample_timesteps "$ES_RESAMPLE_TIMESTEPS" \
  --tfrecord_dir "$ES_GROUP_DIR/tfrecords" --split "$ES_SPLIT" ${TARGET[@]+"${TARGET[@]}"} \
  --lane_graph_dir "$ES_LANE_GRAPH_DIR" \
  --exp_dir "$ES_OUT/$ES_ARM" --start_from "$OUT_DIR" --tag "$ES_ARM" \
  --num_worlds "$ES_NUM_WORLDS" --num_scenarios "$ES_NUM_SCENARIOS" --seed "$ES_SEED" \
  --population_size "$ES_POP" --elite_size "$ES_ELITE" --num_es_iterations "$ES_IT" \
  --replan_interval_steps 10 --visualize_mode none \
  "${FC[@]}" \
  --posthoc_repair "$ES_POSTHOC" --posthoc_window "$ES_POSTHOC_WINDOW" --posthoc_cfg "$ES_POSTHOC_CFG" \
  ${SAVE[@]+"${SAVE[@]}"}
n_done=$(find "$OUT_DIR" -name '*.scenario_*.json' | wc -l)
echo "done: $n_done scene results in $OUT_DIR"
if (( n_done == 0 )); then echo "FATAL: no scene results written" >&2; exit 1; fi
