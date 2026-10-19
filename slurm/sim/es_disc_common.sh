#!/bin/bash
# Shared preamble of slurm/sim/es_disc.sbatch, es_disc_cpu.sbatch and submit_es_disc.sh:
# the default of every ES_* key, the config loader, and es_run_one (one arm x view x
# shard group). Sourced by ABSOLUTE path: Slurm runs a spool copy of the batch script,
# which cannot find files next to itself.
#
# Keys (the submitter writes every one explicitly into a per-task env file):
#   ES_OUT          run root; results go to $ES_OUT/$ES_VIEW/$ES_ARM/<group>/ (resumable)
#   ES_ARM          arm name (directory only; behaviour comes from the keys below)
#   ES_VIEW         log = NPCs replay the log; idm = es_WM's closed-loop IDM
#                   (simulation/idm_closed_loop.py: the planner sees the simulated NPCs)
#   ES_PLANNER      es (Diffusion-ES) | diffusion (the policy alone, one sample per replan)
#   ES_FORECAST     cv | gt | wm_mf  (feeds the ES scorer AND the repair's neighbours)
#   ES_POSTHOC      1 = o_am repair of every candidate before scoring (planner/posthoc_repair.py)
#   ES_DISC_CKPT    non-empty = es_WM's learned discriminator replaces the handcrafted ES reward
#   ES_DISC_WM_CKPT the frozen MF world model whose forecasts the discriminator was trained on
#   ES_ALSO_EVAL    extra evaluation-only NPC models for the log view (empty under idm)
#   ES_GROUP_DIR / ES_SPLIT / ES_NUM_SCENARIOS / ES_WORLD_DT_METHOD / ES_SAVE_TRAJ / ES_SEED
#   ES_NUM_WORLDS / ES_POP / ES_ELITE / ES_IT / ES_RESAMPLE_TIMESTEPS / ES_REPLAN_INTERVAL

ES_ROOT="${ES_ROOT:-/home/yixiz/waymax_rs_es_disc}"
ES_PYTHON="${ES_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/conda_envs/waymax_rs/bin/python}"
ES_LOG_DIR="${ES_LOG_DIR:-/home/yixiz/waymax_rs/logs/es_disc}"
ES_DEFAULT_POLICY=/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest
ES_DEFAULT_MF_WM=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/latest.pt
ES_DEFAULT_DISC_WM=/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt

# es_load_args ARG...: an env FILE is sourced, an array DIR resolves to <dir>/<task>.env,
# VAR=value is exported. Config never travels by `sbatch --export` (it strands jobs here).
es_load_args() {
  local arg f
  for arg in "$@"; do
    if [[ -d "$arg" ]]; then
      [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "FATAL: '$arg' is a dir but not an array job" >&2; return 1; }
      f="$arg/${SLURM_ARRAY_TASK_ID}.env"
      [[ -f "$f" ]] || { echo "FATAL: no $f" >&2; return 1; }
      echo "array task ${SLURM_ARRAY_TASK_ID} -> $f"
      # shellcheck disable=SC1090
      source "$f"
    elif [[ -f "$arg" ]]; then
      # shellcheck disable=SC1090
      source "$arg"
    elif [[ "$arg" == *=* ]]; then
      export "${arg?}"
    fi
  done
}

# es_monitor: every ES_MON_SEC seconds, GPU utilisation/memory, per-process GPU memory and
# the host RSS of each simulation process, into the job's .out (sacct is dead here, so
# this is the only record of what a task really used).
es_monitor() {
  while sleep "${ES_MON_SEC:-300}"; do
    echo "[mon $(date +%H:%M:%S)] gpu util,used,total: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/[mon]   gpu app /'
    local pid
    for pid in $(pgrep -u "$(id -un)" -f run_simulation.py); do
      # only this job's processes (other jobs of mine may share the node)
      tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "SLURM_JOB_ID=${SLURM_JOB_ID:-none}" || continue
      awk -v p="$pid" '/^VmRSS/ {printf "[mon]   rss pid %s %.1f GB\n", p, $2 / 1048576}' "/proc/$pid/status" 2>/dev/null
    done
  done
}

# es_run_one: one arm x view x group from the ES_* keys in the environment.
es_run_one() {
  local out="${ES_OUT:?set ES_OUT}" arm="${ES_ARM:?set ES_ARM}" view="${ES_VIEW:-log}"
  local group_dir="${ES_GROUP_DIR:?set ES_GROUP_DIR}" forecast="${ES_FORECAST:?set ES_FORECAST}"
  local planner="${ES_PLANNER:-es}" split="${ES_SPLIT:-training}" posthoc="${ES_POSTHOC:-0}"
  local also_eval="${ES_ALSO_EVAL-}" dt_method="${ES_WORLD_DT_METHOD:-median}"
  local policy="${ES_DIFFUSION_CKPT:-$ES_DEFAULT_POLICY}" mf_wm="${ES_MF_WM_CKPT:-$ES_DEFAULT_MF_WM}"
  local disc="${ES_DISC_CKPT-}" disc_wm="${ES_DISC_WM_CKPT:-$ES_DEFAULT_DISC_WM}"
  case "$view" in log|idm) ;; *) echo "FATAL: ES_VIEW=$view not log|idm" >&2; return 1 ;; esac
  if [[ "$view" == idm && -n "$also_eval" ]]; then
    echo "FATAL: ES_ALSO_EVAL=$also_eval is evaluation-only; the idm view is already closed loop" >&2; return 1
  fi

  cd "$ES_ROOT"
  echo "repo=$ES_ROOT @ $(git -C "$ES_ROOT" rev-parse --short HEAD) dirty=$(git -C "$ES_ROOT" status --porcelain | wc -l) python=$ES_PYTHON"
  echo "arm=$arm view=$view planner=$planner forecast=$forecast posthoc=$posthoc disc=${disc:-<handcrafted reward>}"
  echo "group=$group_dir split=$split scenes<=${ES_NUM_SCENARIOS:-100000} world_dt=$dt_method also_eval=${also_eval:-<none>} save_traj=${ES_SAVE_TRAJ:-1}"
  echo "es: worlds=${ES_NUM_WORLDS:-5} pop=${ES_POP:-64} elite=${ES_ELITE:-8} it=${ES_IT:-4} rt=${ES_RESAMPLE_TIMESTEPS:-5} replan=${ES_REPLAN_INTERVAL:-10} seed=${ES_SEED:-0}"
  echo "policy=$policy"

  # Every shard of this group must be readable on this node (a node without the
  # /data/user_data/yixiz/autonlab/zfsauton/datasets mount would otherwise finish with 0 scenes and exit 0).
  local f n_bad=0
  for f in "$group_dir"/tfrecords/"$split"/*; do [[ -r "$f" ]] || { echo "unreadable: $f"; n_bad=$((n_bad + 1)); }; done
  if (( n_bad > 0 )); then echo "FATAL: $n_bad shards unreadable on $(hostname)" >&2; return 1; fi

  local fc=()
  case "$forecast" in
    cv|gt) fc=(--other_agent_forecast "$forecast") ;;
    wm_mf)
      echo "wm_mf=$mf_wm (expected_independent)"
      fc=(--other_agent_forecast wm --world_model_ckpt "$mf_wm" --world_model_device auto
          --wm_backend multifuture --wm_refresh_steps 10 --num_collision_worlds 1
          --collision_reduce any --collision_sample_seed 0 --wm_planning_mode expected_independent) ;;
    *) echo "FATAL: ES_FORECAST=$forecast not cv|gt|wm_mf" >&2; return 1 ;;
  esac
  local extra=()
  if [[ -n "$disc" ]]; then
    [[ -r "$disc" && -r "$disc_wm" ]] || { echo "FATAL: discriminator $disc / WM $disc_wm unreadable" >&2; return 1; }
    echo "discriminator=$disc wm=$disc_wm"
    extra+=(--discriminator_ckpt "$disc" --discriminator_wm_ckpt "$disc_wm" --discriminator_device auto)
  fi
  [[ -n "$also_eval" ]] && extra+=(--also_eval_sim_agents "$also_eval")
  [[ "${ES_SAVE_TRAJ:-1}" == 1 ]] && extra+=(--save_trajectory)
  # Without a target list the runner scores every scene of every shard in the group.
  [[ -s "$group_dir/target_scenarios.json" ]] && extra+=(--target_scenario_path "$group_dir/target_scenarios.json")

  local out_dir="$out/$view/$arm/$(basename "$group_dir")"
  mkdir -p "$out_dir"
  echo "out=$out_dir (done so far: $(find "$out_dir" -name '*.scenario_*.json' | wc -l))"
  # XLA command buffers intermittently abort long runs (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED; es_WM run_es.sh).
  export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
  [[ -n "${ES_XLA_MEM_FRACTION:-}" ]] && export XLA_PYTHON_CLIENT_MEM_FRACTION="$ES_XLA_MEM_FRACTION"
  [[ "${ES_XLA_PREALLOCATE:-}" == false ]] && export XLA_PYTHON_CLIENT_PREALLOCATE=false
  echo "xla: mem_fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION:-<default 0.75>} preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE:-<default true>}"
  export PYTHONPATH="$ES_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
  local rc=0
  "$ES_PYTHON" -u simulation/run_simulation.py \
    --planner "$planner" --other_agents "$view" --world_dt_method "$dt_method" \
    --diffusion_ckpt_path "$policy" --resample_timesteps "${ES_RESAMPLE_TIMESTEPS:-5}" \
    --tfrecord_dir "$group_dir/tfrecords" --split "$split" \
    --lane_graph_dir "${ES_LANE_GRAPH_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}" \
    --exp_dir "$out/$view/$arm" --start_from "$out_dir" --tag "$arm" \
    --num_worlds "${ES_NUM_WORLDS:-5}" --num_scenarios "${ES_NUM_SCENARIOS:-100000}" --seed "${ES_SEED:-0}" \
    --population_size "${ES_POP:-64}" --elite_size "${ES_ELITE:-8}" --num_es_iterations "${ES_IT:-4}" \
    --replan_interval_steps "${ES_REPLAN_INTERVAL:-10}" --visualize_mode none \
    "${fc[@]}" \
    --posthoc_repair "$posthoc" --posthoc_window "${ES_POSTHOC_WINDOW:-20}" --posthoc_cfg "${ES_POSTHOC_CFG:-}" \
    ${extra[@]+"${extra[@]}"} || rc=$?
  local n_done
  n_done=$(find "$out_dir" -name '*.scenario_*.json' | wc -l)
  echo "done: rc=$rc, $n_done scene results in $out_dir"
  (( rc == 0 )) || return "$rc"
  if (( n_done == 0 )); then echo "FATAL: no scene results written" >&2; return 1; fi
}
