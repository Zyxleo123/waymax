#!/bin/bash
# Diffusion-ES + world model + o_am repair + es_WM's learned discriminator, every arm scored
# by both of es_WM's evaluators, on fail1500 or WOMD validation shards 0-9. Only calls sbatch.
#
# ARMS (same policy, seed, scenes, world-dt and window protocol):
#   diff                the diffusion policy alone (one sample per replan, no ES)
#   diff_posthoc        + the plan repaired against the PDM cv forecast
#   gt                  Diffusion-ES ranked against the logged future
#   cv / cv_posthoc     Diffusion-ES, PDM constant-velocity forecast (+ repair of every candidate)
#   wm_mf / wm_mf_posthoc   Diffusion-ES, MF world model, expected_independent (+ repair vs MF modes)
#   wm_mf_disc          Diffusion-ES ranked by es_WM's learned WM-conditioned discriminator
#                       (es_WM's "ES + WM + discriminator"; the handcrafted reward is bypassed)
#   wm_mf_disc_posthoc  + every candidate repaired against the MF modes BEFORE the discriminator
#                       scores it: the whole system (es + wm + repair + discriminator)
# WORLD-MODEL COMPARISON (Sagar's MF WM above vs Ishan's, everything else identical):
#   wmI_mf / wmI_mf_posthoc   Ishan's MF WM (mini-hong/waymax_rs@ishan, epoch_0450), expected_independent
#   wmI_mf_disc_posthoc       Ishan's WM for the repair; the discriminator keeps its own Sagar WM
#                             (the inputs it was trained on) -- "Ishan's WM with Sagar's discriminator"
#   wm_mf_tb / wmI_mf_tb      Ishan's recommended ES objective (--offroad_mode margin_tiebreak) with
#                             either WM: a 2x2 that separates the WM from the objective
# VIEWS (es_WM c0fded3's evaluators; each view is its own planning run):
#   log   NPCs replay the log (also re-scored with evaluation-only IDM: idm_* keys)
#   idm   closed-loop IDM (simulation/idm_closed_loop.py): every non-SDC object runs Waymax's
#         IDMRoutePolicy and the planner observes the simulated NPCs
# Every scene JSON also carries es_WM's original failure window (*_oldwin).
# POLICY: mineuih goal_cond_diffusion_policy/epoch_1000 (mini-hong/waymax_rs@new_model architecture and
# map-feature sampling, ported into this worktree; see es_disc_common.sh for why not "latest").
#
# CHAIN: es (GPU arrays; general + preempt) -> compare (CPU, afterany) -> <root>/compare.{json,md}
# The repair and the ES ranking use the planner's MF WM (latest.pt, as in every earlier run);
# the discriminator gets its own frozen copy of epoch_0450.pt, the WM its inputs were trained on.
#
# Usage (lov4: sbatch is local; elsewhere ON RHEA):
#   ES_SMOKE=1 bash slurm/sim/submit_es_disc.sh   # 8 configs x 5 scenes, one GPU job, sequential
#   (no CPU smoke: the policy checkpoint cannot be restored with JAX_PLATFORMS=cpu --
#    orbax "sharding ... Got None", job 87958)
#   ES_SUBSET=val10 bash slurm/sim/submit_es_disc.sh
#   ES_SUBSET=fail1500 bash slurm/sim/submit_es_disc.sh
#   ES_ARMS_SEL=wm_mf_disc,wm_mf_disc_posthoc ES_VIEWS=idm bash ...   # a subset
#   ES_OUT=<existing root> bash ...                                   # resume into a root
#   ES_PACK=1 bash ...         # one config per GPU task (default: both groups share one GPU)
#   ES_PREEMPT_IDX=0,1,2 bash ...  # these task ids go to preempt (qos_preempt caps 10 submitted per user)
set -euo pipefail
ROOT="/zfsauton2/home/yixiz/waymax_rs_es_newmodel"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
source "$ROOT/slurm/sim/es_disc_common.sh"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SMOKE="${ES_SMOKE:-}"
SCR=/zfsauton/scratch/yixiz/waymax_rs/es_posthoc
ES_SUBSET="${ES_SUBSET:-fail1500}"
case "$ES_SUBSET" in
  fail1500) SUBSET="${ES_SUBSET_DIR:-$SCR/fail1500_subset_g2}"; SPLIT=training; EXPECT=1500 ;;
  val10)    SUBSET="${ES_SUBSET_DIR:-$SCR/val10_subset_x5}"; SPLIT=validation; EXPECT=0 ;;
  *) echo "FATAL: ES_SUBSET=$ES_SUBSET not fail1500|val10" >&2; exit 1 ;;
esac
[[ -s "$SUBSET/manifest.json" ]] || { echo "FATAL: $SUBSET has no manifest.json (build it with the es_posthoc prep job)" >&2; exit 1; }
mapfile -t GROUP_DIRS < <(find "$SUBSET" -maxdepth 1 -type d -name 'group_*' | sort -V)
(( ${#GROUP_DIRS[@]} > 0 )) || { echo "FATAL: no group_* under $SUBSET" >&2; exit 1; }
DISC="${ES_DISC_CKPT:-/zfsauton/scratch/sbellad/es_discriminator/safety_progress/safety_progress_v3_20261016_155456_86103/best.pt}"
DISC_WM="${ES_DISC_WM_CKPT:-$ES_DEFAULT_DISC_WM}"
POLICY="${ES_DIFFUSION_CKPT:-$ES_DEFAULT_POLICY}"
MF_WM="${ES_MF_WM_CKPT:-$ES_DEFAULT_MF_WM}"
ISHAN_WM="${ES_ISHAN_WM_CKPT:-$ES_DEFAULT_ISHAN_WM}"
for f in "$DISC" "$DISC_WM" "$MF_WM" "$ISHAN_WM" "$POLICY"; do [[ -e "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }; done
case "$SMOKE" in
  "") OUT="${ES_OUT:-$SCR/runs/nm_${ES_SUBSET}_${STAMP}}" ;;
  1) OUT="${ES_OUT:-$SCR/runs/nm_${ES_SUBSET}_smoke_${STAMP}}" ;;
  *) echo "FATAL: ES_SMOKE=$SMOKE not 1" >&2; exit 1 ;;
esac
# Longest first, so the slow arms start early.
ARMS="${ES_ARMS_SEL:-wm_mf_disc_posthoc,wmI_mf_disc_posthoc,wm_mf_posthoc,wmI_mf_posthoc,cv_posthoc,gt,wm_mf_disc,cv,wm_mf,wmI_mf,wm_mf_tb,wmI_mf_tb,diff_posthoc,diff}"
VIEWS="${ES_VIEWS:-log,idm}"
PACK="${ES_PACK:-groups}"
mkdir -p "$OUT/cfg" "$OUT/env" "$ES_LOG_DIR"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
SBATCH_BIN="$(env -i PATH="$PATH" bash -c 'command -v sbatch' || true)"
[[ -n "$SBATCH_BIN" ]] || { echo "FATAL: no sbatch on PATH (submit from lov4 or rhea)" >&2; exit 1; }
echo "sbatch=$SBATCH_BIN root=$OUT subset=$SUBSET (${#GROUP_DIRS[@]} groups) views=$VIEWS pack=$PACK"
echo "policy=$POLICY"
echo "planner MF WM=$MF_WM; discriminator=$DISC (its WM: $DISC_WM)"

# write_cfg VIEW ARM GROUP_DIR -> path of a self-contained config file (every key explicit)
write_cfg() {
  local view="$1" arm="$2" gdir="$3" pl=es fc ph disc="" om=binary
  case "$arm" in
    diff) pl=diffusion; fc=cv; ph=0 ;; diff_posthoc) pl=diffusion; fc=cv; ph=1 ;;
    gt) fc=gt; ph=0 ;;
    cv) fc=cv; ph=0 ;; cv_posthoc) fc=cv; ph=1 ;;
    wm_mf) fc=wm_mf; ph=0 ;; wm_mf_posthoc) fc=wm_mf; ph=1 ;;
    wm_mf_disc) fc=wm_mf; ph=0; disc="$DISC" ;; wm_mf_disc_posthoc) fc=wm_mf; ph=1; disc="$DISC" ;;
    wmI_mf) fc=wm_mf_ishan; ph=0 ;; wmI_mf_posthoc) fc=wm_mf_ishan; ph=1 ;;
    wmI_mf_disc_posthoc) fc=wm_mf_ishan; ph=1; disc="$DISC" ;;
    wm_mf_tb) fc=wm_mf; ph=0; om=margin_tiebreak ;; wmI_mf_tb) fc=wm_mf_ishan; ph=0; om=margin_tiebreak ;;
    *) echo "FATAL: unknown arm '$arm'" >&2; return 1 ;;
  esac
  local f="$OUT/cfg/${ES_SUBSET}${SMOKE:+_smoke$SMOKE}.${view}.${arm}.$(basename "$gdir").env"
  { kv ES_OUT "$OUT"; kv ES_ARM "$arm"; kv ES_VIEW "$view"; kv ES_PLANNER "$pl"; kv ES_FORECAST "$fc"
    kv ES_POSTHOC "$ph"; kv ES_DISC_CKPT "$disc"; kv ES_DISC_WM_CKPT "$DISC_WM"
    kv ES_ISHAN_WM_CKPT "$ISHAN_WM"; kv ES_OFFROAD_MODE "$om"
    kv ES_ALSO_EVAL "$([[ "$view" == log ]] && echo "${ES_ALSO_EVAL:-idm}" || echo "")"
    kv ES_WORLD_DT_METHOD "${ES_WORLD_DT_METHOD:-median}"; kv ES_SAVE_TRAJ "${ES_SAVE_TRAJ:-1}"
    kv ES_GROUP_DIR "$gdir"; kv ES_SPLIT "$SPLIT"
    kv ES_DIFFUSION_CKPT "$POLICY"; kv ES_MF_WM_CKPT "$MF_WM"; kv ES_SEED "${ES_SEED:-0}"
    kv ES_XLA_MEM_FRACTION "${XLA_FRAC:-}"
    [[ -n "${XLA_PREALLOC:-}" ]] && kv ES_XLA_PREALLOCATE "$XLA_PREALLOC"
    case "$SMOKE" in
      # on-demand JAX allocation, so the monitor reports what each arm really needs
      1) kv ES_NUM_SCENARIOS "${ES_SMOKE_N:-5}"; kv ES_XLA_PREALLOCATE false; kv ES_MON_SEC 60 ;;
      *) kv ES_NUM_SCENARIOS 100000 ;;
    esac; } > "$f"
  echo "$f"
}

if [[ -n "$SMOKE" ]]; then
  g0="${GROUP_DIRS[0]}"
  # log:diff and log:cv_posthoc re-run scenes an earlier run planned (regression check);
  # the rest exercise each new path once: the discriminator with and without the repair,
  # and the closed loop with the discriminator, the diffusion-only repair, gt and the WM.
  SPECS="${ES_SMOKE_SPECS:-log:diff log:cv_posthoc log:wm_mf_disc log:wm_mf_disc_posthoc idm:wm_mf_disc_posthoc idm:diff_posthoc idm:gt idm:wm_mf}"
  cfgs=()
  for s in $SPECS; do cfgs+=("$(write_cfg "${s%%:*}" "${s#*:}" "$g0")"); done
  { kv ES_SUBTASKS "${cfgs[*]}"; kv ES_SUBTASK_MODE seq; } > "$OUT/env/smoke.env"
  SB=(--time=6:00:00)
  [[ -n "${ES_EXCLUDE-gpu30}" ]] && SB+=(--exclude="${ES_EXCLUDE-gpu30}")
  # shellcheck disable=SC2206
  [[ -n "${ES_SMOKE_SB:-}" ]] && SB+=($ES_SMOKE_SB)
  { kv ES_MON_SEC 60; } >> "$OUT/env/smoke.env"
  JID=$(sb --parsable "${SB[@]}" slurm/sim/es_disc.sbatch "$OUT/env/smoke.env")
  echo "+ sbatch ${SB[*]} slurm/sim/es_disc.sbatch $OUT/env/smoke.env -> $JID"
  echo "smoke root $OUT"
  exit 0
fi

# Full run: one task per (view, arm); with ES_PACK=groups its shard groups share the GPU.
IFS=, read -r -a ARM_LIST <<< "$ARMS"
IFS=, read -r -a VIEW_LIST <<< "$VIEWS"
if [[ "$PACK" == groups ]]; then P=${#GROUP_DIRS[@]}; else P=1; fi
# Packed runs share the card: JAX allocates on demand, each capped at its share (with
# preallocation off the fraction is a ceiling), and torch's WMs use what is left.
if (( P > 1 )); then
  XLA_FRAC="${ES_XLA_MEM_FRACTION:-$(awk -v p="$P" 'BEGIN { printf "%.2f", 0.85 / p }')}"
  XLA_PREALLOC=false
fi
i=0
for arm in "${ARM_LIST[@]}"; do
  for view in "${VIEW_LIST[@]}"; do
    cfgs=()
    for gdir in "${GROUP_DIRS[@]}"; do cfgs+=("$(write_cfg "$view" "$arm" "$gdir")"); done
    if (( P > 1 )); then
      { kv ES_SUBTASKS "${cfgs[*]}"; kv ES_SUBTASK_MODE par; } > "$OUT/env/$i.env"
      printf '%2d  %-4s %-20s %d groups on one GPU\n' "$i" "$view" "$arm" "${#cfgs[@]}"; i=$((i + 1))
    else
      for c in "${cfgs[@]}"; do
        cp "$c" "$OUT/env/$i.env"; printf '%2d  %-4s %-20s %s\n' "$i" "$view" "$arm" "$(basename "$c")"; i=$((i + 1))
      done
    fi
  done
done
N=$i
# ES_PREEMPT_IDX: comma list of task ids for the preempt partition (qos_preempt: 10 submitted,
# 8 GPUs per user), the rest go to general. A preempted task is requeued and resumes at its
# last finished batch, so the LONG tasks can go there; the default is none.
PRE_IDX="${ES_PREEMPT_IDX:-}"
GEN_IDX=""
for ((t = 0; t < N; t++)); do [[ ",$PRE_IDX," == *",$t,"* ]] || GEN_IDX="${GEN_IDX:+$GEN_IDX,}$t"; done
for t in ${PRE_IDX//,/ }; do (( t >= 0 && t < N )) || { echo "FATAL: ES_PREEMPT_IDX has $t, tasks are 0-$((N - 1))" >&2; exit 1; }; done
CPUS=$(( 8 * P )); MEM="$(( ${ES_MEM_PER_CFG_GB:-48} * P ))G"
COMMON=(--cpus-per-task="$CPUS" --mem="$MEM")
[[ -n "${ES_EXCLUDE-gpu30}" ]] && COMMON+=(--exclude="${ES_EXCLUDE-gpu30}")
DEPS=()
if [[ -n "$GEN_IDX" ]]; then
  A1=$(sb --parsable "${COMMON[@]}" --array="$GEN_IDX" slurm/sim/es_disc.sbatch "$OUT/env")
  echo "+ sbatch ${COMMON[*]} --array=$GEN_IDX slurm/sim/es_disc.sbatch $OUT/env -> $A1 (general)"
  DEPS+=("$A1")
fi
if [[ -n "$PRE_IDX" ]]; then
  A2=$(sb --parsable "${COMMON[@]}" --partition=preempt --qos=qos_preempt --array="$PRE_IDX" \
    slurm/sim/es_disc.sbatch "$OUT/env")
  echo "+ sbatch ${COMMON[*]} --partition=preempt --qos=qos_preempt --array=$PRE_IDX slurm/sim/es_disc.sbatch $OUT/env -> $A2 (preempt)"
  DEPS+=("$A2")
fi

PAIRS="wm_mf_disc:wm_mf,wm_mf_disc_posthoc:wm_mf_disc,wm_mf_disc_posthoc:wm_mf_posthoc,wm_mf_disc_posthoc:diff"
PAIRS="$PAIRS,wm_mf_disc:diff,wm_mf_posthoc:wm_mf,cv_posthoc:cv,wm_mf:cv,wm_mf_posthoc:cv_posthoc"
PAIRS="$PAIRS,diff_posthoc:diff,cv:diff,gt:cv,wm_mf_posthoc:diff_posthoc"
# Ishan's WM vs Sagar's, and the offroad-objective 2x2
PAIRS="$PAIRS,wmI_mf:wm_mf,wmI_mf_posthoc:wm_mf_posthoc,wmI_mf_disc_posthoc:wm_mf_disc_posthoc,wmI_mf:cv"
PAIRS="$PAIRS,wm_mf_tb:wm_mf,wmI_mf_tb:wmI_mf,wmI_mf_tb:wm_mf_tb,wmI_mf_tb:wm_mf"
{ kv ES_CPU_MODE compare; kv ES_OUT "$OUT"
  kv ES_ARMS "diff,diff_posthoc,gt,cv,cv_posthoc,wm_mf,wm_mf_posthoc,wm_mf_disc,wm_mf_disc_posthoc,wmI_mf,wmI_mf_posthoc,wmI_mf_disc_posthoc,wm_mf_tb,wmI_mf_tb"
  kv ES_VIEWS "$VIEWS"; kv ES_PAIRS "$PAIRS"; kv ES_EXPECT "$EXPECT"
  # fail1500 scenes are in the discriminator's own training set: slice by its split
  [[ "$ES_SUBSET" == fail1500 ]] && kv ES_SPLIT_JSON "${ES_DISC_SPLIT_JSON:-/zfsauton/scratch/sbellad/es_discriminator/training_sets/fail1500_plus_womd_train_s02_11/scenario_split_seed0.json}"
  true; } > "$OUT/compare.env"
DEP="afterany:$(IFS=:; echo "${DEPS[*]}")"
CID=$(sb --parsable --dependency="$DEP" slurm/sim/es_disc_cpu.sbatch "$OUT/compare.env")
echo "+ sbatch --dependency=$DEP slurm/sim/es_disc_cpu.sbatch $OUT/compare.env -> $CID (compare)"
echo "all under $OUT"
