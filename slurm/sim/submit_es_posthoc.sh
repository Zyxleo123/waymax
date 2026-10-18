#!/bin/bash
# Diffusion-ES on fail1500, with and without the o_am post-hoc repair of every
# candidate before elite selection, with and without the multi-future world model.
# Only calls sbatch.
#
# ARMS (same policy, seed, scenes; the forecast feeds BOTH the ES ranking and the repair):
#   diff            the diffusion policy alone (one sample per replan, no ES)
#   diff_posthoc    + the plan repaired against the PDM cv forecast (o_am inside this runner)
#   gt              Diffusion-ES ranked against the logged future (oracle under log replay)
#   cv              Diffusion-ES, PDM-style constant-velocity forecast
#   cv_posthoc      + every candidate repaired (planner/posthoc_repair.py) before scoring
#   wm_mf           Diffusion-ES, MF world model scored expected_independent
#   wm_mf_posthoc   + every candidate repaired against the MF modes (probability-weighted)
#
# CHAIN
#   prep    (CPU) rebuild fail1500 from a finished fail1500 run, split into shard groups
#   es      (GPU array, afterok:prep) one task per (arm, group), resumable
#   compare (CPU, afterany:es) paired success deltas + scene-bootstrap CIs -> compare.json
#
# Policy: pretrain_diffusion_without_subgoal_20260616_013642/latest (user, 2026-10-12),
# NOT the diffusion_baseline ckpt of es_WM's README table -- so absolute rates differ
# from that table; every arm here is rerun on the same policy.
#
# Usage (lov4: sbatch is local; elsewhere ON RHEA):
#   ES_SMOKE=1 bash slurm/sim/submit_es_posthoc.sh     # 10 scenes of group 0 per arm
#   bash slurm/sim/submit_es_posthoc.sh
#   ES_ARMS_SEL=cv,cv_posthoc bash ...                  # a subset of the arms
#   ES_OUT=<existing root> bash ...                     # resume into a previous root
#   ES_SUBSET=val10 bash ...                            # WOMD validation shards 0-9, every scene
#                                                       # (the set o_am-on-policy was validated on)
set -euo pipefail
ROOT="/zfsauton2/home/yixiz/waymax_rs_es_posthoc"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SMOKE="${ES_SMOKE:-}"
SCR=/zfsauton/scratch/yixiz/waymax_rs/es_posthoc
ES_SUBSET="${ES_SUBSET:-fail1500}"
case "$ES_SUBSET" in fail1500|val10) ;; *) echo "FATAL: ES_SUBSET=$ES_SUBSET not fail1500|val10" >&2; exit 1;; esac
VAL_GLOB="${ES_VAL_GLOB:-/zfsauton/datasets/womd/tf_example/validation/validation_tfexample.tfrecord-0000[0-9]-of-00150}"
SOURCE_RUN="${ES_SOURCE_RUN:-/zfsauton/scratch/sbellad/world_model_sim/es_diffusion_wm_mf_epoch0450_expected_independent_fail1500/simulation_es_training_es_diffusion_wm_mf_epoch0450_expected_independent_fail1500_20260919_170306}"
if [[ "$ES_SUBSET" == val10 ]]; then
  PER="${ES_VAL_PER_GROUP:-1}"
  SUBSET="${ES_SUBSET_DIR:-$SCR/val10_subset$([[ $PER -gt 1 ]] && echo "_x$PER")}"; SPLIT=validation
  N_SH="$(compgen -G "$VAL_GLOB" | wc -l)"; GROUPS_N=$(( (N_SH + PER - 1) / PER ))
else
  SUBSET="${ES_SUBSET_DIR:-$SCR/fail1500_subset}"; SPLIT=training
  GROUPS_N="${ES_GROUPS:-6}"
fi
OUT="${ES_OUT:-$SCR/runs/${ES_SUBSET}${SMOKE:+_smoke}_${STAMP}}"
ARMS="${ES_ARMS_SEL:-diff,diff_posthoc,gt,cv,cv_posthoc,wm_mf,wm_mf_posthoc}"
MAX_CONCURRENT="${ES_MAX_CONCURRENT:-24}"
mkdir -p "$OUT/env" /zfsauton2/home/yixiz/waymax_rs/logs/es_posthoc
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

# prep (skipped when the subset already exists: it is deterministic in its source run)
PREP_DEP=()
if [[ ! -s "$SUBSET/manifest.json" ]]; then
  if [[ "$ES_SUBSET" == val10 ]]; then
    { kv ES_CPU_MODE prep_val; kv ES_VAL_GLOB "$VAL_GLOB"; kv ES_SUBSET_DIR "$SUBSET"
      kv ES_VAL_PER_GROUP "$PER"; } > "$OUT/prep.env"
  else
    { kv ES_CPU_MODE prep; kv ES_SOURCE_RUN "$SOURCE_RUN"; kv ES_SUBSET_DIR "$SUBSET"
      kv ES_GROUPS "$GROUPS_N"; } > "$OUT/prep.env"
  fi
  PID=$(sb --parsable slurm/sim/es_posthoc_cpu.sbatch "$OUT/prep.env")
  echo "+ sbatch slurm/sim/es_posthoc_cpu.sbatch $OUT/prep.env -> prep job $PID (CPU) -> $SUBSET"
  PREP_DEP=(--dependency="afterok:$PID")
else
  echo "prep    -> exists: $SUBSET ($(head -c 200 "$SUBSET/manifest.json" | tr -d '\n'))"
fi

i=0
IFS=, read -r -a ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
  pl=es
  case "$arm" in
    diff) pl=diffusion; fc=cv; ph=0 ;; diff_posthoc) pl=diffusion; fc=cv; ph=1 ;;
    gt) fc=gt; ph=0 ;;
    cv) fc=cv; ph=0 ;; cv_posthoc) fc=cv; ph=1 ;;
    wm_mf) fc=wm_mf; ph=0 ;; wm_mf_posthoc) fc=wm_mf; ph=1 ;;
    *) echo "FATAL: unknown arm '$arm'" >&2; exit 1 ;;
  esac
  for ((g = 0; g < GROUPS_N; g++)); do
    [[ -n "$SMOKE" && $g -gt 0 ]] && continue
    { kv ES_OUT "$OUT"; kv ES_ARM "$arm"; kv ES_PLANNER "$pl"; kv ES_FORECAST "$fc"; kv ES_POSTHOC "$ph"
      kv ES_ALSO_EVAL "${ES_ALSO_EVAL:-idm}"
      kv ES_WORLD_DT_METHOD "${ES_WORLD_DT_METHOD:-median}"
      kv ES_SAVE_TRAJ "${ES_SAVE_TRAJ:-1}"
      kv ES_GROUP_DIR "$SUBSET/group_$g"; kv ES_SPLIT "$SPLIT"
      kv ES_NUM_SCENARIOS "$([[ -n "$SMOKE" ]] && echo 10 || echo 100000)"
      kv ES_DIFFUSION_CKPT "${ES_DIFFUSION_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
      kv ES_MF_WM_CKPT "${ES_MF_WM_CKPT:-/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/latest.pt}"
      kv ES_SEED "${ES_SEED:-0}"; } > "$OUT/env/$i.env"
    printf '%2d  %-14s group %d\n' "$i" "$arm" "$g"
    i=$((i + 1))
  done
done
# gpu30 cannot read /zfsauton/datasets (jobs 85314_8-10 died on it, 2026-10-15).
SB_FLAGS=(); [[ -n "${ES_EXCLUDE-gpu30}" ]] && SB_FLAGS+=(--exclude="${ES_EXCLUDE-gpu30}")
AID=$(sb --parsable ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} ${PREP_DEP[@]+"${PREP_DEP[@]}"} --array="0-$((i - 1))%${MAX_CONCURRENT}" \
  slurm/sim/es_posthoc.sbatch "$OUT/env")
echo "+ sbatch --array=0-$((i - 1))%$MAX_CONCURRENT slurm/sim/es_posthoc.sbatch $OUT/env -> array $AID (GPU, $i tasks)"

PAIRS="cv_posthoc:cv,wm_mf_posthoc:wm_mf,wm_mf:cv,wm_mf_posthoc:cv_posthoc,wm_mf_posthoc:cv"
PAIRS="$PAIRS,diff_posthoc:diff,cv:diff,gt:cv,wm_mf_posthoc:diff_posthoc"
{ kv ES_CPU_MODE compare; kv ES_OUT "$OUT"; kv ES_ARMS "$ARMS"; kv ES_PAIRS "$PAIRS"
  kv ES_EXPECT "$([[ -n "$SMOKE" ]] && echo 10 || { [[ "$ES_SUBSET" == fail1500 ]] && echo 1500 || echo 0; })"; } > "$OUT/compare.env"
CID=$(sb --parsable --dependency="afterany:$AID" slurm/sim/es_posthoc_cpu.sbatch "$OUT/compare.env")
echo "compare -> job $CID (CPU, afterany:$AID) -> $OUT/compare.json"
echo "all under $OUT"
