#!/bin/bash
# Re-read finished submit_val_posthoc.sh runs on EXPERT-SOLVABLE scenes only.
# Only calls sbatch.
#
# The validation slice is unlabelled, so some of its failures are scenes the
# logged human cannot solve either (dirty ground truth). The held-out strata were
# built on expert-clean scenes only; this applies the same gate to the slice.
# No policy rollouts are repeated: the existing rows are re-analysed.
#
# CHAIN
#   label   (GPU, a6000) expert log replay on every scene of the FIRST run's
#                        splits file -> expert_rows.jsonl + a restricted copy
#                        <split>_expclean_splits.json (same split name)
#   analyze (CPU, afterok:label, one per run) select_summary_expclean{,_term}.json
#   recheck (CPU, afterok:label) the first run against its ORIGINAL splits file,
#                        tag `recheck` -- must equal select_summary{,_term}.json
#                        (the analysis' new scene restriction is a no-op there)
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_val_expert_filter.sh RUN_DIR [RUN_DIR ...]
# where RUN_DIR is a submit_val_posthoc.sh output dir (holds <split>_splits.json
# and select/). All runs must share one splits file (same scenes).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
(( $# > 0 )) || { echo "usage: $0 RUN_DIR [RUN_DIR ...]" >&2; exit 1; }
SPLIT="${DFT_VAL_SPLIT:-val10}"
EXCLUDE="${DFT_EXCLUDE-gpu28}"        # a5000
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

FIRST="$1"
SPLITS="$FIRST/${SPLIT}_splits.json"
[[ -s "$SPLITS" ]] || { echo "FATAL: missing $SPLITS" >&2; exit 1; }
for d in "$@"; do
  [[ -s "$d/select/select_summary.json" ]] || { echo "FATAL: $d has no finished select/" >&2; exit 1; }
  cmp -s <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["shard_glob"], d["splits"][sys.argv[2]]["scene_ids"])' "$SPLITS" "$SPLIT") \
         <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["shard_glob"], d["splits"][sys.argv[2]]["scene_ids"])' "$d/${SPLIT}_splits.json" "$SPLIT") \
    || { echo "FATAL: $d scores different scenes than $FIRST" >&2; exit 1; }
done
GLOB="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_glob"])' "$SPLITS")"
LDIR="$FIRST/expert"; mkdir -p "$LDIR" "$ROOT/logs/dft"
CLEAN="$LDIR/${SPLIT}_expclean_splits.json"

{ kv DFT_OUT_DIR "$LDIR"; kv DFT_SPLITS_JSON "$SPLITS"; kv DFT_LABEL_SPLITS "$SPLIT"
  kv DFT_CLEAN_SPLITS_OUT "$CLEAN"; kv DFT_SHARD_GLOB "$GLOB"; } > "$LDIR/label.env"
FLAGS=(); [[ -n "$EXCLUDE" ]] && FLAGS+=(--exclude="$EXCLUDE")
echo "+ sbatch --parsable ${FLAGS[*]} rl/diffusion_ft/slurm/expert_label.sbatch $LDIR/label.env"
LID=$(sb --parsable "${FLAGS[@]}" rl/diffusion_ft/slurm/expert_label.sbatch "$LDIR/label.env")
echo "label   -> job $LID (GPU) -> $CLEAN"

for d in "$@"; do
  f="$d/select/analyze_expclean.env"
  { kv DFT_SELECT_DIR "$d/select"; kv DFT_SPLITS_JSON "$CLEAN"
    kv DFT_SELECT_REPEATS 4; kv DFT_SELECT_TAG expclean; } > "$f"
  echo "+ sbatch --parsable --dependency=afterok:$LID rl/diffusion_ft/slurm/select_analysis.sbatch $f"
  AID=$(sb --parsable --dependency="afterok:$LID" rl/diffusion_ft/slurm/select_analysis.sbatch "$f")
  echo "analyze -> job $AID (CPU) -> $d/select/select_summary_expclean_term.json"
done
f="$FIRST/select/analyze_recheck.env"
{ kv DFT_SELECT_DIR "$FIRST/select"; kv DFT_SPLITS_JSON "$SPLITS"
  kv DFT_SELECT_REPEATS 4; kv DFT_SELECT_TAG recheck; } > "$f"
echo "+ sbatch --parsable --dependency=afterok:$LID rl/diffusion_ft/slurm/select_analysis.sbatch $f"
RID=$(sb --parsable --dependency="afterok:$LID" rl/diffusion_ft/slurm/select_analysis.sbatch "$f")
echo "recheck -> job $RID (CPU) -> $FIRST/select/select_summary_recheck_term.json (must equal select_summary_term.json)"
