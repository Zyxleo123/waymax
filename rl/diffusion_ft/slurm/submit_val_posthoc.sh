#!/bin/bash
# The no-rule post-hoc optimization (o_am, README_posthoc_opt.md) vs the base
# policy on an UNLABELLED slice of WOMD validation: by default the first 10
# shards (~2.9k scenes), every scene scored. Only calls sbatch.
#
# Unlike the held-out strata (failure / intermittent / success of the training
# scan, reweighted to the corpus), these scenes carry no labels, so the plain
# solved rate is the corpus rate and the analysis' `marginal2` is exactly that,
# with a scene-bootstrap 95% CI on o_am - base.
#
# CHAIN
#   prep    (CPU) byte-offset indexes for the glob + a one-split splits JSON
#                 (tools/build_shard_splits.py)
#   select  (GPU array, a6000, afterok:prep) arms base and o_am, paired draw for
#                 draw, cv neighbours, seed 4242, 4 repeats -- the README run's
#                 configuration with only the scenes changed
#   analyze (CPU, afterok:select) select_summary{,_term}.json + png
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_val_posthoc.sh
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_val_posthoc.sh   # 4 scenes, 1 repeat
#   DFT_SIM_AGENTS=idm_brake bash ...   # reactive (braking) NPCs; base is re-scored in the same job
#   DFT_VAL_GLOB='.../validation_tfexample.tfrecord-000[0-4]?-of-00150' DFT_VAL_SPLIT=val50 bash ...
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
GLOB="${DFT_VAL_GLOB:-/zfsauton/datasets/womd/tf_example/validation/validation_tfexample.tfrecord-0000[0-9]-of-00150}"
SPLIT="${DFT_VAL_SPLIT:-val10}"
INDEX_DIR="${DFT_INDEX_DIR:-/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
# `none` log-replays the other cars; `idm_brake` makes them brake for the ego
# (rl/diffusion_ft/sim_agents.py). It changes what `overlap` means, so it is in
# the run's name and both arms are always scored under the same simulator.
SIM="${DFT_SIM_AGENTS:-none}"
case "$SIM" in none|idm_brake) ;; *) echo "FATAL: DFT_SIM_AGENTS=$SIM not none|idm_brake" >&2; exit 1;; esac
OUT="${DFT_VAL_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/val_posthoc_${SPLIT}_${SIM}_${STAMP}}"
SPLITS="$OUT/${SPLIT}_splits.json"
AM_CFG="${DFT_ARGMIN_CFG:-lon=1,w_lon_prox=0.05,lon_back_m=1000}"
n_shards="$(compgen -G "$GLOB" | wc -l)"
(( n_shards > 0 )) || { echo "FATAL: no shards match $GLOB" >&2; exit 1; }
mkdir -p "$OUT" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

{ kv DFT_SHARD_GLOB "$GLOB"; kv DFT_INDEX_DIR "$INDEX_DIR"
  kv DFT_SPLIT_NAME "$SPLIT"; kv DFT_SPLITS_JSON "$SPLITS"; } > "$OUT/prep.env"
echo "+ sbatch --parsable rl/diffusion_ft/slurm/prep_shard_splits.sbatch $OUT/prep.env"
PID=$(sb --parsable rl/diffusion_ft/slurm/prep_shard_splits.sbatch "$OUT/prep.env")
echo "prep    -> job $PID (CPU) $n_shards shards -> $SPLITS"

# The base checkpoint dir is required by the runner but never loaded (no ft arms).
OLD=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS" DFT_SPLITS_UPSTREAM=1 \
DFT_SELECT_DIR="$OUT/select" DFT_SELECT_DEP="afterok:$PID" \
DFT_SHARD_GLOB="$GLOB" DFT_INDEX_DIR="$INDEX_DIR" DFT_SELECT_SPLITS="$SPLIT" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,o_am=bank@am:argmin:20" \
DFT_SELECT_BANKS="am=" DFT_SELECT_BANK_OPTS="am=init0,initprev" \
DFT_SELECT_OPT_CFG="" DFT_SELECT_BANK_OPT_CFGS="am=${AM_CFG}" \
DFT_SELECT_BANK_RAMPS="am=20" DFT_SELECT_BANK_BLENDS="am=cubic" \
DFT_SELECT_BANK_NEIGHBORS="am=cv" \
DFT_SIM_AGENTS="$SIM" DFT_DUMP_TRAJ=1 DFT_SELECT_SMOKE="${DFT_SELECT_SMOKE:-}" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-4}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh
echo "all under $OUT; read select/select_summary_term.json (marginal2 = the $SPLIT rate)"
