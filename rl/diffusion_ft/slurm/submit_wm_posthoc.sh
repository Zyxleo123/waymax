#!/bin/bash
# The no-rule post-hoc optimization (o_am, README_posthoc_opt.md) with the other
# cars forecast by es_WM's learned world models instead of constant velocity.
# Only calls sbatch.
#
# WHY. o_am's only honest neighbour source was cv. es_WM (Sagar Bellad) trained a
# deterministic WM and a multi-future Gaussian WM (K=6); in their Diffusion-ES on
# fail1500 the MF WM scored expected_independent closed 4.3 of the 6.1 points
# between cv (46.20) and the GT oracle (52.27). Here the same models feed the
# repair's barrier (rl/diffusion_ft/world_model_forecast.py): wm_mf charges each
# agent's modes by probability -- the expected barrier, the additive analogue of
# their P_i = sum_k p_ik c_ik.
#
# ARMS (one job, paired draws -- every arm sees the same base samples):
#   base    the policy alone
#   o_am    bank am  , neighbours cv      -- the shipped arm, the one to beat
#   o_amD   bank amd , neighbours wm_det
#   o_amM   bank amm , neighbours wm_mf   (expected over modes)
#   o_amL   bank aml , neighbours log     -- the oracle: the ceiling on any forecast
# All share o_am's optimizer config and starts (init0, initprev); only the
# neighbour source differs.
#
# SIMULATOR. DFT_SIM_AGENTS=none (default) log-replays the other cars, like every
# o_am number to date and es_WM's runs; `idm` is waymax's stock IDMRoutePolicy
# (reactive). It changes what `overlap` means, so it is in the run's name and all
# arms are scored under the same one. Under `none` the log arm is EXACT, so o_amL
# is a hard ceiling; under `idm` nothing is.
#
# CHAIN: gate (CPU, tests/test_wm_forecast.py) -> select (GPU array, a6000,
# afterok:gate, reads the verdict) -> analysis + compares (CPU, term-aware).
#
# Usage (from repo root; on lov4 sbatch is local, anywhere else ON RHEA):
#   DFT_WM_SMOKE=1 bash rl/diffusion_ft/slurm/submit_wm_posthoc.sh
#   bash rl/diffusion_ft/slurm/submit_wm_posthoc.sh
#   DFT_SIM_AGENTS=idm bash rl/diffusion_ft/slurm/submit_wm_posthoc.sh
#   DFT_WM_MF_CKPT=/path/epoch_0450.pt bash ...       # another checkpoint
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
for v in $(compgen -v SLURM_); do unset "$v"; done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SIM="${DFT_SIM_AGENTS:-none}"
case "$SIM" in none|idm|idm_brake) ;; *) echo "FATAL: DFT_SIM_AGENTS=$SIM not none|idm|idm_brake" >&2; exit 1;; esac
SMOKE="${DFT_WM_SMOKE:-}"
OUT="${DFT_WM_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select/wm_posthoc_${SIM}${SMOKE:+_smoke}_${STAMP}}"
# Same scene ids as eval_splits_3strata.json, re-pointed at /zfsauton/datasets/womd
# (the eshau copy is gone; the runner refuses a splits file built on another glob).
SPLITS3=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata_womd.json
AM_CFG="${DFT_ARGMIN_CFG:-lon=1,w_lon_prox=0.05,lon_back_m=1000}"
mkdir -p "$OUT/gate" "$ROOT/logs/dft"
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }

{ kv DFT_GATE_VERDICT "$OUT/gate/verdict.txt"; kv DFT_SPLITS_JSON "$SPLITS3"
  kv DFT_WM_JSON "$OUT/gate/test_wm_forecast.json"
  kv DFT_WM_GATE_SCENES "$([[ -n "$SMOKE" ]] && echo 4 || echo 8)"
  kv DFT_WM_DET_CKPT "${DFT_WM_DET_CKPT:-}"; kv DFT_WM_MF_CKPT "${DFT_WM_MF_CKPT:-}"
} > "$OUT/gate/gate.env"
echo "+ sbatch --parsable rl/diffusion_ft/slurm/test_wm_forecast.sbatch $OUT/gate/gate.env"
GID=$(sb --parsable rl/diffusion_ft/slurm/test_wm_forecast.sbatch "$OUT/gate/gate.env")
echo "gate    -> job $GID (CPU) -> $OUT/gate/verdict.txt"

# The base checkpoint dir is required by the runner but never loaded (no ft arms).
OLD=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
DFT_SPLITS3_JSON="$SPLITS3" DFT_SELECT_DIR="$OUT/select" \
DFT_SELECT_DEP="afterok:$GID" DFT_REQUIRE_GATE="$OUT/gate/verdict.txt" \
DFT_SELECT_CKPTS="$OLD/base_s0/iter_000110" DFT_FT_ARMS="" \
DFT_BASE_ARMS="base,o_am=bank@am:argmin:20,o_amD=bank@amd:argmin:20,o_amM=bank@amm:argmin:20,o_amL=bank@aml:argmin:20" \
DFT_SELECT_BANKS="am=;amd=;amm=;aml=" \
DFT_SELECT_BANK_OPTS="am=init0,initprev;amd=init0,initprev;amm=init0,initprev;aml=init0,initprev" \
DFT_SELECT_OPT_CFG="" \
DFT_SELECT_BANK_OPT_CFGS="am=${AM_CFG};amd=${AM_CFG};amm=${AM_CFG};aml=${AM_CFG}" \
DFT_SELECT_BANK_RAMPS="am=20;amd=20;amm=20;aml=20" \
DFT_SELECT_BANK_BLENDS="am=cubic;amd=cubic;amm=cubic;aml=cubic" \
DFT_SELECT_BANK_NEIGHBORS="am=cv;amd=wm_det;amm=wm_mf;aml=log" \
DFT_WM_DET_CKPT="${DFT_WM_DET_CKPT:-}" DFT_WM_MF_CKPT="${DFT_WM_MF_CKPT:-}" \
DFT_SIM_AGENTS="$SIM" DFT_DUMP_TRAJ=1 DFT_SELECT_SMOKE="$SMOKE" \
DFT_NUM_SHARDS="${DFT_NUM_SHARDS:-4}" DFT_MAX_CONCURRENT="${DFT_MAX_CONCURRENT:-4}" \
  bash rl/diffusion_ft/slurm/submit_select_eval.sh

AID="$(cat "$OUT/select/jobid.txt")"
CIDS=()
for pair in "o_amM:o_am" "o_amD:o_am" "o_amL:o_am" "o_amM:o_amD" "o_am:base" "o_amM:base"; do
  arm="${pair%%:*}"; ref="${pair##*:}"
  f="$OUT/cmp_${arm}_vs_${ref}_term.env"
  { kv DFT_CMP_A "$OUT/select"; kv DFT_CMP_B "$OUT/select"; kv DFT_SPLITS_JSON "$SPLITS3"
    kv DFT_CMP_ARM_A "$ref"; kv DFT_CMP_ARM_B "$arm"; kv DFT_CMP_LABEL_A "$ref"; kv DFT_CMP_LABEL_B "$arm"
    kv DFT_CMP_TERM 1; kv DFT_SELECT_REPEATS "$([[ -n "$SMOKE" ]] && echo 1 || echo 4)"
    kv DFT_CMP_OUT "$OUT/select/compare_${arm}_vs_${ref}_term.json"; } > "$f"
  CIDS+=("$(sb --parsable --dependency="afterok:${AID}" rl/diffusion_ft/slurm/compare_select.sbatch "$f")")
done
echo "compare -> jobs ${CIDS[*]} (CPU, afterok:$AID) -> $OUT/select/compare_*_term.json"
echo "all under $OUT; read select/select_summary_term.json and compare_*_term.json"
